"""
ballpark_allocation_stage2.py
=============================

Stage 2 of the ballpark analysis: does park CONFIGURATION change how playing time
is allocated across defensive positions? The batting residual (stage 1) cannot see
fielding; the ALLOCATION residual can -- it is the extra PA a glove position draws
at matched bat. We ask whether that price moves with the park's outfield size.

Design (the positional censored-share Tobit, from meso_core, plus park terms):

    share* = a + b*tau_bat + (position) + (position x triples_factor)
                + (position x run_factor)            <- control for hitter-friendliness
                + age + season + e          (right-censored at 0.90 x full-time)

triples_factor (era-normalized) proxies OUTFIELD size. Directed prediction: the
CENTER-FIELD premium rises with triples (big yards buy outfield range) while the
INFIELD premia (SS, 2B) do not -- that asymmetry is the identification:
  - CF responds, infield flat -> parks reward outfield range (a result; park is
    NOT effectively random for fielding allocation).
  - everything responds together -> generic roster-quality confound.
  - nothing responds -> park is effectively random; the structural exclusion holds.

Caveat: triples_factor still reflects home-roster speed, not park size alone. A
visitor-only size factor (the residual cache already carries bat_home) is the
hardening step IF center field bites -- deferred until then.

Reads pa_allocation_panel.parquet, the unified pa_team_share.parquet, park_profile.csv,
and pa_panel.parquet (home park). Requires scipy.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import meso_core as mc

HOMEPARK = mc.DATA_DIR  / "home_park_by_batter_season.parquet"
PROFILE  = mc.OUTPUT_DIR / "park_profile.csv"
OUT_SUM  = mc.OUTPUT_DIR / "ballpark_allocation_stage2_summary.txt"
MIN_PARK_PA = 1000

L = []
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def P(s=""): print(s); L.append(s)


def home_park_by_batter_season() -> pd.DataFrame:
    """Modal park among a batter's home-team PAs each season ~ his home park."""
    if HOMEPARK.exists():
        return pd.read_parquet(HOMEPARK)
    log("deriving home park per batter-season from pa_panel")
    d = pd.read_parquet(mc.PANEL_PA, columns=["bat_id", "season", "park_id", "bat_home"])
    d = d[d["bat_home"] == 1].dropna(subset=["park_id"])
    d["season"] = d["season"].astype(int)
    g = (d.groupby(["bat_id", "season", "park_id"]).size().reset_index(name="n")
           .sort_values("n"))
    hp = (g.drop_duplicates(["bat_id", "season"], keep="last")[["bat_id", "season", "park_id"]]
            .rename(columns={"bat_id": "batter"}))
    hp["batter"] = hp["batter"].astype(str)
    hp.to_parquet(HOMEPARK, index=False)
    log(f"  {len(hp):,} batter-seasons with a home park")
    return hp


def main():
    OUT_SUM.parent.mkdir(parents=True, exist_ok=True)

    share = pd.read_parquet(mc.SHARE_CACHE)
    share["batter"] = share["batter"].astype(str); share["season"] = share["season"].astype(int)
    PA_TEAM_MEAN = float(share["pa_team_mean"].iloc[0])
    ps = pd.read_parquet(mc.PANEL_ALLOC)
    ps["batter"] = ps["batter"].astype(str); ps["season"] = ps["season"].astype(int)
    ps = ps.merge(share[["batter", "season", "pa_share"]], on=["batter", "season"], how="inner")

    prof = pd.read_csv(PROFILE)
    prof = prof[prof["n"] >= MIN_PARK_PA]; prof["park_id"] = prof["park_id"].astype(str)
    hp = home_park_by_batter_season()
    ps = ps.merge(hp, on=["batter", "season"], how="left")
    ps = ps.merge(prof[["park_id", "run_factor", "triples_factor"]], on="park_id", how="inner")
    ps = ps.reset_index(drop=True)

    full_ref = float(ps["pa_share"].quantile(0.99)); c = mc.HEADLINE_FRAC * full_ref
    to_pct = 100.0 / full_ref
    tri = ps["triples_factor"].to_numpy(float); tri_c = tri - tri.mean(); sd_tri = tri.std()
    run = ps["run_factor"].to_numpy(float); run_c = run - run.mean()

    P("BALLPARK STAGE 2: park configuration and position allocation")
    P("=" * 70)
    P(f"matched {len(ps):,} batter-seasons to a home-park profile; "
      f"{ps['park_id'].nunique()} parks; mean team PA {PA_TEAM_MEAN:,.0f}")
    P(f"full-time ref share={full_ref:.4f}; SD(triples_factor)={sd_tri:.3f}")
    P("Interaction reported as % of full-time per +1 SD of triples_factor.")
    P("Directed test: CF (outfield) should respond; SS/2B (infield) are the placebo.")

    posv = ps["pos"].astype(str).to_numpy()
    age_c = ps["age_c"].to_numpy(float); season_c = ps["season"].to_numpy(float) - 1970.0
    y = ps["pa_share"].to_numpy(float); cl = ps["batter"].to_numpy()
    base = [("const", np.ones(len(ps))), ("tau_bat", ps["tau_bat"].to_numpy(float))]
    pos_cols = [(f"pos_{p}", (posv == p).astype(float)) for p in mc.POS]
    tri_int = [(f"pos_{p}_tri", (posv == p).astype(float) * tri_c) for p in mc.POS]
    run_int = [(f"pos_{p}_run", (posv == p).astype(float) * run_c) for p in mc.POS]
    ctrl = [("tri_main", tri_c), ("run_main", run_c),
            ("age_c", age_c), ("age_c2", age_c ** 2),
            ("season_c", season_c), ("season_c2", season_c ** 2)]
    X, names = mc.design(base + pos_cols + tri_int + run_int + ctrl)
    res, V, cf = mc.fit_tobit(y, X, c, cl)
    P(f"\nfit: {cf*100:.1f}% censored\n")

    P("  position premium at an AVERAGE park, and its shift per +1 SD of outfield size")
    P(f"  {'pos':>10s} {'prem(%FT)':>10s} {'x triples':>11s} {'t':>7s}")
    for p in mc.POS:
        e0, _ = mc.lincom({f"pos_{p}": 1.0}, names, res.x, V)
        ei, sei = mc.lincom({f"pos_{p}_tri": 1.0}, names, res.x, V)
        tag = "  <- CF (treatment)" if p == "CF" else (
              "  <- infield placebo" if p in ("SS", "2B") else "")
        P(f"  {p:>10s} {e0*to_pct:>10.1f} {ei*sd_tri*to_pct:>+11.1f} {ei/sei:>7.1f}{tag}")

    P("")
    P("Read CF vs SS/2B: CF 'x triples' positive & significant with SS/2B ~0 means")
    P("big-outfield parks buy center-field range (a fielding-allocation result). If CF")
    P("is also ~0, park is effectively random for allocation and the structural model")
    P("excludes it cleanly. run_factor interactions are partialled out, so this is not")
    P("a hitter-park artifact.")
    OUT_SUM.write_text("\n".join(L), encoding="utf-8")
    log(f"wrote {OUT_SUM}")


if __name__ == "__main__":
    main()
