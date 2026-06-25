"""
ballpark_residuals.py
=====================

Ballparks, the right way: NOT a park term in the skill model (that would launder
real run-environment out of tau_bat and out of mu_S), but a look at the RESIDUALS
of the main two-way model, segregated by park. Because the residual is taken after
tau_bat, tau_pit, season and age are removed, a park's mean residual is already a
WITHIN-PLAYER, quality-adjusted park factor -- the players have been differenced
out, so it is cleaner than a home/road park factor.

Two hypotheses, two homes:
  - "rewards/penalizes slugging" lives in the wOBA RESIDUAL: a power park lifts a
    power hitter above his linear-weights prediction. This script does that (stage 1).
  - "rewards skill at a defensive position" is a FIELDING claim with no footprint in
    the batting residual; it lives in the ALLOCATION residual (PA at matched bat).
    This script does NOT test it -- it builds the bridge (a per-park profile) that a
    follow-on allocation script consumes (stage 2).

The endogeneity guard. A hitter's slugging is partly PRODUCED by his parks, so we
never bucket on observed slugging. We bucket on a park-exogenous power measure:
era-adjusted career isolated power, pooled over all the parks a hitter played in.
(For the rare single-park career this still leaks; such a hitter's park is also
partly absorbed into his own tau_bat, which only ATTENUATES the park residual --
the safe direction.) Park means are empirical-Bayes shrunk by PA count so a tiny
park cannot post an extreme factor on noise. Identity is Retrosheet park_id: a
redesign under one id blurs configs together (attenuates), a rename splits one park
into two ids (costs power) -- both bias toward null, the safe direction.

Reads pa_panel.parquet and pa_allocation_panel.parquet (for position). Fits the
two-way model once (pyfixest), caches per-PA residuals. Writes a park profile CSV.
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyfixest as pf

PROJECT_DIR = Path(r"C:\baseball_eras")
PANEL   = PROJECT_DIR / "data"   / "pa_panel.parquet"
ALLOC   = PROJECT_DIR / "data"   / "pa_allocation_panel.parquet"
RESID_CACHE = PROJECT_DIR / "data"   / "park_resid_cache.parquet"
OUT_PROFILE = PROJECT_DIR / "output" / "park_profile.csv"
OUT_SUM     = PROJECT_DIR / "output" / "ballpark_residuals_summary.txt"

AGE_LO, AGE_HI = 18, 45
MIN_PARK_PA = 1000     # parks below this are reported but flagged thin
L = []
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def P(s=""): print(s); L.append(s)


def drop_singletons(df: pd.DataFrame, keys) -> pd.DataFrame:
    """Iteratively drop rows unique in any FE key so the estimation sample == df
    and pyfixest's residual vector aligns row-for-row."""
    while True:
        before = len(df)
        for k in keys:
            vc = df[k].value_counts()
            df = df[df[k].isin(vc[vc >= 2].index)]
        if len(df) == before:
            return df.reset_index(drop=True)


def eb_shrink(mean, n, sigma2):
    """Empirical-Bayes shrink group means toward the PA-weighted grand mean.
    sigma2 = PA-level residual variance; sampling var of a group mean ~ sigma2/n."""
    n = np.asarray(n, float); mean = np.asarray(mean, float)
    gm = np.sum(n * mean) / np.sum(n)
    var_obs = np.sum(n * (mean - gm) ** 2) / np.sum(n)
    mean_samp = len(n) * sigma2 / np.sum(n)
    sig2_b = max(var_obs - mean_samp, 1e-12)
    K = sigma2 / sig2_b
    rho = n / (n + K)
    return gm + rho * (mean - gm), K, np.sqrt(sig2_b)


def build_resid_cache() -> pd.DataFrame:
    if RESID_CACHE.exists():
        log(f"loading cached residuals {RESID_CACHE.name} (delete to recompute)")
        return pd.read_parquet(RESID_CACHE)

    log(f"loading {PANEL}")
    df = pd.read_parquet(PANEL, columns=[
        "season", "bat_id", "pit_id", "woba_value", "iw", "bat_age", "pit_age",
        "park_id", "bat_is_pitcher", "double", "triple", "hr"])
    df = df[df["iw"] != 1].dropna(subset=["woba_value", "bat_age", "pit_age",
                                          "bat_id", "pit_id"])
    df["bat_age"] = df["bat_age"].astype(int); df["pit_age"] = df["pit_age"].astype(int)
    df["season"] = df["season"].astype(int)
    df = df[(df["bat_age"].between(AGE_LO, AGE_HI)) & (df["pit_age"].between(AGE_LO, AGE_HI))]

    # park-exogenous power: era-adjusted career isolated power (extra bases / PA),
    # pooled over all parks. ISO numerator per PA = 2B + 2*3B + 3*HR.
    df["iso_num"] = df["double"].fillna(0) + 2 * df["triple"].fillna(0) + 3 * df["hr"].fillna(0)
    seas_iso = df.groupby("season")["iso_num"].transform("mean")
    df["iso_adj"] = df["iso_num"] - seas_iso                       # era-detrended
    pw = df.groupby("bat_id")["iso_adj"].mean()                    # career power level
    df["bat_pow"] = df["bat_id"].map(pw)

    # canonical two-way fit (matches two_way_pyfixest; pitcher-batters kept in the
    # fit so the residual definition is identical, then dropped from park cuts).
    for c in ["season", "bat_id", "pit_id", "bat_age", "pit_age"]:
        df[c] = df[c].astype(str) if c in ("bat_id", "pit_id") else df[c].astype(int)
    bn = df.groupby("bat_id")["season"].nunique(); pn = df.groupby("pit_id")["season"].nunique()
    df = df[df["bat_id"].isin(bn[bn >= 2].index) & df["pit_id"].isin(pn[pn >= 2].index)]
    df = drop_singletons(df, ["season", "bat_id", "pit_id", "bat_age", "pit_age"])
    log(f"  fitting two-way on {len(df):,} PA")
    t0 = time.time()
    m = pf.feols("woba_value ~ 1 | season + bat_id + pit_id + bat_age + pit_age", data=df)
    log(f"  fit in {(time.time()-t0)/60:.1f} min; extracting residuals")
    df["resid"] = np.asarray(m.resid())

    # attach primary position (from allocation panel) by batter-season
    pos = pd.read_parquet(ALLOC, columns=["batter", "season", "pos"])
    pos["batter"] = pos["batter"].astype(str); pos["season"] = pos["season"].astype(int)
    df["season_i"] = df["season"].astype(int)
    df = df.merge(pos.rename(columns={"batter": "bat_id", "season": "season_i"}),
                  on=["bat_id", "season_i"], how="left")

    out = df[["park_id", "season_i", "resid", "bat_pow", "pos", "bat_is_pitcher",
              "hr", "triple"]].rename(columns={"season_i": "season"})
    out.to_parquet(RESID_CACHE, index=False)
    log(f"  cached {RESID_CACHE.name} ({len(out):,} rows)")
    return out


def main():
    OUT_SUM.parent.mkdir(parents=True, exist_ok=True)
    r = build_resid_cache()

    # park cuts on POSITION PLAYERS only (pitchers hitting are not the question)
    r = r[(r["bat_is_pitcher"] != 1)].dropna(subset=["park_id", "resid", "bat_pow"])
    r["park_id"] = r["park_id"].astype(str)
    sigma2 = float(r["resid"].var())
    r["pow_q"] = pd.qcut(r["bat_pow"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])

    P("BALLPARK RESIDUAL PROFILES  (residuals of the two-way wOBA model)")
    P("=" * 70)
    P(f"{len(r):,} position-player PAs; {r['park_id'].nunique():,} parks; "
      f"per-PA residual SD {np.sqrt(sigma2):.3f}")
    P("A park's mean residual = quality-adjusted run factor (players differenced out).")

    # ---- stage 1a: overall park run factor (EB-shrunk) ----
    g = r.groupby("park_id")["resid"].agg(["mean", "size"]).reset_index()
    g.columns = ["park_id", "raw", "n"]
    g["factor"], K_park, sd_park = eb_shrink(g["raw"], g["n"], sigma2)
    P(f"\n--- park run factor (EB-shrunk; K={K_park:,.0f} PA, between-park SD={sd_park:.4f} wOBA) ---")
    P(f"  parks vary by ~{sd_park*100:.1f} wOBA points (1 SD) after removing players.")
    big = g[g["n"] >= MIN_PARK_PA].sort_values("factor")
    P("  lowest (pitcher-friendly):  " +
      ", ".join(f"{x.park_id}{x.factor:+.3f}" for x in big.head(8).itertuples()))
    P("  highest (hitter-friendly):  " +
      ", ".join(f"{x.park_id}{x.factor:+.3f}" for x in big.tail(8).iloc[::-1].itertuples()))

    # ---- stage 1b: does the park REWARD POWER? (the slugging hypothesis) ----
    # within each park, slope of residual on park-exogenous power, EB-shrunk.
    pw_c = r["bat_pow"] - r["bat_pow"].mean()
    r2 = r.assign(pw_c=pw_c, pw_c2=pw_c ** 2, rp=r["resid"] * pw_c)
    sl = r2.groupby("park_id").agg(Spp=("pw_c2", "sum"), Srp=("rp", "sum"),
                                   n=("resid", "size")).reset_index()
    sl = sl[sl["Spp"] > 0]
    sl["slope_raw"] = sl["Srp"] / sl["Spp"]
    # sampling var of slope ~ sigma2 / Spp; shrink slopes toward 0-mean
    sl_eb, K_sl, sd_sl = eb_shrink(sl["slope_raw"], sl["Spp"], sigma2)
    sl["power_slope"] = sl_eb
    P(f"\n--- park POWER-reward slope (resid per unit park-exogenous power; "
      f"EB K={K_sl:,.0f}) ---")
    P("  positive = the park lifts power hitters above their linear-weights wOBA.")
    big_sl = sl[sl["n"] >= MIN_PARK_PA].sort_values("power_slope")
    P("  most power-SUPPRESSING: " +
      ", ".join(f"{x.park_id}{x.power_slope:+.2f}" for x in big_sl.head(8).itertuples()))
    P("  most power-REWARDING:   " +
      ", ".join(f"{x.park_id}{x.power_slope:+.2f}" for x in big_sl.tail(8).iloc[::-1].itertuples()))

    # ---- residual segregated by position x power (what you asked to see) ----
    P("\n--- mean residual by position x power quartile (x1000 wOBA) ---")
    pos_order = ["C", "SS", "2B", "3B", "CF", "CORNER_OF", "1B", "DH"]
    cell = (r.dropna(subset=["pos"]).groupby(["pos", "pow_q"], observed=True)["resid"]
              .mean().mul(1000).unstack())
    cell = cell.reindex([p for p in pos_order if p in cell.index])
    P(cell.to_string(float_format=lambda x: f"{x:+.0f}"))
    P("  (In the BATTING residual, position mostly proxies power composition; the")
    P("   genuine 'park rewards fielding at a position' question is the allocation")
    P("   residual -- stage 2, which consumes the park profile written below.)")

    # ---- bridge artifact: per-park profile for the allocation (position) stage ----
    # triples factor = park 3B rate relative to the CONTEMPORANEOUS league rate
    # (era-normalized), so it proxies park configuration/outfield size rather than
    # the deadball era's leaguewide abundance of triples. NOTE: still not adjusted
    # for home-roster speed; a visitor-only version would isolate the park further
    # (needs bat_home in the cache, i.e. one more fit) -- defer unless stage 2 bites.
    lg_tr = r.groupby("season")["triple"].transform("mean").replace(0, np.nan)
    r = r.assign(tr_rel=r["triple"] / lg_tr)
    tr = r.groupby("park_id").agg(triples_factor=("tr_rel", "mean"),
                                  n=("resid", "size")).reset_index()
    prof = (g[["park_id", "n", "factor"]]
            .merge(sl[["park_id", "power_slope"]], on="park_id", how="left")
            .merge(tr[["park_id", "triples_factor"]], on="park_id", how="left")
            .rename(columns={"factor": "run_factor"}))
    prof.to_csv(OUT_PROFILE, index=False, float_format="%.4f")
    log(f"wrote {OUT_PROFILE}")

    big_prof = prof[prof["n"] >= MIN_PARK_PA]
    P(f"\n--- park profile written ({len(prof):,} parks) ---")
    P(f"  corr(run_factor, power_slope)   = {big_prof['run_factor'].corr(big_prof['power_slope']):+.2f}  "
      "(hitter parks tend to reward power)")
    P(f"  corr(run_factor, triples_factor)= {big_prof['run_factor'].corr(big_prof['triples_factor']):+.2f}")
    P(f"  corr(power_slope, triples_factor)= {big_prof['power_slope'].corr(big_prof['triples_factor']):+.2f}  "
      "(power parks vs big-outfield parks: expect NEGATIVE if distinct)")
    P("")
    P("Stage 2 (separate script): attach each batter-season's home park profile to")
    P("the allocation panel and run the censored-share Tobit with position x park")
    P("interactions -- e.g. is the CF allocation premium larger in high-triples")
    P("parks. If park profiles are symmetric across positions and uncorrelated with")
    P("the allocation residual, park is 'effectively random' for the structural model")
    P("and the exclusion is justified; if the triples/CF interaction is real, it is a")
    P("result. Build that once we see whether triples_factor has usable spread.")

    OUT_SUM.write_text("\n".join(L), encoding="utf-8")
    log(f"wrote {OUT_SUM}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
