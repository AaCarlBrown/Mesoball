"""
dispersion_enriched.py  (exploratory; companion to talent_dispersion / pitcher_dispersion)
=========================================================================================

Does conditioning the talent measure on PARK, the batter x pitcher HANDEDNESS matchup,
and LEAGUE -- things only micro data can do -- change the cross-era dispersion of
batting and pitching talent? Expected: a small, clean reduction (purging park strips a
Coors hitter's thin-air boost out of his tau), biting hardest on extreme-park and
atypical-handedness players, and possibly in the DH era.

For each decade we fit TWO two-way models on the same data and compare:
  baseline : ycol ~ 1 | season + bat_id + pit_id + bat_age + pit_age
  enriched : baseline + park_id + same_hand
and report batter and pitcher true (noise-corrected, PA-weighted) dispersion from each,
in win-equivalent units (tau / wOBAScale * PA_ref / runs_per_win), by decade.

same_hand = 1 if the batter faces a same-handed pitcher (the platoon INTERACTION; a
6-level bat_hand x pit_hand factor would be collinear with the player FEs -- see
add_handedness). League is intentionally NOT a separate term: a ballpark sits in one
league, so park_id already absorbs league-level differences, and including both makes
the fixed-effect system collinear. Two fits per decade, ~2x pitcher_dispersion.py.

Run from C:\\baseball_eras\\github :  py dispersion_enriched.py
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import pyfixest as pf

import meso_core as mc
import talent_dispersion as td      # reuse prep_woba

DECADES = list(range(1910, 2030, 10))
MIN_PA = 1000                       # regular bar, same for batters and pitchers
SCHEMES = ["shapenorm"]             # add "fixedw" for the robustness pass if wanted
PA_REF = 600.0                      # constant scalar -> "wins per 600 PA"
WEIGHTS = next((p for p in [mc.PROJECT_DIR / "wOBA_weights.csv", mc.PROJECT_DIR / "data" / "wOBA_weights.csv"]
                if p.exists()), mc.PROJECT_DIR / "wOBA_weights.csv")
OUT_CSV = mc.OUTPUT_DIR / "dispersion_enriched.csv"
OUT_SUM = mc.OUTPUT_DIR / "dispersion_enriched_summary.txt"
L = []
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def P(s=""): print(s); L.append(s)


def add_handedness(df):
    """The platoon effect is the bat_hand x pit_hand INTERACTION, not a 6-level
    matchup factor: bat_id already absorbs the batter-hand main effect (a batter's
    hand is constant within his id) and pit_id absorbs the pitcher-hand main effect,
    so a full matchup factor is collinear with the player FEs and the demeaner fails.
    The interaction is a single binary -- same- vs opposite-handed -- which varies
    within both ids and is clean. Switch hitters bat opposite, so same_hand = 0."""
    bh = df["bat_hand"].astype(str).str[0].str.upper()
    ph = df["pit_hand"].astype(str).str[0].str.upper()
    same = (bh.isin(["L", "R"])) & (bh == ph)          # S/B (switch) -> opposite -> 0
    df["same_hand"] = same.astype(int)
    df["park_id"] = df["park_id"].astype(str).fillna("UNK")
    return df


def _true_sd(tau, n_i, sigma2, reg_idx):
    tau_r = tau.reindex(reg_idx).dropna(); n_r = n_i.reindex(tau_r.index)
    if len(tau_r) < 2:
        return np.nan, len(tau_r)
    w = n_r.to_numpy(float); t = tau_r.to_numpy(float)
    mu = np.average(t, weights=w)
    raw_var = np.average((t - mu) ** 2, weights=w)
    samp_var = np.average(sigma2 / n_r.to_numpy(float), weights=w)
    return np.sqrt(max(raw_var - samp_var, 0.0)), len(tau_r)


def fit_decade(sub, ycol, enriched):
    rhs = "season + bat_id + pit_id + bat_age + pit_age"
    if enriched:
        rhs += " + park_id + same_hand"
    keep = ["bat_id", "pit_id", "season", "bat_age", "pit_age"]
    if enriched:
        keep += ["park_id", "same_hand"]
    s = sub.dropna(subset=[ycol, "bat_age", "pit_age", "bat_id", "pit_id"]).copy()
    s["bat_age"] = s["bat_age"].astype(int); s["pit_age"] = s["pit_age"].astype(int)
    s = s[s["bat_age"].between(mc.TWO_WAY_AGE_LO, mc.TWO_WAY_AGE_HI)
          & s["pit_age"].between(mc.TWO_WAY_AGE_LO, mc.TWO_WAY_AGE_HI)]
    for c in ["bat_id", "pit_id", "season"] + (["park_id", "same_hand"] if enriched else []):
        s[c] = s[c].astype(str)
    s = mc.drop_singletons(s, keep)
    if s["bat_id"].nunique() < 50 or s["pit_id"].nunique() < 50:
        return None
    m = pf.feols(f"{ycol} ~ 1 | {rhs}", data=s)
    fe = m.fixef(); sigma2 = float(np.var(np.asarray(m.resid())))
    bkey = next(k for k in fe if "bat_id" in k); pkey = next(k for k in fe if "pit_id" in k)
    tb = pd.Series(fe[bkey]); tb.index = tb.index.astype(str)
    tp = pd.Series(fe[pkey]); tp.index = tp.index.astype(str)
    nb = s.groupby("bat_id").size(); npi = s.groupby("pit_id").size()
    b_sd, b_n = _true_sd(tb, nb, sigma2, nb[nb >= MIN_PA].index)
    p_sd, p_n = _true_sd(tp, npi, sigma2, npi[npi >= MIN_PA].index)
    return dict(b_sd=b_sd, b_n=b_n, p_sd=p_sd, p_n=p_n, tb=tb, nb=nb)


def main():
    OUT_SUM.parent.mkdir(parents=True, exist_ok=True)
    w = pd.read_csv(WEIGHTS); w.columns = [c.strip() for c in w.columns]
    w["decade"] = w["Season"] // 10 * 10
    conv = w.groupby("decade").agg(RPW=("R/W", "mean"), ws=("wOBAScale", "mean"))

    log(f"loading {mc.PANEL_PA}")
    df = pd.read_parquet(mc.PANEL_PA, columns=[
        "season", "bat_id", "pit_id", "iw", "bat_age", "pit_age", "park_id",
        "bat_hand", "pit_hand",
        "single", "double", "triple", "hr", "walk", "hbp"])
    df["season"] = df["season"].astype(int)
    df = df[df["iw"] != 1]
    df = td.prep_woba(df)
    df = add_handedness(df)

    def to_win(sd, dec):
        if dec not in conv.index or pd.isna(sd):
            return np.nan
        return sd / conv.loc[dec, "ws"] * PA_REF / conv.loc[dec, "RPW"]

    P("ENRICHED DISPERSION: baseline vs park+handedness+league-purged (win units/600 PA)")
    P("=" * 84)
    rows = []
    for scheme in SCHEMES:
        ycol = f"w_{scheme}"
        P(f"\n--- {scheme} ---")
        P(f"  {'dec':>5} | {'bat base':>8} {'bat enr':>8} {'Δ':>6} | "
          f"{'pit base':>8} {'pit enr':>8} {'Δ':>6} | {'corr':>5} {'moveSD':>7}")
        for d in DECADES:
            sub = df[(df["season"] // 10 * 10) == d]
            if len(sub) < 5000:
                continue
            t0 = time.time()
            base = fit_decade(sub, ycol, enriched=False)
            enr = fit_decade(sub, ycol, enriched=True)
            if not base or not enr:
                continue
            bb, be = to_win(base["b_sd"], d), to_win(enr["b_sd"], d)
            pb, pe = to_win(base["p_sd"], d), to_win(enr["p_sd"], d)
            # how much do individual batter taus move when we enrich?
            j = pd.concat([base["tb"].rename("base"), enr["tb"].rename("enr")], axis=1).dropna()
            reg = base["nb"][base["nb"] >= MIN_PA].index
            j = j.reindex(reg).dropna()
            corr = j["base"].corr(j["enr"]) if len(j) > 2 else np.nan
            move_sd = to_win((j["enr"] - j["base"]).std(), d) if len(j) > 2 else np.nan
            log(f"    {d}s fit pair in {time.time()-t0:.0f}s")
            P(f"  {d:>4}s | {bb:>8.3f} {be:>8.3f} {be-bb:>+6.3f} | "
              f"{pb:>8.3f} {pe:>8.3f} {pe-pb:>+6.3f} | {corr:>5.3f} {move_sd:>7.3f}")
            rows.append(dict(scheme=scheme, decade=d, bat_base=bb, bat_enr=be,
                             pit_base=pb, pit_enr=pe, tau_corr=corr, move_sd_wins=move_sd,
                             nB=enr["b_n"], nP=enr["p_n"]))
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False, float_format="%.5f")
    if len(res):
        P(f"\n  mean change bat {(res['bat_enr']-res['bat_base']).mean():+.4f}, "
          f"pit {(res['pit_enr']-res['pit_base']).mean():+.4f} wins/600PA")
        P(f"  median batter tau correlation base-vs-enriched: {res['tau_corr'].median():.4f}")
        dh = res[res["decade"] >= 1970]
        P(f"  DH-era (1970+) mean bat change: {(dh['bat_enr']-dh['bat_base']).mean():+.4f}")
    OUT_SUM.write_text("\n".join(L), encoding="utf-8")
    log(f"wrote {OUT_CSV}, {OUT_SUM}")


if __name__ == "__main__":
    main()
