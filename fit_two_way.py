"""
fit_two_way.py
==============

Keystone fit of the Mesoball spine. One pass over pa_panel.parquet produces
every object the downstream scripts need from the two-way wOBA model, so nothing
refits it:

  mu_three_schemes.csv     season run environment mu_S under three weighting
                           schemes -- shapenorm (HEADLINE), seasonw, fixedw. The
                           "weighting choice doesn't matter" robustness statement
                           (corr ~ 0.999) is read off this file.
  tau_batter.parquet       shape-norm two-way BATTER fixed effect, mean-anchored.
                           This is the single bat axis for the whole paper:
                           build_allocation_panel.py joins it on (it no longer
                           computes its own batter FE), so the bat that prices
                           fielding is the same bat behind mu_S.
  tau_pitcher.parquet      shape-norm two-way PITCHER fixed effect, mean-anchored.
  age_curves_twoway.parquet  alpha_bat(age), alpha_pit(age) from the headline fit,
                           anchored at age 28 (model component / diagnostic).
  park_resid_cache.parquet per-PA residual of the HEADLINE fit (season, batter,
                           pitcher, ages differenced out) plus the fields the
                           ballpark scripts cut on -- so ballpark_residuals.py
                           reads this instead of refitting.
  fit_two_way_summary.txt

Headline weighting is SHAPE-NORMALIZED: keep each season's event-weight ratios
(the strategic shape) and remove the wOBAScale level normalization (which would
scrub out the run environment we are measuring). All estimation is in wOBA; the
runs/wins layer in meso_core is reporting only and is not touched here.

The model, per scheme:
    woba_value ~ 1 | season + bat_id + pit_id + bat_age + pit_age
on the ex-IBB, ages-[18,45], >=2-season(both) sample, with singletons dropped so
the residual vector aligns to the rows.

Reads only pa_panel.parquet and wOBA_weights.csv (NOT Retrosheet). Run once,
BEFORE build_allocation_panel.py. Requires pyfixest.

    py fit_two_way.py                 # all three schemes (default)
    py fit_two_way.py --schemes shapenorm   # headline only, for quick iteration
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import pyfixest as pf

import meso_core as mc

ANCHOR_AGE = 28
SCHEMES_ALL = ["shapenorm", "seasonw", "fixedw"]
HEADLINE = "shapenorm"

# outputs
OUT_MU      = mc.OUTPUT_DIR / "mu_three_schemes.csv"
OUT_TAU_BAT = mc.DATA_DIR   / "tau_batter.parquet"
OUT_TAU_PIT = mc.DATA_DIR   / "tau_pitcher.parquet"
OUT_AGE     = mc.DATA_DIR   / "age_curves_twoway.parquet"
OUT_RESID   = mc.DATA_DIR   / "park_resid_cache.parquet"
OUT_SUM     = mc.OUTPUT_DIR / "fit_two_way_summary.txt"

L = []
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def P(s=""): print(s); L.append(s)


# ----------------------------------------------------------------------
# Weight schemes (all recomputed from the panel's per-PA event flags)
# ----------------------------------------------------------------------

def reference_event_mix(df_nonibb):
    """All-time PA-weighted per-PA event rates (ex-IBB), aligned to WEIGHT_COLS."""
    n = len(df_nonibb)
    f = {e: df_nonibb[e].sum() / n for e in mc.EVENT_COLS}
    return np.array([f[e] for e in mc.EVENT_COLS]), f


def fixed_weight_vector(w_seasons, pa_by_season):
    """PA-weighted average of the season weight vectors -> one fixed vector."""
    m = w_seasons.merge(pa_by_season, on="season", how="inner")
    tot = m["pa"].sum()
    return {c: float((m[c] * m["pa"]).sum() / tot) for c in mc.WEIGHT_COLS}


def shapenorm_scaling(w_seasons, f_ref_vec, pa_by_season):
    """Per-season k_s = V_bar / V_s, V_s = season weights . reference event mix."""
    w = w_seasons.copy()
    w["V_s"] = (w[mc.WEIGHT_COLS].to_numpy() * f_ref_vec).sum(axis=1)
    m = w.merge(pa_by_season, on="season", how="inner")
    V_bar = float((m["V_s"] * m["pa"]).sum() / m["pa"].sum())
    w["k_s"] = V_bar / w["V_s"]
    return w[["season", "k_s"]], V_bar


def woba_from_weights(df, wcols_by_season):
    """Dot per-PA events with per-row season weights; IBB -> NaN."""
    e = df[mc.EVENT_COLS].to_numpy(float)
    wv = (e * wcols_by_season).sum(axis=1)
    wv[df["iw"].to_numpy() == 1] = np.nan
    return wv


# ----------------------------------------------------------------------
# Fit + FE extraction
# ----------------------------------------------------------------------

def fit_scheme(df, ycol, want_all_fe):
    """Fit the two-way model on ycol. Return dict of mean/age-anchored FE series
    (always season; if want_all_fe also bat_id, pit_id, bat_age, pit_age) and,
    when want_all_fe, the per-row residual vector."""
    t0 = time.time()
    m = pf.feols(f"{ycol} ~ 1 | season + bat_id + pit_id + bat_age + pit_age", data=df)
    log(f"    fit {ycol} in {time.time()-t0:.1f}s")
    fe = m.fixef()
    def grab(key):
        k = next(kk for kk in fe if key in kk)
        s = pd.Series(fe[k]); s.index = s.index.astype(int if key.endswith("age") or key == "season" else str)
        return s
    out = {}
    mu = grab("season").sort_index(); out["season"] = mu - mu.mean()
    resid = None
    if want_all_fe:
        tb = grab("bat_id"); out["bat_id"] = tb - tb.mean()
        tp = grab("pit_id"); out["pit_id"] = tp - tp.mean()
        ab = grab("bat_age").sort_index()
        ap = grab("pit_age").sort_index()
        out["bat_age"] = ab - (ab.loc[ANCHOR_AGE] if ANCHOR_AGE in ab.index else ab.mean())
        out["pit_age"] = ap - (ap.loc[ANCHOR_AGE] if ANCHOR_AGE in ap.index else ap.mean())
        resid = np.asarray(m.resid())
    return out, resid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schemes", default="all",
                    help="'all' (default) or a comma list from shapenorm,seasonw,fixedw")
    args = ap.parse_args()
    schemes = SCHEMES_ALL if args.schemes == "all" else \
        [s.strip() for s in args.schemes.split(",")]
    if HEADLINE not in schemes:
        schemes = [HEADLINE] + schemes
    OUT_MU.parent.mkdir(parents=True, exist_ok=True)
    OUT_RESID.parent.mkdir(parents=True, exist_ok=True)

    # ---- load panel (native woba_value = seasonw) ----
    log(f"loading {mc.PANEL_PA}")
    df = pd.read_parquet(mc.PANEL_PA, columns=[
        "season", "bat_id", "pit_id", "iw", "bat_age", "pit_age",
        "single", "double", "triple", "hr", "walk", "hbp",
        "woba_value", "park_id", "bat_is_pitcher", "bat_home"])
    log(f"  {len(df):,} rows")
    df["bb_unint"] = (df["walk"] - df["iw"]).clip(lower=0)

    # ---- weight schemes, computed on the FULL ex-IBB panel ----
    nonibb = df[df["iw"] != 1]
    f_ref_vec, f_ref = reference_event_mix(nonibb)
    pa_by_season = nonibb.groupby("season").size().rename("pa").reset_index()
    w_seasons = (pd.read_csv(mc.WOBA_CSV, encoding="utf-8-sig")
                 .rename(columns={"Season": "season"})[["season"] + mc.WEIGHT_COLS])
    fixed = fixed_weight_vector(w_seasons, pa_by_season)
    ks, V_bar = shapenorm_scaling(w_seasons, f_ref_vec, pa_by_season)

    # per-row season weights for fixedw / shapenorm
    df = df.merge(w_seasons, on="season", how="left").merge(ks, on="season", how="left")
    if "seasonw" in schemes:
        df["w_seasonw"] = df["woba_value"]                         # native panel wOBA
    if "fixedw" in schemes:
        fw = np.array([fixed[c] for c in mc.WEIGHT_COLS])
        df["w_fixedw"] = woba_from_weights(df, fw[None, :])
    # headline shapenorm always computed
    sn_cols = df[mc.WEIGHT_COLS].to_numpy(float) * df["k_s"].to_numpy()[:, None]
    df["w_shapenorm"] = woba_from_weights(df, sn_cols)

    # ---- common sample filters (identical rows across schemes) ----
    ycols = {s: f"w_{s}" for s in schemes}
    need = list(ycols.values())
    df = df[df["iw"] != 1].dropna(subset=need + ["bat_age", "pit_age", "bat_id", "pit_id"])
    df["bat_age"] = df["bat_age"].astype(int); df["pit_age"] = df["pit_age"].astype(int)
    df["season"] = df["season"].astype(int)
    df = df[df["bat_age"].between(mc.TWO_WAY_AGE_LO, mc.TWO_WAY_AGE_HI)
            & df["pit_age"].between(mc.TWO_WAY_AGE_LO, mc.TWO_WAY_AGE_HI)]
    bn = df.groupby("bat_id")["season"].nunique(); pn = df.groupby("pit_id")["season"].nunique()
    df = df[df["bat_id"].isin(bn[bn >= 2].index) & df["pit_id"].isin(pn[pn >= 2].index)]
    df["bat_id"] = df["bat_id"].astype(str); df["pit_id"] = df["pit_id"].astype(str)
    df = mc.drop_singletons(df, ["season", "bat_id", "pit_id", "bat_age", "pit_age"])
    log(f"  {len(df):,} PA enter the fit; {df['bat_id'].nunique():,} batters, "
        f"{df['pit_id'].nunique():,} pitchers, {df['season'].nunique()} seasons")

    # ---- fit headline first (extract everything), then the others (mu only) ----
    P("FIT_TWO_WAY  (one keystone fit; headline = shape-normalized wOBA)")
    P("=" * 70)
    P(f"PA in fit: {len(df):,};  seasons {df['season'].min()}-{df['season'].max()};  "
      f"V_bar={V_bar:.5f}; k_s in [{ks['k_s'].min():.4f},{ks['k_s'].max():.4f}]")
    mu = {}
    headline_fe, resid = fit_scheme(df, ycols[HEADLINE], want_all_fe=True)
    mu[HEADLINE] = headline_fe["season"]
    for s in schemes:
        if s == HEADLINE:
            continue
        fe_s, _ = fit_scheme(df, ycols[s], want_all_fe=False)
        mu[s] = fe_s["season"]

    # ---- mu_three_schemes.csv (column order matches the existing file) ----
    three = pd.DataFrame({"season": mu[HEADLINE].index, "shapenorm": mu[HEADLINE].values})
    for s in ["seasonw", "fixedw"]:
        if s in mu:
            three = three.merge(
                pd.DataFrame({"season": mu[s].index, s: mu[s].values}),
                on="season", how="outer")
    three = three.sort_values("season")
    three.to_csv(OUT_MU, index=False, float_format="%.5f")
    log(f"wrote {OUT_MU}")

    # ---- tau_batter / tau_pitcher / age curves ----
    tb = headline_fe["bat_id"]; tp = headline_fe["pit_id"]
    pd.DataFrame({"bat_id": tb.index, "tau_bat": tb.values}).to_parquet(OUT_TAU_BAT, index=False)
    pd.DataFrame({"pit_id": tp.index, "tau_pit": tp.values}).to_parquet(OUT_TAU_PIT, index=False)
    ab = headline_fe["bat_age"]; apc = headline_fe["pit_age"]
    age_df = (pd.DataFrame({"age": ab.index, "alpha_bat": ab.values})
              .merge(pd.DataFrame({"age": apc.index, "alpha_pit": apc.values}),
                     on="age", how="outer").sort_values("age"))
    age_df.to_parquet(OUT_AGE, index=False)
    log(f"wrote {OUT_TAU_BAT.name}, {OUT_TAU_PIT.name}, {OUT_AGE.name}")

    # ---- per-PA residual cache for the ballpark scripts ----
    assert len(resid) == len(df), "residual vector misaligned with sample"
    cache = df[["park_id", "season", "bat_id", "bat_is_pitcher",
                "double", "triple", "hr", "bat_home"]].copy()
    cache["resid"] = resid
    cache.to_parquet(OUT_RESID, index=False)
    log(f"wrote {OUT_RESID.name} ({len(cache):,} rows)")

    # ---- summary / key numbers (for diffing against the draft) ----
    tau_sd = float(tb.std())
    P(f"\ntau_bat: n={len(tb):,}  SD(all batters)={tau_sd:.4f} wOBA  "
      f"(regulars' SD reported by build_allocation_panel)")
    P(f"alpha_bat age curve (anchored 28): "
      f"22={ab.get(22, np.nan):+.4f}  32={ab.get(32, np.nan):+.4f}  38={ab.get(38, np.nan):+.4f}")
    P("\n--- mu_S, headline shape-norm, key seasons ---")
    msn = mu[HEADLINE]
    for yr in [1930, 1968, 1969, 2000, 2024]:
        if yr in msn.index:
            P(f"  {yr}: {msn.loc[yr]:+.4f}")
    P(f"  range: [{msn.min():+.4f} ({int(msn.idxmin())}), {msn.max():+.4f} ({int(msn.idxmax())})]")
    if len(mu) > 1:
        P("\n--- scheme correlations (the 'weighting doesn't matter' check) ---")
        cmp = three.dropna()
        for s in ["seasonw", "fixedw"]:
            if s in cmp.columns:
                P(f"  corr(shapenorm, {s}) = {cmp['shapenorm'].corr(cmp[s]):+.4f}")
    OUT_SUM.write_text("\n".join(L), encoding="utf-8")
    log(f"wrote {OUT_SUM}")


if __name__ == "__main__":
    main()
