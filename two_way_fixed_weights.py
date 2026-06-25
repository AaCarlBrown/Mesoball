"""
two_way_fixed_weights.py
========================

Robustness check: refit the two-way (and per-role) season models using a
SINGLE FIXED wOBA weight vector instead of FanGraphs' season-specific
weights.

Rationale: season-specific weights (with wOBAScale) are designed to make
wOBA comparable RELATIVE TO each season's league average -- i.e. they
deliberately scrub out year-to-year run-environment differences. Since
mu_S is meant to MEASURE the run environment, using season-specific
weights partly defines away the dependent variable. A fixed weight vector
scores every PA on one consistent yardstick, so mu_S reflects the actual
change in the rate and mix of offensive events across seasons.

Fixed weights used here: the PA-weighted average of the season weight
vectors across the panel's coverage. (PA weighting so the average reflects
the seasons that actually contribute data, not ancient low-PA seasons.)

The panel already stores per-PA event flags (single, double, triple, hr,
walk, iw, hbp), so we recompute woba_value WITHOUT touching retrosheet:

    woba_value = wBB*(walk - iw) + wHBP*hbp + w1B*single
               + w2B*double + w3B*triple + wHR*hr
    (IBBs: woba_value set to NaN, excluded -- FanGraphs convention)

Then run the same three fits as two_way_pyfixest.py.

Inputs:
    C:\\baseball_eras\\data\\pa_panel.parquet      (must have event flags + iw)
    C:\\baseball_eras\\wOBA_weights.csv

Outputs:
    C:\\baseball_eras\\data\\mu_two_way_fixedw.parquet
    C:\\baseball_eras\\data\\mu_bat_only_fixedw.parquet
    C:\\baseball_eras\\data\\mu_pit_only_fixedw.parquet
    C:\\baseball_eras\\output\\mu_compare_fixedw.csv
    C:\\baseball_eras\\output\\mu_seasonw_vs_fixedw.csv   (side-by-side w/ the season-weight fit)
    C:\\baseball_eras\\output\\two_way_fixed_weights_summary.txt

Requires: pip install pyfixest
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf


PROJECT_DIR = Path(r"C:\baseball_eras")
PANEL_PATH = PROJECT_DIR / "data" / "pa_panel.parquet"
WOBA_CSV   = PROJECT_DIR / "wOBA_weights.csv"

# Existing season-weight fit (for side-by-side); optional
MU_SEASONW = PROJECT_DIR / "data" / "mu_two_way.parquet"

OUT_MU_TW   = PROJECT_DIR / "data"   / "mu_two_way_fixedw.parquet"
OUT_MU_BAT  = PROJECT_DIR / "data"   / "mu_bat_only_fixedw.parquet"
OUT_MU_PIT  = PROJECT_DIR / "data"   / "mu_pit_only_fixedw.parquet"
OUT_COMPARE = PROJECT_DIR / "output" / "mu_compare_fixedw.csv"
OUT_SIDE    = PROJECT_DIR / "output" / "mu_seasonw_vs_fixedw.csv"
OUT_SUMMARY = PROJECT_DIR / "output" / "two_way_fixed_weights_summary.txt"

AGE_LO, AGE_HI = 18, 45
ANCHOR_AGE = 28

WEIGHT_COLS = ["wBB", "wHBP", "w1B", "w2B", "w3B", "wHR"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_fixed_weights(panel: pd.DataFrame) -> dict:
    """
    PA-weighted average of season weight vectors, where the PA weight for
    each season is the number of (non-IBB) PAs that season contributes to
    the panel.
    """
    log("computing PA-weighted-average fixed weight vector")
    w = pd.read_csv(WOBA_CSV, encoding="utf-8-sig").rename(columns={"Season": "season"})
    w = w[["season"] + WEIGHT_COLS].copy()

    pa_by_season = (
        panel[panel["iw"] != 1]
        .groupby("season").size().rename("pa").reset_index()
    )
    w = w.merge(pa_by_season, on="season", how="inner")
    log(f"  {len(w)} seasons overlap between weights file and panel "
        f"({w.season.min()}-{w.season.max()})")

    fixed = {}
    total_pa = w["pa"].sum()
    for c in WEIGHT_COLS:
        fixed[c] = float((w[c] * w["pa"]).sum() / total_pa)
    log("  fixed weights (PA-weighted average):")
    for c in WEIGHT_COLS:
        log(f"    {c} = {fixed[c]:.4f}")
    return fixed


def load_and_reweight(fixed: dict) -> pd.DataFrame:
    log(f"loading {PANEL_PATH}")
    df = pd.read_parquet(PANEL_PATH, columns=[
        "season", "bat_id", "pit_id", "iw",
        "bat_age", "pit_age",
        "single", "double", "triple", "hr", "walk", "hbp",
    ])
    log(f"  {len(df):,} rows")

    # Recompute woba_value with fixed weights
    log("recomputing woba_value with fixed weights")
    walk_unintentional = (df["walk"] - df["iw"]).clip(lower=0)
    df["woba_value"] = (
        fixed["wBB"]  * walk_unintentional
        + fixed["wHBP"] * df["hbp"]
        + fixed["w1B"]  * df["single"]
        + fixed["w2B"]  * df["double"]
        + fixed["w3B"]  * df["triple"]
        + fixed["wHR"]  * df["hr"]
    ).astype("float64")
    # IBBs excluded
    df.loc[df["iw"] == 1, "woba_value"] = np.nan

    # Drop event-flag columns we no longer need
    df = df.drop(columns=["single", "double", "triple", "hr", "walk", "hbp"])

    # Same filters as the season-weight fit
    df = df[df["iw"] != 1]
    df = df.dropna(subset=["woba_value", "bat_age", "pit_age", "bat_id", "pit_id"])
    df["bat_age"] = df["bat_age"].astype(int)
    df["pit_age"] = df["pit_age"].astype(int)
    df["season"]  = df["season"].astype(int)
    df = df[(df["bat_age"] >= AGE_LO) & (df["bat_age"] <= AGE_HI)
            & (df["pit_age"] >= AGE_LO) & (df["pit_age"] <= AGE_HI)]
    log(f"  {len(df):,} after IBB/missing/age filter")

    bat_nseason = df.groupby("bat_id")["season"].nunique()
    pit_nseason = df.groupby("pit_id")["season"].nunique()
    keep_bat = set(bat_nseason[bat_nseason >= 2].index)
    keep_pit = set(pit_nseason[pit_nseason >= 2].index)
    before = len(df)
    df = df[df["bat_id"].isin(keep_bat) & df["pit_id"].isin(keep_pit)]
    log(f"  {before:,} -> {len(df):,} after dropping one-season-player rows")

    df["bat_id"] = df["bat_id"].astype(str)
    df["pit_id"] = df["pit_id"].astype(str)
    return df


def fit(df: pd.DataFrame, formula: str, label: str) -> dict:
    log(f"=== fitting {label}: {formula} ===")
    t0 = time.time()
    mod = pf.feols(formula, data=df)
    log(f"  fit done in {time.time()-t0:.1f}s; extracting FEs")
    fe = mod.fixef()
    out = {"label": label, "fe": {}}
    for fe_key, d in fe.items():
        name = fe_key[2:-1] if fe_key.startswith("C(") and fe_key.endswith(")") else fe_key
        out["fe"][name] = pd.Series(d, name=name)
    return out


def reanchor_mu(s: pd.Series) -> pd.Series:
    s = s.copy(); s.index = s.index.astype(int); s = s.sort_index()
    return s - s.mean()


def main() -> None:
    OUT_MU_TW.parent.mkdir(parents=True, exist_ok=True)
    OUT_COMPARE.parent.mkdir(parents=True, exist_ok=True)

    # Load panel once just to compute PA weights (need season + iw)
    base = pd.read_parquet(PANEL_PATH, columns=["season", "iw"])
    fixed = compute_fixed_weights(base)
    del base

    df = load_and_reweight(fixed)

    res_tw  = fit(df, "woba_value ~ 1 | season + bat_id + pit_id + bat_age + pit_age", "two_way")
    res_bat = fit(df, "woba_value ~ 1 | season + bat_id + bat_age", "bat_only")
    res_pit = fit(df, "woba_value ~ 1 | season + pit_id + pit_age", "pit_only")

    mu_tw  = reanchor_mu(res_tw["fe"]["season"])
    mu_bat = reanchor_mu(res_bat["fe"]["season"])
    mu_pit = reanchor_mu(res_pit["fe"]["season"])

    pa_per_season = df.groupby("season").size()
    pa_per_season.index = pa_per_season.index.astype(int)

    def to_df(mu):
        o = pd.DataFrame({"season": mu.index, "mu": mu.values})
        o["n_pa"] = o["season"].map(pa_per_season).fillna(0).astype(int)
        return o

    to_df(mu_tw ).to_parquet(OUT_MU_TW,  index=False)
    to_df(mu_bat).to_parquet(OUT_MU_BAT, index=False)
    to_df(mu_pit).to_parquet(OUT_MU_PIT, index=False)

    cmp = pd.DataFrame({
        "season": mu_tw.index,
        "mu_two_way":  mu_tw.values,
        "mu_bat_only": mu_bat.reindex(mu_tw.index).values,
        "mu_pit_only": mu_pit.reindex(mu_tw.index).values,
    })
    cmp.to_csv(OUT_COMPARE, index=False, float_format="%.5f")
    log(f"wrote {OUT_COMPARE}")

    # Side-by-side with the season-weight two-way fit, if present
    side = cmp[["season", "mu_two_way"]].rename(columns={"mu_two_way": "mu_fixedw"})
    if MU_SEASONW.exists():
        sw = pd.read_parquet(MU_SEASONW)[["season", "mu"]].rename(columns={"mu": "mu_seasonw"})
        side = sw.merge(side, on="season", how="outer").sort_values("season")
        side["diff"] = side["mu_fixedw"] - side["mu_seasonw"]
        corr = side[["mu_seasonw", "mu_fixedw"]].corr().iloc[0, 1]
    else:
        corr = np.nan
    side.to_csv(OUT_SIDE, index=False, float_format="%.5f")
    log(f"wrote {OUT_SIDE}")

    lines = []
    P = lines.append
    P("Fixed-weight robustness check (two-way + per-role)")
    P("=" * 70)
    P("Fixed weights (PA-weighted average of season weights):")
    for c in WEIGHT_COLS:
        P(f"  {c} = {fixed[c]:.4f}")
    P("")
    P(f"two_way  mu range: [{mu_tw.min():+.5f}, {mu_tw.max():+.5f}]")
    P(f"bat_only mu range: [{mu_bat.min():+.5f}, {mu_bat.max():+.5f}]")
    P(f"pit_only mu range: [{mu_pit.min():+.5f}, {mu_pit.max():+.5f}]")
    if not np.isnan(corr):
        P("")
        P(f"Correlation between season-weight and fixed-weight two-way mu_S: {corr:+.4f}")
        P(f"Max |diff| (fixedw - seasonw): {side['diff'].abs().max():.5f}")
        P(f"Mean |diff|: {side['diff'].abs().mean():.5f}")
    P("")
    P("=== fixed-weight mu_S (all three models) ===")
    P(cmp.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    if not np.isnan(corr):
        P("")
        P("=== season-weight vs fixed-weight two-way, side by side ===")
        P(side.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {OUT_SUMMARY}")
    for s in lines[:30]:
        print(s)
    print(f"\n... (full output in {OUT_SUMMARY})")


if __name__ == "__main__":
    main()
