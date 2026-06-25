"""
two_way_shape_norm.py
=====================

Third weighting scheme: "shape-normalized" season weights.

Motivation: season-specific FanGraphs weights bundle two things --
  (1) the SHAPE: the ratios among wBB, w1B, w2B, w3B, wHR, which encode
      what a player should have been trying to do in that run environment
      (e.g. swing for the fences when HRs are disproportionately valuable);
  (2) the LEVEL: the wOBAScale normalization that forces league wOBA to
      equal league OBP each year, which deliberately scrubs out the
      season-to-season run environment we are trying to measure.

We want to KEEP the shape and REMOVE the level normalization.

Method (Option A -- equal value on a fixed reference event mix):
  1. Compute a reference event-frequency vector f_ref from the panel:
     the PA-weighted all-time average per-PA rates of
        BB(unintentional), HBP, 1B, 2B, 3B, HR.
  2. For each season s, take its FanGraphs weight vector w_s (the six
     event weights) and compute the dot product
        V_s = sum_e  w_s,e * f_ref,e
     i.e. the wOBA that season's weights would assign to the reference
     batting line.
  3. Choose a target constant V_bar = PA-weighted-average of V_s across
     seasons (so the rescaled weights stay on the same scale as before).
  4. Scale season s's entire weight vector by k_s = V_bar / V_s.
     Now every season's weights assign the SAME total value to the
     reference event mix (level normalized), but the ratios among events
     within a season are untouched (shape preserved).
  5. Recompute woba_value per PA with the rescaled season weights:
        woba_value = k_{s} * [ wBB_s*(walk-iw) + wHBP_s*hbp + w1B_s*single
                             + w2B_s*double + w3B_s*triple + wHR_s*hr ]
  6. Re-run the two-way / bat-only / pit-only fits.

Comparison: outputs alongside the season-weight and fixed-weight two-way
series so all three weighting schemes can be plotted together.

Inputs:
    C:\\baseball_eras\\data\\pa_panel.parquet     (event flags + iw + season + ages + ids)
    C:\\baseball_eras\\wOBA_weights.csv
    C:\\baseball_eras\\data\\mu_two_way.parquet          (season-weight fit, optional)
    C:\\baseball_eras\\data\\mu_two_way_fixedw.parquet   (fixed-weight fit, optional)

Outputs:
    C:\\baseball_eras\\data\\mu_two_way_shapenorm.parquet
    C:\\baseball_eras\\data\\mu_bat_only_shapenorm.parquet
    C:\\baseball_eras\\data\\mu_pit_only_shapenorm.parquet
    C:\\baseball_eras\\output\\mu_compare_shapenorm.csv
    C:\\baseball_eras\\output\\mu_three_schemes.csv      (season vs fixed vs shapenorm, two-way)
    C:\\baseball_eras\\output\\two_way_shape_norm_summary.txt

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

MU_SEASONW = PROJECT_DIR / "data" / "mu_two_way.parquet"
MU_FIXEDW  = PROJECT_DIR / "data" / "mu_two_way_fixedw.parquet"

OUT_MU_TW   = PROJECT_DIR / "data"   / "mu_two_way_shapenorm.parquet"
OUT_MU_BAT  = PROJECT_DIR / "data"   / "mu_bat_only_shapenorm.parquet"
OUT_MU_PIT  = PROJECT_DIR / "data"   / "mu_pit_only_shapenorm.parquet"
OUT_COMPARE = PROJECT_DIR / "output" / "mu_compare_shapenorm.csv"
OUT_THREE   = PROJECT_DIR / "output" / "mu_three_schemes.csv"
OUT_SUMMARY = PROJECT_DIR / "output" / "two_way_shape_norm_summary.txt"

AGE_LO, AGE_HI = 18, 45
WEIGHT_COLS = ["wBB", "wHBP", "w1B", "w2B", "w3B", "wHR"]
EVENT_COLS  = ["bb_unint", "hbp", "single", "double", "triple", "hr"]  # aligns with WEIGHT_COLS


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    OUT_MU_TW.parent.mkdir(parents=True, exist_ok=True)
    OUT_COMPARE.parent.mkdir(parents=True, exist_ok=True)

    log(f"loading {PANEL_PATH}")
    df = pd.read_parquet(PANEL_PATH, columns=[
        "season", "bat_id", "pit_id", "iw",
        "bat_age", "pit_age",
        "single", "double", "triple", "hr", "walk", "hbp",
    ])
    log(f"  {len(df):,} rows")

    df["bb_unint"] = (df["walk"] - df["iw"]).clip(lower=0)

    # --- Step 1: reference event mix (PA-weighted all-time per-PA rates) ---
    # Exclude IBBs from the denominator (FanGraphs convention).
    log("computing reference event mix (all-time per-PA rates, ex-IBB)")
    non_ibb = df[df["iw"] != 1]
    n_pa_ref = len(non_ibb)
    f_ref = {}
    f_ref["bb_unint"] = non_ibb["bb_unint"].sum() / n_pa_ref
    f_ref["hbp"]      = non_ibb["hbp"].sum()      / n_pa_ref
    f_ref["single"]   = non_ibb["single"].sum()   / n_pa_ref
    f_ref["double"]   = non_ibb["double"].sum()   / n_pa_ref
    f_ref["triple"]   = non_ibb["triple"].sum()   / n_pa_ref
    f_ref["hr"]       = non_ibb["hr"].sum()        / n_pa_ref
    log("  reference rates per PA:")
    for e in EVENT_COLS:
        log(f"    {e:9s} = {f_ref[e]:.5f}")

    # --- Step 2-4: per-season scaling constants ---
    log(f"loading {WOBA_CSV.name} and computing per-season scaling")
    w = pd.read_csv(WOBA_CSV, encoding="utf-8-sig").rename(columns={"Season": "season"})
    w = w[["season"] + WEIGHT_COLS].copy()

    # V_s = sum_e w_s,e * f_ref,e   (value the season's weights assign to ref mix)
    f_ref_vec = np.array([f_ref["bb_unint"], f_ref["hbp"], f_ref["single"],
                          f_ref["double"], f_ref["triple"], f_ref["hr"]])
    w["V_s"] = (w[WEIGHT_COLS].to_numpy() * f_ref_vec).sum(axis=1)

    # PA per season to weight the target constant
    pa_by_season = non_ibb.groupby("season").size().rename("pa").reset_index()
    w = w.merge(pa_by_season, on="season", how="inner")
    V_bar = float((w["V_s"] * w["pa"]).sum() / w["pa"].sum())
    w["k_s"] = V_bar / w["V_s"]
    log(f"  target V_bar = {V_bar:.5f}")
    log(f"  k_s range: [{w['k_s'].min():.4f}, {w['k_s'].max():.4f}]")

    # Rescaled weights per season
    for c in WEIGHT_COLS:
        w[c + "_n"] = w[c] * w["k_s"]

    # --- Step 5: recompute woba_value per PA with rescaled season weights ---
    log("recomputing woba_value with shape-normalized season weights")
    wn = w[["season"] + [c + "_n" for c in WEIGHT_COLS]].copy()
    df = df.merge(wn, on="season", how="left")

    df["woba_value"] = (
        df["wBB_n"]  * df["bb_unint"]
        + df["wHBP_n"] * df["hbp"]
        + df["w1B_n"]  * df["single"]
        + df["w2B_n"]  * df["double"]
        + df["w3B_n"]  * df["triple"]
        + df["wHR_n"]  * df["hr"]
    ).astype("float64")
    df.loc[df["iw"] == 1, "woba_value"] = np.nan

    # --- Filters (same as other fits) ---
    df = df[df["iw"] != 1]
    df = df.dropna(subset=["woba_value", "bat_age", "pit_age", "bat_id", "pit_id"])
    df["bat_age"] = df["bat_age"].astype(int)
    df["pit_age"] = df["pit_age"].astype(int)
    df["season"]  = df["season"].astype(int)
    df = df[(df["bat_age"] >= AGE_LO) & (df["bat_age"] <= AGE_HI)
            & (df["pit_age"] >= AGE_LO) & (df["pit_age"] <= AGE_HI)]
    bat_n = df.groupby("bat_id")["season"].nunique()
    pit_n = df.groupby("pit_id")["season"].nunique()
    keep_bat = set(bat_n[bat_n >= 2].index)
    keep_pit = set(pit_n[pit_n >= 2].index)
    df = df[df["bat_id"].isin(keep_bat) & df["pit_id"].isin(keep_pit)]
    df["bat_id"] = df["bat_id"].astype(str)
    df["pit_id"] = df["pit_id"].astype(str)
    log(f"  {len(df):,} PAs enter the fit")

    def fit(formula, label):
        log(f"=== fitting {label} ===")
        t0 = time.time()
        mod = pf.feols(formula, data=df)
        log(f"  done in {time.time()-t0:.1f}s")
        fe = mod.fixef()
        season_key = next(k for k in fe if "season" in k)
        s = pd.Series(fe[season_key])
        s.index = s.index.astype(int)
        s = s.sort_index()
        return s - s.mean()

    mu_tw  = fit("woba_value ~ 1 | season + bat_id + pit_id + bat_age + pit_age", "two_way")
    mu_bat = fit("woba_value ~ 1 | season + bat_id + bat_age", "bat_only")
    mu_pit = fit("woba_value ~ 1 | season + pit_id + pit_age", "pit_only")

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

    # Three-scheme side-by-side (two-way only)
    three = cmp[["season", "mu_two_way"]].rename(columns={"mu_two_way": "shapenorm"})
    if MU_SEASONW.exists():
        sw = pd.read_parquet(MU_SEASONW)[["season","mu"]].rename(columns={"mu":"seasonw"})
        three = three.merge(sw, on="season", how="outer")
    if MU_FIXEDW.exists():
        fw = pd.read_parquet(MU_FIXEDW)[["season","mu"]].rename(columns={"mu":"fixedw"})
        three = three.merge(fw, on="season", how="outer")
    three = three.sort_values("season")
    three.to_csv(OUT_THREE, index=False, float_format="%.5f")

    lines = []
    P = lines.append
    P("Shape-normalized weighting (Option A)")
    P("=" * 70)
    P("Reference event rates per PA (all-time, ex-IBB):")
    for e in EVENT_COLS:
        P(f"  {e:9s} = {f_ref[e]:.5f}")
    P(f"Target V_bar = {V_bar:.5f}")
    P(f"k_s range: [{w['k_s'].min():.4f}, {w['k_s'].max():.4f}]")
    P("")
    P(f"two_way  mu range: [{mu_tw.min():+.5f}, {mu_tw.max():+.5f}]")
    P(f"bat_only mu range: [{mu_bat.min():+.5f}, {mu_bat.max():+.5f}]")
    P(f"pit_only mu range: [{mu_pit.min():+.5f}, {mu_pit.max():+.5f}]")
    P("")
    if {"seasonw","fixedw"}.issubset(three.columns):
        c1 = three[["shapenorm","seasonw"]].corr().iloc[0,1]
        c2 = three[["shapenorm","fixedw"]].corr().iloc[0,1]
        P(f"corr(shapenorm, seasonw) = {c1:+.4f}")
        P(f"corr(shapenorm, fixedw)  = {c2:+.4f}")
        P("")
    P("=== two-way mu_S, three schemes side by side ===")
    P(three.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {OUT_SUMMARY}")
    for s in lines[:25]:
        print(s)
    print(f"\n... full output in {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
