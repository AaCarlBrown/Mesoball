"""
two_way_pyfixest.py
===================

Same model as two_way_season_fit.py but solved with pyfixest, which
handles the high-dimensional fixed effects properly via alternating
projection demeaning. No anchor hacks, fast convergence, well-tested.

Three fits:

  1. two_way: woba_value ~ 1 | season + bat_id + pit_id + bat_age + pit_age
  2. bat_only: woba_value ~ 1 | season + bat_id + bat_age
  3. pit_only: woba_value ~ 1 | season + pit_id + pit_age

In each case the season fixed effects (mu_S) are the parameter of
interest. Other FEs are nuisance.

Anchoring: pyfixest drops one level per FE by default. We re-anchor
mu_S so its mean is 0 over fitted seasons, and re-anchor the age
curves so age 28 is 0.

Filters:
  - IBBs excluded (woba_value NaN)
  - Ages [18, 45]
  - Drop players with PAs in only one season

Inputs:
    C:\\baseball_eras\\data\\pa_panel.parquet

Outputs:
    C:\\baseball_eras\\data\\mu_two_way.parquet
    C:\\baseball_eras\\data\\mu_bat_only.parquet
    C:\\baseball_eras\\data\\mu_pit_only.parquet
    C:\\baseball_eras\\data\\alpha_bat.parquet
    C:\\baseball_eras\\data\\alpha_pit.parquet
    C:\\baseball_eras\\output\\mu_compare.csv
    C:\\baseball_eras\\output\\two_way_pyfixest_summary.txt

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

OUT_MU_TW    = PROJECT_DIR / "data"   / "mu_two_way.parquet"
OUT_MU_BAT   = PROJECT_DIR / "data"   / "mu_bat_only.parquet"
OUT_MU_PIT   = PROJECT_DIR / "data"   / "mu_pit_only.parquet"
OUT_ALPHA_B  = PROJECT_DIR / "data"   / "alpha_bat.parquet"
OUT_ALPHA_P  = PROJECT_DIR / "data"   / "alpha_pit.parquet"
OUT_COMPARE  = PROJECT_DIR / "output" / "mu_compare.csv"
OUT_SUMMARY  = PROJECT_DIR / "output" / "two_way_pyfixest_summary.txt"

AGE_LO, AGE_HI = 18, 45
ANCHOR_AGE = 28


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_panel() -> pd.DataFrame:
    log(f"loading {PANEL_PATH}")
    df = pd.read_parquet(PANEL_PATH, columns=[
        "season", "bat_id", "pit_id", "woba_value", "iw",
        "bat_age", "pit_age",
    ])
    log(f"  {len(df):,} rows")

    df = df[df["iw"] != 1]
    df = df.dropna(subset=["woba_value", "bat_age", "pit_age",
                           "bat_id", "pit_id"])
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
    log(f"  {len(keep_bat):,} batters, {len(keep_pit):,} pitchers, "
        f"{df['season'].nunique()} seasons")

    # pyfixest wants string-typed FE columns for stable id'ing
    df["season"]  = df["season"].astype(int)
    df["bat_id"]  = df["bat_id"].astype(str)
    df["pit_id"]  = df["pit_id"].astype(str)
    df["bat_age"] = df["bat_age"].astype(int)
    df["pit_age"] = df["pit_age"].astype(int)
    return df


def fit(df: pd.DataFrame, formula: str, label: str) -> dict:
    log(f"=== fitting {label}: {formula} ===")
    t0 = time.time()
    mod = pf.feols(formula, data=df)
    log(f"  pyfixest fit done in {time.time() - t0:.1f}s")

    # Extract fixed-effect levels
    log("  extracting fixed effects")
    fe = mod.fixef()
    # pyfixest wraps FE names as 'C(name)' and uses string-typed level keys
    # even when the original column is int. Unwrap both.
    out = {"label": label, "fe": {}}
    for fe_key, d in fe.items():
        # fe_key looks like 'C(season)' -> extract 'season'
        fe_name = fe_key[2:-1] if fe_key.startswith("C(") and fe_key.endswith(")") else fe_key
        s = pd.Series(d, name=fe_name)
        out["fe"][fe_name] = s
    return out


def reanchor_mu(s: pd.Series) -> pd.Series:
    # Season levels: cast index to int, sort, mean-anchor.
    s = s.copy()
    s.index = s.index.astype(int)
    s = s.sort_index()
    return s - s.mean()


def reanchor_age(s: pd.Series, anchor_age: int) -> pd.Series:
    s = s.copy()
    s.index = s.index.astype(int)
    s = s.sort_index()
    if anchor_age in s.index:
        return s - s.loc[anchor_age]
    return s - s.mean()


def main() -> None:
    OUT_MU_TW.parent.mkdir(parents=True, exist_ok=True)
    OUT_COMPARE.parent.mkdir(parents=True, exist_ok=True)

    df = load_panel()

    res_tw  = fit(df, "woba_value ~ 1 | season + bat_id + pit_id + bat_age + pit_age",
                  "two_way")
    res_bat = fit(df, "woba_value ~ 1 | season + bat_id + bat_age",
                  "bat_only")
    res_pit = fit(df, "woba_value ~ 1 | season + pit_id + pit_age",
                  "pit_only")

    # Build mu_S series for each
    mu_tw  = reanchor_mu(res_tw["fe"]["season"])
    mu_bat = reanchor_mu(res_bat["fe"]["season"])
    mu_pit = reanchor_mu(res_pit["fe"]["season"])

    # Age curves come only from two-way (consistent with the primary mu)
    ab = reanchor_age(res_tw["fe"]["bat_age"], ANCHOR_AGE)
    ap = reanchor_age(res_tw["fe"]["pit_age"], ANCHOR_AGE)

    # PA counts per season
    pa_per_season = df.groupby("season").size()
    pa_per_season.index = pa_per_season.index.astype(int)
    pa_per_season = pa_per_season.sort_index()

    def to_df(mu: pd.Series) -> pd.DataFrame:
        out = pd.DataFrame({"season": mu.index, "mu": mu.values})
        out["n_pa"] = out["season"].map(pa_per_season).fillna(0).astype(int)
        return out

    to_df(mu_tw ).to_parquet(OUT_MU_TW,  index=False)
    to_df(mu_bat).to_parquet(OUT_MU_BAT, index=False)
    to_df(mu_pit).to_parquet(OUT_MU_PIT, index=False)
    pd.DataFrame({"age": ab.index, "alpha": ab.values}).to_parquet(OUT_ALPHA_B, index=False)
    pd.DataFrame({"age": ap.index, "alpha": ap.values}).to_parquet(OUT_ALPHA_P, index=False)
    log("wrote parquet outputs")

    cmp = pd.DataFrame({
        "season": mu_tw.index,
        "mu_two_way":  mu_tw.values,
        "mu_bat_only": mu_bat.reindex(mu_tw.index).values,
        "mu_pit_only": mu_pit.reindex(mu_tw.index).values,
        "n_pa":        [int(pa_per_season.get(s, 0)) for s in mu_tw.index],
    })
    cmp.to_csv(OUT_COMPARE, index=False, float_format="%.5f")
    log(f"wrote {OUT_COMPARE}")

    # Summary
    lines = []
    P = lines.append
    P("Two-way (and per-role) season-level fits via pyfixest")
    P("=" * 70)
    P(f"  seasons: {int(mu_tw.index.min())}-{int(mu_tw.index.max())}  ({len(mu_tw)})")
    P(f"  mu_two_way  range: [{mu_tw.min():+.5f}, {mu_tw.max():+.5f}]")
    P(f"  mu_bat_only range: [{mu_bat.min():+.5f}, {mu_bat.max():+.5f}]")
    P(f"  mu_pit_only range: [{mu_pit.min():+.5f}, {mu_pit.max():+.5f}]")
    P("")
    P("=== mu_S table (all three models side by side) ===")
    P(cmp.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    P("")
    P("=== alpha_bat (two-way model) ===")
    P(pd.DataFrame({"age": ab.index, "alpha_bat": ab.values})
      .to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    P("")
    P("=== alpha_pit (two-way model) ===")
    P(pd.DataFrame({"age": ap.index, "alpha_pit": ap.values})
      .to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    P("")
    P("=== correlations between mu series ===")
    P(cmp[["mu_two_way", "mu_bat_only", "mu_pit_only"]].corr()
      .to_string(float_format=lambda x: f"{x:+.4f}"))

    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {OUT_SUMMARY}")
    for s in lines[:25]:
        print(s)
    print(f"\n... (full output in {OUT_SUMMARY})")


if __name__ == "__main__":
    main()
