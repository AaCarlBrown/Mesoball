"""
u_build_cohort_cube.py
======================

Step 1 of the cohort-cube approach: aggregate the PA panel to the
(pitcher birth year, batter birth year, season) level.

Each cell:
    n_pa            number of PAs in this triple
    sum_woba        sum of wOBA values
    mean_woba       PA-weighted mean wOBA (sum_woba / n_pa)
    pit_age         season - pit_birth_year
    bat_age         season - bat_birth_year

Filtering:
    - exclude IBBs from the wOBA calculation (FanGraphs convention)
    - drop rows with missing birth years (~5 batters out of 15M PAs)

Output:
    data/cohort_cube.parquet
    output/u_build_cohort_cube_summary.txt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(r"C:\baseball_eras")
PANEL_PATH = PROJECT_DIR / "data" / "pa_panel.parquet"
OUT_CUBE   = PROJECT_DIR / "data" / "cohort_cube.parquet"
OUT_SUMMARY = PROJECT_DIR / "output" / "u_build_cohort_cube_summary.txt"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exclude-pitcher-batters", action="store_true",
                    help="Drop PAs where the batter is also a pitcher")
    args = ap.parse_args()

    log(f"loading {PANEL_PATH.name}")
    panel = pd.read_parquet(PANEL_PATH, columns=[
        "season", "bat_birth_year", "pit_birth_year",
        "woba_value", "iw", "bat_is_pitcher",
    ])
    log(f"  {len(panel):,} PA rows")

    # Drop IBBs (FanGraphs wOBA convention)
    before = len(panel)
    panel = panel[panel["iw"] != 1]
    log(f"  excluded IBBs: {before:,} -> {len(panel):,}")

    if args.exclude_pitcher_batters:
        before = len(panel)
        panel = panel[panel["bat_is_pitcher"] == 0]
        log(f"  excluded pitchers-batting PAs: {before:,} -> {len(panel):,}")

    # Drop missing birth years
    before = len(panel)
    panel = panel.dropna(subset=["bat_birth_year", "pit_birth_year"])
    log(f"  dropped rows with missing birth year: {before:,} -> {len(panel):,}")

    log("aggregating by (pit_birth_year, bat_birth_year, season)")
    cube = (
        panel.groupby(
            ["pit_birth_year", "bat_birth_year", "season"], observed=True
        )
        .agg(
            n_pa=("woba_value", "size"),
            sum_woba=("woba_value", "sum"),
        )
        .reset_index()
    )
    cube["mean_woba"] = cube["sum_woba"] / cube["n_pa"]
    cube["pit_age"] = (cube["season"] - cube["pit_birth_year"]).astype(int)
    cube["bat_age"] = (cube["season"] - cube["bat_birth_year"]).astype(int)
    log(f"  {len(cube):,} cells")

    OUT_CUBE.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    cube.to_parquet(OUT_CUBE, index=False)
    log(f"wrote {OUT_CUBE}")

    # ---- Summary ----
    lines = []
    P = lines.append
    P("Cohort cube build (step 1 of cube approach)")
    P("=" * 70)
    P(f"exclude_pitcher_batters: {args.exclude_pitcher_batters}")
    P("")
    P(f"Total PAs in cube:    {cube['n_pa'].sum():,}")
    P(f"Total cells:          {len(cube):,}")
    P(f"Median PA per cell:   {int(cube['n_pa'].median()):,}")
    P(f"Mean PA per cell:     {int(cube['n_pa'].mean()):,}")
    P(f"Max PA per cell:      {int(cube['n_pa'].max()):,}")
    P("")
    P(f"Pitcher birth years:  {cube['pit_birth_year'].min()} - {cube['pit_birth_year'].max()}")
    P(f"Batter birth years:   {cube['bat_birth_year'].min()} - {cube['bat_birth_year'].max()}")
    P(f"Seasons:              {cube['season'].min()} - {cube['season'].max()}")
    P("")

    # cells per (pit_by, bat_by) pair across seasons
    pair_counts = cube.groupby(["pit_birth_year", "bat_birth_year"]).agg(
        n_seasons=("season", "nunique"),
        total_pa=("n_pa", "sum"),
    ).reset_index()
    P(f"Distinct (pit_by, bat_by) pairs: {len(pair_counts):,}")
    P(f"  median seasons covered per pair: {int(pair_counts['n_seasons'].median())}")
    P(f"  pairs with >= 5 seasons:         {(pair_counts['n_seasons'] >= 5).sum():,}")
    P(f"  pairs with >= 10 seasons:        {(pair_counts['n_seasons'] >= 10).sum():,}")
    P("")

    P("=== top 15 (pit_by, bat_by) pairs by total PA ===")
    P(pair_counts.nlargest(15, "total_pa").to_string(index=False))
    P("")

    # example slice: one specific (pit_by, bat_by) pair across seasons
    # Pick a well-covered modern pair
    best = pair_counts.nlargest(1, "total_pa").iloc[0]
    pby, bby = int(best["pit_birth_year"]), int(best["bat_birth_year"])
    P(f"=== example slice: pit_by={pby}, bat_by={bby}, across all seasons ===")
    slc = cube[(cube["pit_birth_year"] == pby) & (cube["bat_birth_year"] == bby)]
    slc = slc.sort_values("season")
    P(slc[["season", "pit_age", "bat_age", "n_pa", "mean_woba"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {OUT_SUMMARY}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
