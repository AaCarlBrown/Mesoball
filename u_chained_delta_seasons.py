"""
u_chained_delta_seasons.py
==========================

Step 2 of the era-comparison project. Reproduce the chained-delta method
from Chapter 19 of *Wrong Number*, using the PA panel built in step 1,
expressed in wOBA points instead of runs-per-9-innings.

Method (per the chapter):

  For each consecutive season pair (Y, Y+1):
    1. delta_bat: PA-weighted change in wOBA among batters who appeared
       in both seasons. Holds batter identity fixed; reflects how much
       easier or harder it was to hit, plus any age effect.
    2. delta_pit: PA-weighted change in wOBA-against among pitchers who
       appeared in both. Holds pitcher identity fixed; reflects how much
       easier or harder it was to pitch, plus any age effect.
    3. (delta_bat - delta_pit) / 2 is the change in net hitting talent
       relative to net pitching talent, with league-wide scoring shifts
       cancelled.

  To strip the age effect, do steps 1-2 within each age cohort and then
  weight-average across ages. The chapter's logic: a 27-year-old batter
  in Y compared with the same player at 28 in Y+1, averaged with a
  27-year-old pitcher in Y compared with himself at 28 in Y+1, cancels
  the symmetric age effects to the extent that batter and pitcher aging
  curves resemble each other.

  Chain the per-pair deltas to produce a single talent index across all
  seasons.

Inputs:
    C:\\baseball_eras\\data\\pa_panel.parquet

Outputs:
    C:\\baseball_eras\\data\\delta_by_season.parquet      (long format: season, age, delta_bat, delta_pit, weights)
    C:\\baseball_eras\\data\\talent_index.parquet         (season, delta_pair, talent_index)
    C:\\baseball_eras\\output\\u_chained_delta_seasons_summary.txt

Usage:
    py u_chained_delta_seasons.py
    py u_chained_delta_seasons.py --no-age-adjust    # skip the age-cohort step
    py u_chained_delta_seasons.py --exclude-pitcher-batters
    py u_chained_delta_seasons.py --exclude-ibb      # drop IBBs from PA totals
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\baseball_eras")
PANEL_PATH = PROJECT_DIR / "data" / "pa_panel.parquet"

OUT_DELTAS = PROJECT_DIR / "data" / "delta_by_season.parquet"
OUT_INDEX = PROJECT_DIR / "data" / "talent_index.parquet"
OUT_SUMMARY = PROJECT_DIR / "output" / "u_chained_delta_seasons_summary.txt"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Per-player-season aggregation
# ---------------------------------------------------------------------------

def player_season_aggregates(
    panel: pd.DataFrame,
    role: str,            # 'bat' or 'pit'
    exclude_ibb: bool,
    age_convention: str,  # 'chapter' or 'baseball'
) -> pd.DataFrame:
    """
    Aggregate the PA panel to (player_id, season, age) rows with PA-weighted
    wOBA. For pitchers, this is wOBA-against; for batters, wOBA.

    Age conventions:
        'chapter'  : season - birth_year, ignoring birth month. Matches the
                     convention used in Chapter 19 of *Wrong Number*.
        'baseball' : age on June 30 of the season (sabermetric standard).
                     The PA panel already has bat_age / pit_age computed
                     this way.

    Returns columns: player_id, season, age, n_pa, woba.
    """
    df = panel
    if exclude_ibb:
        df = df[df["iw"] != 1]
    else:
        df = df.assign(woba_value=df["woba_value"].fillna(0.0))

    id_col = "bat_id" if role == "bat" else "pit_id"

    df = df.copy()
    if age_convention == "chapter":
        by_col = "bat_birth_year" if role == "bat" else "pit_birth_year"
        df["age"] = (df["season"] - df[by_col]).astype("Int64")
    elif age_convention == "baseball":
        age_src = "bat_age" if role == "bat" else "pit_age"
        df["age"] = df[age_src].astype("Int64")
    else:
        raise ValueError(f"unknown age_convention: {age_convention}")

    g = df.groupby([id_col, "season", "age"], dropna=False)
    out = g.agg(
        n_pa=("woba_value", "size"),
        woba=("woba_value", "mean"),
    ).reset_index()
    out = out.rename(columns={id_col: "player_id"})
    out = out.dropna(subset=["age"])
    out["age"] = out["age"].astype(int)
    return out


# ---------------------------------------------------------------------------
# Pair-season delta
# ---------------------------------------------------------------------------

def compute_paired_deltas(
    agg: pd.DataFrame,
    role: str,
) -> pd.DataFrame:
    """
    For every (season Y, season Y+1) pair, compute each player's wOBA
    change. Returns long-format dataframe with:
        season         (= Y, the earlier season)
        age            (= player's age in Y)
        player_id
        n_pa_y, n_pa_y1
        woba_y, woba_y1
        delta          (= woba_y1 - woba_y)
        weight         (= min(n_pa_y, n_pa_y1) -- "double-counting" min weight is
                        a common convention; alternatively use geometric mean.
                        Using min gives less weight to a guy who had 600 PA
                        in Y and 50 in Y+1.)
    """
    next_year = agg.copy()
    next_year["season"] = next_year["season"] - 1  # so the same player_id matches
    next_year = next_year.rename(columns={
        "n_pa": "n_pa_y1",
        "woba": "woba_y1",
        "age": "age_y1",
    })

    merged = agg.merge(
        next_year[["player_id", "season", "n_pa_y1", "woba_y1", "age_y1"]],
        on=["player_id", "season"],
        how="inner",
    )
    merged = merged.rename(columns={"n_pa": "n_pa_y", "woba": "woba_y"})
    merged["delta"] = merged["woba_y1"] - merged["woba_y"]
    merged["weight"] = np.minimum(merged["n_pa_y"], merged["n_pa_y1"])
    merged["role"] = role
    return merged


# ---------------------------------------------------------------------------
# Cohort-weighted season-pair deltas
# ---------------------------------------------------------------------------

def cohort_weighted_pair_delta(
    bat_pairs: pd.DataFrame,
    pit_pairs: pd.DataFrame,
    age_adjust: bool,
) -> pd.DataFrame:
    """
    For each season Y, compute the chained-delta talent change from Y to Y+1
    in wOBA points (positive = talent improved).

    Derivation:
        Let B = hitter-pool improvement, P = pitcher-pool improvement,
        G = game-factor favoring offense. Then:
            delta_woba(same batter)  = -P + G
            delta_woba(same pitcher) = +B + G
        So (delta_pit - delta_bat) / 2 = (B + P) / 2 = average talent improvement.

    Within-cohort weighting (individual player pairs):
        Each player-pair (Y, Y+1) is weighted by (PA_Y + PA_Y+1). The cohort
        delta is the weighted mean of the per-player deltas.

    Cross-cohort weighting (across ages within a season pair):
        Per the chapter, each cohort's pair_delta estimate is weighted by
        the product of all four cohort PA sums:
            cohort_weight(a) = bat_PA_Y(a) * bat_PA_Y+1(a+1)
                             * pit_PA_Y(a) * pit_PA_Y+1(a+1)
        This concentrates weight on prime-age cohorts where all four pools
        are large.

    With age_adjust=False: just weight individual pairs by PA_Y + PA_Y+1
        and average bat / pit separately, then combine.
    """
    if not age_adjust:
        rows = []
        for season, gb in bat_pairs.groupby("season"):
            w_bat_indiv = gb["n_pa_y"] + gb["n_pa_y1"]
            d_bat = np.average(gb["delta"], weights=w_bat_indiv)
            sub = pit_pairs[pit_pairs.season == season]
            if len(sub) == 0:
                continue
            w_pit_indiv = sub["n_pa_y"] + sub["n_pa_y1"]
            d_pit = np.average(sub["delta"], weights=w_pit_indiv)
            rows.append({
                "season": season,
                "delta_bat": d_bat,
                "delta_pit": d_pit,
                "delta_pair": (d_pit - d_bat) / 2,   # positive = talent improved
                "w_bat": w_bat_indiv.sum(),
                "w_pit": w_pit_indiv.sum(),
            })
        return pd.DataFrame(rows)

    # ---- Age-adjusted ----
    # In-cohort weight on each individual player pair is PA_Y + PA_Y+1.
    # We also need the cohort-total PA sums in BOTH years to form the
    # cross-cohort product weight.

    # Per-cohort batter pair delta (weighted by individual PA_Y + PA_Y+1)
    bp = bat_pairs.copy()
    bp["pa_sum"] = bp["n_pa_y"] + bp["n_pa_y1"]
    bat_by = (
        bp.groupby(["season", "age"])
          .apply(lambda g: pd.Series({
              "delta_bat": np.average(g["delta"], weights=g["pa_sum"]),
              "bat_pa_y":  g["n_pa_y"].sum(),
              "bat_pa_y1": g["n_pa_y1"].sum(),
          }), include_groups=False)
          .reset_index()
    )

    pp = pit_pairs.copy()
    pp["pa_sum"] = pp["n_pa_y"] + pp["n_pa_y1"]
    pit_by = (
        pp.groupby(["season", "age"])
          .apply(lambda g: pd.Series({
              "delta_pit": np.average(g["delta"], weights=g["pa_sum"]),
              "pit_pa_y":  g["n_pa_y"].sum(),
              "pit_pa_y1": g["n_pa_y1"].sum(),
          }), include_groups=False)
          .reset_index()
    )

    cohort = bat_by.merge(pit_by, on=["season", "age"], how="inner")
    cohort["pair_delta"] = (cohort["delta_pit"] - cohort["delta_bat"]) / 2
    # Chapter's weighting: product of all four PA totals
    cohort["cohort_weight"] = (
        cohort["bat_pa_y"]  * cohort["bat_pa_y1"]
        * cohort["pit_pa_y"] * cohort["pit_pa_y1"]
    ).astype("float64")

    rows = []
    for season, gb in cohort.groupby("season"):
        if gb["cohort_weight"].sum() == 0:
            continue
        rows.append({
            "season": season,
            "delta_bat":  np.average(gb["delta_bat"],  weights=gb["cohort_weight"]),
            "delta_pit":  np.average(gb["delta_pit"],  weights=gb["cohort_weight"]),
            "delta_pair": np.average(gb["pair_delta"], weights=gb["cohort_weight"]),
            "bat_pa_y":   gb["bat_pa_y"].sum(),
            "pit_pa_y":   gb["pit_pa_y"].sum(),
            "n_ages":     len(gb),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

def chain_index(pair_deltas: pd.DataFrame) -> pd.DataFrame:
    """
    Build a talent index by cumulatively summing the pair deltas. The
    season-Y row of pair_deltas describes the Y -> Y+1 change, so the
    index for season Y+1 is the sum of all pair deltas through Y.

    Returns a dataframe with columns: season, talent_index (anchored so
    the first season is 0).
    """
    pair_deltas = pair_deltas.sort_values("season").reset_index(drop=True)
    seasons = sorted(set(pair_deltas["season"]) | set(pair_deltas["season"] + 1))
    idx = pd.Series(0.0, index=seasons, dtype=float)
    cum = 0.0
    for _, row in pair_deltas.iterrows():
        cum += row["delta_pair"]
        idx.loc[row["season"] + 1] = cum
    return pd.DataFrame({"season": idx.index, "talent_index": idx.values})


# ---------------------------------------------------------------------------
# Eras
# ---------------------------------------------------------------------------

ERAS = [
    ("Deadball",   1910, 1919),   # chapter's "Dead Ball" 1901-1919; data starts 1910
    ("Live Ball",  1920, 1939),
    ("WWII",       1940, 1946),
    ("Integration",1947, 1960),
    ("Expansion",  1961, 1976),
    ("Free agency",1977, 1993),
    ("Steroid",    1994, 2005),
    ("Post-steroid",2006, 2024),
]


def era_summary(pair_deltas: pd.DataFrame) -> pd.DataFrame:
    """
    For each era, compute the average yearly improvement in talent
    (wOBA points per year).
    """
    rows = []
    for name, lo, hi in ERAS:
        sub = pair_deltas[(pair_deltas["season"] >= lo) & (pair_deltas["season"] < hi)]
        if len(sub) == 0:
            continue
        rows.append({
            "era": name,
            "from": lo,
            "to": hi,
            "n_pairs": len(sub),
            "avg_yearly_improvement": sub["delta_pair"].mean(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-age-adjust", action="store_true",
                    help="Skip the age-cohort weighting step")
    ap.add_argument("--exclude-pitcher-batters", action="store_true",
                    help="Drop PAs where the batter is also a pitcher")
    ap.add_argument("--exclude-ibb", action="store_true",
                    help="Drop intentional walks entirely")
    ap.add_argument("--age-convention", choices=["chapter", "baseball"],
                    default="chapter",
                    help="'chapter' = season - birth_year (default, matches the book); "
                         "'baseball' = age on June 30 (sabermetric standard)")
    args = ap.parse_args()

    log(f"loading {PANEL_PATH.name}")
    panel = pd.read_parquet(PANEL_PATH, columns=[
        "season", "bat_id", "pit_id",
        "bat_birth_year", "pit_birth_year",
        "woba_value", "iw", "bat_is_pitcher",
    ])
    log(f"  {len(panel):,} PA rows")

    if args.exclude_pitcher_batters:
        before = len(panel)
        panel = panel[panel["bat_is_pitcher"] == 0]
        log(f"  excluded pitchers batting: {before:,} -> {len(panel):,}")

    log("aggregating per (batter, season, age)")
    bat_agg = player_season_aggregates(panel, "bat", args.exclude_ibb,
                                       args.age_convention)
    log(f"  {len(bat_agg):,} batter-season-age rows")

    log("aggregating per (pitcher, season, age)")
    pit_agg = player_season_aggregates(panel, "pit", args.exclude_ibb,
                                       args.age_convention)
    log(f"  {len(pit_agg):,} pitcher-season-age rows")

    log("computing batter pair deltas")
    bat_pairs = compute_paired_deltas(bat_agg, "bat")
    log(f"  {len(bat_pairs):,} batter pair rows")

    log("computing pitcher pair deltas")
    pit_pairs = compute_paired_deltas(pit_agg, "pit")
    log(f"  {len(pit_pairs):,} pitcher pair rows")

    age_adjust = not args.no_age_adjust
    log(f"computing cohort-weighted pair deltas (age_adjust={age_adjust})")
    pair_deltas = cohort_weighted_pair_delta(bat_pairs, pit_pairs, age_adjust)
    log(f"  {len(pair_deltas):,} season-pair rows")

    log("chaining to talent index")
    idx = chain_index(pair_deltas)

    log("writing outputs")
    OUT_DELTAS.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    pair_deltas.to_parquet(OUT_DELTAS, index=False)
    idx.to_parquet(OUT_INDEX, index=False)

    # Summary text
    lines: list[str] = []
    P = lines.append
    P("Chained-delta era comparison (step 2)")
    P("=" * 70)
    P(f"age_convention:           {args.age_convention}")
    P(f"age_adjust:               {age_adjust}")
    P(f"exclude_pitcher_batters:  {args.exclude_pitcher_batters}")
    P(f"exclude_ibb:              {args.exclude_ibb}")
    P("")
    P(f"Seasons covered: {pair_deltas['season'].min()}-{pair_deltas['season'].max()+1}")
    P(f"Number of season-pair deltas: {len(pair_deltas)}")
    P("")
    P("=== per-season pair deltas (wOBA points, Y -> Y+1) ===")
    P(pair_deltas.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    P("")
    P("=== era averages (wOBA points per year) ===")
    eras = era_summary(pair_deltas)
    P(eras.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    P("")
    P("=== talent index (cumulative wOBA points from start) ===")
    P(idx.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {OUT_SUMMARY}")
    # Print just era summary + a few sanity-check season pairs
    print()
    print("=== era averages (wOBA points per year) ===")
    print(eras.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    print()
    print("=== specific season pairs the chapter discusses ===")
    target_pairs = [1940, 1941, 1942, 1943, 1944, 1945,  # WWII
                    1960, 1961, 1968, 1969, 1976, 1977, 1992, 1993, 1997, 1998]  # expansions
    chk = pair_deltas[pair_deltas["season"].isin(target_pairs)][
        ["season", "delta_bat", "delta_pit", "delta_pair"]
    ]
    print(chk.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))


if __name__ == "__main__":
    main()
