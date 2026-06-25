"""
u_build_pa_panel.py
===================

Step 1 of the era-comparison project. Build a single plate-appearance-level
parquet panel from the retrosheet data, with all the joins we'll need for
the regression model in later steps:

    game_id, date, season, gametype, park,
    bat_id, pit_id,
    bat_birth_date, pit_birth_date,
    bat_age, pit_age,                         (age on June 30 of season)
    bat_birth_year, pit_birth_year,
    bat_hand, pit_hand,
    bat_team, pit_team, bat_home,
    is_pa_outcome columns: single, double, triple, hr, walk, ibb, hbp, k,
                           sh, sf, roe, xi, fc, othout
    bat_is_pitcher (1 if the batter is in pitching.csv that season)
    woba_value (FanGraphs convention, season-specific weights)

Filter:
    - lgID in {'AL', 'NL'} for both home and visiting team
    - gametype == 'regular'
    - pa == 1 (the row is a completed plate appearance)
    - intentional walks: KEPT in the panel with woba_value = NaN, so the
      modeler can decide whether to drop them. (FanGraphs convention is
      to exclude IBBs from the wOBA denominator.)

Output:
    C:\\baseball_eras\\data\\pa_panel.parquet
    C:\\baseball_eras\\output\\u_build_pa_panel_summary.txt

Usage:
    py u_build_pa_panel.py
    py u_build_pa_panel.py --chunksize 2000000     # default 1M
    py u_build_pa_panel.py --dry-run               # don't write parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(r"C:\baseball_eras")
RETRO_DIR = Path(r"C:\overnight_effect_data\retrosheet")

PLAYS_CSV = RETRO_DIR / "plays.csv"
GAMEINFO_CSV = RETRO_DIR / "gameinfo.csv"
BIOFILE_CSV = RETRO_DIR / "biofile0.csv"
PITCHING_CSV = RETRO_DIR / "pitching.csv"
TEAMS_LAHMAN_CSV = PROJECT_DIR / "Teams.csv"
WOBA_CSV = PROJECT_DIR / "wOBA_weights.csv"

OUT_PARQUET = PROJECT_DIR / "data" / "pa_panel.parquet"
OUT_SUMMARY = PROJECT_DIR / "output" / "u_build_pa_panel_summary.txt"

LEAGUES_KEEP = {"AL", "NL"}

# Columns we read from plays.csv -- a subset of its 177 columns.
PLAYS_USECOLS = [
    "gid", "inning", "top_bot",
    "batter", "pitcher",
    "bathand", "pithand",
    "batteam", "pitteam",
    "pa",
    "single", "double", "triple", "hr",
    "walk", "iw", "hbp", "k",
    "sh", "sf", "roe", "xi", "fc", "othout",
    "date",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def baseball_age(birth_yyyymmdd: pd.Series, season: pd.Series) -> pd.Series:
    """
    Standard baseball age: how old the player was on June 30 of the season.
    birth_yyyymmdd is an Int64 series of YYYYMMDD ints (with <NA> for
    unknowns). season is an int series of YYYY.
    """
    by = (birth_yyyymmdd // 10000).astype("Int64")
    bm = ((birth_yyyymmdd // 100) % 100).astype("Int64")
    bd = (birth_yyyymmdd % 100).astype("Int64")
    age = season.astype("Int64") - by
    # If birthday is after June 30, subtract one year.
    after_june30 = (bm > 6) | ((bm == 6) & (bd > 30))
    age = age.mask(after_june30, age - 1)
    return age


def load_team_league_lookup() -> pd.DataFrame:
    """
    Load Lahman's Teams.csv and produce a (season, team_retro) -> lgID
    lookup table.
    """
    log(f"loading {TEAMS_LAHMAN_CSV.name}")
    t = pd.read_csv(TEAMS_LAHMAN_CSV, usecols=["yearID", "lgID", "teamIDretro"])
    t = t.rename(columns={"yearID": "season", "teamIDretro": "team", "lgID": "lg"})
    t = t.dropna(subset=["team"])
    return t


def load_gameinfo(team_lg: pd.DataFrame) -> pd.DataFrame:
    """
    Load gameinfo.csv, attach league for home and visiting teams, filter
    to regular-season AL/NL games.
    """
    log(f"loading {GAMEINFO_CSV.name}")
    gi = pd.read_csv(
        GAMEINFO_CSV,
        usecols=["gid", "season", "gametype", "visteam", "hometeam", "site"],
    )
    log(f"  {len(gi):,} games before filter")

    # Attach league for home team and visiting team.
    gi = gi.merge(
        team_lg.rename(columns={"team": "hometeam", "lg": "home_lg"}),
        on=["season", "hometeam"],
        how="left",
    )
    gi = gi.merge(
        team_lg.rename(columns={"team": "visteam", "lg": "vis_lg"}),
        on=["season", "visteam"],
        how="left",
    )

    n_unknown = (gi["home_lg"].isna() | gi["vis_lg"].isna()).sum()
    log(f"  {n_unknown:,} games have an unknown league for at least one team")

    keep = (
        (gi["gametype"] == "regular")
        & gi["home_lg"].isin(LEAGUES_KEEP)
        & gi["vis_lg"].isin(LEAGUES_KEEP)
    )
    gi = gi[keep].copy()
    log(f"  {len(gi):,} regular-season AL/NL games kept")
    return gi[["gid", "season", "site", "hometeam", "visteam", "home_lg", "vis_lg"]]


def load_biofile() -> pd.DataFrame:
    """Player id -> birth date, bats, throws."""
    log(f"loading {BIOFILE_CSV.name}")
    bio = pd.read_csv(
        BIOFILE_CSV,
        usecols=["id", "birthdate", "bats", "throws"],
        dtype={"birthdate": "Int64"},
    )
    bio = bio.rename(columns={"id": "player_id"})
    log(f"  {len(bio):,} player records")
    return bio


def load_pitchers_set() -> pd.DataFrame:
    """
    Build (player_id, season) set of player-seasons where the player
    appeared as a pitcher in any game. pitching.csv is per-game and has
    columns id, gid, date (YYYYMMDD) but no explicit season column, so
    we derive season from date.
    """
    log(f"loading {PITCHING_CSV.name}")
    p = pd.read_csv(
        PITCHING_CSV,
        usecols=["id", "date"],
        dtype={"id": "string", "date": "Int64"},
    )
    p["season"] = (p["date"] // 10000).astype("Int64")
    p = p.drop(columns=["date"])
    p = p.rename(columns={"id": "player_id"})
    p["bat_is_pitcher"] = 1
    p = p.drop_duplicates(["player_id", "season"])
    log(f"  {len(p):,} distinct (pitcher, season) pairs")
    return p


def load_woba_weights() -> pd.DataFrame:
    """
    Season-specific FanGraphs wOBA weights. Returns dataframe indexed by
    season with columns wBB, wHBP, w1B, w2B, w3B, wHR.
    """
    log(f"loading {WOBA_CSV.name}")
    w = pd.read_csv(WOBA_CSV, encoding="utf-8-sig")  # strips BOM
    w = w.rename(columns={"Season": "season"})
    keep = ["season", "wBB", "wHBP", "w1B", "w2B", "w3B", "wHR"]
    w = w[keep].copy()
    log(f"  {len(w):,} season-weight rows ({w.season.min()}-{w.season.max()})")
    return w


# ---------------------------------------------------------------------------
# Chunk processor for plays.csv
# ---------------------------------------------------------------------------

def stream_plays(
    valid_gids: set[str],
    chunksize: int,
) -> Iterator[pd.DataFrame]:
    """
    Stream plays.csv in chunks, filtering to PA-ending rows whose gid is
    in our AL/NL regular-season set.

    The dtype dict keeps memory under control -- plays.csv has 177 columns
    and we only want ~25 of them. usecols + dtype lets pandas skip the rest.
    """
    log(f"streaming {PLAYS_CSV.name} in chunks of {chunksize:,}")
    reader = pd.read_csv(
        PLAYS_CSV,
        usecols=PLAYS_USECOLS,
        dtype={
            "gid": "string",
            "batter": "string",
            "pitcher": "string",
            "bathand": "string",
            "pithand": "string",
            "batteam": "string",
            "pitteam": "string",
            "top_bot": "Int8",
            "inning": "Int16",
            "date": "Int64",
            # PA outcome flags: small ints
            "pa": "Int8",
            "single": "Int8", "double": "Int8", "triple": "Int8", "hr": "Int8",
            "walk": "Int8", "iw": "Int8", "hbp": "Int8", "k": "Int8",
            "sh": "Int8", "sf": "Int8", "roe": "Int8", "xi": "Int8",
            "fc": "Int8", "othout": "Int8",
        },
        chunksize=chunksize,
        low_memory=False,
    )
    for i, ch in enumerate(reader, 1):
        n_in = len(ch)
        ch = ch[ch["pa"] == 1]
        ch = ch[ch["gid"].isin(valid_gids)]
        log(f"  chunk {i}: {n_in:,} rows -> {len(ch):,} PA after filter")
        if len(ch) == 0:
            continue
        yield ch


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build_panel(chunksize: int, dry_run: bool) -> None:
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    team_lg = load_team_league_lookup()
    games = load_gameinfo(team_lg)
    bio = load_biofile()
    pitchers = load_pitchers_set()
    woba = load_woba_weights()

    valid_gids = set(games["gid"])
    games_idx = games.set_index("gid")

    parts: list[pd.DataFrame] = []
    rows_in = 0
    rows_out = 0

    for ch in stream_plays(valid_gids, chunksize):
        rows_in += len(ch)
        # Drop the row-level 'date' from plays (it's a per-row dup of game date);
        # we'll attach season/site from games_idx.
        ch = ch.drop(columns=["date"])
        ch = ch.join(games_idx[["season", "site", "hometeam", "visteam"]], on="gid")

        # Bat home/away
        ch["bat_home"] = (ch["top_bot"] == 1).astype("Int8")

        # Drop the ones we no longer need
        ch = ch.drop(columns=["top_bot"])

        # Join batter bio
        ch = ch.merge(
            bio.rename(columns={
                "player_id": "batter",
                "birthdate": "bat_birthdate",
                "bats": "bat_bats",
                "throws": "bat_throws",
            }),
            on="batter", how="left",
        )
        # Join pitcher bio
        ch = ch.merge(
            bio.rename(columns={
                "player_id": "pitcher",
                "birthdate": "pit_birthdate",
                "bats": "pit_bats",
                "throws": "pit_throws",
            }),
            on="pitcher", how="left",
        )

        # Compute ages
        ch["bat_age"] = baseball_age(ch["bat_birthdate"], ch["season"])
        ch["pit_age"] = baseball_age(ch["pit_birthdate"], ch["season"])
        ch["bat_birth_year"] = (ch["bat_birthdate"] // 10000).astype("Int64")
        ch["pit_birth_year"] = (ch["pit_birthdate"] // 10000).astype("Int64")

        # Pitcher-as-batter flag
        ch = ch.merge(
            pitchers.rename(columns={"player_id": "batter"}),
            on=["batter", "season"], how="left",
        )
        ch["bat_is_pitcher"] = ch["bat_is_pitcher"].fillna(0).astype("Int8")

        # wOBA value (only for non-IBB PAs)
        ch = ch.merge(woba, on="season", how="left")

        # Standard FanGraphs wOBA: 1B/2B/3B/HR/BB/HBP get their weights,
        # everything else (outs, K, SF, SH, ROE, XI, FC) is zero, and IBB
        # is excluded entirely (woba_value = NaN, denominator excluded).
        ch["woba_value"] = (
            ch["wBB"]  * (ch["walk"] - ch["iw"]).clip(lower=0)
            + ch["wHBP"] * ch["hbp"]
            + ch["w1B"]  * ch["single"]
            + ch["w2B"]  * ch["double"]
            + ch["w3B"]  * ch["triple"]
            + ch["wHR"]  * ch["hr"]
        )
        # Mark IBBs with NaN so the modeler can exclude them cleanly.
        ch.loc[ch["iw"] == 1, "woba_value"] = np.nan

        # Drop the weight columns -- we won't keep them per-row.
        ch = ch.drop(columns=["wBB", "wHBP", "w1B", "w2B", "w3B", "wHR"])

        # Rename for the final schema
        ch = ch.rename(columns={
            "gid": "game_id",
            "batter": "bat_id",
            "pitcher": "pit_id",
            "bathand": "bat_hand",
            "pithand": "pit_hand",
            "batteam": "bat_team",
            "pitteam": "pit_team",
            "site": "park_id",
        })

        rows_out += len(ch)
        parts.append(ch)

    log("concatenating chunks...")
    panel = pd.concat(parts, ignore_index=True)
    log(f"  final panel: {len(panel):,} PA rows, {panel.memory_usage(deep=True).sum() / 1e9:.2f} GB in memory")

    if not dry_run:
        log(f"writing {OUT_PARQUET}")
        panel.to_parquet(OUT_PARQUET, index=False, compression="zstd")
        log(f"  done: {OUT_PARQUET.stat().st_size / 1e9:.2f} GB on disk")

    write_summary(panel, rows_in, rows_out)


def write_summary(panel: pd.DataFrame, rows_in: int, rows_out: int) -> None:
    lines: list[str] = []
    p = lines.append

    p("PA Panel build summary")
    p("=" * 70)
    p(f"Rows read from plays.csv (pre-filter): not counted (streamed)")
    p(f"PA rows kept after game-set + pa==1 filter: {rows_in:,}")
    p(f"Final PA rows in panel: {rows_out:,}")
    p("")

    p("=== seasons ===")
    p(f"  range: {panel['season'].min()} - {panel['season'].max()}")
    by_season = panel.groupby("season").size()
    p(f"  median PAs/season: {int(by_season.median()):,}")
    p(f"  min  PAs/season:   {int(by_season.min()):,}  ({by_season.idxmin()})")
    p(f"  max  PAs/season:   {int(by_season.max()):,}  ({by_season.idxmax()})")
    p("")

    p("=== player coverage ===")
    n_bat = panel["bat_id"].nunique()
    n_pit = panel["pit_id"].nunique()
    p(f"  distinct batters:  {n_bat:,}")
    p(f"  distinct pitchers: {n_pit:,}")
    p("")

    p("=== missing-data rates ===")
    for col in ["bat_birth_year", "pit_birth_year", "bat_age", "pit_age",
                "bat_hand", "pit_hand"]:
        n_miss = panel[col].isna().sum()
        p(f"  {col:<20} missing: {n_miss:>10,d}  ({100*n_miss/len(panel):.2f}%)")
    p("")

    p("=== age distribution (batters) ===")
    p(panel["bat_age"].describe().to_string())
    p("")
    p("=== age distribution (pitchers) ===")
    p(panel["pit_age"].describe().to_string())
    p("")

    p("=== PA outcome counts ===")
    for col in ["single", "double", "triple", "hr", "walk", "iw", "hbp",
                "k", "sh", "sf", "roe", "xi", "fc", "othout"]:
        n = int(panel[col].sum())
        p(f"  {col:<10} {n:>12,d}  ({100*n/len(panel):.2f}%)")
    p("")

    p("=== wOBA value summary (excluding IBBs) ===")
    p(panel["woba_value"].describe().to_string())
    p("")

    p("=== pitchers batting ===")
    n_pb = int(panel["bat_is_pitcher"].sum())
    p(f"  PAs where batter is also a pitcher: {n_pb:,}  ({100*n_pb/len(panel):.2f}%)")
    p("")

    p("=== leagues ===")
    # Pull league via game_id is expensive, skip for now -- gameinfo filter
    # guaranteed AL/NL.
    p("  AL/NL only (enforced at game-filter stage)")
    p("")

    p("=== head ===")
    p(panel.head(10).to_string())

    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {OUT_SUMMARY}")
    print("\n".join(lines[:30]))
    print(f"...  (full summary at {OUT_SUMMARY})")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--dry-run", action="store_true",
                    help="Do not write the parquet output")
    args = ap.parse_args()

    t0 = time.time()
    build_panel(args.chunksize, args.dry_run)
    log(f"total runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
