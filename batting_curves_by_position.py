"""
batting_curves_by_position.py
=============================

Descriptive first step for the age-and-league-dependent aging-curve model.

We posit: observed batting at age a is a mixture of (latent skill curve
g(a) + player level) censored/graded by a fielding-dependent selection
floor, where selection operates continuously through PLAYING TIME (PAs):
starters -> platoon -> bench -> pinch hitter -> out. Good fielders have a
low batting floor (survive on defense), so we observe their weak-bat ages;
bat-first players (1B/DH/corner OF) only appear near their batting peak.

This script does NOT model anything yet. It produces the raw descriptive
material to look at before choosing an estimator:

  For each (position group, age) -- and optionally split by era -- report
    - mean wOBA (PA-weighted)            : observed batting curve
    - total PA                           : exposure/selection curve
    - number of distinct players
    - PA per player                      : starter vs bench tilt

Position groups (from retrosheet batter fielding position bat_f, the
position the batter occupied IN THAT GAME; we take each player-season's
MODAL position to assign a season-level group, so a PA is grouped by the
player's primary position that year, not his role in that one game):

    C  = catcher (2)
    IF_premium = SS, 2B, 3B (6,4,5)   [up-the-middle/hot-corner gloves]
    CF = center field (8)
    CORNER_OF = LF, RF (7,9)
    1B = first base (3)
    DH = designated hitter (10)
    P  = pitcher (1)   [excluded from batting analysis but reported]

We assign each (player, season) a primary position = the position at which
he made the most PAs that season. PAs are then grouped by that primary
position, so a shortstop who occasionally DHs is "IF_premium" all year.

Data source: retrosheet (panel lacks per-PA position). We read plays.csv
for bat_f, batter, gid; gameinfo for season + league; biofile for age.

Eras (coarse, editable):
    pre-integration  1910-1946
    integration      1947-1968
    expansion/FA     1969-1993
    steroid          1994-2005
    modern           2006-2025

Outputs:
    C:\\baseball_eras\\output\\batting_curves_by_position.csv
        cols: era, league, pos_group, age, n_pa, mean_woba, n_players, pa_per_player
    C:\\baseball_eras\\output\\batting_curves_by_position_summary.txt

Notes:
  - IBBs excluded from wOBA (set NaN), included nowhere in mean.
  - wOBA computed with season-specific FanGraphs weights here (descriptive
    only; weighting scheme doesn't matter for curve SHAPE comparisons).
  - Pitchers (pos 1) excluded from the batting curves but their PA counts
    are reported in a separate row group for reference.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\baseball_eras")
RETRO_DIR   = Path(r"C:\overnight_effect_data\retrosheet")

PLAYS_CSV    = RETRO_DIR / "plays.csv"
GAMEINFO_CSV = RETRO_DIR / "gameinfo.csv"
BIOFILE_CSV  = RETRO_DIR / "biofile0.csv"
TEAMS_CSV    = PROJECT_DIR / "Teams.csv"
WOBA_CSV     = PROJECT_DIR / "wOBA_weights.csv"

OUT_CSV = PROJECT_DIR / "output" / "batting_curves_by_position.csv"
OUT_SUM = PROJECT_DIR / "output" / "batting_curves_by_position_summary.txt"

AGE_LO, AGE_HI = 18, 44

POS_GROUP = {
    1:  "P",
    2:  "C",
    3:  "1B",
    4:  "2B",
    5:  "3B",
    6:  "SS",
    7:  "CORNER_OF",   # LF
    8:  "CF",
    9:  "CORNER_OF",   # RF
    10: "DH",
}

ERAS = [
    ("pre_integration", 1910, 1946),
    ("integration",     1947, 1968),
    ("expansion_FA",    1969, 1993),
    ("steroid",         1994, 2005),
    ("modern",          2006, 2025),
]

WEIGHT_COLS = ["wBB", "wHBP", "w1B", "w2B", "w3B", "wHR"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def baseball_age(bd, season):
    by = bd // 10000; bm = (bd // 100) % 100; bdd = bd % 100
    age = season - by
    after = (bm > 6) | ((bm == 6) & (bdd > 30))
    return age - after.astype(int)


def era_of(season):
    for name, lo, hi in ERAS:
        if lo <= season <= hi:
            return name
    return "other"


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # --- plays: batter, position, hit-type flags, gid ---
    log("reading plays.csv")
    pcols = ["gid", "batter", "bat_f", "pa",
             "single", "double", "triple", "hr", "walk", "iw", "hbp"]
    plays = pd.read_csv(
        PLAYS_CSV,
        usecols=lambda c: c in pcols,
        dtype={"gid": "string", "batter": "string", "bat_f": "Int16", "pa": "Int8",
               "single": "Int8", "double": "Int8", "triple": "Int8", "hr": "Int8",
               "walk": "Int8", "iw": "Int8", "hbp": "Int8"},
        low_memory=False,
    )
    plays = plays[plays["pa"] == 1]
    log(f"  {len(plays):,} PA rows")

    # --- gameinfo: season, league of home (for AL/NL tag of the game) ---
    log("reading gameinfo.csv")
    gi = pd.read_csv(GAMEINFO_CSV,
                     usecols=["gid", "season", "gametype", "hometeam"],
                     dtype={"gid": "string", "season": "Int32",
                            "gametype": "string", "hometeam": "string"})
    gi = gi[gi["gametype"] == "regular"]
    plays = plays.merge(gi[["gid", "season", "hometeam"]], on="gid", how="inner")
    log(f"  {len(plays):,} regular-season PA rows")

    # batter league via Teams.csv on (season, hometeam)? No -- we want the
    # batter's own league. Simplest: map hometeam... but batter may be away.
    # For a descriptive split we use the batter's TEAM league. We don't have
    # batteam here; approximate league by home park league is wrong for away
    # batters. Instead, skip league split for v1 and report league=ALL.
    plays["league"] = "ALL"

    # --- biofile: age ---
    log("reading biofile")
    bio = pd.read_csv(BIOFILE_CSV, usecols=["id", "birthdate"],
                      dtype={"id": "string", "birthdate": "Int64"})
    bio = bio.rename(columns={"id": "batter", "birthdate": "bd"})
    plays = plays.merge(bio, on="batter", how="left")
    plays = plays.dropna(subset=["bd", "season"])
    plays["age"] = baseball_age(plays["bd"].astype("int64"),
                                plays["season"].astype("int64"))
    plays = plays[(plays["age"] >= AGE_LO) & (plays["age"] <= AGE_HI)]
    log(f"  {len(plays):,} PA rows with valid age")

    # --- primary position per (player, season) = modal bat_f by PA count ---
    log("assigning primary position per (player, season)")
    pos_counts = (plays.dropna(subset=["bat_f"])
                       .groupby(["batter", "season", "bat_f"]).size()
                       .rename("n").reset_index())
    # pick max-n position per player-season
    idx = pos_counts.groupby(["batter", "season"])["n"].idxmax()
    primary = pos_counts.loc[idx, ["batter", "season", "bat_f"]].rename(
        columns={"bat_f": "primary_pos"})
    plays = plays.merge(primary, on=["batter", "season"], how="left")
    plays["pos_group"] = plays["primary_pos"].map(POS_GROUP).fillna("UNK")

    # --- wOBA with season weights ---
    log("computing wOBA (season weights)")
    w = pd.read_csv(WOBA_CSV, encoding="utf-8-sig").rename(columns={"Season": "season"})
    w = w[["season"] + WEIGHT_COLS]
    plays = plays.merge(w, on="season", how="left")
    bb_u = (plays["walk"] - plays["iw"]).clip(lower=0)
    plays["woba_value"] = (
        plays["wBB"]*bb_u + plays["wHBP"]*plays["hbp"] + plays["w1B"]*plays["single"]
        + plays["w2B"]*plays["double"] + plays["w3B"]*plays["triple"] + plays["wHR"]*plays["hr"]
    )
    plays.loc[plays["iw"] == 1, "woba_value"] = np.nan
    plays = plays[plays["iw"] != 1]

    plays["era"] = plays["season"].astype(int).map(era_of)

    # --- aggregate ---
    log("aggregating by (era, pos_group, age)")
    grp = plays.groupby(["era", "pos_group", "age"], observed=True)
    agg = grp.agg(
        n_pa=("woba_value", "size"),
        sum_woba=("woba_value", "sum"),
        n_players=("batter", "nunique"),
    ).reset_index()
    agg["mean_woba"] = agg["sum_woba"] / agg["n_pa"]
    agg["pa_per_player"] = agg["n_pa"] / agg["n_players"]
    agg["league"] = "ALL"
    agg = agg[["era", "league", "pos_group", "age", "n_pa", "mean_woba",
               "n_players", "pa_per_player"]]
    agg.to_csv(OUT_CSV, index=False, float_format="%.5f")
    log(f"wrote {OUT_CSV}")

    # --- summary 1: position-level by era (prime-age wOBA), the key view ---
    lines = []
    P = lines.append
    P("Observed batting by position group and era")
    P("=" * 70)
    P("Primary position = modal fielding position (by PA) per player-season.")
    P("wOBA is PA-weighted, season weights.")
    P("")

    groups = ["C", "SS", "2B", "3B", "CF", "CORNER_OF", "1B", "DH"]
    era_names = [e[0] for e in ERAS]

    # PRIME-AGE LEVEL (ages 26-31) by position x era -- the floor over time.
    P("=== Prime-age (26-31) PA-weighted mean wOBA, by position x era ===")
    P("(This is the 'positional floor' and how it moves over time.)")
    P("")
    prime = plays[(plays["age"] >= 26) & (plays["age"] <= 31)]
    pe = (prime.groupby(["pos_group", "era"], observed=True)
               .agg(n_pa=("woba_value", "size"), sw=("woba_value", "sum"))
               .reset_index())
    pe["mw"] = pe["sw"] / pe["n_pa"]
    # also a league-wide prime-age baseline per era to show RELATIVE position
    base = (prime.groupby("era", observed=True)
                 .agg(n_pa=("woba_value", "size"), sw=("woba_value", "sum")))
    base["mw"] = base["sw"] / base["n_pa"]
    header = f"{'pos':>10s}  " + "  ".join(f"{e[:9]:>9s}" for e in era_names)
    P(header)
    for g in groups:
        cells = []
        for en in era_names:
            row = pe[(pe["pos_group"] == g) & (pe["era"] == en)]
            if len(row) and int(row["n_pa"].iloc[0]) >= 500:
                cells.append(f"{row['mw'].iloc[0]:>9.4f}")
            else:
                cells.append(f"{'--':>9s}")
        P(f"{g:>10s}  " + "  ".join(cells))
    # baseline row
    cells = []
    for en in era_names:
        if en in base.index:
            cells.append(f"{base.loc[en,'mw']:>9.4f}")
        else:
            cells.append(f"{'--':>9s}")
    P(f"{'ALL':>10s}  " + "  ".join(cells))
    P("")

    # RELATIVE to league baseline (position minus all) -- isolates the
    # positional floor net of run-environment changes.
    P("=== Same, RELATIVE to that era's all-position prime-age mean ===")
    P("(positive = position hit better than league average that era)")
    P("")
    P(header)
    for g in groups:
        cells = []
        for en in era_names:
            row = pe[(pe["pos_group"] == g) & (pe["era"] == en)]
            if len(row) and int(row["n_pa"].iloc[0]) >= 500 and en in base.index:
                rel = row["mw"].iloc[0] - base.loc[en, "mw"]
                cells.append(f"{rel:>+9.4f}")
            else:
                cells.append(f"{'--':>9s}")
        P(f"{g:>10s}  " + "  ".join(cells))
    P("")

    # --- summary 2: full age curves by position x era to CSV only (already
    #     written); print SS age curve across eras as the worked example. ---
    P("=== Worked example: SS observed wOBA age curve, by era ===")
    P("(the shortstop-as-hitting-position change over time)")
    P("")
    ss = (plays[plays["pos_group"] == "SS"]
              .groupby(["era", "age"], observed=True)
              .agg(n_pa=("woba_value", "size"), sw=("woba_value", "sum"))
              .reset_index())
    ss["mw"] = ss["sw"] / ss["n_pa"]
    P(f"{'age':>4s}  " + "  ".join(f"{e[:9]:>9s}" for e in era_names))
    for age in range(22, 39):
        cells = []
        for en in era_names:
            row = ss[(ss["era"] == en) & (ss["age"] == age)]
            if len(row) and int(row["n_pa"].iloc[0]) >= 300:
                cells.append(f"{row['mw'].iloc[0]:>9.4f}")
            else:
                cells.append(f"{'--':>9s}")
        P(f"{age:>4d}  " + "  ".join(cells))

    OUT_SUM.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {OUT_SUM}")
    print("\n".join(lines))
    print(f"... full position x era x age grid in {OUT_CSV}")


if __name__ == "__main__":
    main()
