"""
dh_age_distribution.py
======================

Descriptive check: what is the age distribution of designated hitters?

The DH mechanism assumes DH PAs skew old. This script verifies that
directly from retrosheet, by identifying PAs where the batter's fielding
"position" is DH (retrosheet field-position code 10), then tabulating the
age distribution of DH PAs vs all position-player PAs.

It first tries to identify the DH-PA flag from the PA panel (in case a
fielding-position column exists). If not present, it reads the needed
columns from retrosheet plays.csv + gameinfo.csv + biofile, computing
the batter's age on the fly.

Retrosheet position encoding for the batter:
  Many retrosheet 'plays.csv' builds carry a column giving the batter's
  defensive position (e.g. 'batter_fld_pos', 'pos', or 'dh'). Position
  code 10 = designated hitter. We auto-detect which column is present.

Outputs:
    C:\\baseball_eras\\output\\dh_age_distribution.csv
        age, dh_pa, nondh_pos_pa, dh_share_of_age, age_share_within_dh
    C:\\baseball_eras\\output\\dh_age_distribution_summary.txt

Usage:
    py dh_age_distribution.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"C:\baseball_eras")
RETRO_DIR   = Path(r"C:\overnight_effect_data\retrosheet")
PANEL_PATH  = PROJECT_DIR / "data" / "pa_panel.parquet"

PLAYS_CSV    = RETRO_DIR / "plays.csv"
GAMEINFO_CSV = RETRO_DIR / "gameinfo.csv"
BIOFILE_CSV  = RETRO_DIR / "biofile0.csv"

OUT_CSV = PROJECT_DIR / "output" / "dh_age_distribution.csv"
OUT_SUM = PROJECT_DIR / "output" / "dh_age_distribution_summary.txt"

AGE_LO, AGE_HI = 18, 45

# Candidate column names in plays.csv that encode the batter's position.
POS_CANDIDATES = ["bat_f", "batter_fld_pos", "bat_fld_cd", "bat_field_pos",
                  "batpos", "bat_pos", "pos", "dh"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def baseball_age(birth_yyyymmdd, season):
    by = birth_yyyymmdd // 10000
    bm = (birth_yyyymmdd // 100) % 100
    bd = birth_yyyymmdd % 100
    age = season - by
    after = (bm > 6) | ((bm == 6) & (bd > 30))
    return age - after.astype(int)


def detect_pos_column() -> str | None:
    # Peek at plays.csv header
    hdr = pd.read_csv(PLAYS_CSV, nrows=0)
    for c in POS_CANDIDATES:
        if c in hdr.columns:
            return c
    # Show what's available so the user can tell us
    log("Could not auto-detect a batter-position column in plays.csv.")
    log("Columns containing 'pos', 'fld', 'dh', or 'bat':")
    for c in hdr.columns:
        cl = c.lower()
        if any(k in cl for k in ["pos", "fld", "dh", "bat"]):
            log(f"    {c}")
    return None


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    pos_col = detect_pos_column()
    if pos_col is None:
        log("Aborting: tell me which plays.csv column holds the batter's "
            "fielding position (DH = code 10) and I'll wire it in.")
        return
    log(f"using batter-position column: {pos_col!r}")
    _vc = pd.read_csv(PLAYS_CSV, usecols=[pos_col], nrows=500000)[pos_col].value_counts(dropna=False)
    log(f"  position-code frequencies (first 500k rows):\n{_vc.to_string()}")

    # --- read plays: gid, batter, position, and enough to get season+age ---
    log("reading plays.csv (batter, position, date)")
    usecols = ["gid", "batter", pos_col]
    # date may be on plays or gameinfo; prefer gameinfo for season
    plays = pd.read_csv(PLAYS_CSV, usecols=lambda c: c in usecols + ["pa"],
                        dtype={"gid": "string", "batter": "string",
                               pos_col: "Int16", "pa": "Int8"},
                        low_memory=False)
    plays = plays[plays["pa"] == 1]
    log(f"  {len(plays):,} PA rows")

    log("reading gameinfo.csv (gid -> season)")
    gi = pd.read_csv(GAMEINFO_CSV, usecols=["gid", "season", "gametype"],
                     dtype={"gid": "string", "season": "Int32"})
    gi = gi[gi["gametype"] == "regular"]
    plays = plays.merge(gi[["gid", "season"]], on="gid", how="inner")
    log(f"  {len(plays):,} PA rows in regular-season games")

    log("reading biofile (batter -> birthdate)")
    bio = pd.read_csv(BIOFILE_CSV, usecols=["id", "birthdate"],
                      dtype={"id": "string", "birthdate": "Int64"})
    bio = bio.rename(columns={"id": "batter", "birthdate": "bdate"})
    plays = plays.merge(bio, on="batter", how="left")
    plays = plays.dropna(subset=["bdate", "season"])
    plays["age"] = baseball_age(plays["bdate"].astype("int64"),
                                plays["season"].astype("int64"))
    plays = plays[(plays["age"] >= AGE_LO) & (plays["age"] <= AGE_HI)]
    log(f"  {len(plays):,} PA rows with valid age in [{AGE_LO},{AGE_HI}]")

    # Classify: DH (pos==10), pitcher (pos==1), else position player
    plays["is_dh"]      = (plays[pos_col] == 10)
    plays["is_pitcher"] = (plays[pos_col] == 1)
    plays["is_pos"]     = ~plays["is_dh"] & ~plays["is_pitcher"]

    log("tabulating by age")
    g = plays.groupby("age")
    tab = pd.DataFrame({
        "dh_pa":        g["is_dh"].sum().astype(int),
        "pos_pa":       g["is_pos"].sum().astype(int),
        "pitcher_pa":   g["is_pitcher"].sum().astype(int),
    })
    tab["total_nonpitcher"] = tab["dh_pa"] + tab["pos_pa"]
    tab["dh_share_of_age"]  = tab["dh_pa"] / tab["total_nonpitcher"].replace(0, np.nan)
    tab["age_share_within_dh"] = tab["dh_pa"] / tab["dh_pa"].sum()
    tab["age_share_within_pos"] = tab["pos_pa"] / tab["pos_pa"].sum()
    tab = tab.reset_index()
    tab.to_csv(OUT_CSV, index=False, float_format="%.5f")
    log(f"wrote {OUT_CSV}")

    # Summary stats
    def wmean(values, weights):
        w = weights.to_numpy(dtype=float)
        v = values.to_numpy(dtype=float)
        return (v * w).sum() / w.sum()

    dh_mean_age  = wmean(tab["age"], tab["dh_pa"])
    pos_mean_age = wmean(tab["age"], tab["pos_pa"])

    lines = []
    P = lines.append
    P("Age distribution of designated hitters")
    P("=" * 70)
    P(f"DH-PA mean age:           {dh_mean_age:.2f}")
    P(f"Position-player-PA mean:  {pos_mean_age:.2f}")
    P(f"Difference (DH older by): {dh_mean_age - pos_mean_age:+.2f} years")
    P("")
    P(f"Total DH PAs:        {int(tab['dh_pa'].sum()):,}")
    P(f"Total pos-player PAs:{int(tab['pos_pa'].sum()):,}")
    P("")
    P("Share of each age's non-pitcher PAs that are DH PAs,")
    P("and how DH PAs are distributed across ages vs position players:")
    P("")
    P(f"{'age':>4s}  {'dh_pa':>10s}  {'pos_pa':>11s}  {'dh_share':>9s}  "
      f"{'dh_agedist':>10s}  {'pos_agedist':>11s}")
    for _, r in tab.iterrows():
        P(f"{int(r['age']):>4d}  {int(r['dh_pa']):>10,d}  {int(r['pos_pa']):>11,d}  "
          f"{r['dh_share_of_age']:>9.4f}  {r['age_share_within_dh']:>10.4f}  "
          f"{r['age_share_within_pos']:>11.4f}")
    OUT_SUM.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {OUT_SUM}")
    print("\n".join(lines[:12]))
    print(f"... full table in {OUT_SUM}")


if __name__ == "__main__":
    main()
