"""
fetch_fielding_external.py  (run at home; needs network)
=======================================================

Pulls the external, modern fielding-validation metrics via pybaseball and keys
them to Retrosheet player ids so they merge onto the allocation panel:

  Outs Above Average (OAA)  -- Baseball Savant, 2016+, the gold-standard fielding
     metric, summed across the positions a player manned in a season.
  Sprint speed              -- Baseball Savant, the baserunning / athleticism
     control (a proxy for baserunning value; swap in FanGraphs BsR later if the
     site cooperates).

Both come keyed by MLBAM id; chadwick_register() supplies key_mlbam -> key_retro,
so the output is keyed by the same batter id as pa_panel. Catcher is unavailable
in the OAA leaderboard and is simply absent here. Writes fielding_external.csv:
  batter (Retrosheet id), season, oaa, frp (fielding runs prevented),
  sprint_speed, n_pos (positions manned).

Savant rate-limits, so the script sleeps between calls; a full 2016-2025 pull
takes a few minutes. Re-runs are cheap to cache by editing YEARS.
"""
from __future__ import annotations

import time

import pandas as pd

import meso_core as mc

try:
    import pybaseball as pb
except ImportError:
    raise SystemExit("pip install pybaseball")

YEARS = list(range(2016, 2026))
OAA_POS = ["1B", "2B", "3B", "SS", "LF", "CF", "RF"]   # catcher not in the OAA leaderboard
OUT = mc.DATA_DIR / "fielding_external.csv"
SLEEP = 1.0


def main():
    mc.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ---- OAA: loop year x position, sum to player-season ----
    oaa_rows = []
    for y in YEARS:
        for pos in OAA_POS:
            try:
                d = pb.statcast_outs_above_average(y, pos, min_att=1)
            except Exception as e:
                print(f"  [OAA {y} {pos}] skipped: {str(e)[:80]}"); continue
            if d is None or not len(d):
                continue
            keep = d[["player_id", "outs_above_average", "fielding_runs_prevented"]].copy()
            keep["season"] = y
            oaa_rows.append(keep)
            print(f"  OAA {y} {pos}: {len(keep)}")
            time.sleep(SLEEP)
    oaa = pd.concat(oaa_rows, ignore_index=True)
    oaa = (oaa.groupby(["player_id", "season"])
           .agg(oaa=("outs_above_average", "sum"),
                frp=("fielding_runs_prevented", "sum"),
                n_pos=("outs_above_average", "size")).reset_index())
    print(f"OAA player-seasons: {len(oaa):,}")

    # ---- sprint speed: one call per year ----
    sp_rows = []
    for y in YEARS:
        try:
            s = pb.statcast_sprint_speed(y, min_opp=10)
            s = s[["player_id", "sprint_speed"]].copy(); s["season"] = y
            sp_rows.append(s); print(f"  sprint {y}: {len(s)}")
            time.sleep(SLEEP)
        except Exception as e:
            print(f"  [sprint {y}] skipped: {str(e)[:80]}")
    sprint = pd.concat(sp_rows, ignore_index=True) if sp_rows else \
        pd.DataFrame(columns=["player_id", "sprint_speed", "season"])

    ext = oaa.merge(sprint, on=["player_id", "season"], how="left")

    # ---- map MLBAM id -> Retrosheet id via Chadwick ----
    reg = pb.chadwick_register(save=True)
    xwalk = (reg[["key_mlbam", "key_retro"]].dropna()
             .astype({"key_mlbam": "Int64"}).drop_duplicates("key_mlbam"))
    ext["player_id"] = pd.to_numeric(ext["player_id"], errors="coerce").astype("Int64")
    ext = ext.merge(xwalk, left_on="player_id", right_on="key_mlbam", how="left")
    matched = ext["key_retro"].notna().mean()
    print(f"chadwick match: {matched*100:.1f}% of player-seasons mapped to a Retrosheet id")
    ext = ext.dropna(subset=["key_retro"]).rename(columns={"key_retro": "batter"})

    # ---- baserunning runs (BsR) from FanGraphs, keyed via key_fangraphs (optional) ----
    # FanGraphs throttles cloud IPs; this often works from a home connection and
    # is skipped gracefully if it 403s. BsR is the proper baserunning control;
    # sprint_speed above is the reliable fallback.
    fg_xwalk = (reg[["key_fangraphs", "key_retro"]].dropna()
                .astype({"key_fangraphs": "Int64"}).drop_duplicates("key_fangraphs"))
    bsr_rows = []
    for y in YEARS:
        try:
            b = pb.batting_stats(y, y, qual=1)
            col = "IDfg" if "IDfg" in b.columns else ("playerid" if "playerid" in b.columns else None)
            if col and "BsR" in b.columns:
                bb = b[[col, "BsR"]].copy(); bb.columns = ["key_fangraphs", "bsr"]; bb["season"] = y
                bsr_rows.append(bb); print(f"  BsR {y}: {len(bb)}")
                time.sleep(SLEEP)
        except Exception as e:
            print(f"  [BsR {y}] skipped (FanGraphs): {str(e)[:70]}")
    if bsr_rows:
        bsr = pd.concat(bsr_rows, ignore_index=True)
        bsr["key_fangraphs"] = pd.to_numeric(bsr["key_fangraphs"], errors="coerce").astype("Int64")
        bsr = bsr.merge(fg_xwalk, on="key_fangraphs", how="left").dropna(subset=["key_retro"])
        bsr = bsr.rename(columns={"key_retro": "batter"})[["batter", "season", "bsr"]]
        ext = ext.merge(bsr, on=["batter", "season"], how="left")
        print(f"BsR merged for {ext['bsr'].notna().mean()*100:.0f}% of player-seasons")
    else:
        print("BsR unavailable (FanGraphs); validate_fielding will use sprint speed")
        ext["bsr"] = pd.NA

    out = ext[["batter", "season", "oaa", "frp", "sprint_speed", "bsr", "n_pos"]].sort_values(
        ["season", "batter"])
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(out):,} player-seasons, {out['batter'].nunique():,} players, "
          f"{int(out['season'].min())}-{int(out['season'].max())})")


if __name__ == "__main__":
    main()
