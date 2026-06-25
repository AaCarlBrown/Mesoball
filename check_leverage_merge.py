"""
check_leverage_merge.py
=======================
Verify leverage_panel.parquet joins onto pa_panel.parquet by (game_id, pa_seq).
Reports the merge rate and, if it is clean, a few sanity checks that the joined
base-out-score state is real rather than coincidentally aligned.

Run:  py check_leverage_merge.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\baseball_eras")
PA = ROOT / "data" / "pa_panel.parquet"
LEV = ROOT / "data" / "leverage_panel.parquet"


def main():
    print(f"pa_panel : {PA} (exists={PA.exists()})")
    print(f"leverage : {LEV} (exists={LEV.exists()})")
    if not (PA.exists() and LEV.exists()):
        print("missing a file; edit paths at top."); return

    pa = pd.read_parquet(PA)
    # pa_panel uses 'game_id'; leverage_panel uses 'gid'. Align names.
    if "game_id" in pa.columns and "gid" not in pa.columns:
        pa = pa.rename(columns={"game_id": "gid"})
    # pa_seq = within-game running index in FILE ORDER (same construction as leverage)
    pa = pa.reset_index(drop=True)
    pa["pa_seq"] = pa.groupby("gid").cumcount()
    print(f"pa_panel rows: {len(pa):,}  games: {pa['gid'].nunique():,}")

    lev = pd.read_parquet(LEV)
    print(f"leverage rows: {len(lev):,}  games: {lev['gid'].nunique():,}")

    merged = pa.merge(lev, on=["gid", "pa_seq"], how="left", suffixes=("", "_lev"))
    rate = merged["base_state"].notna().mean()
    print(f"\nMERGE RATE: {rate*100:.2f}%  ({merged['base_state'].notna().sum():,} of {len(merged):,})")

    # where do misses concentrate? (era, and games present in pa but not lev)
    miss = merged[merged["base_state"].isna()]
    if len(miss):
        print(f"\nmisses: {len(miss):,}")
        if "season" in merged.columns:
            by = miss.groupby(miss["season"] // 10 * 10).size().sort_values(ascending=False).head(6)
            print("  by decade (top):")
            for d, n in by.items():
                print(f"    {int(d)}s: {n:,}")
        pa_games = set(pa["gid"]); lev_games = set(lev["gid"])
        print(f"  pa games not in leverage: {len(pa_games - lev_games):,}")
        # within a shared game, do PA counts match? (key-length mismatch = PA-def differs)
        shared = list(pa_games & lev_games)[:5000]
        pac = pa[pa["gid"].isin(shared)].groupby("gid").size()
        levc = lev[lev["gid"].isin(shared)].groupby("gid").size()
        cmp = pd.DataFrame({"pa": pac, "lev": levc}).dropna()
        mismatch = (cmp["pa"] != cmp["lev"]).mean()
        print(f"  of {len(cmp):,} shared games sampled, {mismatch*100:.1f}% have differing PA counts")
        print("  (high % differing -> PA-row definitions differ; that's the thing to align)")

    # sanity checks on the matched rows
    ok = merged[merged["base_state"].notna()]
    if len(ok):
        print("\nsanity checks on matched rows:")
        # 1. outs in 0..2
        print(f"  outs distribution: {ok['outs'].value_counts().sort_index().to_dict()}")
        # 2. base_state 0..7
        print(f"  base_state range: {int(ok['base_state'].min())}..{int(ok['base_state'].max())}"
              f"  (mean occ bits ~ {ok['base_state'].apply(lambda x: bin(int(x)).count('1')).mean():.2f})")
        # 3. HR should score: when the prior was a HR the score should move; check that
        #    bases-empty share is plausible (~ half of PAs leadoff-ish)
        print(f"  bases-empty share: {(ok['base_state'] == 0).mean()*100:.1f}%  "
              f"(typical ~55-60%)")
        # 4. late_close share
        if "late_close" in ok.columns:
            print(f"  late_close share: {ok['late_close'].mean()*100:.1f}%  (typical ~8-12%)")
        # 5. does woba_value rise with men on? (RE expectation: more value created with runners)
        if "woba_value" in ok.columns:
            on = ok[ok["base_state"] > 0]["woba_value"].mean()
            empty = ok[ok["base_state"] == 0]["woba_value"].mean()
            print(f"  mean woba_value bases-empty {empty:.3f} vs men-on {on:.3f}")


if __name__ == "__main__":
    main()
