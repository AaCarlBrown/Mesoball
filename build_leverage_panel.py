"""
build_leverage_panel.py   (overnight rerun)
===========================================

Extracts the per-PA situational state the clutch/leverage analysis needs -- inning,
half, outs, base occupancy, score differential -- which the main panel build drops,
and writes leverage_panel.parquet keyed by (gid, pa_seq) for merging onto pa_panel.

THE NAMING PROBLEM, HANDLED.  plays.csv has 177 columns; the main build reads only
~25 friendly ones, so I do not know what the outs / runners / score columns are
called in YOUR file. So this script:
  1. prints the FULL column list (and writes it to leverage_columns.txt) FIRST,
  2. auto-detects the situational columns by trying common name patterns,
  3. logs exactly which columns it chose,
  4. proceeds if it found them; if not, it still writes the column list so we can
     set the mapping by hand tomorrow -- the run is never wasted.
If auto-detection picks wrong, set the names in CONFIG below and rerun.

PA ALIGNMENT.  pa_seq = the within-game running index of PA-terminating rows, in
file order, defined by the SAME outcome flags the main build uses (single..othout).
pa_panel rows are in the same file order, so tomorrow we add the identical
pa_seq to pa_panel via groupby('gid').cumcount() and merge on (gid, pa_seq).

LEVERAGE.  Tonight we extract the STATE fields (the expensive part). A coarse
leverage proxy (late-and-close + a base-out weight) is computed now; the exact
Tango Leverage Index is a cheap downstream step once the fields are in hand.

Run:  py build_leverage_panel.py
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(r"C:\baseball_eras")
RETRO_DIR = Path(r"C:\overnight_effect_data\retrosheet")
PLAYS_CSV = next((p for p in [RETRO_DIR / "plays.csv",
                              PROJECT_DIR / "data" / "plays.csv",
                              PROJECT_DIR / "plays.csv"] if p.exists()),
                 RETRO_DIR / "plays.csv")
OUT_PARQUET = PROJECT_DIR / "data" / "leverage_panel.parquet"
OUT_COLS = PROJECT_DIR / "output" / "leverage_columns.txt"
CHUNK = 1_000_000

# ---- CONFIG: locked from leverage_columns.txt (your plays.csv names) -----------
CONFIG = dict(gid="gid", inning="inning", half="top_bot", outs="outs_pre",
              run1="br1_pre", run2="br2_pre", run3="br3_pre",
              bat_score=None, fld_score=None,            # not present; use home/away
              home_score="score_h", away_score="score_v")
# extra fields to carry through if present (count is logged for the modern era)
EXTRA_COLS = ["balls", "strikes", "count"]

# outcome flags that mark a PA-terminating row (the main build's friendly names)
PA_FLAGS = ["single", "double", "triple", "hr", "walk", "ibb", "hbp", "k",
            "sh", "sf", "roe", "xi", "fc", "othout"]

# candidate name patterns for auto-detection (lowercased substring match, in order)
PATTERNS = dict(
    gid=["gid", "game_id", "gameid"],
    inning=["inning", "inn_ct", "inning_ct", "inn"],
    half=["half", "top_bot", "bat_home_id", "battedteam", "bat_team_id", "top"],
    outs=["outs_ct", "outs_pre", "outs", "out_ct", "pre_outs", "start_outs"],
    run1=["run1_id", "br1_pre", "first_runner", "on_1b", "base1", "runner_1b", "r1"],
    run2=["run2_id", "br2_pre", "second_runner", "on_2b", "base2", "runner_2b", "r2"],
    run3=["run3_id", "br3_pre", "third_runner", "on_3b", "base3", "runner_3b", "r3"],
    bat_score=["bat_score", "bat_score_ct", "batting_score"],
    fld_score=["fld_score", "fld_score_ct", "field_score", "pitching_score"],
    home_score=["home_score", "home_score_ct", "score_home"],
    away_score=["away_score", "vis_score", "away_score_ct", "score_away", "vis_score_ct"],
)


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}")


def detect(cols):
    lower = {c.lower(): c for c in cols}
    chosen = {}
    for key, pats in PATTERNS.items():
        if CONFIG.get(key):
            chosen[key] = CONFIG[key]; continue
        for p in pats:
            hit = next((orig for low, orig in lower.items() if p == low), None) \
                or next((orig for low, orig in lower.items() if p in low), None)
            if hit:
                chosen[key] = hit; break
    return chosen


def base_state_code(r1, r2, r3, n):
    """3-bit base occupancy 0..7 from runner columns. A base is occupied unless its
    field is an empty marker. Handles both runner-id strings (empty='' when vacant)
    and 0/1 occupancy columns."""
    EMPTY = {"", "0", "0.0", "nan", "none", "<na>", "?", "-1"}
    def occ(s):
        if s is None:
            return np.zeros(n, dtype=np.int8)
        v = pd.Series(s)
        isna = v.isna()
        sv = v.astype(str).str.strip().str.lower()
        occupied = ~(isna | sv.isin(EMPTY))     # NaN/NA counts as an empty base
        return occupied.to_numpy().astype(np.int8)
    return occ(r1) + 2 * occ(r2) + 4 * occ(r3)


def main():
    OUT_COLS.parent.mkdir(parents=True, exist_ok=True)
    if not PLAYS_CSV.exists():
        log(f"plays.csv not found at {PLAYS_CSV}. Edit PLAYS_CSV at top."); return
    header = pd.read_csv(PLAYS_CSV, nrows=0).columns.tolist()
    OUT_COLS.write_text("\n".join(header), encoding="utf-8")
    log(f"plays.csv has {len(header)} columns; written to {OUT_COLS}")
    chosen = detect(header)
    log("auto-detected situational columns:")
    for k in PATTERNS:
        log(f"    {k:>11}: {chosen.get(k, '*** NOT FOUND ***')}")
    have_flags = [f for f in PA_FLAGS if f in header]
    log(f"PA-flag columns present: {have_flags}")

    essential = ["gid", "inning", "outs"]
    missing_ess = [k for k in essential if k not in chosen]
    if missing_ess or not have_flags:
        log(f"MISSING ESSENTIALS {missing_ess or ''} {'(no PA flags)' if not have_flags else ''}.")
        log("Column list is saved. Set names in CONFIG from leverage_columns.txt and rerun.")
        return

    have_extra = [c for c in EXTRA_COLS if c in header]
    usecols = sorted(set([chosen[k] for k in chosen] + have_flags + have_extra))
    score_mode = "batfld" if ("bat_score" in chosen and "fld_score" in chosen) else \
                 ("homeaway" if ("home_score" in chosen and "away_score" in chosen) else None)
    log(f"score mode: {score_mode}; count fields: {have_extra}; reading {len(usecols)} columns")

    parts = []
    seq_counter = {}            # running PA index per gid across chunks
    rows_in = rows_pa = 0
    for ch in pd.read_csv(PLAYS_CSV, usecols=usecols, chunksize=CHUNK, low_memory=False):
        rows_in += len(ch)
        is_pa = np.zeros(len(ch), dtype=bool)
        for f in have_flags:
            is_pa |= (ch[f].fillna(0).to_numpy() != 0)
        ch = ch[is_pa].copy()
        rows_pa += len(ch)
        g = ch[chosen["gid"]].to_numpy()
        # within-gid running PA index continued across chunks
        seq = np.empty(len(ch), dtype=np.int32)
        # vectorized cumcount within chunk, offset by prior counts per gid
        order = pd.Series(np.arange(len(ch)))
        cc = ch.groupby(chosen["gid"]).cumcount().to_numpy()
        base = np.array([seq_counter.get(x, 0) for x in g])
        seq = base + cc
        # update counters
        last = pd.Series(seq, index=g).groupby(level=0).max() + 1
        for gid_v, mx in last.items():
            seq_counter[gid_v] = int(mx)

        out = pd.DataFrame({"gid": g, "pa_seq": seq,
                            "inning": pd.to_numeric(ch[chosen["inning"]], errors="coerce").to_numpy(),
                            "outs": pd.to_numeric(ch[chosen["outs"]], errors="coerce").to_numpy()})
        if "half" in chosen:
            out["half_raw"] = ch[chosen["half"]].to_numpy()
        out["base_state"] = base_state_code(ch.get(chosen.get("run1")),
                                            ch.get(chosen.get("run2")),
                                            ch.get(chosen.get("run3")), len(ch))
        if score_mode == "batfld":
            out["score_diff"] = ch[chosen["bat_score"]].to_numpy() - ch[chosen["fld_score"]].to_numpy()
        elif score_mode == "homeaway":
            sh = pd.to_numeric(ch[chosen["home_score"]], errors="coerce").to_numpy()
            sv = pd.to_numeric(ch[chosen["away_score"]], errors="coerce").to_numpy()
            if "half" in chosen:
                # batting team = home when bottom half. top_bot: 1/'B'/'bot' = bottom.
                hv = pd.Series(ch[chosen["half"]]).astype(str).str.lower().str.strip()
                bat_home = hv.isin(["1", "b", "bot", "bottom", "true"]).to_numpy()
                bat = np.where(bat_home, sh, sv); fld = np.where(bat_home, sv, sh)
                out["score_diff"] = bat - fld          # batting team's lead (+) / deficit (-)
            else:
                out["score_diff"] = sh - sv
        for c in have_extra:
            out[c] = pd.to_numeric(ch[c], errors="coerce").to_numpy()
        parts.append(out)
        log(f"  processed {rows_in:,} rows ({rows_pa:,} PAs)")

    panel = pd.concat(parts, ignore_index=True)
    # coarse leverage proxy (refine to exact Tango LI downstream)
    base_out_w = {0: 0.8, 1: 1.0, 2: 1.1, 3: 1.2, 4: 1.1, 5: 1.3, 6: 1.4, 7: 1.5}
    w = panel["base_state"].map(base_out_w).fillna(1.0) * (1.0 + 0.15 * (2 - panel["outs"].clip(0, 2)))
    late = (panel["inning"] >= 7).astype(float)
    close = (panel.get("score_diff", pd.Series(0, index=panel.index)).abs() <= 1).astype(float) \
        if "score_diff" in panel.columns else 1.0
    panel["lev_proxy"] = w * (1 + late) * (1 + 0.5 * close if "score_diff" in panel.columns else 1)
    panel["late_close"] = (late.astype(bool) & (close.astype(bool) if "score_diff" in panel.columns else True))

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PARQUET, index=False, compression="zstd")
    log(f"wrote {OUT_PARQUET}: {len(panel):,} PA rows, {panel['gid'].nunique():,} games")
    log("TO MERGE TOMORROW: add pa_seq to pa_panel via")
    log("  pa = pd.read_parquet('pa_panel.parquet')")
    log("  pa['pa_seq'] = pa.groupby('gid').cumcount()   # same file order")
    log("  pa = pa.merge(leverage_panel, on=['gid','pa_seq'], how='left')")
    log("  (verify the merge rate is ~100%; if not, the PA-flag filter differs and we align it)")


if __name__ == "__main__":
    t0 = time.time()
    main()
    log(f"total runtime: {(time.time()-t0)/60:.1f} min")
