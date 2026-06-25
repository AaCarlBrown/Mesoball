"""
clutch.py  (exploratory)
========================

Do hitters perform worse when it matters -- and do *clutch hitters* exist?
Cramer (1977) found no persistent clutch ability; Tango et al. (2007) found a
small real spread. Both worked without the move that matters here: a batter in a
high-leverage spot faces the *closer*, not an average pitcher, so a naive "clutch
penalty" is partly just facing better arms. We net the opponent out.

  value_t  ->  v_bs   = value - season mean                  (environment)
           ->  v_b    = v_bs  - batter mean(v_bs)            ("raw": net season+batter)
           ->  resid  = v_b   - pitcher mean(v_b)            ("purged": also net pitcher)

  raw clutch    = mean(v_b   | high leverage) - mean(v_b   | low)     <- includes opponent selection
  purged clutch = mean(resid | high leverage) - mean(resid | low)     <- opponent netted out
  selection     = raw - purged                                        <- the closer effect

High leverage = late and close (inning >= 7, |score_diff| <= 1); a base-out/leverage-
proxy tertile split is reported alongside. Cross-player clutch skill (per-batter
purged high-minus-low), EB-shrunk with an upper bound, answers Cramer directly.

Run:  py clutch.py --synthetic   |   py clutch.py
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\baseball_eras")
PA = ROOT / "data" / "pa_panel.parquet"
LEV = ROOT / "data" / "leverage_panel.parquet"
OUTPUT_DIR = ROOT / "output" if (ROOT / "output").exists() else Path(".")
MIN_HL = 50            # min high- and low-leverage PAs for a batter to enter the spread

L: list[str] = []
def P(s: str = "") -> None:
    print(s); L.append(s)


def purge_layers(df: pd.DataFrame, val: np.ndarray):
    """Return (v_b 'raw' net season+batter, resid 'purged' also net pitcher)."""
    s = pd.Series(val)
    v = s - s.groupby(df["season"].to_numpy()).transform("mean")
    v_b = v - v.groupby(df["batter"].to_numpy()).transform("mean")
    resid = v_b - v_b.groupby(df["pitcher"].to_numpy()).transform("mean")
    return v_b.to_numpy(), resid.to_numpy()


def eb_spread(per_player: pd.DataFrame, col: str):
    """per_player has columns: value (the per-batter clutch diff), w (weight ~ n_eff),
    samp_var (sampling variance of that batter's diff). Returns mean, true SD, UB."""
    w = per_player["w"].clip(lower=1)
    mean = np.average(per_player[col], weights=w)
    total_var = np.average((per_player[col] - mean) ** 2, weights=w)
    samp_var = np.average(per_player["samp_var"], weights=w)
    true_var = max(total_var - samp_var, 0.0)
    se_total = total_var * np.sqrt(2.0 / max(len(per_player), 1))
    ub = np.sqrt(max(true_var + 1.645 * se_total, 0.0))
    return mean, np.sqrt(true_var), np.sqrt(samp_var), ub


def analyze(df: pd.DataFrame, val: np.ndarray, hi: np.ndarray, label: str):
    df = df.copy()
    v_b, resid = purge_layers(df, val)
    df["v_b"] = v_b; df["resid"] = resid; df["hi"] = hi.astype(bool)

    # aggregate within-everything high-minus-low
    raw = df.loc[df["hi"], "v_b"].mean() - df.loc[~df["hi"], "v_b"].mean()
    purged = df.loc[df["hi"], "resid"].mean() - df.loc[~df["hi"], "resid"].mean()
    P(f"\n  == {label} ==")
    P(f"  high-leverage share: {df['hi'].mean()*100:.1f}%   PAs {len(df):,}")
    P(f"  RAW    clutch (net season+batter)      : {raw:+.5f}")
    P(f"  PURGED clutch (also net pitcher)       : {purged:+.5f}   <- the opponent-netted effect")
    P(f"  selection (raw - purged) = closer eff  : {raw - purged:+.5f}")

    # cross-player clutch skill: per-batter purged high-minus-low
    g = df.groupby("batter")
    hi_mean = df[df["hi"]].groupby("batter")["resid"].agg(["mean", "size"])
    lo_mean = df[~df["hi"]].groupby("batter")["resid"].agg(["mean", "size"])
    j = hi_mean.join(lo_mean, lsuffix="_hi", rsuffix="_lo").dropna()
    j = j[(j["size_hi"] >= MIN_HL) & (j["size_lo"] >= MIN_HL)]
    var_all = df.groupby("batter")["resid"].var()
    j = j.join(var_all.rename("v"))
    j["diff"] = j["mean_hi"] - j["mean_lo"]
    j["samp_var"] = j["v"] * (1.0 / j["size_hi"] + 1.0 / j["size_lo"])
    j["w"] = 1.0 / j["samp_var"].clip(lower=j["samp_var"][j["samp_var"] > 0].median())
    mean_c, true_sd, samp_sd, ub = eb_spread(j.reset_index(), "diff")
    P(f"  clutch-skill: {len(j):,} batters (>= {MIN_HL} PA each side)")
    P(f"    pooled mean high-minus-low : {mean_c:+.5f}")
    P(f"    between-player true SD (EB): {true_sd:.5f}   (sampling SD {samp_sd:.5f})")
    P(f"    true SD approx 95% upper   : {ub:.5f}")
    return j


def load_merged():
    pa = pd.read_parquet(PA)
    if "game_id" in pa.columns and "gid" not in pa.columns:
        pa = pa.rename(columns={"game_id": "gid"})
    ren = {}
    if "batter" not in pa.columns and "bat_id" in pa.columns: ren["bat_id"] = "batter"
    if "pitcher" not in pa.columns and "pit_id" in pa.columns: ren["pit_id"] = "pitcher"
    pa = pa.rename(columns=ren).reset_index(drop=True)
    pa["pa_seq"] = pa.groupby("gid").cumcount()
    lev = pd.read_parquet(LEV)[["gid", "pa_seq", "inning", "outs", "base_state",
                                "score_diff", "late_close", "lev_proxy"]]
    df = pa.merge(lev, on=["gid", "pa_seq"], how="left", suffixes=("", "_lev"))
    if "bat_is_pitcher" in df.columns:
        df = df[df["bat_is_pitcher"] == 0]
    df = df.dropna(subset=["base_state", "woba_value"]).reset_index(drop=True)
    return df


def run_real():
    print(f"pa: {PA.exists()}  lev: {LEV.exists()}")
    if not (PA.exists() and LEV.exists()):
        print("missing inputs"); return
    df = load_merged()
    P("CLUTCH / LEVERAGE  (opponent-purged)")
    P("=" * 64)
    P(f" {len(df):,} non-pitcher PAs with leverage state")
    hi = df["late_close"].fillna(False).to_numpy().astype(bool)
    analyze(df, df["woba_value"].to_numpy(), hi, "late & close (inning>=7, |diff|<=1)")
    # leverage-proxy top tertile as an alternative high-leverage definition
    if "lev_proxy" in df.columns and df["lev_proxy"].notna().any():
        thr = df["lev_proxy"].quantile(2 / 3)
        analyze(df, df["woba_value"].to_numpy(),
                (df["lev_proxy"] >= thr).to_numpy(), "lev_proxy top tertile")
    (OUTPUT_DIR / "clutch_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\nwrote clutch_summary.txt")


def synthetic():
    rng = np.random.default_rng(5)
    n = 3_000_000
    batter = rng.integers(0, 6000, n)
    season = rng.integers(1990, 2025, n)
    bskill = rng.normal(0, 0.03, 6000)[batter]
    pq_all = rng.normal(0, 0.03, 4000)              # pitcher quality (value allowed)
    hi = rng.random(n) < 0.12                        # 12% high leverage
    # CONFOUND done right: high-leverage PAs are assigned to BETTER pitchers (low pq).
    good = np.argsort(pq_all)[:800]                  # the 800 best (closers)
    pitcher = rng.integers(0, 4000, n)
    n_hi = hi.sum()
    pitcher[hi] = rng.choice(good, n_hi)             # high-lev faces the good tail
    pq_eff = pq_all[pitcher]
    # planted TRUE within-batter clutch penalty -0.010 + heterogeneous clutch skill
    clutch_skill = rng.normal(0, 0.006, 6000)[batter]
    true_clutch = (-0.010 + clutch_skill) * hi
    val = 0.320 + bskill + pq_eff + true_clutch + rng.normal(0, 0.30, n)
    df = pd.DataFrame({"batter": batter, "pitcher": pitcher, "season": season,
                       "woba_value": val})
    P("SYNTHETIC  (planted purged clutch -0.010; high-leverage faces better pitchers;")
    P("            planted clutch-skill SD 0.006)")
    P("=" * 64)
    analyze(df, df["woba_value"].to_numpy(), hi, "synthetic late&close")
    P("\n  expect: PURGED ~ -0.010, RAW more negative (selection = facing the closer),")
    P("          clutch-skill true SD ~ 0.006")
    (OUTPUT_DIR / "clutch_summary.txt").write_text("\n".join(L), encoding="utf-8")


def main():
    if "--synthetic" in sys.argv:
        synthetic(); return
    run_real()


if __name__ == "__main__":
    main()
