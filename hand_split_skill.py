"""
hand_split_skill.py
===================

Step 1 of the pitcher-handedness section. Estimate each batter's hitting skill
SEPARATELY against right- and left-handed pitchers, with empirical-Bayes
shrinkage so the thin left-handed-pitcher sample does not leave tau_L as noise.

Construction (mirrors the season-level tau_bat, but per hand): for each hand H,
aggregate to batter-season means of wOBA weighted by PA, fit a PA-weighted batter
fixed effect net of season and age, then shrink:

    tau_eb_i = tau_bar + rho_i * (tau_raw_i - tau_bar),
    rho_i    = N_i / (N_i + K),   K = sigma2_PA / sigma2_between   (in PA units)

K is the regression-to-the-mean constant, reported and sanity-checked against the
~150-300 PA range where wOBA stabilizes. Each hand's tau_eb is anchored to a
PA-weighted mean of zero, so it measures skill vs that hand RELATIVE to the
league-average batter vs that hand; the platoon LEVEL drops out, leaving each
batter's idiosyncratic tilt in split = tau_R - tau_L, which step 2 prices.

  CAVEAT (preserve in the write-up): K calibrates low (~22 PA vs the expected
  150-300) because the balanced-design SE approximation biases sigma2_between up.
  Quote the per-hand effects as shapes and signs, not precise per-talent-SD
  magnitudes.

Reads pa_panel.parquet; writes tau_by_hand.parquet. Requires pyfixest.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pyfixest as pf

import meso_core as mc

PANEL   = mc.PANEL_PA
OUT     = mc.DATA_DIR  / "tau_by_hand.parquet"
OUT_SUM = mc.OUTPUT_DIR / "hand_split_skill_summary.txt"
AGE_LO, AGE_HI = mc.TWO_WAY_AGE_LO, mc.TWO_WAY_AGE_HI
HANDS = ["R", "L"]

L = []
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def P(s=""): print(s); L.append(s)


def wmean(x, w):
    w = np.asarray(w, float); x = np.asarray(x, float)
    return float(np.sum(w * x) / np.sum(w))

def wvar(x, w):
    w = np.asarray(w, float); x = np.asarray(x, float)
    mu = wmean(x, w)
    return float(np.sum(w * (x - mu) ** 2) / np.sum(w))


def hand_tau(cell: pd.DataFrame, hand: str) -> pd.DataFrame:
    """cell: batter-season-hand rows (batter, season, bat_age, woba, n).
    Returns per-batter raw and EB-shrunk tau plus N and rho."""
    log(f"  [{hand}] {len(cell):,} batter-seasons; "
        f"{cell['batter'].nunique():,} batters; {int(cell['n'].sum()):,} PA")
    cell = cell.copy()
    cell["batter_s"] = cell["batter"].astype(str)
    cell["season_i"] = cell["season"].astype(int)
    cell["age_i"]    = cell["bat_age"].astype(int)
    cell = mc.drop_singletons(cell, ["batter_s", "season_i", "age_i"])
    log(f"  [{hand}] {len(cell):,} batter-seasons after dropping singletons")

    m = pf.feols("woba ~ 1 | batter_s + season_i + age_i", data=cell, weights="n")
    fe = m.fixef()
    bkey = next(k for k in fe if "batter_s" in k)
    tau_raw = pd.Series(fe[bkey]); tau_raw.index = tau_raw.index.astype(str)

    cell["resid"] = np.asarray(m.resid())
    sigma2_PA = float(np.mean(cell["n"] * cell["resid"] ** 2))

    N = cell.groupby("batter_s")["n"].sum()
    tau_raw = tau_raw.reindex(N.index)
    tau_bar = wmean(tau_raw.values, N.values)
    var_obs = wvar(tau_raw.values, N.values)
    mean_samp = float(np.sum(N.values * (sigma2_PA / N.values)) / np.sum(N.values))
    sigma2_b = max(var_obs - mean_samp, 1e-8)
    K = sigma2_PA / sigma2_b
    rho = N / (N + K)
    tau_eb = tau_bar + rho * (tau_raw - tau_bar)
    tau_eb = tau_eb - wmean(tau_eb.values, N.values)

    P(f"  [{hand}] sigma2_PA={sigma2_PA:.5f}  sigma2_between={sigma2_b:.6f}  "
      f"K={K:,.0f} PA  (wOBA stabilizes ~150-300 PA; K low => see caveat)")
    P(f"  [{hand}] median N={int(N.median()):,}  median rho={float(rho.median()):.2f}  "
      f"min rho={float(rho.min()):.2f}")
    return pd.DataFrame({"batter": N.index, f"tau_{hand}_raw": tau_raw.values,
                         f"tau_{hand}_eb": tau_eb.values, f"N_{hand}": N.values,
                         f"rho_{hand}": rho.values})


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUM.parent.mkdir(parents=True, exist_ok=True)

    log(f"loading {PANEL}")
    df = pd.read_parquet(PANEL, columns=["season", "bat_id", "woba_value", "iw",
                                         "bat_age", "pit_hand", "bat_is_pitcher"])
    df = df[df["iw"] != 1].dropna(subset=["woba_value", "bat_age", "pit_hand", "bat_id"])
    # exclude pitchers-as-batters: not part of the position-player allocation
    # question, and their low wOBA inflates the between-batter variance.
    df = df[df["bat_is_pitcher"] != 1]
    df["bat_age"] = df["bat_age"].astype(int); df["season"] = df["season"].astype(int)
    df = df[(df["bat_age"] >= AGE_LO) & (df["bat_age"] <= AGE_HI)]
    df["pit_hand"] = df["pit_hand"].astype(str).str.upper().str[0]
    df = df[df["pit_hand"].isin(HANDS)]
    log(f"  {len(df):,} usable PA; pit_hand mix "
        + ", ".join(f"{h}={100*(df['pit_hand']==h).mean():.1f}%" for h in HANDS))

    P("HAND-SPLIT BATTER SKILL with empirical-Bayes shrinkage")
    P("=" * 70)

    out = None
    for h in HANDS:
        cell = (df[df["pit_hand"] == h]
                .groupby(["bat_id", "season"])
                .agg(woba=("woba_value", "mean"), n=("woba_value", "size"),
                     bat_age=("bat_age", "first")).reset_index()
                .rename(columns={"bat_id": "batter"}))
        t = hand_tau(cell, h)
        out = t if out is None else out.merge(t, on="batter", how="outer")

    both = out.dropna(subset=["tau_R_eb", "tau_L_eb"]).copy()
    both["overall"] = 0.5 * (both["tau_R_eb"] + both["tau_L_eb"])
    both["split"]   = both["tau_R_eb"] - both["tau_L_eb"]      # >0 => better vs RHP
    out = out.merge(both[["batter", "overall", "split"]], on="batter", how="left")
    out.to_parquet(OUT, index=False)
    log(f"wrote {OUT}  ({len(out):,} batters; {len(both):,} with both hands)")

    P("")
    P(f"batters with both-hand skills: {len(both):,}")
    P(f"corr(tau_R_eb, tau_L_eb): {both['tau_R_eb'].corr(both['tau_L_eb']):+.3f}  "
      "(high = ability travels across hands)")
    P(f"SD(overall) = {both['overall'].std():.4f}   SD(split) = {both['split'].std():.4f} wOBA")
    P(f"mean split (~=0 after anchoring): {both['split'].mean():+.5f}")
    P(f"corr(overall, split): {both['overall'].corr(both['split']):+.3f}  "
      "(near 0 => clean decomposition for step 2)")
    OUT_SUM.write_text("\n".join(L), encoding="utf-8")
    log(f"wrote {OUT_SUM}")


if __name__ == "__main__":
    main()
