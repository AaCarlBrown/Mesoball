"""
hot_hand.py  (exploratory)
==========================

Is there a hot hand in batting -- does recent success predict the next outcome,
beyond a player's baseline? Cramer (1977) and Albright (1993, JASA) said no, from
low-powered year-to-year tests; Miller & Sanjurjo (2018, Econometrica) showed the
canonical hot-hand test is biased TOWARD the null by a finite-sample streak-
selection bias. We redo it with the whole population, the bias correction, and the
move the originals could not make: testing serial dependence in the OPPONENT- and
ENVIRONMENT-PURGED residual, so a "streak" is not just a soft stretch of schedule.

ESTIMATOR.  For each batter, order his PAs in time and form the residual
    r_t = value_t - season_eff - batter_eff - pitcher_eff      (additive two-way purge)
then take the within-player lag-1 autocorrelation rho1. Under i.i.d. the sample
lag-1 autocorrelation of an n-length series demeaned by its own mean has expectation
-1/(n-1) -- a NEGATIVE bias, exactly the Miller-Sanjurjo streak-selection bias at
streak length 1. The bias-corrected per-player statistic is rho1_i + 1/(n_i-1).

  * pooled mean of the corrected statistic  -> is there an average hot hand?
  * EB-shrunk SD across players              -> do persistently streaky hitters exist?
    (the Cramer "do clutch/streaky hitters exist" question, with real power)

A binary on-base permutation test (Miller-Sanjurjo's own unbiased procedure) runs on
a player subsample as a GVT-comparable cross-check.

NOTE.  PAs are ordered by (date, gid, inning); the leverage rerun will add a within-
game PA sequence index to sharpen ties (rare for lag-1). Residual uses an additive
two-way purge computed from the panel; swap in the full two-way FE fitted values via
a tau file if desired (set QUALITY).

Run:
  py hot_hand.py --synthetic   # self-test: recover planted hot hand, show the bias
  py hot_hand.py               # real panel
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\baseball_eras")

def _first_existing(cands, default):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return Path(default)

PANEL_PA = _first_existing([ROOT / "data" / "pa_panel.parquet"], ROOT / "data" / "pa_panel.parquet")
WEIGHTS = _first_existing([ROOT / "wOBA_weights.csv", ROOT / "data" / "wOBA_weights.csv",
                           ROOT / "github" / "wOBA_weights.csv", "wOBA_weights.csv"],
                          ROOT / "wOBA_weights.csv")
QUALITY = None
OUTPUT_DIR = ROOT / "output" if (ROOT / "output").exists() else Path(".")
MIN_PA = 200          # players with at least this many career PAs enter the spread

L: list[str] = []
def P(s: str = "") -> None:
    print(s); L.append(s)

W_CANON = dict(wBB=0.69, wHBP=0.72, w1B=0.89, w2B=1.27, w3B=1.62, wHR=2.10)


def value_per_pa(df: pd.DataFrame, w: pd.DataFrame | None) -> np.ndarray:
    # prefer the panel's own per-PA value (paper-consistent) if present
    if "woba_value" in df.columns:
        return df["woba_value"].fillna(0).to_numpy(float)
    if w is not None:
        w = w.rename(columns={"Season": "season"})
        d = df.merge(w[["season", "wBB", "wHBP", "w1B", "w2B", "w3B", "wHR"]], on="season", how="left")
        cols = {k: d[k].fillna(W_CANON[k]) for k in W_CANON}
    else:
        d = df; cols = {k: W_CANON[k] for k in W_CANON}
    return (cols["wBB"] * d.get("walk", 0).fillna(0)
            + cols["wHBP"] * d.get("hbp", 0).fillna(0)
            + cols["w1B"] * d.get("single", 0).fillna(0)
            + cols["w2B"] * d.get("double", 0).fillna(0)
            + cols["w3B"] * d.get("triple", 0).fillna(0)
            + cols["wHR"] * d.get("hr", 0).fillna(0)).to_numpy()


def purge(df: pd.DataFrame, val: np.ndarray) -> np.ndarray:
    """Additive two-way purge: subtract season, then batter, then pitcher effects
    (sequential FWL). Approximation to the full two-way FE residual; adequate for a
    serial-dependence test and fast on 15M rows."""
    s = pd.Series(val)
    seas = s.groupby(df["season"].to_numpy()).transform("mean")
    r = s - seas
    bat = r.groupby(df["batter"].to_numpy()).transform("mean")
    r = r - bat
    pit = r.groupby(df["pitcher"].to_numpy()).transform("mean")
    r = r - pit
    return r.to_numpy()


def lag1_by_player(pid: np.ndarray, r: np.ndarray):
    """Per-player lag-1 autocorrelation of r, with the i.i.d. bias 1/(n-1) added back.
    INPUT MUST ALREADY BE SORTED by (player, time). Returns batter, n, rho1, corrected."""
    df = pd.DataFrame({"pid": pid, "r": r})
    df["r_lag"] = df.groupby("pid")["r"].shift(1)
    pm = df.groupby("pid")["r"].transform("mean")
    df["rdm"] = df["r"] - pm
    df["rdm_lag"] = df["r_lag"] - pm
    num = (df["rdm"] * df["rdm_lag"]).groupby(df["pid"]).sum()
    den = (df["rdm"] ** 2).groupby(df["pid"]).sum()
    n = df.groupby("pid").size()
    rho1 = (num / den).replace([np.inf, -np.inf], np.nan)
    corrected = rho1 + 1.0 / (n - 1)
    return pd.DataFrame({"batter": rho1.index, "n": n.values,
                         "rho1": rho1.values, "corrected": corrected.values}).dropna()


def eb_spread(stat: pd.DataFrame) -> tuple[float, float, float, float]:
    """Pooled mean, EB true between-player SD (floored at 0), sampling SD, and an
    approximate one-sided 95% UPPER BOUND on the true SD -- so we report 'spread is
    below X' instead of a bare 0.00000 when the between-player variance is undetectable."""
    w = (stat["n"] - 1).clip(lower=1)
    mean = np.average(stat["corrected"], weights=w)
    total_var = np.average((stat["corrected"] - mean) ** 2, weights=w)
    samp_var = np.average(1.0 / stat["n"], weights=w)
    true_var = max(total_var - samp_var, 0.0)
    P_eff = len(stat)
    se_total = total_var * np.sqrt(2.0 / max(P_eff, 1))     # SE of a variance estimate
    true_var_ub = max(total_var - samp_var, 0.0) + 1.645 * se_total
    return mean, np.sqrt(true_var), np.sqrt(samp_var), np.sqrt(max(true_var_ub, 0.0))


def measured_effects(big: pd.DataFrame) -> dict:
    """Measured next-PA effects on the opponent-purged residual scale, within batter:
      (a) residual after a HR vs after a K in the prior PA  -> the hot hand, in value
          units, measured rather than inferred from the slope;
      (b) the platoon split: residual vs opposite- vs same-handed pitchers.
    Both within-batter demeaned so they share the same currency. big must be sorted
    by (batter, time) and carry resid, hr, k, bat_hand, pit_hand."""
    d = big.copy()
    d["resid_dm"] = d["resid"] - d.groupby("batter")["resid"].transform("mean")
    out = {}
    # (a) hot hand: prior-PA outcome category, shifted within batter
    if {"hr", "k"}.issubset(d.columns):
        cat = np.where(d["hr"].fillna(0) == 1, "HR",
                       np.where(d["k"].fillna(0) == 1, "K", "other"))
        d["prior_cat"] = pd.Series(cat, index=d.index).groupby(d["batter"]).shift(1)
        m_hr = d.loc[d["prior_cat"] == "HR", "resid_dm"].mean()
        m_k = d.loc[d["prior_cat"] == "K", "resid_dm"].mean()
        out["hr_minus_k_next"] = m_hr - m_k
        out["after_hr"] = m_hr; out["after_k"] = m_k
    # (b) platoon split on the same residual scale (switch hitters bat opposite)
    if {"bat_hand", "pit_hand"}.issubset(d.columns):
        bh = d["bat_hand"].astype(str); ph = d["pit_hand"].astype(str)
        same = ((bh == ph) & bh.isin(["L", "R"]))
        opp = ~same
        m_opp = d.loc[opp, "resid_dm"].mean()
        m_same = d.loc[same, "resid_dm"].mean()
        out["platoon_opp_minus_same"] = m_opp - m_same
    return out


def binary_permutation_check(pid, onbase, n_perm=200, max_players=4000, seed=0):
    """GVT-comparable: streak-1 conditional difference D = P(1|prev1)-P(1|prev0),
    compared to each player's own permutation null (Miller-Sanjurjo unbiased test).
    INPUT MUST ALREADY BE SORTED by (player, time). Pooled standardized effect over a
    player subsample."""
    rng = np.random.default_rng(seed)
    pid_s, x_s = pid, onbase.astype(np.int8)
    uniq, starts = np.unique(pid_s, return_index=True)
    bounds = list(starts) + [len(pid_s)]
    keep = (set(rng.choice(len(uniq), max_players, replace=False))
            if len(uniq) > max_players else set(range(len(uniq))))
    def Dstat(x):
        prev, cur = x[:-1], x[1:]
        n1 = prev.sum(); n0 = len(prev) - n1
        if n1 == 0 or n0 == 0:
            return np.nan
        return cur[prev == 1].mean() - cur[prev == 0].mean()
    zs = []
    for i in range(len(uniq)):
        if i not in keep:
            continue
        x = x_s[bounds[i]:bounds[i + 1]]
        if len(x) < 50 or x.sum() < 5 or x.sum() > len(x) - 5:
            continue
        d0 = Dstat(x)
        if np.isnan(d0):
            continue
        null = np.array([Dstat(rng.permutation(x)) for _ in range(n_perm)])
        null = null[~np.isnan(null)]
        if len(null) < 30:
            continue
        sd = null.std()
        if sd > 0:
            zs.append((d0 - null.mean()) / sd)
    zs = np.array(zs)
    return zs.mean() if len(zs) else np.nan, len(zs)


def analyze(df, val, label, w_for_value=None):
    df = df.copy().reset_index(drop=True)
    df["val"] = val
    df["onbase"] = ((df.get("single", 0).fillna(0) + df.get("double", 0).fillna(0)
                     + df.get("triple", 0).fillna(0) + df.get("hr", 0).fillna(0)
                     + df.get("walk", 0).fillna(0) + df.get("hbp", 0).fillna(0)) > 0).astype(int)
    df["resid"] = purge(df, df["val"].to_numpy())
    # restrict to batters with enough PAs, then sort by (batter, time)
    big = df[df.groupby("batter")["batter"].transform("size") >= MIN_PA].copy()
    time_cols = [c for c in ["date", "gid", "inning"] if c in big.columns]
    if time_cols == ["inning"] or not time_cols:
        P(f"\n  ABORT: only {time_cols} available to order PAs -- inning alone scrambles")
        P("  the within-player sequence and would give a meaningless autocorrelation.")
        P("  Need a date or game-id column. Not running the test.")
        return pd.DataFrame()
    big = big.sort_values(["batter"] + time_cols, kind="stable").reset_index(drop=True)

    stat = lag1_by_player(big["batter"].to_numpy(), big["resid"].to_numpy())
    mean_c, true_sd, samp_sd, true_sd_ub = eb_spread(stat)
    raw_mean = np.average(stat["rho1"], weights=(stat["n"] - 1).clip(lower=1))

    P(f"\n  == {label} ==")
    P(f"  players (>= {MIN_PA} PA): {len(stat):,}   |  ordered by {time_cols}")
    P(f"  pooled lag-1 autocorr  UNCORRECTED : {raw_mean:+.5f}   <- shows the M-S bias")
    P(f"  pooled lag-1 autocorr  CORRECTED   : {mean_c:+.5f}     <- the hot-hand estimate")
    P(f"  between-player true SD (EB)        : {true_sd:.5f}   (sampling SD {samp_sd:.5f})")
    P(f"  between-player SD, approx 95% upper bound : {true_sd_ub:.5f}")
    z, nz = binary_permutation_check(big["batter"].to_numpy(), big["onbase"].to_numpy())
    P(f"  binary on-base permutation z (mean over {nz} players): {z:+.4f}")

    me = measured_effects(big)
    if "hr_minus_k_next" in me:
        P(f"\n  MEASURED next-PA effect (same residual scale, within batter):")
        P(f"    after HR : {me['after_hr']:+.5f}   after K : {me['after_k']:+.5f}")
        P(f"    HR - K next-PA value  : {me['hr_minus_k_next']:+.5f}   <- the practical hot-hand size")
    if "platoon_opp_minus_same" in me:
        P(f"    platoon (opp - same hand) : {me['platoon_opp_minus_same']:+.5f}   <- same currency, for comparison")
        if "hr_minus_k_next" in me and me["platoon_opp_minus_same"]:
            P(f"    ratio hot-hand / platoon  : {me['hr_minus_k_next']/me['platoon_opp_minus_same']:.2f}")
    return stat


def run_real():
    print(f"panel: {PANEL_PA} (exists={PANEL_PA.exists()})")
    if not PANEL_PA.exists():
        print("Edit ROOT/PANEL_PA at top."); return
    df = pd.read_parquet(PANEL_PA)
    ren = {}
    if "batter" not in df.columns and "bat_id" in df.columns: ren["bat_id"] = "batter"
    if "pitcher" not in df.columns and "pit_id" in df.columns: ren["pit_id"] = "pitcher"
    df = df.rename(columns=ren).reset_index(drop=True)
    print("panel columns:", sorted(df.columns.tolist()))
    # canonicalize chronological keys (the test needs PAs in TIME order)
    date_c = next((c for c in ["date", "game_date", "gdate", "gamedate", "gameday", "gamedate_int"]
                   if c in df.columns), None)
    gid_c = next((c for c in ["gid", "game_id", "gameid", "game"] if c in df.columns), None)
    if date_c and date_c != "date":
        df = df.rename(columns={date_c: "date"})
    if gid_c and gid_c != "gid":
        df = df.rename(columns={gid_c: "gid"})
    if not date_c and not gid_c:
        print("\nNO chronological key (date / game id) found among the columns above.")
        print("The hot-hand test needs PAs in time order; inning alone scrambles the")
        print("sequence and the result is meaningless. Tell me which column holds the")
        print("game date or game id and I'll map it (or add it in the leverage rerun).")
        return
    print(f"using chronological keys: date={date_c}, game={gid_c} (+ inning)")
    need = ["batter", "pitcher", "season", "inning"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        print(f"missing columns {miss}; available: {sorted(df.columns)[:40]}"); return
    if "bat_is_pitcher" in df.columns:
        df = df[df["bat_is_pitcher"] == 0].reset_index(drop=True)
    w = pd.read_csv(WEIGHTS) if WEIGHTS.exists() else None
    val = value_per_pa(df, w)
    P("HOT HAND  (opponent-purged residual, Miller-Sanjurjo bias-corrected)")
    P("=" * 72)
    P(f" {len(df):,} non-pitcher PAs")
    stat = analyze(df, val, "all batters, career PA sequence")
    stat.to_csv(OUTPUT_DIR / "hot_hand_player.csv", index=False, float_format="%.5f")
    (OUTPUT_DIR / "hot_hand_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\nwrote hot_hand_summary.txt, hot_hand_player.csv")


def synthetic():
    rng = np.random.default_rng(3)
    P("SYNTHETIC TEST"); P("=" * 72)
    for phi, name in [(0.0, "i.i.d. (NO hot hand)"), (0.15, "AR(1) phi=0.15 (hot hand)")]:
        rows = []
        for pid in range(3000):
            n = rng.integers(200, 700)
            e = rng.normal(0, 1, n); r = np.empty(n); r[0] = e[0]
            for t in range(1, n):
                r[t] = phi * r[t - 1] + e[t]            # planted serial dependence
            rows.append(pd.DataFrame({"batter": pid, "season": 2000 + np.arange(n) // 600,
                                      "date": 20000000 + np.arange(n),   # monotonic time
                                      "gid": np.arange(n),
                                      "inning": np.arange(n) % 9 + 1, "pitcher": rng.integers(0, 2000, n),
                                      "single": (rng.random(n) < 0.15 + 0.02 * r).astype(int),
                                      "double": 0, "triple": 0, "hr": 0, "walk": 0, "hbp": 0,
                                      "_r": r}))
        df = pd.concat(rows, ignore_index=True)
        # use the planted residual directly as 'value' so recovery is checkable
        st = analyze(df, df["_r"].to_numpy(), name)
        P(f"  (planted phi = {phi:+.2f}; CORRECTED pooled should be ~ that, UNCORRECTED biased low)\n")
    (OUTPUT_DIR / "hot_hand_summary.txt").write_text("\n".join(L), encoding="utf-8")


def main():
    if "--synthetic" in sys.argv:
        synthetic(); return
    run_real()


if __name__ == "__main__":
    main()
