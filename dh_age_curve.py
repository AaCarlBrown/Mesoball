"""
dh_age_curve.py
===============

Does the designated hitter rule flatten the batter aging curve?

Mechanism (per discussion):
  The DH redistributes plate appearances. An aging slugger (good bat,
  eroding glove) bats whether he plays the field or DHs, so the DH rule
  is roughly PA-neutral FOR HIM. But moving him to DH opens a field
  position for a good-glove / weak-bat player, who thereby accumulates
  PAs he would not have gotten without the DH. Those weak-bat fielders
  tend to gain PAs loaded toward later career (the roster can carry a
  weak bat because the DH shoulders the offense). Their bat was never
  the source of their value, so their within-player decline is shallow.
  Adding their age deltas to the older-age pool should FLATTEN the
  batter age curve at older ages in DH context relative to non-DH.

Design (difference-in-differences in spirit):
  The DH rule follows the BALLPARK, not the batter's league: the rule in
  effect for a PA is determined by the HOME team's league (pre-2022), so
  interleague games are handled correctly and we need not stop at 1996.
  We tag each PA "DH" or "noDH" by whether the DH was in force in that
  game (home AL park pre-2022, or any park 2022+), and estimate the batter
  age curve separately by DH context in three regimes:

    (1) SPLIT     1973-2021: AL parks use the DH, NL parks do not. The
                  clean natural experiment. Expect the DH-context curve
                  flatter at old ages than the no-DH context.
    (2) BOTH_NODH 1947-1972: no park uses the DH. Control: the two
                  context curves should coincide (the AL/NL park split is
                  a placebo here).
    (3) BOTH_DH   2022-2025: every park uses the DH. Control, but the
                  window is too short for full-age coverage; included for
                  completeness, not leaned on.

  Within each (regime, context) cell we fit the two-way model
        woba = mu_season + tau_bat + tau_pit + alpha_bat(age) + alpha_pit(age)
  and extract alpha_bat(age). Player and season effects absorb level and
  run-environment differences, so any context difference in alpha_bat is
  about the SHAPE of within-player aging, which is what the DH mechanism
  predicts.

  For the two control regimes, where the DH context is uniform across all
  parks, we instead split by the HOME team's league (AL park vs NL park)
  as a placebo: there is no rule difference, so the curves should match.

  Pitcher-batters are EXCLUDED throughout (bat_is_pitcher == 0); pitcher
  hitting is analyzed separately elsewhere.

Inputs:
    C:\\baseball_eras\\data\\pa_panel.parquet
      needs: season, bat_id, pit_id, woba_value, iw, bat_age, pit_age,
             bat_is_pitcher, and a league-of-game column.
      The build script stores home/visiting teams but the per-PA league
      is derivable from bat_home + team league. To keep this script
      self-contained we expect a column `bat_league` (the league the
      batter's team played in that game). If it is absent we derive it
      from `bat_team` + Teams.csv (see derive_bat_league).

Outputs:
    C:\\baseball_eras\\data\\dh_alpha_bat.parquet
        regime, league, age, alpha_bat, n_pa
    C:\\baseball_eras\\output\\dh_age_curve.csv
    C:\\baseball_eras\\output\\dh_age_curve_summary.txt

Requires: pip install pyfixest
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf


PROJECT_DIR = Path(r"C:\baseball_eras")
PANEL_PATH = PROJECT_DIR / "data" / "pa_panel.parquet"
TEAMS_CSV  = PROJECT_DIR / "Teams.csv"

OUT_ALPHA = PROJECT_DIR / "data"   / "dh_alpha_bat.parquet"
OUT_CSV   = PROJECT_DIR / "output" / "dh_age_curve.csv"
OUT_SUM   = PROJECT_DIR / "output" / "dh_age_curve_summary.txt"

AGE_LO, AGE_HI = 20, 42   # tighter than the main fit; old-age tail is the focus
ANCHOR_AGE = 27

REGIMES = [
    ("SPLIT",     1973, 2021),   # AL parks DH, NL parks no DH
    ("BOTH_NODH", 1947, 1972),   # no park uses DH
    ("BOTH_DH",   2022, 2025),   # every park uses DH (short window)
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_home_team(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine the HOME team's retrosheet id for each PA. The DH rule
    follows the home park. If the panel has an explicit 'hometeam'
    column, use it. Otherwise derive it from bat_team/pit_team and
    bat_home: if the batter is home (bat_home==1) the home team is
    bat_team, else pit_team.
    """
    if "hometeam" in df.columns:
        df["home_team"] = df["hometeam"]
        return df
    log("deriving home_team from bat_team/pit_team + bat_home")
    df["home_team"] = np.where(df["bat_home"] == 1, df["bat_team"], df["pit_team"])
    return df


def add_home_league(df: pd.DataFrame) -> pd.DataFrame:
    """Join home_team + season to Teams.csv to get the home team's league."""
    log("deriving home-park league from home_team + Teams.csv")
    t = pd.read_csv(TEAMS_CSV, usecols=["yearID", "lgID", "teamIDretro"])
    t = t.dropna(subset=["teamIDretro"]).rename(
        columns={"yearID": "season", "teamIDretro": "home_team", "lgID": "home_league"})
    df = df.merge(t, on=["season", "home_team"], how="left")
    return df


def load_panel() -> pd.DataFrame:
    log(f"loading {PANEL_PATH}")
    base = ["season", "bat_id", "pit_id", "woba_value", "iw",
            "bat_age", "pit_age", "bat_is_pitcher"]
    head = pd.read_parquet(PANEL_PATH, columns=None).head(0)
    extra = []
    for c in ["hometeam", "bat_team", "pit_team", "bat_home"]:
        if c in head.columns:
            extra.append(c)
    df = pd.read_parquet(PANEL_PATH, columns=base + extra)
    log(f"  {len(df):,} rows")

    df = df[df["iw"] != 1]
    df = df[df["bat_is_pitcher"] == 0]          # EXCLUDE pitcher-batters
    df = df.dropna(subset=["woba_value", "bat_age", "pit_age", "bat_id", "pit_id"])
    df["bat_age"] = df["bat_age"].astype(int)
    df["pit_age"] = df["pit_age"].astype(int)
    df["season"]  = df["season"].astype(int)
    df = df[(df["bat_age"] >= AGE_LO) & (df["bat_age"] <= AGE_HI)
            & (df["pit_age"] >= AGE_LO) & (df["pit_age"] <= AGE_HI)]

    df = add_home_team(df)
    df = add_home_league(df)
    df = df[df["home_league"].isin(["AL", "NL"])]

    # DH in effect: home AL park (pre-2022) or any park 2022+.
    df["dh_context"] = np.where(
        (df["season"] >= 2022) | (df["home_league"] == "AL"),
        "DH", "noDH")
    log(f"  {len(df):,} after filters (ex-IBB, ex-pitcher-batters, age, AL/NL home)")
    return df


def fit_alpha_bat(df: pd.DataFrame, label: str) -> pd.Series:
    """Two-way fit; return alpha_bat(age) anchored at ANCHOR_AGE."""
    # drop one-season players within this subsample
    bn = df.groupby("bat_id")["season"].nunique()
    pn = df.groupby("pit_id")["season"].nunique()
    df = df[df["bat_id"].isin(bn[bn >= 2].index) & df["pit_id"].isin(pn[pn >= 2].index)]
    d = df.copy()
    d["bat_id"] = d["bat_id"].astype(str)
    d["pit_id"] = d["pit_id"].astype(str)
    log(f"    {label}: {len(d):,} PAs")
    mod = pf.feols("woba_value ~ 1 | season + bat_id + pit_id + bat_age + pit_age", data=d)
    fe = mod.fixef()
    key = next(k for k in fe if "bat_age" in k)
    s = pd.Series(fe[key]); s.index = s.index.astype(int); s = s.sort_index()
    if ANCHOR_AGE in s.index:
        s = s - s.loc[ANCHOR_AGE]
    else:
        s = s - s.mean()
    return s


def main() -> None:
    OUT_ALPHA.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = load_panel()

    rows = []
    curves = {}
    for regime, lo, hi in REGIMES:
        sub = df[(df["season"] >= lo) & (df["season"] <= hi)]
        if len(sub) == 0:
            log(f"regime {regime}: no data, skipping")
            continue
        # In SPLIT, the meaningful split is DH context. In the controls,
        # the DH context is uniform, so we split by home-park league as a
        # placebo (should show no gap).
        if regime == "SPLIT":
            split_col, cats = "dh_context", ["DH", "noDH"]
        else:
            split_col, cats = "home_league", ["AL", "NL"]
        for cat in cats:
            cell = sub[sub[split_col] == cat]
            if len(cell) < 50000:
                log(f"regime {regime} {cat}: only {len(cell):,} PAs, skipping")
                continue
            log(f"fitting regime={regime} {split_col}={cat}")
            alpha = fit_alpha_bat(cell, f"{regime}-{cat}")
            pa_by_age = cell.groupby("bat_age").size()
            curves[(regime, cat)] = alpha
            for age, a in alpha.items():
                rows.append({"regime": regime, "context": cat, "age": int(age),
                             "alpha_bat": float(a),
                             "n_pa": int(pa_by_age.get(age, 0))})

    out = pd.DataFrame(rows)
    out.to_parquet(OUT_ALPHA, index=False)
    out.to_csv(OUT_CSV, index=False, float_format="%.5f")
    log(f"wrote {OUT_CSV}")

    # Summary: AL-minus-NL alpha gap by age, per regime
    lines = []
    P = lines.append
    P("DH effect on batter aging curve")
    P("=" * 70)
    P(f"Ages {AGE_LO}-{AGE_HI}, anchored at {ANCHOR_AGE}. Pitcher-batters excluded.")
    P("Prediction: in SPLIT regime, AL (DH) curve is flatter at old ages")
    P("(alpha_bat higher / less negative at 35+) than NL (no DH).")
    P("Controls BOTH_NODH and BOTH_DH should show ~no AL-NL gap.")
    P("")
    for regime, lo, hi in REGIMES:
        if regime == "SPLIT":
            c1, c2 = "DH", "noDH"
        else:
            c1, c2 = "AL", "NL"
        if (regime, c1) in curves and (regime, c2) in curves:
            a1 = curves[(regime, c1)]
            a2 = curves[(regime, c2)]
            ages = sorted(set(a1.index) & set(a2.index))
            P(f"=== {regime} ({lo}-{hi}):  {c1}_alpha  {c2}_alpha  ({c1}-{c2}) ===")
            for age in ages:
                gap = a1.loc[age] - a2.loc[age]
                P(f"  age {age:>2d}:  {a1.loc[age]:+.5f}  {a2.loc[age]:+.5f}  "
                  f"{gap:+.5f}")
            old = [a for a in ages if a >= 35]
            if old:
                gap_old = np.mean([a1.loc[a] - a2.loc[a] for a in old])
                P(f"  mean ({c1}-{c2}) over ages 35+: {gap_old:+.5f}")
            P("")
    OUT_SUM.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {OUT_SUM}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
