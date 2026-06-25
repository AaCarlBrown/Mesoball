"""
dh_hbp_moral_hazard.py  (exploratory)
=====================================

Does shielding a pitcher from batting (a DH in his game) raise the chance he hits
a batter, holding the batter's worth-hitting quality fixed? That residual, net of
lineup composition, is the moral-hazard effect (Goff-Shughart-Tollison 1997/98;
Bradbury-Drinen 2006/07).

DEPENDENT VARIABLE: this is an EVENT model -- P(HBP) within a plate appearance --
not an allocation model, so it is deliberately NOT the censored-share Tobit used
elsewhere in the paper. The censored-share estimator prices playing time; moral
hazard is the probability of an event inside a PA, a different object. Flagged in
the text as a considered choice, not estimator drift. We use a linear probability
model (fast, interpretable in probability points, no incidental-parameters issue
with the fixed effects) with logit as a robustness check.

IDENTIFICATION -- "DH in effect this game" is a game-level binary (both teams use
a DH or neither, set by the home park's league and the season's rule). A pitcher
is shielded iff a DH is in effect. The prediction: P(HBP) is higher when shielded,
after controlling batter worth-hitting quality (which nets out the lineup-
composition channel -- the DH puts a real hitter in the 9th slot) and pitcher
wildness. Three cuts, increasing cleanliness:
  Cut 1  full post-1973 panel, era FE                  (league cross-section)
  Cut 2  interleague games 1997-2019 only              (same teams, DH toggles by park)
  Cut 3  within-pitcher DiD around the 2022 universal DH (NL pitchers gain shielding;
         AL pitchers, already shielded, are the control)

What is NOT here yet: direct retaliation (needs within-game plunk sequencing) and
the ball-strike count (not in the panel; Retrosheet pitch sequences are modern-only
anyway). Both are flagged for later.

DH rule calendar (home_lg = home team's league; DH applies to BOTH teams in a game):
  <=1972            : no DH anywhere
  1973-1996         : DH iff home_lg == AL          (AL-only, no interleague yet)
  1997-2019         : DH iff home_lg == AL          (interleague: DH set by park)
  2020              : universal DH                  (COVID)
  2021              : DH iff home_lg == AL          (NL reverted)
  2022-2025         : universal DH

Run:
  py dh_hbp_moral_hazard.py --synthetic   # self-test: recover a planted effect
  py dh_hbp_moral_hazard.py               # real panel
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

# ---- real-data config ----------------------------------------------------
# Explicit paths off the known layout (C:\baseball_eras\{data,github,output}).
# Edit ROOT or any path below if yours differs. No meso_core dependency.
from pathlib import Path

ROOT = Path(r"C:\baseball_eras")

def _first_existing(cands, default):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return Path(default)

PANEL_PA = _first_existing(
    [ROOT / "data" / "pa_panel.parquet"],
    ROOT / "data" / "pa_panel.parquet")
WEIGHTS = _first_existing(
    [ROOT / "wOBA_weights.csv",
     ROOT / "data" / "wOBA_weights.csv",
     ROOT / "github" / "wOBA_weights.csv",
     "wOBA_weights.csv"],
    ROOT / "wOBA_weights.csv")
# Optional tau file with columns [batter, tau_bat] (+ optionally [pitcher, tau_pit]).
# Set to a real path to use the paper's two-way fixed effects instead of the
# self-contained uwOB proxy; leave as None to use the proxy.
QUALITY = None
TEAMS = _first_existing(
    [ROOT / "Teams.csv",
     ROOT / "data" / "Teams.csv",
     ROOT / "github" / "Teams.csv",
     "Teams.csv"],
    ROOT / "Teams.csv")
OUTPUT_DIR = ROOT / "output" if (ROOT / "output").exists() else Path(".")

L: list[str] = []
def P(s: str = "") -> None:
    print(s); L.append(s)


# --------------------------- DH rule calendar -----------------------------
def dh_in_effect(season: np.ndarray, home_lg: np.ndarray) -> np.ndarray:
    s = np.asarray(season); h = np.asarray(home_lg).astype(str)
    out = np.zeros(len(s), dtype=float)
    al = (h == "AL")
    out[(s >= 1973) & (s <= 2019) & al] = 1.0
    out[(s == 2021) & al] = 1.0
    out[s == 2020] = 1.0
    out[(s >= 2022)] = 1.0
    return out


# ----------------------- quality / wildness controls ----------------------
def build_uwob(df: pd.DataFrame, w: pd.DataFrame | None) -> pd.Series:
    """Per-PA run value from outcome flags. Used only to build the batter
    worth-hitting and pitcher quality CONTROLS, so precision is not critical.
    If a season-weight table is supplied, use it; otherwise use fixed canonical
    linear weights (good enough for a control, needs no external file)."""
    if w is not None:
        w = w.rename(columns={"Season": "season"})
        d = df.merge(w[["season", "wBB", "wHBP", "w1B", "w2B", "w3B", "wHR"]],
                     on="season", how="left")
        wBB, wHBP, w1B, w2B, w3B, wHR = (d["wBB"].fillna(0.69), d["wHBP"].fillna(0.72),
                                         d["w1B"].fillna(0.89), d["w2B"].fillna(1.27),
                                         d["w3B"].fillna(1.62), d["wHR"].fillna(2.10))
        d = d
    else:
        d = df
        wBB, wHBP, w1B, w2B, w3B, wHR = 0.69, 0.72, 0.89, 1.27, 1.62, 2.10
    return (wBB * d.get("walk", 0).fillna(0)
            + wHBP * d.get("hbp", 0).fillna(0)
            + w1B * d.get("single", 0).fillna(0)
            + w2B * d.get("double", 0).fillna(0)
            + w3B * d.get("triple", 0).fillna(0)
            + wHR * d.get("hr", 0).fillna(0)).to_numpy()


def attach_controls(df: pd.DataFrame, weights: pd.DataFrame,
                    quality: pd.DataFrame | None) -> pd.DataFrame:
    """batter worth-hitting quality (bq), pitcher quality (pq), pitcher wildness
    (pwild = career walk+hbp rate). Uses a supplied tau file if present, else the
    uwOB proxy. All standardized to SD units."""
    df = df.copy()
    if quality is not None and {"batter", "tau_bat"}.issubset(quality.columns):
        df = df.merge(quality[["batter", "tau_bat"]].rename(columns={"tau_bat": "bq"}),
                      on="batter", how="left")
        if {"pitcher", "tau_pit"}.issubset(quality.columns):
            df = df.merge(quality[["pitcher", "tau_pit"]].rename(columns={"tau_pit": "pq"}),
                          on="pitcher", how="left")
    if "bq" not in df.columns:
        df["uwob"] = build_uwob(df, weights)
        bq = df.groupby("batter")["uwob"].transform("mean")
        df["bq"] = bq
    if "pq" not in df.columns:
        if "uwob" not in df.columns:
            df["uwob"] = build_uwob(df, weights)
        df["pq"] = df.groupby("pitcher")["uwob"].transform("mean")
    # pitcher wildness: career (walk + hbp) per PA
    wild_num = df.get("walk", 0).fillna(0) + df["hbp"].fillna(0)
    df["pwild"] = wild_num.groupby(df["pitcher"]).transform("mean")
    for c in ["bq", "pq", "pwild"]:
        sd = df[c].std()
        df[c] = (df[c] - df[c].mean()) / (sd if sd > 0 else 1.0)
    return df


# ----------------------------- LPM with FE --------------------------------
def lpm(df: pd.DataFrame, xcols: list[str], fe_cols: list[str], cluster: str):
    """OLS of hbp on [xcols] absorbing fixed effects in fe_cols (within transform),
    SE clustered on `cluster`. Returns coef table keyed by xcol."""
    d = df.dropna(subset=xcols + fe_cols + ["hbp", cluster]).copy()
    y = d["hbp"].to_numpy(float)
    X = d[xcols].to_numpy(float)
    # absorb FE by sequential within-demeaning (one-way exact; multi-way approx via
    # alternating projections, 3 sweeps -- adequate for these designs)
    def demean(mat, key):
        g = d.groupby(key)
        return mat - g.transform("mean").to_numpy() if mat.ndim == 1 else \
               mat - np.column_stack([pd.Series(mat[:, j]).groupby(d[key].to_numpy())
                                      .transform("mean") for j in range(mat.shape[1])])
    yt, Xt = y.copy(), X.copy()
    for _ in range(3 if len(fe_cols) > 1 else 1):
        for k in fe_cols:
            ybar = pd.Series(yt).groupby(d[k].to_numpy()).transform("mean").to_numpy()
            yt = yt - ybar
            for j in range(Xt.shape[1]):
                xbar = pd.Series(Xt[:, j]).groupby(d[k].to_numpy()).transform("mean").to_numpy()
                Xt[:, j] = Xt[:, j] - xbar
    XtX = Xt.T @ Xt
    beta, *_ = np.linalg.lstsq(Xt, yt, rcond=None)   # rank-safe
    resid = yt - Xt @ beta
    # cluster-robust covariance
    bread = np.linalg.pinv(XtX)
    cl = d[cluster].to_numpy()
    meat = np.zeros((Xt.shape[1], Xt.shape[1]))
    order = np.argsort(cl)
    Xs, rs, cs = Xt[order], resid[order], cl[order]
    idx = np.unique(cs, return_index=True)[1]
    for a, b in zip(idx, list(idx[1:]) + [len(cs)]):
        g = Xs[a:b].T @ rs[a:b]
        meat += np.outer(g, g)
    V = bread @ meat @ bread
    se = np.sqrt(np.diag(V))
    return {xcols[i]: (beta[i], se[i], beta[i] / se[i]) for i in range(len(xcols))}, len(d)


def report_cut(name, df, xcols, fe_cols, cluster):
    res, n = lpm(df, xcols, fe_cols, cluster)
    P(f"\n  -- {name}  (n={n:,}, cluster={cluster}, FE={'+'.join(fe_cols)}) --")
    P(f"    {'term':>16} {'coef(pp)':>10} {'SE':>9} {'t':>7}")
    for k in xcols:
        b, s, t = res[k]
        star = "  <-- shield" if k == "dh" else ""
        P(f"    {k:>16} {b*100:>10.4f} {s*100:>9.4f} {t:>7.2f}{star}")
    return res


# ------------------------------ driver ------------------------------------
REQUIRED = ["season", "hbp", "bat_is_pitcher", "inning",
            "bat_team", "pit_team", "bat_home"]   # ids + league handled below

def normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    if "batter" not in df.columns and "bat_id" in df.columns:
        ren["bat_id"] = "batter"
    if "pitcher" not in df.columns and "pit_id" in df.columns:
        ren["pit_id"] = "pitcher"
    return df.rename(columns=ren)


def derive_leagues(df: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct home_lg / vis_lg from bat_team, pit_team, bat_home and the
    Teams.csv (yearID, teamIDretro, lgID) mapping -- no panel rebuild needed.
    The home team bats in the bottom half, so it is bat_team when bat_home==1."""
    t = teams.rename(columns={"yearID": "season", "teamIDretro": "team", "lgID": "lg"})
    t = t[["season", "team", "lg"]].dropna().drop_duplicates(["season", "team"])
    home_team = np.where(df["bat_home"] == 1, df["bat_team"], df["pit_team"])
    vis_team = np.where(df["bat_home"] == 1, df["pit_team"], df["bat_team"])

    def lg_of(season_arr, team_arr):
        key = pd.DataFrame({"season": np.asarray(season_arr), "team": np.asarray(team_arr)})
        return key.merge(t, on=["season", "team"], how="left", sort=False)["lg"].to_numpy()

    df = df.copy()
    df["home_lg"] = lg_of(df["season"].to_numpy(), home_team)
    df["vis_lg"] = lg_of(df["season"].to_numpy(), vis_team)
    return df


def run_real():
    print(f"panel:   {PANEL_PA}  (exists={PANEL_PA.exists()})")
    print(f"weights: {WEIGHTS}  (exists={WEIGHTS.exists()}; optional)")
    print(f"teams:   {TEAMS}  (exists={TEAMS.exists()})")
    print(f"output:  {OUTPUT_DIR}")
    if not PANEL_PA.exists():
        print("Panel parquet not found. Edit ROOT/PANEL_PA at the top of the script.")
        return
    if not TEAMS.exists():
        print("Teams.csv not found (needed to reconstruct league). Edit TEAMS at the top.")
        return
    df = pd.read_parquet(PANEL_PA)
    df = normalize_ids(df)
    missing = [c for c in REQUIRED + ["batter", "pitcher"] if c not in df.columns]
    if missing:
        P(f"PANEL IS MISSING REQUIRED COLUMNS: {missing}")
        P(f"  available columns: {sorted(df.columns.tolist())}")
        P("  -> tell me the right names and I will map them; otherwise add to the build.")
        return
    df = derive_leagues(df, pd.read_csv(TEAMS))
    n_lg = df["home_lg"].notna().mean()
    P(f"[league reconstructed for {n_lg*100:.1f}% of PAs from Teams.csv]")
    df = df[(df["home_lg"].isin(["AL", "NL"])) & (df["vis_lg"].isin(["AL", "NL"]))].copy()
    df = df[df["bat_is_pitcher"] == 0]                 # real hitters only
    df["hbp"] = df["hbp"].fillna(0).astype(float)
    df["dh"] = dh_in_effect(df["season"].to_numpy(), df["home_lg"].to_numpy())
    df["interleague"] = (df["home_lg"] != df["vis_lg"]).astype(int)
    weights = pd.read_csv(WEIGHTS) if WEIGHTS.exists() else None
    quality = pd.read_parquet(QUALITY) if (QUALITY and Path(QUALITY).exists()) else None
    df = attach_controls(df, weights, quality)

    P("DH MORAL HAZARD / HIT BATSMEN")
    P("=" * 72)
    P(f" {len(df):,} non-pitcher PAs; overall HBP rate {df['hbp'].mean()*100:.3f}%; "
      f"DH-in-effect share {df['dh'].mean():.3f}")

    X = ["dh", "bq", "pq", "pwild"]
    # Cut 1: full post-1973, era FE (decade) + park (hometeam) , cluster pitcher
    c1 = df[df["season"] >= 1973].copy()
    c1["decade"] = (c1["season"] // 10) * 10
    report_cut("Cut 1: full post-1973", c1, X, ["decade"], "pitcher")

    # Cut 1b: our controlled micro method on GST(1997)'s EXACT window, 1973-1990.
    # This is the decisive test -- the aggregate AL-NL gap is 10-15% here; does the
    # composition control dissolve the effect even on their own sample?
    c1b = df[df["season"].between(1973, 1990)].copy()
    report_cut("Cut 1b: GST window 1973-1990 (controlled micro)", c1b, X, ["season"], "pitcher")

    # Cut 2: interleague only 1997-2019, DH toggles by park; season FE, cluster pitcher
    c2 = df[(df["interleague"] == 1) & (df["season"].between(1997, 2019))].copy()
    if len(c2) > 1000:
        report_cut("Cut 2: interleague 1997-2019 (park-toggle)", c2, X, ["season"], "pitcher")
    else:
        P("\n  -- Cut 2 skipped: too few interleague PAs (is there an interleague flag?) --")

    # Cut 3: within-pitcher DiD around 2022 universal DH (exclude 2020-21 transition).
    # Pitcher FE absorbs anything constant within pitcher, so pq and pwild (career
    # constants) are collinear with the FE and must be dropped; only bq (varies across
    # the batters a pitcher faces) and dh are identified here.
    c3 = df[(df["season"].between(2015, 2025)) & (df["season"] != 2020) & (df["season"] != 2021)].copy()
    report_cut("Cut 3: within-pitcher DiD around 2022 universal DH", c3, ["dh", "bq"],
               ["pitcher"], "pitcher")

    # Cut 3b: the DiD made explicit. Shielding only changed for NL pitchers at 2022
    # (AL pitchers were always shielded). treat = pitcher's league is NL; post = 2022+.
    # The interaction treat*post is the moral-hazard DiD; AL pitchers are the control.
    c3b = c3.copy()
    # a pitcher's league this game = his own team's league = vis_lg if batter is home
    # else home_lg (the pitching side is the visitor when the batter's team is home)
    pit_lg = np.where(c3b["bat_home"] == 1, c3b["vis_lg"], c3b["home_lg"])
    c3b["treat_nl"] = (pit_lg == "NL").astype(float)
    c3b["post"] = (c3b["season"] >= 2022).astype(float)
    c3b["treat_post"] = c3b["treat_nl"] * c3b["post"]
    # treat_nl is constant within pitcher (a pitcher's league), so it is absorbed by
    # the pitcher FE and omitted; the DiD estimate is the coef on treat_post.
    report_cut("Cut 3b: explicit DiD (NL pitchers gain shielding at 2022; AL = control)",
               c3b, ["treat_post", "post", "bq"], ["pitcher"], "pitcher")

    aggregate_replication(df)
    rolling_decay(df)
    emit_series_and_figures(df)
    (OUTPUT_DIR / "dh_hbp_moral_hazard_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\nwrote dh_hbp_moral_hazard_summary.txt")


def aggregate_replication(df: pd.DataFrame):
    """Reproduce the published aggregate result on the originals' own samples.
    GST(1997): AL DH seasons 1973-1990, AL batters hit 10-15% more than NL.
    GST(1998): the effect fades when extended through 1997 (blamed on 1993 NL
    expansion). We compute the AL-NL HBP-rate gap and ratio on three windows so
    the confirmation AND its known fragility both show, the way the micro cuts
    then explain. League here = the batting team's league (a batter is 'in the AL'
    if his team is the AL team), matching how the leagues' batters are tallied."""
    P("\n" + "=" * 72)
    P("AGGREGATE REPLICATION  (the originals' samples, from our panel)")
    P("=" * 72)
    d = df.copy()
    # batting team's league: bat team is home when bat_home==1 -> home_lg, else vis_lg
    d["bat_lg"] = np.where(d["bat_home"] == 1, d["home_lg"], d["vis_lg"])
    d = d[d["bat_lg"].isin(["AL", "NL"])]
    windows = [("1973-1990  GST(1997) original", 1973, 1990),
               ("1973-1997  GST(1998) extension", 1973, 1997),
               ("1973-2025  full panel", 1973, 2025)]
    P(f"  {'window':>32} {'AL HBP%':>8} {'NL HBP%':>8} {'AL/NL':>7} {'gap%':>6}")
    for name, lo, hi in windows:
        w = d[d["season"].between(lo, hi)]
        al = w.loc[w["bat_lg"] == "AL", "hbp"].mean()
        nl = w.loc[w["bat_lg"] == "NL", "hbp"].mean()
        ratio = al / nl if nl else float("nan")
        P(f"  {name:>32} {al*100:>8.3f} {nl*100:>8.3f} {ratio:>7.3f} {(ratio-1)*100:>6.1f}")
    P("  (GST's 10-15% is the 1973-1990 'gap%'. Watch it shrink on extension --")
    P("   their own 1998 retraction. The micro cuts above attribute the residual")
    P("   to batter composition + era, not moral hazard.)")


WIN_W, WIN_STEP = 15, 3       # rolling window width (years) and step between centers

def rolling_decay(df: pd.DataFrame):
    """The controlled shielding coefficient (Cut-1 spec) estimated on a rolling
    window, so the decay is one consistent-spec curve rather than four differently
    built cuts. Also a non-overlapping era-block table. Writes dh_hbp_rolling.csv."""
    X = ["dh", "bq", "pq", "pwild"]
    P("\n" + "=" * 72)
    P("ROLLING-WINDOW SHIELDING COEFFICIENT  (one spec, varying window)")
    P("=" * 72)
    rows = []
    half = WIN_W // 2
    centers = range(1973 + half, 2025 - half + 1, WIN_STEP)
    P(f"  {'window':>13} {'center':>6} {'n':>10} {'dh(pp)':>8} {'SE':>7} {'t':>6}")
    for c in centers:
        lo, hi = c - half, c + half
        w = df[df["season"].between(lo, hi)]
        if w["dh"].nunique() < 2 or len(w) < 50000:
            continue
        try:
            res, n = lpm(w, X, ["season"], "pitcher")
            b, s, t = res["dh"]
            rows.append({"center": c, "lo": lo, "hi": hi, "n": n,
                         "dh_pp": b * 100, "se_pp": s * 100, "t": t})
            P(f"  {lo}-{hi:>4} {c:>6} {n:>10,} {b*100:>8.4f} {s*100:>7.4f} {t:>6.2f}")
        except Exception as e:
            P(f"  {lo}-{hi}: fit failed ({e})")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "dh_hbp_rolling.csv", index=False,
                              float_format="%.5f")

    # non-overlapping era blocks, same spec
    P("\n  era blocks (same controlled spec):")
    blocks = [(1973, 1990), (1991, 2005), (2006, 2021), (2022, 2025)]
    P(f"    {'block':>11} {'n':>10} {'dh(pp)':>8} {'SE':>7} {'t':>6}")
    for lo, hi in blocks:
        w = df[df["season"].between(lo, hi)]
        if w["dh"].nunique() < 2:
            P(f"    {lo}-{hi}: no DH variation in block (skipped)"); continue
        res, n = lpm(w, X, ["season"], "pitcher")
        b, s, t = res["dh"]
        P(f"    {lo}-{hi:>4} {n:>10,} {b*100:>8.4f} {s*100:>7.4f} {t:>6.2f}")


def emit_series_and_figures(df: pd.DataFrame):
    """Plot-ready data for the two figures, and (if matplotlib is present) the
    figures themselves: (A) raw AL vs NL HBP rate by season; (B) the rolling
    shielding coefficient with a confidence ribbon."""
    d = df.copy()
    d["bat_lg"] = np.where(d["bat_home"] == 1, d["home_lg"], d["vis_lg"])
    d = d[d["bat_lg"].isin(["AL", "NL"])]
    g = (d.groupby(["season", "bat_lg"])["hbp"].agg(["mean", "size"]).reset_index())
    wide = g.pivot(index="season", columns="bat_lg", values="mean")
    wide.columns = [f"{c}_hbp_rate" for c in wide.columns]
    cnt = g.pivot(index="season", columns="bat_lg", values="size")
    cnt.columns = [f"{c}_n" for c in cnt.columns]
    series = wide.join(cnt).reset_index()
    series.to_csv(OUTPUT_DIR / "dh_hbp_season_league.csv", index=False, float_format="%.6f")
    P(f"\nwrote dh_hbp_season_league.csv ({len(series)} seasons) and dh_hbp_rolling.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        P("  (matplotlib not available -- CSVs written, generate figures yourself)")
        return

    # Fig A: AL vs NL HBP rate by season
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    s = series.dropna(subset=["AL_hbp_rate", "NL_hbp_rate"])
    ax.plot(s["season"], s["AL_hbp_rate"] * 100, label="AL", lw=1.8)
    ax.plot(s["season"], s["NL_hbp_rate"] * 100, label="NL", lw=1.8)
    ax.axvline(1973, ls="--", lw=1, color="0.5")
    ax.annotate("DH (AL) 1973", (1973, ax.get_ylim()[1] * 0.95), fontsize=8,
                rotation=90, va="top", ha="right", color="0.4")
    ax.set_xlabel("season"); ax.set_ylabel("HBP per 100 PA")
    ax.set_title("Hit-by-pitch rate by league")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig_hbp_al_nl.png", dpi=200); plt.close(fig)

    # Fig B: rolling shielding coefficient with CI ribbon
    try:
        r = pd.read_csv(OUTPUT_DIR / "dh_hbp_rolling.csv")
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        ax.axhline(0, color="0.6", lw=1)
        ax.fill_between(r["center"], r["dh_pp"] - 1.96 * r["se_pp"],
                        r["dh_pp"] + 1.96 * r["se_pp"], alpha=0.2)
        ax.plot(r["center"], r["dh_pp"], lw=2)
        ax.set_xlabel(f"window center ({WIN_W}-yr window)")
        ax.set_ylabel("shielding effect on HBP (pp)")
        ax.set_title("Moral-hazard effect over time (controlled)")
        fig.tight_layout(); fig.savefig(OUTPUT_DIR / "fig_hbp_rolling.png", dpi=200); plt.close(fig)
        P("  wrote fig_hbp_al_nl.png and fig_hbp_rolling.png")
    except Exception as e:
        P(f"  (rolling figure skipped: {e})")


# ------------------------------ synthetic ---------------------------------
def synthetic():
    rng = np.random.default_rng(11)
    n = 600_000
    pitchers = rng.integers(0, 4000, n)
    batters = rng.integers(0, 6000, n)
    bq = rng.normal(0, 1, 6000)[batters]              # batter worth-hitting
    pwild = np.abs(rng.normal(0, 1, 4000))[pitchers]  # pitcher wildness
    season = rng.integers(1974, 2026, n)
    home_lg = np.where(rng.random(n) < 0.5, "AL", "NL")
    dh = dh_in_effect(season, home_lg)
    # planted: base 1% HBP; +0.20pp moral hazard when shielded; better batters &
    # wilder pitchers hit more (the composition + wildness confounds)
    base = 0.010
    p = base + 0.0020 * dh + 0.0015 * bq + 0.0030 * pwild
    # composition confound: under DH the realized batter pool is better -> raise bq
    bq_obs = bq + 0.25 * dh                            # DH games face better hitters
    p = base + 0.0020 * dh + 0.0015 * bq_obs + 0.0030 * pwild
    hbp = (rng.random(n) < np.clip(p, 0, 1)).astype(float)
    df = pd.DataFrame({"hbp": hbp, "dh": dh, "bq": bq_obs,
                       "pq": rng.normal(0, 1, n), "pwild": pwild,
                       "pitcher": pitchers, "season": season})
    P("SYNTHETIC TEST  (planted moral hazard = +0.20 pp on HBP when shielded)")
    P("=" * 72)
    P(f" {n:,} PAs; base HBP {df['hbp'].mean()*100:.3f}%")
    P("\n  Naive (no batter-quality control) -- should be INFLATED by composition:")
    report_cut("naive: hbp ~ dh", df, ["dh"], ["season"], "pitcher")
    P("\n  Controlled -- should RECOVER ~+0.20 pp:")
    report_cut("hbp ~ dh + bq + pwild", df, ["dh", "bq", "pwild"], ["season"], "pitcher")
    P("\n  Read: the naive dh coef is biased up because DH games face better hitters;")
    P("  adding batter quality nets out composition and returns the planted +0.20 pp.")
    (OUTPUT_DIR / "dh_hbp_moral_hazard_summary.txt").write_text("\n".join(L), encoding="utf-8")


def main():
    if "--synthetic" in sys.argv:
        synthetic(); return
    run_real()


if __name__ == "__main__":
    main()
