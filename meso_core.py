"""
meso_core.py
============

Shared core for the Mesoball build. Everything the result scripts need that
is not specific to one exhibit lives here, so there is exactly one copy of the
estimator, one copy of the units conversion, and one place for the constants.

Three blocks:

  1. ESTIMATOR -- the censored-share Tobit and its helpers (rcs_basis,
     neg_ll_and_grad, per_obs_scores, fit_tobit, lincom, design,
     within_season_vif). These are lifted verbatim from the headline
     positional fit and the auxiliary-tests script, which carried byte-identical
     copies; consolidating them here is the whole point of the cleanup. Numbers
     are unchanged by construction.

  2. UNITS (reporting only) -- season Pythagenpat runs-per-win, wOBA->runs,
     runs->wins. NOTHING ANALYTIC DEPENDS ON THIS BLOCK. All estimation is done
     in wOBA; these functions exist solely to put a wOBA or PA-equivalent number
     into runs/wins for the reader. PA_FULL is a presentational anchor, not the
     censoring threshold (those are different objects -- see PA_FULL note).

  3. CONSTANTS / PATHS -- era boundaries, spline knots, position map, censor
     fractions, expansion years, integration phase-in, file locations.

The estimator is a RIGHT-CENSORED normal (Tobit) on a player's SHARE of his
team's plate appearances (PA / team PA, ex-IBB). Latent:
    share* = X @ beta + e,   e ~ N(0, sigma^2)
Observed:
    share  = share*          if share* <  c
             censored at c    if share* >= c     (only "share* >= c" is known)
SEs are clustered by batter via a BHHH score sandwich. Linear combinations of
coefficients (premia, age-spline evaluations) come through `lincom`.

Requires numpy, pandas, scipy. pyfixest is needed by the build scripts that
import this, not by the module itself.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows consoles default to cp1252, which raises UnicodeEncodeError on glyphs
# like the approximately-equal sign. Force UTF-8 on the output streams so any
# script that imports meso_core can print freely. Guarded: a no-op where the
# streams don't support reconfigure (e.g. already-wrapped pipes).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

__all__ = [
    # estimator
    "rcs_basis", "neg_ll_and_grad", "per_obs_scores", "fit_tobit",
    "lincom", "design", "within_season_vif", "drop_singletons",
    # units (reporting only)
    "runs_per_win_by_season", "woba_to_runs", "runs_to_wins",
    # phase-in / shock helpers
    "integ_phasein", "exp_intensity", "era_of", "baseball_age",
    # paths
    "PROJECT_DIR", "RETRO_DIR", "DATA_DIR", "OUTPUT_DIR",
    "PANEL_PA", "PANEL_ALLOC", "SHARE_CACHE", "WOBA_CSV", "TEAMS_CSV",
    # constants
    "REF_POS", "REF_AGE", "POS", "ALLPOS", "GLOVE", "POS_GROUP",
    "ERAS", "ERA_NAMES", "SPLINE_KNOTS", "HEADLINE_FRAC", "THRESH_FRACS",
    "EXP_YEARS", "WEIGHT_COLS", "EVENT_COLS", "PYTHAG_EXP", "PA_FULL",
    "TWO_WAY_AGE_LO", "TWO_WAY_AGE_HI", "ALLOC_AGE_LO", "ALLOC_AGE_HI",
    "INTEG_CENTER", "INTEG_SCALE",
]

# ======================================================================
# 3. CONSTANTS / PATHS
#    (declared first so the rest of the module can reference them)
# ======================================================================

# --- file locations -------------------------------------------------------
# PROJECT_DIR is the repo root. It DEFAULTS to the directory containing this
# module, so a fresh clone is self-contained and runs regardless of the current
# working directory (paths resolve off this file, not off where you launch).
# To reuse an existing data tree instead, set the MESOBALL_DIR environment
# variable -- e.g. set MESOBALL_DIR=C:\baseball_eras to pick up an already-built
# data\ and output\ (and the wOBA_weights.csv / Teams.csv) under that root.
#
# RETRO_DIR holds the Retrosheet CSVs and is NOT shipped in the repo (license +
# size). Only the build scripts read it; override with MESOBALL_RETRO if needed.
PROJECT_DIR = Path(os.environ.get("MESOBALL_DIR", Path(__file__).resolve().parent))
RETRO_DIR   = Path(os.environ.get("MESOBALL_RETRO", r"C:\overnight_effect_data\retrosheet"))

DATA_DIR    = PROJECT_DIR / "data"      # generated parquet artifacts
OUTPUT_DIR  = PROJECT_DIR / "output"    # generated summaries, result CSVs, plots

PANEL_PA    = DATA_DIR / "pa_panel.parquet"             # 15.4M-row PA panel
PANEL_ALLOC = DATA_DIR / "pa_allocation_panel.parquet"  # player-season allocation panel
SHARE_CACHE = DATA_DIR / "pa_team_share.parquet"        # PA share + by-hand + league
WOBA_CSV    = PROJECT_DIR / "wOBA_weights.csv"          # small input; lives beside the scripts
TEAMS_CSV   = PROJECT_DIR / "Teams.csv"                 # small input; lives beside the scripts

# --- positions ------------------------------------------------------------
REF_POS = "CORNER_OF"                                    # baseline; premia are RELATIVE to it
REF_AGE = 27
POS    = ["C", "SS", "2B", "3B", "CF", "1B", "DH"]       # all non-reference positions
ALLPOS = POS                                             # alias for readability in scripts
GLOVE  = ["C", "SS", "2B", "3B", "CF", "1B"]             # positions that take shock interactions (DH excluded)

# Retrosheet fielding-position code -> position group. Corner OF (7,9) is the
# reference; pitchers (1) are dropped from the allocation analysis.
POS_GROUP = {
    1: "P", 2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS",
    7: "CORNER_OF", 8: "CF", 9: "CORNER_OF", 10: "DH",
}

# --- eras -----------------------------------------------------------------
ERAS = [
    ("pre_integration", 1910, 1946),
    ("integration",     1947, 1968),
    ("expansion_FA",    1969, 1993),
    ("steroid",         1994, 2005),
    ("modern",          2006, 2025),
]
ERA_NAMES = [e[0] for e in ERAS]

# --- estimator settings ---------------------------------------------------
SPLINE_KNOTS  = [23.0, 27.0, 30.0, 33.0, 37.0]           # RCS interior knots (exact age)
HEADLINE_FRAC = 0.90                                     # censor at 0.90 x full-time reference
THRESH_FRACS  = [0.85, 0.90, 0.95]                       # M1 robustness strip

# --- age windows (deliberately different by stage) ------------------------
TWO_WAY_AGE_LO, TWO_WAY_AGE_HI = 18, 45                  # two-way wOBA model
ALLOC_AGE_LO,   ALLOC_AGE_HI   = 21, 40                  # allocation panel

# --- shock helpers --------------------------------------------------------
EXP_YEARS    = [1961, 1962, 1969, 1977, 1993, 1998]      # expansion seasons
INTEG_CENTER = 1954.0                                    # integration logistic center
INTEG_SCALE  = 3.0                                       # integration logistic scale

# --- wOBA weight columns (FanGraphs convention, IBB excluded) -------------
WEIGHT_COLS = ["wBB", "wHBP", "w1B", "w2B", "w3B", "wHR"]
EVENT_COLS  = ["bb_unint", "hbp", "single", "double", "triple", "hr"]  # aligned with WEIGHT_COLS

# --- units (reporting only) -----------------------------------------------
PYTHAG_EXP = 0.713      # runs_per_win = 2 * RPG^PYTHAG_EXP  (Pythagenpat)

# PA_FULL is a PRESENTATIONAL anchor, not the censoring threshold.
# The censoring threshold is a SHARE (0.90 x the 99th-pct PA share ~ 0.106 of
# team PA ~ 597 PA on the average team). PA_FULL is the assumed length of a full
# season of plate appearances used only to turn a per-PA wOBA gap into season
# runs for the reader. Under the no-wOBAScale-division shape-norm treatment,
# 0.100 wOBA x 600 = 60 runs, which is the anchor the paper quotes. Change here
# (e.g. to the full-time reference ~660) if a different convention is preferred;
# it never enters an estimate.
PA_FULL = 600


# ======================================================================
# 1. ESTIMATOR
# ======================================================================

def rcs_basis(x, knots):
    """Restricted cubic spline basis (Harrell parameterization) with linear tails.

    Returns an (n, len(knots)-1) array: column 0 is x itself, the remaining
    columns are the truncated-cube terms. NaNs in x propagate to NaN rows.
    """
    t = np.asarray(knots, float); k = len(t); x = np.asarray(x, float)
    cols = [x.copy()]
    tk, tk1 = t[-1], t[-2]; denom = tk - tk1; scale = (t[-1] - t[0]) ** 2
    cube = lambda u: np.where(u > 0, u ** 3, 0.0)
    for j in range(k - 2):
        tj = t[j]
        cols.append((cube(x - tj) - cube(x - tk1) * (tk - tj) / denom
                     + cube(x - tk) * (tk1 - tj) / denom) / scale)
    B = np.column_stack(cols); B[np.isnan(x), :] = np.nan
    return B


def neg_ll_and_grad(theta, y, X, c, cen):
    """Censored-normal negative log-likelihood and gradient.

    theta = [beta (K), log_sigma]; y = share; X = design; c = censor point;
    cen = boolean (share >= c). Uncensored obs contribute the normal density;
    censored obs contribute log P(share* >= c) = log Phi((X@beta - c)/sigma).
    """
    K = X.shape[1]; beta = theta[:K]; log_s = theta[K]; s = np.exp(log_s)
    m = X @ beta; unc = ~cen
    g = np.zeros(K + 1); ll = 0.0
    r = (y[unc] - m[unc]) / s
    ll += np.sum(-log_s - 0.5 * np.log(2 * np.pi) - 0.5 * r ** 2)
    g[:K] += ((r / s)[:, None] * X[unc]).sum(axis=0)
    g[K]  += np.sum(r ** 2 - 1.0)
    a = (m[cen] - c) / s
    ll += np.sum(norm.logcdf(a))
    lam = np.exp(norm.logpdf(a) - norm.logcdf(a))
    g[:K] += ((lam / s)[:, None] * X[cen]).sum(axis=0)
    g[K]  += np.sum(-lam * a)
    return -ll, -g


def per_obs_scores(theta, y, X, c, cen):
    """Per-observation score matrix (n, K+1) for the BHHH clustered sandwich."""
    K = X.shape[1]; beta = theta[:K]; s = np.exp(theta[K]); m = X @ beta
    G = np.zeros((len(y), K + 1)); unc = ~cen
    r = (y[unc] - m[unc]) / s
    iu = np.where(unc)[0]
    G[iu[:, None], np.arange(K)] = (r / s)[:, None] * X[unc]
    G[iu, K] = r ** 2 - 1.0
    a = (m[cen] - c) / s
    lam = np.exp(norm.logpdf(a) - norm.logcdf(a))
    ic = np.where(cen)[0]
    G[ic[:, None], np.arange(K)] = (lam / s)[:, None] * X[cen]
    G[ic, K] = -lam * a
    return G


def fit_tobit(y, X, c, cluster):
    """Fit the censored-share Tobit.

    Returns (res, V, censor_fraction) where res is the scipy OptimizeResult
    (res.x = [beta, log_sigma]) and V is the (K+1, K+1) covariance clustered by
    `cluster` (a length-n array of cluster ids, e.g. batter).
    """
    cen = y >= c
    beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
    theta0 = np.append(beta0, np.log((y - X @ beta0).std()))
    res = optimize.minimize(neg_ll_and_grad, theta0, args=(y, X, c, cen),
                            jac=True, method="L-BFGS-B",
                            options={"maxiter": 4000, "ftol": 1e-11})
    G = per_obs_scores(res.x, y, X, c, cen)
    bread = np.linalg.inv(G.T @ G)
    S = pd.DataFrame(G).groupby(cluster).sum().to_numpy()
    V = bread @ (S.T @ S) @ bread
    return res, V, cen.mean()


def lincom(weights, names, theta, V):
    """Estimate and clustered SE of sum_t w_t * coef_t.

    `weights` maps term name -> weight; `names` is the column-name list of the
    design (so coefficient index = names.index(term)). Returns (None, None) if
    any requested term is absent from the fit (e.g. an empty era cell). V is the
    (K+1, K+1) matrix; only the beta block is used.
    """
    K = len(names)
    Vb = V[:K, :K]
    c = np.zeros(K); est = 0.0
    for t, w in weights.items():
        if t not in names:
            return None, None
        i = names.index(t); c[i] = w; est += w * theta[i]
    return est, float(np.sqrt(c @ Vb @ c))


def design(cols):
    """Stack named columns into (matrix, names).

    `cols` is a list of (name, array) pairs. Returns (X, names).
    """
    names = [n for n, _ in cols]
    X = np.column_stack([np.asarray(v, float) for _, v in cols])
    return X, names


def within_season_vif(season, cols):
    """Variance-inflation factor of each column after removing season means.

    Design-only diagnostic (no dependence on the estimator), so it matches the
    OLS-era VIF checks exactly. `cols` is a list of (name, array) pairs.
    """
    M = np.column_stack([np.asarray(v, float) for _, v in cols])
    names = [n for n, _ in cols]
    df = pd.DataFrame(M, columns=names); df["__s"] = np.asarray(season)
    Z = df[names] - df.groupby("__s")[names].transform("mean")
    Z = Z.to_numpy(); out = {}
    for j, n in enumerate(names):
        others = np.delete(Z, j, axis=1)
        A = np.column_stack([np.ones(len(Z)), others])
        b, *_ = np.linalg.lstsq(A, Z[:, j], rcond=None)
        ss_res = ((Z[:, j] - A @ b) ** 2).sum()
        ss_tot = ((Z[:, j] - Z[:, j].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out[n] = np.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2)
    return out


def drop_singletons(df, keys):
    """Iteratively drop rows that are unique in any FE key.

    After this, every level of every key in `keys` has >= 2 observations, so a
    pyfixest residual vector from a model with those fixed effects aligns
    row-for-row with the returned frame. Returns a fresh-indexed copy.
    """
    while True:
        before = len(df)
        for k in keys:
            vc = df[k].value_counts()
            df = df[df[k].isin(vc[vc >= 2].index)]
        if len(df) == before:
            return df.reset_index(drop=True)


# ======================================================================
# Phase-in / shock helpers (shared by the allocation + attribution scripts)
# ======================================================================

def integ_phasein(season):
    """Integration phase-in: logistic in season, centered at INTEG_CENTER."""
    s = np.asarray(season, float)
    return 1.0 / (1.0 + np.exp(-(s - INTEG_CENTER) / INTEG_SCALE))


def exp_intensity(season, decay):
    """Expansion intensity: sum of decaying pulses from each expansion season."""
    s = np.asarray(season, float); out = np.zeros_like(s)
    for y in EXP_YEARS:
        d = s - y
        out += np.where(d >= 0, np.exp(-d / decay), 0.0)
    return out


def era_of(season):
    """Map a season to its era name (or 'other' if outside all windows)."""
    for n, lo, hi in ERAS:
        if lo <= season <= hi:
            return n
    return "other"


def baseball_age(birthdate_yyyymmdd, season):
    """Baseball age (as of June 30) from an integer YYYYMMDD birthdate and season.

    Vectorized; `birthdate_yyyymmdd` and `season` may be arrays.
    """
    bd = np.asarray(birthdate_yyyymmdd)
    s = np.asarray(season)
    by = bd // 10000
    bm = (bd // 100) % 100
    bdd = bd % 100
    a = s - by
    after = (bm > 6) | ((bm == 6) & (bdd > 30))
    return a - after.astype(int)


# ======================================================================
# 2. UNITS  (REPORTING ONLY -- nothing analytic depends on this block)
# ======================================================================

def runs_per_win_by_season(teams_csv=None, leagues=("AL", "NL")):
    """Season runs-per-win via Pythagenpat: runs_per_win = 2 * RPG^PYTHAG_EXP.

    RPG = runs per game counting BOTH teams = 2 * sum(R) / sum(G) over the
    league's teams in that season (sum(G) double-counts each game, one entry per
    team, so 2*sum(R)/sum(G) is the per-game runs of both teams combined).

    `teams_csv` defaults to the module's TEAMS_CSV, resolved at call time so a
    reassigned path (e.g. via MESOBALL_DIR or in tests) is honored. Returns a
    pandas Series indexed by season. Reporting only.
    """
    if teams_csv is None:
        teams_csv = TEAMS_CSV
    t = pd.read_csv(teams_csv, encoding="utf-8-sig", usecols=["yearID", "lgID", "R", "G"])
    if leagues is not None:
        t = t[t["lgID"].isin(leagues)]
    g = t.groupby("yearID").agg(R=("R", "sum"), G=("G", "sum"))
    rpg = 2.0 * g["R"] / g["G"]                 # runs per game, both teams
    rpw = 2.0 * rpg ** PYTHAG_EXP
    rpw.index = rpw.index.astype(int)
    rpw.name = "runs_per_win"
    return rpw.sort_index()


def woba_to_runs(delta_woba, pa=PA_FULL, wobascale=None):
    """Convert a per-PA wOBA difference into season runs over `pa` plate appearances.

    HEADLINE (shape-normalized wOBA): leave wobascale=None. Shape-normalized
    wOBA is already in run-value units, so runs = delta_woba * pa (no division).
    With the defaults, 0.100 wOBA over 600 PA -> 60 runs.

    NON-HEADLINE (seasonw / fixedw schemes): pass the season's wobascale; those
    weights are OBP-scaled, so runs = (delta_woba / wobascale) * pa.

    Reporting only.
    """
    d = np.asarray(delta_woba, float)
    if wobascale is not None:
        d = d / float(wobascale)
    return d * pa


def runs_to_wins(runs, season=None, rpw=None, rpw_table=None):
    """Convert runs to wins.

    Provide ONE of: an explicit `rpw` (runs per win), or a `season` together
    with an `rpw_table` (the Series from runs_per_win_by_season). For a pooled
    cross-era figure, pass an explicit `rpw` (default reference ~9.6 modern) and
    footnote the era range. Reporting only.
    """
    runs = np.asarray(runs, float)
    if rpw is None:
        if season is None or rpw_table is None:
            raise ValueError("pass rpw, or (season and rpw_table)")
        rpw = float(rpw_table.loc[int(season)])
    return runs / rpw
