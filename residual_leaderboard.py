"""
residual_leaderboard.py
=======================

Who did teams play MORE (or less) than their bat, position, and age justify?

The allocation model has NO player fixed effect: it predicts a player's share of
his team's plate appearances from his career bat (tau_bat), his position, a
position-specific age curve, and the season. Whatever is left over -- actual share
minus predicted share -- is playing time the model cannot account for. Summed over a
career it is a single, model-derived number: time on the field bought by something
other than the bat, at a position whose average glove is ALREADY credited and at an
age whose average decline is ALREADY credited.

Positive  => valued beyond the bat: glove, durability/iron-man health, clutch or
             leadership reputation, fan appeal, a cheap contract a team kept playing.
Negative  => played less than the profile predicts: injury-shortened seasons,
             late-career benchings, platoon usage, negative intangibles.
(It is a "why did this guy play so much?" index, not a pure defensive metric -- it
blends every off-bat reason at once. Career gaps from war service show up as MISSING
seasons, not negative residuals, except where a season was cut in half.)

Prediction uses the SAME censored-share Tobit (M3: tau_bat + position + restricted-
cubic-spline age x position + season), so the residual is consistent with the paper's
estimator. Residual = actual share - latent linear prediction (X @ beta). Aggregated
per player two ways:
  total_fts  = sum over seasons of residual / full-time-share
               -> career "extra full-time seasons" beyond the profile (the headline)
  mean_pct   = PA-weighted mean of residual / full-time-share x 100
               -> typical % of a full-time workload above prediction, per season

Reads pa_allocation_panel.parquet, pa_team_share.parquet (from
share_tobit_positional), and biofile0.csv for names. Requires scipy.
"""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

PROJECT_DIR = Path(r"C:\baseball_eras")
RETRO_DIR   = Path(r"C:\overnight_effect_data\retrosheet")
PANEL       = PROJECT_DIR / "data"   / "pa_allocation_panel.parquet"
SHARE_CACHE = PROJECT_DIR / "data"   / "pa_team_share.parquet"
BIOFILE     = RETRO_DIR / "biofile0.csv"
OUT_CSV     = PROJECT_DIR / "output" / "residual_leaderboard.csv"
OUT_SUM     = PROJECT_DIR / "output" / "residual_leaderboard_summary.txt"
RESID_SEASON = PROJECT_DIR / "data"   / "residual_by_season.parquet"

REF_POS = "CORNER_OF"
REF_AGE = 27
POS = ["C", "SS", "2B", "3B", "CF", "1B", "DH"]
SPLINE_KNOTS = [23.0, 27.0, 30.0, 33.0, 37.0]
HEADLINE_FRAC = 0.90
MIN_CAREER_PA = 3000          # established players only, so names are recognizable
TOPN = 40
L = []
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def P(s=""): print(s); L.append(s)


def rcs_basis(x, knots):
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
    K = X.shape[1]; beta = theta[:K]; log_s = theta[K]; s = np.exp(log_s)
    m = X @ beta; unc = ~cen; g = np.zeros(K + 1); ll = 0.0
    r = (y[unc] - m[unc]) / s
    ll += np.sum(-log_s - 0.5 * np.log(2 * np.pi) - 0.5 * r ** 2)
    g[:K] += ((r / s)[:, None] * X[unc]).sum(axis=0); g[K] += np.sum(r ** 2 - 1.0)
    a = (m[cen] - c) / s; ll += np.sum(norm.logcdf(a))
    lam = np.exp(norm.logpdf(a) - norm.logcdf(a))
    g[:K] += ((lam / s)[:, None] * X[cen]).sum(axis=0); g[K] += np.sum(-lam * a)
    return -ll, -g


def fit_tobit(y, X, c):
    cen = y >= c
    beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
    theta0 = np.append(beta0, np.log((y - X @ beta0).std()))
    res = optimize.minimize(neg_ll_and_grad, theta0, args=(y, X, c, cen),
                            jac=True, method="L-BFGS-B",
                            options={"maxiter": 4000, "ftol": 1e-11})
    return res, cen.mean()


def load_names() -> pd.Series:
    """Retrosheet biofile -> player_id : 'First Last'. Column names vary across
    Retrosheet/Chadwick vintages, so resolve them flexibly and never crash: if
    names can't be found we fall back to ids so the leaderboard still writes."""
    bio = pd.read_csv(BIOFILE, dtype=str, encoding="utf-8-sig")
    low = {c.lower().strip(): c for c in bio.columns}
    print("  biofile columns:", list(bio.columns))
    BLOCK = ("game", "date", "debut", "final", "year", "day", "time", "death",
             "birth", "city", "state", "team", "pos", "bat", "throw", "play", "id")
    def find(exacts, subs):
        for k in exacts:
            if k in low:
                return low[k]
        for lc, orig in low.items():
            if any(t in lc for t in subs) and not any(b in lc for b in BLOCK):
                return orig
        return None
    idc    = find(["id", "playerid", "retroid"], [])
    fullc  = find(["fullname", "playername", "name"], [])   # full-name cols only; never usename
    lastc  = find(["last", "lname", "last_name", "namelast", "lastname", "surname"], ["last", "lname", "surname"])
    firstc = find(["first", "fname", "first_name", "namefirst", "firstname", "given"], ["first", "fname", "given"])
    print(f"  resolved -> id={idc!r} full={fullc!r} first={firstc!r} last={lastc!r}")
    if idc is None:
        print("  WARNING: no id column; names will be ids")
        return pd.Series(dtype=str)
    ids = bio[idc].astype(str).str.strip()
    if firstc and lastc:
        nm = (bio[firstc].fillna("").str.strip() + " " + bio[lastc].fillna("").str.strip()).str.strip()
    elif fullc:
        nm = bio[fullc].fillna("").str.strip()
    elif lastc:
        nm = bio[lastc].fillna("").str.strip()
    else:
        print("  WARNING: name columns not found; using ids")
        nm = ids
    return pd.Series(nm.values, index=ids.values)


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    share = pd.read_parquet(SHARE_CACHE)
    full_ref = float(share["pa_share"].quantile(0.99))
    c = HEADLINE_FRAC * full_ref
    ps = pd.read_parquet(PANEL)
    ps["batter"] = ps["batter"].astype(str); ps["season"] = ps["season"].astype(int)
    ps = ps.merge(share, on=["batter", "season"], how="inner").reset_index(drop=True)
    log(f"{len(ps):,} player-seasons; full-time ref share {full_ref:.4f}")

    posv = ps["pos"].astype(str).to_numpy()
    seasons = sorted(ps["season"].unique())
    age = ps["age_exact"].to_numpy(float) - REF_AGE
    B = rcs_basis(age, [k - REF_AGE for k in SPLINE_KNOTS]); nb = B.shape[1]

    cols = [("const", np.ones(len(ps))), ("tau_bat", ps["tau_bat"].to_numpy(float))]
    cols += [(f"pos_{p}", (posv == p).astype(float)) for p in POS]
    cols += [(f"s{b}", B[:, b]) for b in range(nb)]
    cols += [(f"pos_{p}_s{b}", (posv == p).astype(float) * B[:, b]) for p in POS for b in range(nb)]
    cols += [(f"yr_{s}", (ps["season"] == s).to_numpy(float)) for s in seasons[1:]]
    X = np.column_stack([v for _, v in cols])
    y = ps["pa_share"].to_numpy(float)

    log("fitting M3 censored-share Tobit for the prediction")
    res, cf = fit_tobit(y, X, c)
    beta = res.x[:X.shape[1]]
    pred = X @ beta                                  # latent predicted share
    ps["resid"] = y - pred                           # actual minus predicted
    ps["resid_fts"] = ps["resid"] / full_ref         # in full-time-season units
    log(f"fit done ({cf*100:.1f}% censored); mean resid {ps['resid'].mean():+.5f} (~0 expected)")

    # season-grain residual consumed by validate_fielding.py (which must run after this script)
    ps[["batter", "season", "pa", "pa_share", "resid_fts"]].to_parquet(RESID_SEASON, index=False)
    log(f"wrote {RESID_SEASON} ({len(ps):,} batter-seasons)")

    # ---- aggregate per player ----
    def agg(g):
        w = g["pa"].to_numpy(float)
        return pd.Series({
            "career_pa": int(g["pa"].sum()),
            "first_season": int(g["season"].min()),
            "last_season": int(g["season"].max()),
            "n_seasons": int(g["season"].nunique()),
            "total_fts": float(g["resid_fts"].sum()),
            "mean_pct": float(np.sum(w * g["resid_fts"].to_numpy()) / w.sum() * 100),
        })
    car = ps.groupby("batter").apply(agg).reset_index()
    # primary position = PA-weighted modal position over the career
    pp = (ps.groupby(["batter", "pos"])["pa"].sum().reset_index()
            .sort_values("pa").drop_duplicates("batter", keep="last")[["batter", "pos"]])
    car = car.merge(pp, on="batter", how="left")
    car = car[car["career_pa"] >= MIN_CAREER_PA].copy()
    for cc in ["career_pa", "first_season", "last_season", "n_seasons"]:
        car[cc] = car[cc].astype(int)

    names = load_names()
    car["name"] = car["batter"].map(names).fillna(car["batter"])
    car["span"] = car["first_season"].astype(str) + "-" + car["last_season"].astype(str)
    car = car.sort_values("total_fts", ascending=False).reset_index(drop=True)

    keep = ["name", "pos", "span", "n_seasons", "career_pa", "mean_pct", "total_fts", "batter"]
    car[keep].to_csv(OUT_CSV, index=False, float_format="%.3f")
    log(f"wrote {OUT_CSV} ({len(car):,} players with >= {MIN_CAREER_PA:,} career PA)")

    def show(df, title):
        P(f"\n--- {title} ---")
        P(f"  {'player':24s} {'pos':>4s} {'span':>10s} {'PA':>7s} {'mean%':>6s} {'tot FTS':>8s}")
        for r in df.itertuples():
            P(f"  {r.name[:24]:24s} {r.pos:>4s} {r.span:>10s} {r.career_pa:>7,d} "
              f"{r.mean_pct:>+6.1f} {r.total_fts:>+8.2f}")

    P("CAREER ALLOCATION RESIDUALS  (played more/less than bat + position + age + season predict)")
    P("=" * 84)
    P(f"{len(car):,} players, >= {MIN_CAREER_PA:,} career PA. total_fts = career extra")
    P("full-time-seasons of playing time beyond the profile; mean% = per-season rate.")
    show(car.head(TOPN), f"MOST valued beyond the bat (top {TOPN})")
    show(car.tail(TOPN).iloc[::-1], f"LEAST valued / played less than predicted (bottom {TOPN})")
    OUT_SUM.write_text("\n".join(L), encoding="utf-8")
    log(f"wrote {OUT_SUM}")
    print("\n".join(L[:60]))
    print(f"\n... full leaderboard in {OUT_CSV}")


if __name__ == "__main__":
    main()
