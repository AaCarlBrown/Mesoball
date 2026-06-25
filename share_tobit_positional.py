"""
share_tobit_positional.py
=========================

One estimator throughout: a RIGHT-CENSORED normal (Tobit) on the player's SHARE
of his team's plate appearances (PA / team PA, ex-IBB), fitting all three
positional specifications under the same model.

  M1  level-only positional premium                 (reported at 3 thresholds)
  M2  position-by-era premium                        (headline threshold)
  M3  spline-in-age x position premium               (headline threshold)

Why share + censoring: raw season PA piles up against a ceiling near a full-time
workload, which attenuates OLS slopes. The share divides out the season schedule
and team-offense level so the ceiling is a near-constant share (~1/9 of team PA);
the Tobit then treats a player at or above the censoring threshold c as evidence
that his LATENT demand was >= c, contributing P(share* >= c) to the likelihood
rather than fitting his observed value -- keeping the everyday regulars in as
lower-bound evidence without fitting rest/injury noise in the 90-100% region.

Latent:   share* = a + b*tau_bat + (position) + (era or age or season) + e,  e~N(0,s^2)
Observed: share  = share*       if share* <  c
                   censored at c if share* >= c   (only share* >= c is known)

The censored-normal log-likelihood and gradient are written out in full. Per-cell
premia for M2 are single coefficients (a nested position-within-era
parameterization, so the SE is read directly and the DH's empty early-era cells
simply do not exist). For M3 each age premium is a linear combination of a
position main and its spline terms, whose SE is computed from the full covariance
matrix. SEs are clustered by batter via a score (BHHH) sandwich.

Reads the existing panel; reads plays.csv once (then caches the share). Requires
scipy. Writes only a summary.
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
PLAYS_CSV    = RETRO_DIR / "plays.csv"
GAMEINFO_CSV = RETRO_DIR / "gameinfo.csv"
PANEL        = PROJECT_DIR / "data"   / "pa_allocation_panel.parquet"
SHARE_CACHE  = PROJECT_DIR / "data"   / "pa_team_share.parquet"
TEAMS_CSV    = PROJECT_DIR / "Teams.csv"
OUT          = PROJECT_DIR / "output" / "share_tobit_positional_summary.txt"

REF_POS = "CORNER_OF"
REF_AGE = 27
POS = ["C", "SS", "2B", "3B", "CF", "1B", "DH"]          # report order; ref = CORNER_OF
ERAS = ["pre_integration", "integration", "expansion_FA", "steroid", "modern"]
SPLINE_KNOTS = [23.0, 27.0, 30.0, 33.0, 37.0]
GRID = list(range(22, 39, 2))
THRESH_FRACS = [0.85, 0.90, 0.95]    # M1 robustness strip
HEADLINE_FRAC = 0.90                 # single estimator for M2 / M3

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


# ----------------------------------------------------------------------
# Censored-normal negative log-likelihood and gradient.
# theta = [beta (K), log_sigma];  y = share, X = design, cen = (share >= c).
# ----------------------------------------------------------------------
def neg_ll_and_grad(theta, y, X, c, cen):
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
    """Fit censored-normal; return (theta, V) with V clustered by batter."""
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
    """Estimate and clustered SE of sum_t w_t coef_t; None if any term absent.
    V is (K+1)x(K+1) including log_sigma; combinations use only the beta block."""
    K = len(names)
    Vb = V[:K, :K]
    c = np.zeros(K); est = 0.0
    for t, w in weights.items():
        if t not in names:
            return None, None
        i = names.index(t); c[i] = w; est += w * theta[i]
    return est, float(np.sqrt(c @ Vb @ c))


def design(ps, cols):
    """Stack named columns into (matrix, names)."""
    names = [n for n, _ in cols]
    X = np.column_stack([np.asarray(v, float) for _, v in cols])
    return X, names


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # ---- team-PA share (cached) ----
    if SHARE_CACHE.exists():
        log(f"loading cached team share {SHARE_CACHE.name} (delete it to recompute)")
        share = pd.read_parquet(SHARE_CACHE)
        PA_TEAM_MEAN = float(share["pa_team_mean"].iloc[0])
    else:
        log("reading plays.csv for team denominator")
        pl = pd.read_csv(PLAYS_CSV, usecols=lambda x: x in ["gid", "batter", "batteam", "pa", "iw"],
                         dtype={"gid": "string", "batter": "string", "batteam": "string",
                                "pa": "Int8", "iw": "Int8"}, low_memory=False)
        pl = pl[pl["pa"] == 1]
        gi = pd.read_csv(GAMEINFO_CSV, usecols=["gid", "season", "gametype"],
                         dtype={"gid": "string", "season": "Int32", "gametype": "string"})
        gi = gi[gi["gametype"] == "regular"]
        pl = pl.merge(gi[["gid", "season"]], on="gid", how="inner")
        pl = pl[pl["iw"] != 1]
        pl["season"] = pl["season"].astype(int)
        log(f"  {len(pl):,} ex-IBB regular-season PA rows")
        pst = pl.groupby(["batter", "season", "batteam"]).size().reset_index(name="n")
        team_tot = pl.groupby(["season", "batteam"]).size().reset_index(name="team_pa")
        pst = pst.merge(team_tot, on=["season", "batteam"])
        agg = pst.groupby(["batter", "season"]).agg(
            denom=("team_pa", "sum"), pa_chk=("n", "sum")).reset_index()
        agg["pa_share"] = agg["pa_chk"] / agg["denom"]
        PA_TEAM_MEAN = float(team_tot["team_pa"].mean())
        agg["pa_team_mean"] = PA_TEAM_MEAN
        # primary-team league (AL/NL), needed by dh_null_bridge.py and dh_split_table.py
        tlg = pd.read_csv(TEAMS_CSV, usecols=["yearID", "teamIDretro", "lgID"]).rename(
            columns={"yearID": "season", "teamIDretro": "batteam", "lgID": "league"})
        tlg["season"] = tlg["season"].astype(int)
        prim = (pst.sort_values("n").drop_duplicates(["batter", "season"], keep="last")
                    [["batter", "season", "batteam"]]
                    .merge(tlg, on=["season", "batteam"], how="left"))
        agg = agg.merge(prim[["batter", "season", "league"]], on=["batter", "season"], how="left")
        share = agg[["batter", "season", "pa_share", "pa_chk", "pa_team_mean", "league"]]
        share.to_parquet(SHARE_CACHE, index=False)
        log(f"  cached share to {SHARE_CACHE.name}")

    ps = pd.read_parquet(PANEL)
    ps["batter"] = ps["batter"].astype(str); ps["season"] = ps["season"].astype(int)
    n0 = len(ps)
    ps = ps.merge(share, on=["batter", "season"], how="inner").reset_index(drop=True)
    P("SHARE-TOBIT POSITIONAL FIT  (one estimator: censored-share, batter-clustered SEs)")
    P("=" * 72)
    P(f"panel rows {n0:,}; matched {len(ps):,}; pa-vs-numerator match "
      f"{(ps['pa'] == ps['pa_chk']).mean()*100:.2f}%; mean team PA {PA_TEAM_MEAN:,.0f}")

    full_ref = float(ps["pa_share"].quantile(0.99))
    c_head = HEADLINE_FRAC * full_ref
    P(f"full-time reference (99th pct) = {full_ref:.4f};  headline c = "
      f"{HEADLINE_FRAC:.2f} x ref = {c_head:.4f}")

    # ---- shared design pieces ----
    ps["pos"] = pd.Categorical(ps["pos"].astype(str), categories=[REF_POS] + POS)
    posv = ps["pos"].astype(str).to_numpy()
    erav = ps["era"].astype(str).to_numpy()
    seasons = sorted(ps["season"].unique())
    yr_cols = [(f"yr_{s}", (ps["season"] == s).to_numpy()) for s in seasons[1:]]
    age_c = ps["age_c"].to_numpy(float) if "age_c" in ps else (ps["age_exact"].to_numpy(float) - REF_AGE)
    season_c = (ps["season"].to_numpy(float) - 1970.0)
    base = [("const", np.ones(len(ps))), ("tau_bat", ps["tau_bat"].to_numpy(float))]
    pos_cols = [(f"pos_{p}", (posv == p).astype(float)) for p in POS]
    y = ps["pa_share"].to_numpy(float)
    cl = ps["batter"].to_numpy()
    SC = PA_TEAM_MEAN  # share -> PA-equivalent

    # ---- M1: level-only, at the three thresholds ----
    X1, n1 = design(ps, base + pos_cols + yr_cols)
    P("\n--- M1 positional premium vs corner OF, PA-equivalent (est / SE / t) ---")
    P("  (OLS-PA reference: SS 110, 2B 73, CF 61, 3B 52, 1B 16, C -44, DH -75)")
    P("  " + "".join(f"{p:>13s}" for p in POS))
    tau_sd = float(ps["tau_bat"].std())
    for frac in THRESH_FRACS:
        res, V, cf = fit_tobit(y, X1, frac * full_ref, cl)
        cells = []
        for p in POS:
            e, se = lincom({f"pos_{p}": 1.0}, n1, res.x, V)
            cells.append(f"{e*SC:>5.0f}({se*SC:>3.0f},t{e/se:>4.1f})")
        P(f"  c={frac:.2f} ({cf*100:4.1f}% cens) " + " ".join(cells))
        if abs(frac - HEADLINE_FRAC) < 1e-9:
            # bat slope at the headline threshold, in share and PA-equivalent units
            b_tau, se_tau = lincom({"tau_bat": 1.0}, n1, res.x, V)
            P(f"  bat slope (tau_bat): {b_tau:.4f} share/unit (SE {se_tau:.4f}); "
              f"per +1 SD of tau_bat = {b_tau*tau_sd*SC:+.0f} PA-equivalent "
              f"(SE {se_tau*tau_sd*SC:.0f}); tau_bat SD = {tau_sd:.4f}")

    # ---- M2: position-by-era, nested cells (headline c) ----
    era_main = [(f"era_{e}", (erav == e).astype(float)) for e in ERAS[1:]]
    cell_cols, have = [], {}
    for p in POS:
        for e in ERAS:
            col = ((posv == p) & (erav == e)).astype(float)
            if col.sum() > 0:
                cell_cols.append((f"cell_{p}_{e}", col)); have[(p, e)] = True
    ctrl2 = [("age_c", age_c), ("age_c2", age_c ** 2),
             ("season_c", season_c), ("season_c2", season_c ** 2)]
    X2, n2 = design(ps, base + era_main + cell_cols + ctrl2)
    res2, V2, cf2 = fit_tobit(y, X2, c_head, cl)
    P(f"\n--- M2 premium vs corner OF by era, PA-equivalent, est(SE)  [c={HEADLINE_FRAC:.2f}, {cf2*100:.1f}% cens] ---")
    P("  " + " ".join(f"{e[:9]:>13s}" for e in ERAS))
    for p in POS:
        cells = []
        for e in ERAS:
            if (p, e) in have:
                est, se = lincom({f"cell_{p}_{e}": 1.0}, n2, res2.x, V2)
                cells.append(f"{est*SC:>6.0f}({se*SC:>3.0f})")
            else:
                cells.append(f"{'--':>11s}")
        P(f"  {p:>4s} " + " ".join(cells))

    # ---- M3: spline-in-age x position (headline c) ----
    knots_c = [k - REF_AGE for k in SPLINE_KNOTS]
    B = rcs_basis(ps["age_exact"].to_numpy(float) - REF_AGE, knots_c)
    nb = B.shape[1]
    sp_main = [(f"s{b}", B[:, b]) for b in range(nb)]
    sp_int = [(f"pos_{p}_s{b}", (posv == p).astype(float) * B[:, b])
              for p in POS for b in range(nb)]
    X3, n3 = design(ps, base + pos_cols + sp_main + sp_int + yr_cols)
    res3, V3, cf3 = fit_tobit(y, X3, c_head, cl)
    gB = rcs_basis(np.array(GRID, float) - REF_AGE, knots_c)
    P(f"\n--- M3 premium vs corner OF by age, PA-equivalent, est(SE)  [c={HEADLINE_FRAC:.2f}, {cf3*100:.1f}% cens] ---")
    P("  " + " ".join(f"{a:>9d}" for a in GRID))
    for p in POS:
        cells = []
        for gi_ in range(len(GRID)):
            w = {f"pos_{p}": 1.0}
            for b in range(nb):
                w[f"pos_{p}_s{b}"] = float(gB[gi_, b])
            est, se = lincom(w, n3, res3.x, V3)
            cells.append(f"{est*SC:>4.0f}({se*SC:>2.0f})")
        P(f"  {p:>4s} " + " ".join(cells))

    P("\nOne estimator throughout: censored-share Tobit, c = 0.90 x full-time ref,")
    P("SEs batter-clustered (BHHH score sandwich). PA-equivalent = share x mean team PA.")
    OUT.write_text("\n".join(L), encoding="utf-8")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
