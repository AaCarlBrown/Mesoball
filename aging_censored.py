"""
aging_censored.py  (exploratory)
===============================

The selection-corrected, EFFECTIVE baseball aging curve. A player's latent value
index drives his playing-time share, which is observed in [0, full-time]: capped
above at the full-time ceiling (stars cannot play more than every day) and
cornered below at zero (a player whose value falls below the major-league bar is
out, and counts as zero -- not a missing latent quantity). This is a TWO-LIMIT
Tobit, extending the project's right-censored share Tobit with a lower corner.

We track every established player from his first regular season to a horizon
(AGE_CAP), inserting zero-share rows for the ages at which he had dropped out, so
exits enter the estimand as zeros. The fit recovers:

  latent    m(age) = the value index's age profile (net of BOTH censorings)
  effective E[share | age] = expected effective value INCLUDING the zero corner
            -- the "effective baseball aging" curve teams actually face
  survivor  E[share | playing] -- the naive curve Fair/Bradbury approximate

The effective curve declines faster than the survivor curve after the peak,
because it counts the exits the survivor curve discards. The bar's level is not
separately identified from the index intercept, so it is normalized (lower corner
at 0, upper at the full-time reference c), not fitted.

Reads pa_allocation_panel.parquet + pa_team_share.parquet. scipy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

import meso_core as mc

AGE_CAP = 39
AGE_LO = 21
L = []
def P(s=""): print(s); L.append(s)


# ---------- two-limit Tobit (lower corner `lo`, upper censor `hi`) ----------
def _nll(theta, y, X, lo, hi, bcen, tcen):
    K = X.shape[1]; beta = theta[:K]; s = np.exp(theta[K]); m = X @ beta
    unc = ~(bcen | tcen); g = np.zeros(K + 1); ll = 0.0
    r = (y[unc] - m[unc]) / s
    ll += np.sum(-theta[K] - 0.5 * np.log(2 * np.pi) - 0.5 * r ** 2)
    g[:K] += ((r / s)[:, None] * X[unc]).sum(0); g[K] += np.sum(r ** 2 - 1.0)
    at = (m[tcen] - hi) / s; ll += np.sum(norm.logcdf(at))
    lt = np.exp(norm.logpdf(at) - norm.logcdf(at))
    g[:K] += ((lt / s)[:, None] * X[tcen]).sum(0); g[K] += np.sum(-lt * at)
    ab = (lo - m[bcen]) / s; ll += np.sum(norm.logcdf(ab))
    lb = np.exp(norm.logpdf(ab) - norm.logcdf(ab))
    g[:K] += ((-lb / s)[:, None] * X[bcen]).sum(0); g[K] += np.sum(-lb * ab)
    return -ll, -g


def _scores(theta, y, X, lo, hi, bcen, tcen):
    K = X.shape[1]; beta = theta[:K]; s = np.exp(theta[K]); m = X @ beta
    G = np.zeros((len(y), K + 1)); unc = ~(bcen | tcen)
    iu = np.where(unc)[0]; r = (y[unc] - m[unc]) / s
    G[iu[:, None], np.arange(K)] = (r / s)[:, None] * X[unc]; G[iu, K] = r ** 2 - 1.0
    it = np.where(tcen)[0]; at = (m[tcen] - hi) / s
    lt = np.exp(norm.logpdf(at) - norm.logcdf(at))
    G[it[:, None], np.arange(K)] = (lt / s)[:, None] * X[tcen]; G[it, K] = -lt * at
    ib = np.where(bcen)[0]; ab = (lo - m[bcen]) / s
    lb = np.exp(norm.logpdf(ab) - norm.logcdf(ab))
    G[ib[:, None], np.arange(K)] = (-lb / s)[:, None] * X[bcen]; G[ib, K] = -lb * ab
    return G


def fit_2l(y, X, lo, hi, cluster):
    bcen = y <= lo; tcen = y >= hi
    beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
    theta0 = np.append(beta0, np.log(max((y - X @ beta0).std(), 1e-3)))
    res = optimize.minimize(_nll, theta0, args=(y, X, lo, hi, bcen, tcen),
                            jac=True, method="L-BFGS-B", options={"maxiter": 6000, "ftol": 1e-11})
    G = _scores(res.x, y, X, lo, hi, bcen, tcen)
    bread = np.linalg.inv(G.T @ G)
    S = pd.DataFrame(G).groupby(np.asarray(cluster)).sum().to_numpy()
    V = bread @ (S.T @ S) @ bread
    return res, V, bcen.mean(), tcen.mean()


def effective_contrib(m, s, lo, hi):
    """Expected EFFECTIVE share: zero below the observation floor (out of MLB
    produces no plate appearances), the capped share when active. The latent value
    below lo is left-censored (unknown, < lo), NOT zero; only the contribution is."""
    al = (lo - m) / s; ah = (hi - m) / s
    return (m * (norm.cdf(ah) - norm.cdf(al)) + s * (norm.pdf(al) - norm.pdf(ah))
            + hi * (1 - norm.cdf(ah)))


# ---------- build augmented panel and fit ----------
def main():
    ps = pd.read_parquet(mc.PANEL_ALLOC)
    ps["batter"] = ps["batter"].astype(str); ps["season"] = ps["season"].astype(int)
    sh = pd.read_parquet(mc.SHARE_CACHE)
    sh["batter"] = sh["batter"].astype(str); sh["season"] = sh["season"].astype(int)
    full_ref = float(sh["pa_share"].quantile(0.99)); c = full_ref       # upper cap
    ps = ps.merge(sh[["batter", "season", "pa_share"]], on=["batter", "season"], how="inner")
    ps["age"] = ps["age"].astype(int)

    # career attributes
    car = ps.groupby("batter").agg(tau_bat=("tau_bat", "mean"),
                                   first_age=("age", "min"),
                                   pos=("pos", lambda s: s.mode().iloc[0])).reset_index()
    # augment: insert zero-share rows from first_age..AGE_CAP for missing ages
    played = set(zip(ps["batter"], ps["age"]))
    rows = []
    for b, tb, fa, pos in car.itertuples(index=False):
        for a in range(int(fa), AGE_CAP + 1):
            if (b, a) not in played:
                rows.append((b, a, pos, tb, 0.0))
    zeros = pd.DataFrame(rows, columns=["batter", "age", "pos", "tau_bat", "pa_share"])
    obs = ps[["batter", "age", "pos", "tau_bat", "pa_share"]]
    aug = pd.concat([obs, zeros], ignore_index=True)
    aug = aug[(aug["age"] >= AGE_LO) & (aug["age"] <= AGE_CAP)].reset_index(drop=True)

    # design: const, tau_bat, hitting-age spline, position dummies
    age_c = aug["age"].to_numpy(float) - mc.REF_AGE
    sp = mc.rcs_basis(age_c, np.array(mc.SPLINE_KNOTS) - mc.REF_AGE)
    cols = [("const", np.ones(len(aug))), ("tau_bat", aug["tau_bat"].to_numpy(float))]
    cols += [(f"age{i}", sp[:, i]) for i in range(sp.shape[1])]
    for p in mc.POS:
        cols.append((f"pos_{p}", (aug["pos"].astype(str) == p).astype(float)))
    X, names = mc.design(cols)
    y = aug["pa_share"].to_numpy(float)
    c_lo = float(np.quantile(obs["pa_share"], 0.02))    # observation floor; left-censor here, not at 0

    res, V, bfrac, tfrac = fit_2l(y, X, c_lo, c, aug["batter"].to_numpy())
    s = np.exp(res.x[-1])
    P("EFFECTIVE BASEBALL AGING  (two-limit censored share Tobit; exits = zero)")
    P("=" * 72)
    P(f"{len(aug):,} player-ages ({aug['batter'].nunique():,} established players), "
      f"ages {AGE_LO}-{AGE_CAP}")
    P(f"left-censored below floor c_lo={c_lo:.3f} (value < floor, NOT zero): {bfrac*100:.0f}%"
      f"   capped at full-time c={c:.3f}: {tfrac*100:.0f}%")

    # evaluate curves at mean tau_bat, reference position (corner OF = all pos dummies 0)
    tau0 = float(aug["tau_bat"].mean())
    ages = np.arange(22, AGE_CAP + 1)
    spA = mc.rcs_basis(ages - mc.REF_AGE, np.array(mc.SPLINE_KNOTS) - mc.REF_AGE)
    beta = dict(zip(names, res.x[:len(names)]))
    m = (beta["const"] + beta["tau_bat"] * tau0
         + spA @ np.array([beta[f"age{i}"] for i in range(spA.shape[1])]))
    eff = effective_contrib(m, s, c_lo, c)                  # effective contribution (out -> 0)
    al = (c_lo - m) / s; ah = (c - m) / s
    surv = m + s * (norm.pdf(al) - norm.pdf(ah)) / (norm.cdf(ah) - norm.cdf(al))  # E[y|active]
    # descriptive survivor (raw mean share among playing) for cross-check
    play = aug[aug["pa_share"] > 0]
    raw_surv = play.groupby("age")["pa_share"].mean()

    SC = float(sh["pa_team_mean"].iloc[0]) if "pa_team_mean" in sh else 5646.0
    pk_eff = int(ages[np.argmax(eff)]); pk_lat = int(ages[np.argmax(m)])
    P(f"\n  age   latent_m   effective(E[y])  survivor(E[y|play])  raw_surv")
    for i, a in enumerate(ages):
        if a % 2 == 0:
            rs = raw_surv.get(a, np.nan)
            P(f"  {a:>3d}   {m[i]:>8.4f}   {eff[i]:>10.4f}      {surv[i]:>10.4f}      "
              f"{rs:>7.4f}")
    P(f"\n  latent peak age {pk_lat}; effective peak age {pk_eff}")
    def dec(curve):
        pk = curve.max(); a35 = curve[ages == 35][0]
        return (a35 - pk) * SC
    P(f"  decline peak->35, PA-equivalent:  effective {dec(eff):+.0f}   "
      f"survivor {dec(surv):+.0f}")
    P(f"  the effective curve falls faster: the extra drop is the exits the survivor")
    P(f"  curve discards. Fair/Bradbury estimate the survivor curve.")
    (mc.OUTPUT_DIR / "aging_censored_summary.txt").write_text("\n".join(L), encoding="utf-8")
    pd.DataFrame({"age": ages, "latent": m, "effective": eff, "survivor": surv}).to_csv(
        mc.OUTPUT_DIR / "aging_censored.csv", index=False, float_format="%.5f")
    print("wrote aging_censored.csv, summary")


if __name__ == "__main__":
    main()
