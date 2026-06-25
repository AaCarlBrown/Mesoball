"""
platoon_allocation_tobit.py
===========================

Step 2 of the pitcher-handedness analysis. Does a batter's playing time AGAINST A
HAND respond to his skill against THAT hand, holding overall skill fixed -- and is
that platoon-driven allocation fading as starters throw fewer innings?

For each hand H we fit the same censored-share Tobit used for the positional
results, on the batter's share of his team's PA against hand H:

    share_H* = a + g*overall_i + b_H*ownadv_H,i + (position) + (age) + (decade) + e
    ownadv_H = tau_H - tau_other      (own-hand advantage; R: tau_R - tau_L)
    overall  = (tau_R + tau_L)/2

b_H is the platoon coefficient: extra share against hand H per unit of being-
better-against-hand-H, with overall quality partialled out. If teams allocate on
handedness b_H > 0; if they are handedness-blind only overall matters and b_H = 0.
The "ownadv vs overall" split (rather than tau_R and tau_L side by side) is the
near-orthogonal reparameterization, so b_H is identified off the platoon tilt, not
off the shared ability the two skills have in common.

The prediction. A lineup is set against a KNOWN starter, so handedness is
plannable only for the share of PA the starter throws. As starters yield to the
bullpen earlier, less of the game is plannable, so b_H should fade. We estimate
b_H by decade (nested cells, SE read directly) and relate it to a bullpenning
index -- pitchers used per team-game -- rather than to bare calendar time, and we
do NOT impose monotonicity: platooning may have risen with analytics before the
bullpen era pulled it back.

Units. Everything is reported as PERCENT OF A FULL-TIME WORKLOAD (share / the
hand's full-time reference), because the vs-RHP and vs-LHP PA pools differ in size
and only this unit is comparable across the two splits.

Reads pa_allocation_panel.parquet, tau_by_hand.parquet; reads plays.csv once
(caches the hand shares and the bullpen index). Requires scipy.
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
PANEL        = PROJECT_DIR / "data" / "pa_allocation_panel.parquet"
TAU_HAND     = PROJECT_DIR / "data" / "tau_by_hand.parquet"
SHARE_CACHE  = PROJECT_DIR / "data" / "pa_team_share_byhand.parquet"
BULLPEN_CACHE = PROJECT_DIR / "data" / "bullpen_index.csv"
OUT          = PROJECT_DIR / "output" / "platoon_allocation_tobit_summary.txt"

REF_POS = "CORNER_OF"
REF_AGE = 27
POS = ["C", "SS", "2B", "3B", "CF", "1B", "DH"]
HEADLINE_FRAC = 0.90
HANDS = ["R", "L"]
L = []
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def P(s=""): print(s); L.append(s)


# ---- Tobit core (shared with share_tobit_*.py; to be deduped at cleanup) ----
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
    r = (y[unc] - m[unc]) / s; iu = np.where(unc)[0]
    G[iu[:, None], np.arange(K)] = (r / s)[:, None] * X[unc]; G[iu, K] = r ** 2 - 1.0
    a = (m[cen] - c) / s
    lam = np.exp(norm.logpdf(a) - norm.logcdf(a)); ic = np.where(cen)[0]
    G[ic[:, None], np.arange(K)] = (lam / s)[:, None] * X[cen]; G[ic, K] = -lam * a
    return G

def fit_tobit(y, X, c, cluster):
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
    K = len(names); Vb = V[:K, :K]; cc = np.zeros(K); est = 0.0
    for t, w in weights.items():
        if t not in names: return None, None
        i = names.index(t); cc[i] = w; est += w * theta[i]
    return est, float(np.sqrt(cc @ Vb @ cc))

def design(cols):
    return np.column_stack([np.asarray(v, float) for _, v in cols]), [n for n, _ in cols]


def build_shares_and_bullpen():
    """Hand-specific team-PA shares + a per-season bullpen index, from one read."""
    if SHARE_CACHE.exists() and BULLPEN_CACHE.exists():
        log("loading cached hand shares + bullpen index (delete to recompute)")
        return pd.read_parquet(SHARE_CACHE), pd.read_csv(BULLPEN_CACHE)
    log("reading plays.csv (hand shares + bullpen index)")
    pl = pd.read_csv(PLAYS_CSV,
        usecols=lambda x: x in ["gid", "batter", "pitcher", "batteam", "pitteam",
                                "pithand", "pa", "iw"],
        dtype={"gid": "string", "batter": "string", "pitcher": "string",
               "batteam": "string", "pitteam": "string", "pithand": "string",
               "pa": "Int8", "iw": "Int8"}, low_memory=False)
    pl = pl[pl["pa"] == 1]
    gi = pd.read_csv(GAMEINFO_CSV, usecols=["gid", "season", "gametype"],
                     dtype={"gid": "string", "season": "Int32", "gametype": "string"})
    gi = gi[gi["gametype"] == "regular"]
    pl = pl.merge(gi[["gid", "season"]], on="gid", how="inner")
    pl["season"] = pl["season"].astype(int)
    pl["pithand"] = pl["pithand"].astype(str).str.upper().str[0]

    # bullpen index: distinct pitchers per team-game, season mean (before dropping cols)
    ppg = (pl.groupby(["gid", "pitteam"])["pitcher"].nunique().reset_index(name="npit")
             .merge(gi[["gid", "season"]], on="gid").astype({"season": int}))
    bull = ppg.groupby("season")["npit"].mean().reset_index(name="pitchers_per_game")

    # hand shares: ex-IBB, by pitcher hand
    pl = pl[(pl["iw"] != 1) & pl["pithand"].isin(HANDS)]
    pst = pl.groupby(["batter", "season", "batteam", "pithand"]).size().reset_index(name="n")
    team = pl.groupby(["season", "batteam", "pithand"]).size().reset_index(name="team_pa")
    pst = pst.merge(team, on=["season", "batteam", "pithand"])
    agg = (pst.groupby(["batter", "season", "pithand"])
              .agg(num=("n", "sum"), denom=("team_pa", "sum")).reset_index())
    agg["share"] = agg["num"] / agg["denom"]
    share = agg[["batter", "season", "pithand", "share", "num"]]
    share.to_parquet(SHARE_CACHE, index=False)
    bull.to_csv(BULLPEN_CACHE, index=False)
    log(f"  cached {SHARE_CACHE.name} and {BULLPEN_CACHE.name}")
    return share, bull


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    share, bull = build_shares_and_bullpen()

    ps = pd.read_parquet(PANEL, columns=["batter", "season", "pos", "age_c", "era"])
    ps["batter"] = ps["batter"].astype(str); ps["season"] = ps["season"].astype(int)
    tau = pd.read_parquet(TAU_HAND)[["batter", "tau_R_eb", "tau_L_eb",
                                     "overall", "split", "N_R", "N_L"]]
    tau["batter"] = tau["batter"].astype(str)
    tau = tau.dropna(subset=["overall", "split"])

    P("PLATOON ALLOCATION  (censored-share Tobit per pitcher hand, batter-clustered)")
    P("=" * 74)
    P(f"both-hand batters with skills: {len(tau):,}")
    P("b_H = extra share vs hand H per unit own-hand advantage, overall skill held")
    P("fixed. Reported as % of a full-time workload per +1 SD of own-hand advantage.")

    decades = list(range(1910, 2030, 10))
    bull["decade"] = (bull["season"] // 10 * 10)
    bull_dec = bull.groupby("decade")["pitchers_per_game"].mean()

    b_by_decade = {}     # hand -> {decade: (pct, se)}
    for h in HANDS:
        sub = share[share["pithand"] == h].merge(ps, on=["batter", "season"], how="inner")
        sub = sub.merge(tau, on="batter", how="inner").reset_index(drop=True)
        sub["ownadv"] = sub["split"] if h == "R" else -sub["split"]
        full_ref = float(sub["share"].quantile(0.99))
        c = HEADLINE_FRAC * full_ref
        sd_own = float(sub["ownadv"].std()); sd_all = float(sub["overall"].std())
        to_pct = 100.0 / full_ref
        sub["decade"] = (sub["season"] // 10 * 10).astype(int)
        decs = [d for d in decades if (sub["decade"] == d).sum() >= 200]

        posv = sub["pos"].astype(str).to_numpy()
        age_c = sub["age_c"].to_numpy(float)
        y = sub["share"].to_numpy(float); cl = sub["batter"].to_numpy()
        dec_main = [(f"dec_{d}", (sub["decade"] == d).to_numpy(float)) for d in decs[1:]]
        pos_cols = [(f"pos_{p}", (posv == p).astype(float)) for p in POS]
        base = [("const", np.ones(len(sub))), ("overall", sub["overall"].to_numpy(float))]
        ctrl = [("age_c", age_c), ("age_c2", age_c ** 2)]

        # pooled b_H
        Xp, npn = design(base + [("ownadv", sub["ownadv"].to_numpy(float))]
                         + dec_main + pos_cols + ctrl)
        resp, Vp, cf = fit_tobit(y, Xp, c, cl)
        bo, seo = lincom({"ownadv": 1.0}, npn, resp.x, Vp)
        go, gse = lincom({"overall": 1.0}, npn, resp.x, Vp)
        P(f"\n--- hand {h}: {len(sub):,} batter-seasons, {cf*100:.1f}% censored; "
          f"full-time ref share={full_ref:.4f} ---")
        P(f"  SD(own-hand adv)={sd_own:.4f}  SD(overall)={sd_all:.4f} wOBA")
        P(f"  overall skill : +1 SD -> {go*sd_all*to_pct:+.1f}% of full-time "
          f"(t={go/gse:.1f})")
        P(f"  POOLED b_{h}   : +1 SD own-hand adv -> {bo*sd_own*to_pct:+.1f}% of "
          f"full-time (t={bo/seo:.1f})   [platoon allocation signal]")

        # b_H by decade (nested cells: one ownadv coef per decade)
        own_dec = [(f"own_{d}", (sub["ownadv"].to_numpy(float)) * (sub["decade"] == d).to_numpy(float))
                   for d in decs]
        Xd, nnd = design(base + own_dec + dec_main + pos_cols + ctrl)
        resd, Vd, _ = fit_tobit(y, Xd, c, cl)
        P(f"  b_{h} by decade (% of full-time per +1 SD own-hand adv):")
        P(f"  {'decade':>7s} {'pitchers/g':>11s} {'b (%full)':>10s} {'t':>6s}")
        b_by_decade[h] = {}
        for d in decs:
            e, se = lincom({f"own_{d}": 1.0}, nnd, resd.x, Vd)
            pct = e * sd_own * to_pct
            b_by_decade[h][d] = (pct, se * sd_own * to_pct)
            ppgv = bull_dec.get(d, np.nan)
            P(f"  {d:>7d} {ppgv:>11.2f} {pct:>10.1f} {e/se:>6.1f}")

    # link platoon signal to bullpenning: corr across decades, pooled over hands
    P("\n--- platoon signal vs bullpenning ---")
    rows = []
    for h in HANDS:
        for d, (pct, _) in b_by_decade[h].items():
            rows.append({"hand": h, "decade": d, "b_pct": pct,
                         "pitchers_per_game": bull_dec.get(d, np.nan)})
    R = pd.DataFrame(rows).dropna()
    if len(R) >= 4:
        rr = R["b_pct"].corr(R["pitchers_per_game"])
        P(f"  corr(b, pitchers-per-game) across decade-hand cells: {rr:+.3f}")
        P("  (prediction: NEGATIVE -- platoon allocation fades as bullpen use rises)")
        bh = R.groupby("decade")["b_pct"].mean()
        P(f"  b averaged over hands, early vs late: "
          f"{bh.iloc[0]:+.1f}% ({int(bh.index[0])}s) -> {bh.iloc[-1]:+.1f}% ({int(bh.index[-1])}s)")
    P("\nIf b is significantly > 0, teams demonstrably allocate playing time on")
    P("pitcher handedness -> it is neither trivial nor random and earns a paragraph")
    P("(or a control). If b is small / insignificant or attenuates with bullpenning,")
    P("that justifies holding pitcher handedness OUT of the structural model.")
    OUT.write_text("\n".join(L), encoding="utf-8")
    log(f"wrote {OUT}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
