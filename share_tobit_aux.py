"""
share_tobit_aux.py
==================

The three auxiliary tests, migrated to the SAME estimator as the headline fits:
a right-censored (Tobit) model on the player's SHARE of his team's plate
appearances, censored at 90% of a full-time workload, SEs clustered by batter.
Previously these ran on the PA count with OLS; this puts the whole paper on one
estimator. The identifying structure of each test is unchanged -- season fixed
effects absorb each shock's main effect and the shock enters only as a position
interaction -- so only the dependent variable and the estimator change. The
within-season VIF checks are a function of the design matrix alone, hence
estimator-independent; they are reported unchanged.

  DH null      pa_share ~ tau_bat + C(pos) + age + dh_regime:pos + i_exp:pos
                          + i_integ:pos | season        (dh:pos is the test)
  Contact      M1 + BIP_z:pos ;  M2 adds season-trend:pos (horse race) ;
               M3 per-decade position premia vs decade balls-in-play rate
  Expansion    + exp_intensity:pos at decay = 3 and 5 years

Interactions run over the six glove positions C/SS/2B/3B/CF/1B (reference =
corner OF; DH excluded from the shock interactions as in the originals). Premia
and shock coefficients are reported in PA-equivalent units (share x mean team PA)
so they line up with the prior count-scale numbers. Builds its own augmented
share+league cache from plays.csv (one read, then cached). Requires scipy.
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
TEAMS_CSV    = PROJECT_DIR / "Teams.csv"
PANEL        = PROJECT_DIR / "data"   / "pa_allocation_panel.parquet"
AUG_CACHE    = PROJECT_DIR / "data"   / "pa_team_share_aug.parquet"
OUT          = PROJECT_DIR / "output" / "share_tobit_aux_summary.txt"

REF_POS = "CORNER_OF"
REF_AGE = 27
GLOVE = ["C", "SS", "2B", "3B", "CF", "1B"]      # shock-interaction positions
ALLPOS = ["C", "SS", "2B", "3B", "CF", "1B", "DH"]
EXP_YEARS = [1961, 1962, 1969, 1977, 1993, 1998]
HEADLINE_FRAC = 0.90

L = []
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def P(s=""): print(s); L.append(s)

def integ_phasein(s):     return 1.0 / (1.0 + np.exp(-(s - 1954) / 3.0))
def exp_intensity(s, decay):
    s = np.asarray(s, float); out = np.zeros_like(s)
    for y in EXP_YEARS:
        d = s - y; out += np.where(d >= 0, np.exp(-d / decay), 0.0)
    return out


# ---- censored-normal NLL + gradient (identical to the headline script) ----
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

def per_obs_scores(theta, y, X, c, cen):
    K = X.shape[1]; beta = theta[:K]; s = np.exp(theta[K]); m = X @ beta
    G = np.zeros((len(y), K + 1)); unc = ~cen
    r = (y[unc] - m[unc]) / s; iu = np.where(unc)[0]
    G[iu[:, None], np.arange(K)] = (r / s)[:, None] * X[unc]; G[iu, K] = r ** 2 - 1.0
    a = (m[cen] - c) / s; lam = np.exp(norm.logpdf(a) - norm.logcdf(a)); ic = np.where(cen)[0]
    G[ic[:, None], np.arange(K)] = (lam / s)[:, None] * X[cen]; G[ic, K] = -lam * a
    return G

def fit_tobit(y, X, c, cluster):
    cen = y >= c
    beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
    theta0 = np.append(beta0, np.log((y - X @ beta0).std()))
    res = optimize.minimize(neg_ll_and_grad, theta0, args=(y, X, c, cen), jac=True,
                            method="L-BFGS-B", options={"maxiter": 4000, "ftol": 1e-11})
    G = per_obs_scores(res.x, y, X, c, cen); bread = np.linalg.inv(G.T @ G)
    S = pd.DataFrame(G).groupby(cluster).sum().to_numpy(); V = bread @ (S.T @ S) @ bread
    return res, V, cen.mean()

def lincom(weights, names, theta, V):
    K = len(names); Vb = V[:K, :K]; c = np.zeros(K); est = 0.0
    for t, w in weights.items():
        if t not in names: return None, None
        i = names.index(t); c[i] = w; est += w * theta[i]
    return est, float(np.sqrt(c @ Vb @ c))

def design(cols): return np.column_stack([np.asarray(v, float) for _, v in cols]), [n for n, _ in cols]

def within_season_vif(season, cols):
    """VIF of each column after removing season means (matches the OLS-era check)."""
    M = np.column_stack([np.asarray(v, float) for _, v in cols])
    df = pd.DataFrame(M, columns=[n for n, _ in cols]); df["__s"] = np.asarray(season)
    Z = df[[n for n, _ in cols]] - df.groupby("__s")[[n for n, _ in cols]].transform("mean")
    Z = Z.to_numpy(); out = {}
    for j, (n, _) in enumerate(cols):
        others = np.delete(Z, j, axis=1)
        A = np.column_stack([np.ones(len(Z)), others])
        b, *_ = np.linalg.lstsq(A, Z[:, j], rcond=None)
        ss_res = ((Z[:, j] - A @ b) ** 2).sum(); ss_tot = ((Z[:, j] - Z[:, j].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out[n] = np.inf if r2 >= 1 - 1e-12 else 1 / (1 - r2)
    return out


def build_aug_cache():
    if AUG_CACHE.exists():
        log(f"loading augmented cache {AUG_CACHE.name} (delete to recompute)")
        return pd.read_parquet(AUG_CACHE)
    log("reading plays.csv for share + league")
    pl = pd.read_csv(PLAYS_CSV, usecols=lambda x: x in ["gid", "batter", "batteam", "pa", "iw"],
                     dtype={"gid": "string", "batter": "string", "batteam": "string",
                            "pa": "Int8", "iw": "Int8"}, low_memory=False)
    pl = pl[pl["pa"] == 1]
    gi = pd.read_csv(GAMEINFO_CSV, usecols=["gid", "season", "gametype"],
                     dtype={"gid": "string", "season": "Int32", "gametype": "string"})
    gi = gi[gi["gametype"] == "regular"]
    pl = pl.merge(gi[["gid", "season"]], on="gid", how="inner")
    pl = pl[pl["iw"] != 1]; pl["season"] = pl["season"].astype(int)
    pst = pl.groupby(["batter", "season", "batteam"]).size().reset_index(name="n")
    team_tot = pl.groupby(["season", "batteam"]).size().reset_index(name="team_pa")
    pst = pst.merge(team_tot, on=["season", "batteam"])
    agg = pst.groupby(["batter", "season"]).agg(denom=("team_pa", "sum"), pa_chk=("n", "sum")).reset_index()
    agg["pa_share"] = agg["pa_chk"] / agg["denom"]; agg["pa_team_mean"] = float(team_tot["team_pa"].mean())
    # modal team -> league
    idx = pst.groupby(["batter", "season"])["n"].idxmax()
    modal = pst.loc[idx, ["batter", "season", "batteam"]]
    t = pd.read_csv(TEAMS_CSV, usecols=["yearID", "lgID", "teamIDretro"]).dropna(subset=["teamIDretro"])
    t = t.rename(columns={"yearID": "season", "teamIDretro": "batteam", "lgID": "league"})
    modal = modal.merge(t, on=["season", "batteam"], how="left")
    agg = agg.merge(modal[["batter", "season", "league"]], on=["batter", "season"], how="left")
    out = agg[["batter", "season", "pa_share", "pa_chk", "pa_team_mean", "league"]]
    out.to_parquet(AUG_CACHE, index=False); log(f"  cached {AUG_CACHE.name}")
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    share = build_aug_cache()
    PA_TEAM_MEAN = float(share["pa_team_mean"].iloc[0]); SC = PA_TEAM_MEAN

    ps = pd.read_parquet(PANEL)
    ps["batter"] = ps["batter"].astype(str); ps["season"] = ps["season"].astype(int)
    ps = ps.merge(share, on=["batter", "season"], how="inner").reset_index(drop=True)
    ps = ps[ps["pos"].astype(str).isin([REF_POS] + ALLPOS)].copy().reset_index(drop=True)
    posv = ps["pos"].astype(str).to_numpy()
    age_c = ps["age_c"].to_numpy(float) if "age_c" in ps else (ps["age_exact"].to_numpy(float) - REF_AGE)
    seasons = sorted(ps["season"].unique())
    y = ps["pa_share"].to_numpy(float); cl = ps["batter"].to_numpy(); seas = ps["season"].to_numpy()
    full_ref = float(ps["pa_share"].quantile(0.99)); c_head = HEADLINE_FRAC * full_ref

    base = [("const", np.ones(len(ps))), ("tau_bat", ps["tau_bat"].to_numpy(float)),
            ("age_c", age_c), ("age_c2", age_c ** 2)]
    pos_cols = [(f"pos_{p}", (posv == p).astype(float)) for p in ALLPOS]
    yr_cols = [(f"yr_{s}", (ps["season"] == s).to_numpy().astype(float)) for s in seasons[1:]]

    P("SHARE-TOBIT AUXILIARY TESTS  (one estimator: censored-share, c=0.90, batter-clustered SE)")
    P("=" * 76)
    P(f"matched {len(ps):,} player-seasons; mean team PA {PA_TEAM_MEAN:,.0f}; "
      f"headline c = {c_head:.4f}")

    # ================= DH NULL =================
    lg = ps["league"].to_numpy()
    dh_regime = (((lg == "AL") & (seas >= 1973) & (seas <= 2021)) | (seas >= 2022)).astype(float)
    i_exp3 = exp_intensity(seas, 3.0); i_integ = integ_phasein(seas)
    dh_int = [(f"x_dh_{p}", dh_regime * (posv == p)) for p in GLOVE]
    exp_int = [(f"x_exp_{p}", i_exp3 * (posv == p)) for p in GLOVE]
    int_int = [(f"x_integ_{p}", i_integ * (posv == p)) for p in GLOVE]
    Xd, nd = design(base + pos_cols + dh_int + exp_int + int_int + yr_cols)
    resd, Vd, cfd = fit_tobit(y, Xd, c_head, cl)
    vif_dh = within_season_vif(seas, dh_int + exp_int + int_int)
    P(f"\n--- DH null: dh_regime:pos, PA-equivalent  [{cfd*100:.1f}% cens] ---")
    P("  (prior count/OLS: C +2.0, SS -12.8, 2B -10.4, 3B -12.2, CF +3.8, 1B +15.2; VIF~1.8)")
    P(f"  {'pos':>4s} {'dh:pos':>8s} {'SE':>5s} {'t':>6s} {'VIF':>6s}")
    for p in GLOVE:
        e, se = lincom({f"x_dh_{p}": 1.0}, nd, resd.x, Vd)
        P(f"  {p:>4s} {e*SC:>8.1f} {se*SC:>5.1f} {e/se:>6.2f} {vif_dh[f'x_dh_{p}']:>6.1f}")
    for stub in ["dh", "exp", "integ"]:
        vs = [vif_dh[f"x_{stub}_{p}"] for p in GLOVE]
        P(f"  {stub:>6s}:pos VIF {min(vs):.1f}-{max(vs):.1f}")

    # ================= CONTACT =================
    t = pd.read_csv(TEAMS_CSV, usecols=["yearID", "AB", "SO", "HR", "BB", "HBP", "SF"])
    g = t.groupby("yearID").sum(numeric_only=True)
    g["PA"] = g["AB"] + g["BB"] + g["HBP"] + g["SF"]
    g["BIP_rate"] = (g["AB"] - g["SO"] - g["HR"]) / g["PA"]
    env = g["BIP_rate"].reset_index().rename(columns={"yearID": "season"})
    ps2 = ps.merge(env, on="season", how="left")
    bip = ps2["BIP_rate"].to_numpy(float)
    bip_z = (bip - np.nanmean(bip)) / np.nanstd(bip)
    season_c = (seas.astype(float) - 1970.0)
    bip_int = [(f"bip_{p}", bip_z * (posv == p)) for p in GLOVE]
    trend_int = [(f"trend_{p}", (season_c / 10.0) * (posv == p)) for p in GLOVE]
    # M1
    Xc, nc = design(base + pos_cols + bip_int + yr_cols)
    resc, Vc, cfc = fit_tobit(y, Xc, c_head, cl)
    vif_c = within_season_vif(seas, bip_int)
    P(f"\n--- Contact M1: BIP_z:pos, PA-equivalent per +1 SD balls-in-play  [{cfc*100:.1f}% cens] ---")
    P("  (prior count/OLS M1: 2B +18 t3.4, 3B +16 t3.2)")
    P(f"  {'pos':>4s} {'bip:pos':>8s} {'SE':>5s} {'t':>6s} {'VIF':>6s}")
    for p in GLOVE:
        e, se = lincom({f"bip_{p}": 1.0}, nc, resc.x, Vc)
        P(f"  {p:>4s} {e*SC:>8.1f} {se*SC:>5.1f} {e/se:>6.2f} {vif_c[f'bip_{p}']:>6.1f}")
    # M2 horse race
    Xc2, nc2 = design(base + pos_cols + bip_int + trend_int + yr_cols)
    resc2, Vc2, _ = fit_tobit(y, Xc2, c_head, cl)
    P("--- Contact M2: BIP_z:pos AFTER trend:pos (does contact beat generic time?) ---")
    P(f"  {'pos':>4s} {'bip:pos':>8s} {'SE':>5s} {'t':>6s}")
    for p in GLOVE:
        e, se = lincom({f"bip_{p}": 1.0}, nc2, resc2.x, Vc2)
        P(f"  {p:>4s} {e*SC:>8.1f} {se*SC:>5.1f} {e/se:>6.2f}")
    # M3 per-decade premia vs decade BIP
    dec = (seas // 10 * 10).astype(int)
    decs = sorted(np.unique(dec))
    dec_main = [(f"dec_{d}", (dec == d).astype(float)) for d in decs[1:]]
    cell, have = [], {}
    for p in GLOVE:
        for d in decs:
            col = ((posv == p) & (dec == d)).astype(float)
            if col.sum() > 0: cell.append((f"cd_{p}_{d}", col)); have[(p, d)] = True
    Xm3, nm3 = design(base + dec_main + cell)
    resm3, Vm3, _ = fit_tobit(y, Xm3, c_head, cl)
    decbip = env.set_index("season")["BIP_rate"]
    P("--- Contact M3: per-decade position premium (PA-equiv) vs decade BIP_rate ---")
    P("  decade  BIP   " + " ".join(f"{p:>6s}" for p in GLOVE))
    prem = {p: [] for p in GLOVE}; bipv = []
    for d in decs:
        b = float(decbip.loc[decbip.index // 10 * 10 == d].mean()); bipv.append(b)
        cells = []
        for p in GLOVE:
            if (p, d) in have:
                e, _ = lincom({f"cd_{p}_{d}": 1.0}, nm3, resm3.x, Vm3)
                prem[p].append(e * SC); cells.append(f"{e*SC:>6.0f}")
            else:
                prem[p].append(np.nan); cells.append(f"{'--':>6s}")
        P(f"  {d:>5d}  {b:.2f}  " + " ".join(cells))
    P("  corr(premium, BIP_rate) across decades:")
    for p in GLOVE:
        a = np.array(prem[p]); bb = np.array(bipv); ok = ~np.isnan(a)
        r = np.corrcoef(a[ok], bb[ok])[0, 1] if ok.sum() > 2 else np.nan
        P(f"    {p:>4s}: {r:+.2f}")

    # ================= EXPANSION =================
    P("\n--- Expansion: exp_intensity:pos, PA-equivalent  ---")
    P("  (prior count/OLS, decay 3/5: SS +21/+20, 2B +20/+14, 3B +22/+14, CF +29/+26, 1B -6/-9, C +4/+5)")
    for decay in [3.0, 5.0]:
        iexp = exp_intensity(seas, decay)
        e_int = [(f"e_{p}", iexp * (posv == p)) for p in GLOVE]
        Xe, ne = design(base + pos_cols + e_int + yr_cols)
        rese, Ve, cfe = fit_tobit(y, Xe, c_head, cl)
        vif_e = within_season_vif(seas, e_int)
        P(f"  decay={decay:.0f}yr  [{cfe*100:.1f}% cens]   {'pos':>4s} {'exp:pos':>8s} {'SE':>5s} {'t':>6s} {'VIF':>6s}")
        for p in GLOVE:
            e, se = lincom({f"e_{p}": 1.0}, ne, rese.x, Ve)
            P(f"               {p:>4s} {e*SC:>8.1f} {se*SC:>5.1f} {e/se:>6.2f} {vif_e[f'e_{p}']:>6.1f}")

    P("\nAll three auxiliary tests now on the censored-share Tobit, c=0.90, batter-clustered.")
    P("VIFs are within-season (design-only, hence identical to the OLS-era checks).")
    OUT.write_text("\n".join(L), encoding="utf-8")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
