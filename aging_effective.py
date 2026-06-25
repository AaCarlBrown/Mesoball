"""
aging_effective.py  (exploratory)
=================================

Selection-corrected PA-share aging curve.

The within-player curve fit on ACTIVE seasons only is survivor-biased: the
age-a coefficient is identified from players who were still regulars at a, who
are the ones whose ability held up, so the late-age decline reads too shallow.

Fix (per the design we settled on):

  latent share index   y*_{ia} = phi_i + g(age) + eps,   eps ~ N(0, s^2)

  * phi_i  = player career level, estimated on ACTIVE seasons by the within
             (fixed-effect) estimator and PLUGGED IN as an offset -- so the
             censored refit of g(.) avoids the incidental-parameters problem of
             a Tobit with thousands of player dummies.
  * active season (share>0): uncensored obs of y*, right-censored only at the
             full-time ceiling c.
  * PERFORMANCE exit: ONE left-censored row at the first gone-age (latent <= tau,
             the roster bar). Just one row, not a string of zeros to age 40 --
             that filler is what forced the spurious young peak in the earlier
             single-equation attempt.
  * INJURY exit: dropped from the sample (not censored).

The "a player drops out a little below last year" intuition is delivered by
phi_i sitting in the index: a high-phi star crosses tau only after a large
(g+eps) decline; a marginal regular crosses it after a small one. We do not need
a moving, player-specific threshold.

Injury vs performance for TERMINAL exits (we cannot classify them directly):
weight the censored row by its performance-probability  1 - r(age), where r(age)
is the observed RETURN RATE among players who gapped at that age (active at a,
absent at a+1, active again later). r is ~1 for the young (their gaps are
injuries and they come back) and ~0 for the old. Internal gaps (active on both
sides) are injuries by construction and contribute no row at all.
Constant-pi (e.g. 5%, 10%) is available as a robustness anchor, but it cannot
separate young from old, so r(age) is the headline.

Run:
  py aging_effective.py --synthetic     # self-test: recover a planted curve
  py aging_effective.py                 # real data
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

# ---- real-data config (only used in the non-synthetic path) --------------
try:
    import meso_core as mc
    SHARE_CACHE = mc.SHARE_CACHE          # pa_team_share.parquet: batter, season, pa_share
    ALLOC_PANEL = mc.PANEL_ALLOC          # pa_allocation_panel.parquet: batter, season, age, ...
    OUTPUT_DIR = mc.OUTPUT_DIR
    HAVE_CORE = True
except Exception:
    HAVE_CORE = False
    from pathlib import Path
    OUTPUT_DIR = Path(".")

MIN_REG_PA = 300            # "regular" bar: a season counts as active if PA >= this
AGE_LO, AGE_HI = 21, 40
REF_AGE = 27
KNOTS = np.array([23, 27, 31, 35], float)   # restricted-cubic-spline knots (age)
LOW_PCTILE = 0.05          # tau = this percentile of active shares (robustness lever)
CONST_PI = [0.05, 0.10]    # constant injury-share robustness anchors

L: list[str] = []
def P(s: str = "") -> None:
    print(s); L.append(s)


# ----------------------------- spline ------------------------------------
def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline basis (Harrell), k-1 columns for k knots; first
    column is linear. Mirrors the project's rcs_basis."""
    k = len(knots); x = np.asarray(x, float)
    kn = knots; out = [x]
    last, prev = kn[-1], kn[-2]
    for j in range(k - 2):
        t = kn[j]
        def cube(z, c):
            d = z - c
            return np.where(d > 0, d ** 3, 0.0)
        term = (cube(x, t)
                - cube(x, prev) * (last - t) / (last - prev)
                + cube(x, last) * (prev - t) / (last - prev))
        out.append(term / (last - kn[0]) ** 2)
    return np.column_stack(out)


def design_age(ages: np.ndarray) -> np.ndarray:
    """[const, rcs(age-REF_AGE)]; anchored so g is identified up to the constant."""
    sp = rcs_basis(ages.astype(float) - REF_AGE, KNOTS - REF_AGE)
    return np.column_stack([np.ones(len(ages)), sp])


# ------------------- two-limit weighted Tobit (offset) -------------------
def _nll(theta, z, X, lo, hi, w, ucen, bcen, tcen):
    """z = y - offset (offset = phi_i already removed). lo, hi are per-row
    thresholds on z (lo = tau - phi_i, hi = c - phi_i). w = row weights."""
    K = X.shape[1]; beta = theta[:K]; s = np.exp(theta[K]); m = X @ beta
    g = np.zeros(K + 1); ll = 0.0
    # uncensored
    r = (z[ucen] - m[ucen]) / s
    ll += np.sum(w[ucen] * (-theta[K] - 0.5 * np.log(2 * np.pi) - 0.5 * r ** 2))
    g[:K] += ((w[ucen] * r / s)[:, None] * X[ucen]).sum(0)
    g[K] += np.sum(w[ucen] * (r ** 2 - 1.0))
    # upper-censored (z >= hi)
    at = (m[tcen] - hi[tcen]) / s
    ll += np.sum(w[tcen] * norm.logcdf(at))
    lt = np.exp(norm.logpdf(at) - norm.logcdf(at))
    g[:K] += ((w[tcen] * lt / s)[:, None] * X[tcen]).sum(0)
    g[K] += np.sum(w[tcen] * (-lt * at))
    # lower-censored (z <= lo)  -- the performance-exit rows
    ab = (lo[bcen] - m[bcen]) / s
    ll += np.sum(w[bcen] * norm.logcdf(ab))
    lb = np.exp(norm.logpdf(ab) - norm.logcdf(ab))
    g[:K] += ((-w[bcen] * lb / s)[:, None] * X[bcen]).sum(0)
    g[K] += np.sum(w[bcen] * (-lb * ab))
    return -ll, -g


def _scores(theta, z, X, lo, hi, w, ucen, bcen, tcen):
    K = X.shape[1]; beta = theta[:K]; s = np.exp(theta[K]); m = X @ beta
    G = np.zeros((len(z), K + 1))
    iu = np.where(ucen)[0]; r = (z[iu] - m[iu]) / s
    G[iu[:, None], np.arange(K)] = (w[iu] * r / s)[:, None] * X[iu]
    G[iu, K] = w[iu] * (r ** 2 - 1.0)
    it = np.where(tcen)[0]; at = (m[it] - hi[it]) / s
    lt = np.exp(norm.logpdf(at) - norm.logcdf(at))
    G[it[:, None], np.arange(K)] = (w[it] * lt / s)[:, None] * X[it]
    G[it, K] = w[it] * (-lt * at)
    ib = np.where(bcen)[0]; ab = (lo[ib] - m[ib]) / s
    lb = np.exp(norm.logpdf(ab) - norm.logcdf(ab))
    G[ib[:, None], np.arange(K)] = (-w[ib] * lb / s)[:, None] * X[ib]
    G[ib, K] = w[ib] * (-lb * ab)
    return G


def fit_tobit(z, X, lo, hi, w, bcen, tcen, cluster):
    ucen = ~(bcen | tcen)
    beta0, *_ = np.linalg.lstsq(X[ucen], z[ucen], rcond=None)
    theta0 = np.append(beta0, np.log(max((z[ucen] - X[ucen] @ beta0).std(), 1e-3)))
    res = optimize.minimize(_nll, theta0, args=(z, X, lo, hi, w, ucen, bcen, tcen),
                            jac=True, method="L-BFGS-B",
                            options={"maxiter": 8000, "ftol": 1e-12})
    G = _scores(res.x, z, X, lo, hi, w, ucen, bcen, tcen)
    bread = np.linalg.inv(G.T @ G)
    S = pd.DataFrame(G).groupby(np.asarray(cluster)).sum().to_numpy()
    V = bread @ (S.T @ S) @ bread
    return res.x, V


# ------------------- curve evaluation helpers ----------------------------
def curve_from_beta(beta_age) -> pd.Series:
    ages = np.arange(22, AGE_HI + 1)
    g = design_age(ages) @ beta_age
    s = pd.Series(g, index=ages)
    return (s - s.max()) * 1000.0    # vs-peak, x1000 (wOBA-points-like display)


def naive_within_curve(act: pd.DataFrame) -> pd.Series:
    """Within-player FE curve on active seasons only (the survivor-biased one)."""
    X = design_age(act["age"].to_numpy())
    # within transform by player
    y = act["share"].to_numpy(float)
    pid = act["pid"].to_numpy()
    dfm = pd.DataFrame(X); dfm["pid"] = pid; dfm["y"] = y
    Xd = dfm.groupby("pid").transform(lambda c: c - c.mean())
    Xw = Xd.drop(columns=["y"]).to_numpy()[:, 1:]   # drop const (killed by demeaning)
    yw = Xd["y"].to_numpy()
    bw, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    beta_age = np.concatenate([[0.0], bw])          # const irrelevant for vs-peak
    return curve_from_beta(beta_age), beta_age


def player_offsets(act: pd.DataFrame, beta_age_within) -> pd.Series:
    """phi_i = mean_i(share) - mean_i(age-design) . beta_age  (FE level)."""
    X = design_age(act["age"].to_numpy())
    pred_shape = X[:, 1:] @ beta_age_within[1:]
    tmp = pd.DataFrame({"pid": act["pid"].to_numpy(),
                        "share": act["share"].to_numpy(float),
                        "shape": pred_shape})
    gm = tmp.groupby("pid").agg(ms=("share", "mean"), msh=("shape", "mean"))
    return (gm["ms"] - gm["msh"])      # phi_i indexed by pid


# ------------------------- build career rows -----------------------------
def build_rows(act: pd.DataFrame):
    """From active (pid, age, share) rows, build the estimation table:
    active rows + one performance-exit censored row per terminal exit, weighted
    by 1 - r(age). Internal gaps contribute nothing. Returns (rows_df, r_by_age)."""
    act = act.sort_values(["pid", "age"])
    # --- return rate r(age): among gap events at age a, fraction that return ---
    gap_total = {}; gap_return = {}
    last_age = {}
    careers = {pid: g["age"].to_numpy() for pid, g in act.groupby("pid")}
    for pid, ages in careers.items():
        ages = np.unique(ages)
        last_age[pid] = ages.max()
        aset = set(ages.tolist())
        for a in ages:
            if (a + 1) not in aset:                 # a gap event begins after age a
                gap_total[a] = gap_total.get(a, 0) + 1
                if any(x > a + 1 for x in ages):    # returns later => injury/personal
                    gap_return[a] = gap_return.get(a, 0) + 1
    r_by_age = {a: gap_return.get(a, 0) / n for a, n in gap_total.items() if n > 0}

    # --- terminal-exit performance rows ---
    exit_rows = []
    for pid, ages in careers.items():
        a_last = int(np.max(ages))
        a_exit = a_last + 1
        if a_exit > AGE_HI:                          # exits past horizon: no info
            continue
        r = r_by_age.get(a_exit, r_by_age.get(a_last, 0.0))
        w_perf = 1.0 - r                             # performance fraction
        if w_perf <= 1e-6:
            continue
        exit_rows.append({"pid": pid, "age": a_exit, "share": np.nan,
                          "w": w_perf, "kind": "exit"})
    act2 = act[["pid", "age", "share"]].copy()
    act2["w"] = 1.0; act2["kind"] = "active"
    rows = pd.concat([act2, pd.DataFrame(exit_rows)], ignore_index=True)
    rows = rows[(rows["age"] >= AGE_LO) & (rows["age"] <= AGE_HI)].reset_index(drop=True)
    return rows, r_by_age


def fit_censored(rows: pd.DataFrame, phi: pd.Series, tau: float, c: float,
                 const_pi: float | None = None):
    """Fit the offset two-limit Tobit. const_pi overrides r(age) weights with a
    flat injury share on exit rows (robustness)."""
    rows = rows.copy()
    rows["phi"] = rows["pid"].map(phi).astype(float)
    rows = rows.dropna(subset=["phi"])
    if const_pi is not None:
        rows.loc[rows["kind"] == "exit", "w"] = 1.0 - const_pi
    age = rows["age"].to_numpy()
    X = design_age(age)
    phi_v = rows["phi"].to_numpy()
    w = rows["w"].to_numpy(float)
    is_exit = (rows["kind"] == "exit").to_numpy()
    share = rows["share"].to_numpy(float)
    # z = y - phi ; thresholds on z
    z = np.where(is_exit, 0.0, share - phi_v)        # z irrelevant for censored rows
    lo = tau - phi_v
    hi = c - phi_v
    bcen = is_exit                                   # left-censored = perf exits
    tcen = (~is_exit) & (share >= c)                 # ceiling
    z = np.where(tcen, 0.0, z)
    beta, V = fit_tobit(z, X, lo, hi, w, bcen, tcen, rows["pid"].to_numpy())
    return beta[:X.shape[1]], beta, V


# ------------------------------ synthetic --------------------------------
def synthetic():
    rng = np.random.default_rng(7)
    n = 2000
    phi = rng.normal(0.075, 0.018, n)                # career levels around full-time-ish
    # planted TRUE curve: peak 27, gentle then steep decline, mild early rise
    def g_true(a):
        a = np.asarray(a, float)
        return -0.0009 * (a - 27) ** 2 - 0.00010 * np.clip(a - 27, 0, None) ** 3
    tau = 0.045                                       # roster bar
    c = 0.115                                         # full-time ceiling
    s_eps = 0.012
    rows = []
    for i in range(n):
        start = rng.integers(21, 26)
        a = start
        while a <= 42:
            # injury hazard: high young, ~0 old
            h_inj = max(0.0, 0.12 - 0.004 * (a - 21))
            if rng.random() < h_inj:
                # injury gap: skip one year, maybe return
                if rng.random() < 0.7 and a + 1 <= 42:
                    a += 2
                    continue
                else:
                    break                              # injury-terminal
            lat = phi[i] + g_true(a) + rng.normal(0, s_eps)
            if lat < tau:
                break                                  # performance exit
            share = min(lat, c)
            rows.append({"pid": i, "age": int(a), "share": float(share)})
            a += 1
    act = pd.DataFrame(rows)
    act = act[(act["age"] >= AGE_LO) & (act["age"] <= AGE_HI)]
    true_curve = pd.Series({ag: g_true(ag) for ag in range(22, AGE_HI + 1)})
    true_curve = (true_curve - true_curve.max()) * 1000.0
    run_and_report(act, c, tau_pctile=LOW_PCTILE, true_curve=true_curve,
                   header=f"SYNTHETIC RECOVERY TEST  (planted peak=27, "
                          f"planted decline 27->35 = {true_curve[35]:.1f} pts)")


# ------------------------------ real data --------------------------------
def load_real() -> tuple[pd.DataFrame, float]:
    sh = pd.read_parquet(SHARE_CACHE)
    sh["batter"] = sh["batter"].astype(str); sh["season"] = sh["season"].astype(int)
    al = pd.read_parquet(ALLOC_PANEL, columns=["batter", "season", "age"])
    al["batter"] = al["batter"].astype(str); al["season"] = al["season"].astype(int)
    df = sh.merge(al[["batter", "season", "age"]], on=["batter", "season"], how="left")
    # if age missing from alloc panel for some seasons, derive nothing -- keep what we have
    df = df.dropna(subset=["age", "pa_share"])
    df["age"] = df["age"].astype(int)
    # active = regular seasons; full-time ceiling c from the 99th pct share
    c = float(sh["pa_share"].quantile(0.99))
    # keep batters who were ever regular
    if "pa" in df.columns:
        reg = df[df["pa"] >= MIN_REG_PA]
    else:
        reg = df[df["pa_share"] >= df["pa_share"].quantile(0.25)]
    everreg = set(reg["batter"])
    keep = df[df["batter"].isin(everreg)].copy()
    keep = keep.rename(columns={"batter": "pid", "pa_share": "share"})
    cols = ["pid", "age", "share"] + (["pa"] if "pa" in keep.columns else [])
    keep = keep[cols]
    keep = keep[(keep["age"] >= AGE_LO) & (keep["age"] <= AGE_HI)]
    return keep, c, ("pa" in keep.columns)


REG_SHARE_FALLBACK = MIN_REG_PA / 6150.0   # ~0.0488: "regular" share floor when the
#                                             share cache carries no PA-count column


def make_active(df: pd.DataFrame, regulars: bool, reg_share: float | None = None) -> pd.DataFrame:
    """All-PA active sample, or regulars-only. Under regulars-only a slide into
    part-time becomes a non-active year -- so the decline to part-time is CENSORED
    (an exit event) rather than observed as a low share. Uses the PA count if the
    share cache has one, else a share-floor proxy (300 PA / ~6150 team PA)."""
    if not regulars:
        return df[["pid", "age", "share"]].copy()
    if "pa" in df.columns:
        a = df[df["pa"] >= MIN_REG_PA]
    else:
        thr = REG_SHARE_FALLBACK if reg_share is None else reg_share
        a = df[df["share"] >= thr]
    return a[["pid", "age", "share"]].copy()


# --------------------------- driver / report -----------------------------
def run_and_report(act, c, tau_pctile, true_curve=None, header="EFFECTIVE AGING"):
    P(header); P("=" * 80)
    tau = float(np.quantile(act["share"], tau_pctile))
    rows, r_by_age = build_rows(act)
    n_exit = int((rows["kind"] == "exit").shape[0] and (rows["kind"] == "exit").sum())
    P(f" {act['pid'].nunique():,} players, {len(act):,} active seasons, "
      f"{n_exit:,} performance-exit rows; tau={tau:.4f}, c={c:.4f}")

    naive, beta_within = naive_within_curve(act)
    phi = player_offsets(act, beta_within)
    beta_age, _, _ = fit_censored(rows, phi, tau, c)
    cens = curve_from_beta(beta_age)

    ages_show = [23, 25, 27, 29, 31, 33, 35, 37]
    if true_curve is not None:
        P(f"\n  {'age':>4} {'TRUE':>8} {'naive(active)':>16} {'censored+inj':>16}")
        for a in ages_show:
            P(f"  {a:>4d} {true_curve.get(a, np.nan):>8.1f} "
              f"{naive.get(a, np.nan):>16.1f} {cens.get(a, np.nan):>16.1f}")
        P(f"\n  peak: TRUE 27 | naive {int(naive.idxmax())} | censored {int(cens.idxmax())}")
        P(f"  decline 27->35:  TRUE {true_curve[35]:.1f} | "
          f"naive {naive[35]:.1f} | censored {cens[35]:.1f}")
    else:
        P(f"\n  {'age':>4} {'naive(active)':>16} {'censored+inj':>16}  (vs peak, x1000)")
        for a in ages_show:
            P(f"  {a:>4d} {naive.get(a, np.nan):>16.1f} {cens.get(a, np.nan):>16.1f}")
        P(f"\n  peak age: naive {int(naive.idxmax())} | censored {int(cens.idxmax())}")
        P(f"  decline 27->35:  naive {naive[35]:.1f} | censored {cens[35]:.1f}")
        # robustness: constant-pi injury share
        P("\n  robustness (constant injury share pi on exits):")
        for pi in CONST_PI:
            b, _, _ = fit_censored(rows, phi, tau, c, const_pi=pi)
            cc = curve_from_beta(b)
            P(f"    pi={pi:.2f}:  decline 27->35 = {cc[35]:.1f}  (peak {int(cc.idxmax())})")
        # show r(age) the injury exemption is using
        P("\n  return rate r(age) used for injury exemption (1-r = perf weight):")
        for a in [22, 24, 26, 28, 30, 32, 34, 36]:
            P(f"    age {a}: r={r_by_age.get(a, float('nan')):.2f}")

    out = pd.DataFrame({"age": list(range(22, AGE_HI + 1))})
    out["naive_vs_peak_x1000"] = out["age"].map(naive)
    out["censored_vs_peak_x1000"] = out["age"].map(cens)
    if true_curve is not None:
        out["true_vs_peak_x1000"] = out["age"].map(true_curve)
    out.to_csv(OUTPUT_DIR / "aging_effective.csv", index=False, float_format="%.2f")
    (OUTPUT_DIR / "aging_effective_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote aging_effective.csv, aging_effective_summary.txt")


def tau_strip(act, c, label, pctiles=(0.05, 0.10, 0.25)):
    """Re-fit the censored curve across lower-censor thresholds tau = p{5,10,25} of
    the active-share distribution. naive is fixed (tau-free); only the censored
    decline and peak move. Brackets how much of the steepening is the tau choice."""
    P(f"\n  -- tau strip: {label}  "
      f"({act['pid'].nunique():,} players, {len(act):,} active seasons) --")
    naive, beta_within = naive_within_curve(act)
    phi = player_offsets(act, beta_within)
    rows, _ = build_rows(act)                      # tau-independent; build once
    P(f"    {'pctile':>6} {'tau':>8} {'naive 27->35':>13} {'censored 27->35':>16} {'peak':>5}")
    for p in pctiles:
        tau = float(np.quantile(act["share"], p))
        b, _, _ = fit_censored(rows, phi, tau, c)
        cc = curve_from_beta(b)
        P(f"    {p*100:>5.0f}% {tau:>8.4f} {naive[35]:>13.1f} "
          f"{cc[35]:>16.1f} {int(cc.idxmax()):>5d}")


def main():
    if "--synthetic" in sys.argv:
        synthetic(); return
    if not HAVE_CORE:
        print("meso_core not importable; run with --synthetic, or set SHARE_CACHE/ALLOC_PANEL.")
        return
    df, c, has_pa = load_real()
    act_all = make_active(df, regulars=False)
    run_and_report(act_all, c, tau_pctile=LOW_PCTILE,
                   header="EFFECTIVE (selection-corrected) PA-SHARE AGING CURVE")
    P("\n" + "=" * 80)
    P("ROBUSTNESS STRIPS  (tau = lower-censor threshold; naive is tau-free)")
    P("=" * 80)
    tau_strip(act_all, c, "all-PA active (decline-to-part-time OBSERVED)")
    act_reg = make_active(df, regulars=True)
    note = ">=300 PA" if has_pa else f">= {REG_SHARE_FALLBACK:.4f} share proxy (~300 PA)"
    tau_strip(act_reg, c, f"regulars-only {note} (decline-to-part-time CENSORED)")
    (OUTPUT_DIR / "aging_effective_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("rewrote aging_effective_summary.txt with robustness strips")


if __name__ == "__main__":
    main()
