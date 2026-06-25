"""
pitcher_dispersion.py  (exploratory; companion to talent_dispersion.py)
======================================================================

Graphs the dispersion of PITCHING skill (tau_pit) by decade alongside batting
dispersion, as a companion to the Gould section. If Gould's "talent compressed
against a right wall" were a general law, pitching should show it too.

The per-decade two-way fit already estimates the pitcher fixed effect, so this
pulls tau_pit from the SAME fits as the batting curve (no extra fitting), with the
identical noise correction:
    Var(true) = Var(tau_hat) - mean(sigma^2 / n_j),  PA-weighted over regular pitchers.
    (regular = >= 1000 decade PA, same bar as batters -- a full-time pitcher faces a
    comparable ~700 PA/season to a regular hitter's ~600.)

CAVEAT (read before interpreting). Pitcher dispersion conflates talent with the
changing starter/reliever ROLE structure. A modern closer's tau_pit comes from
short, high-leverage, favorable-matchup bursts, creating a fat low-tau tail that
inflates dispersion for usage reasons, not talent. Batters have no analogue. So a
rising pitching curve is NOT clean evidence against Gould compression. A starters-
only cut (needs a role flag, a small rerun) is the refinement if the curve looks
usage-driven; reported here with the caveat stated.

Reuses talent_dispersion.prep_woba and the same model. Requires pyfixest.

Run:  py pitcher_dispersion.py
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import pyfixest as pf

import meso_core as mc
import fit_two_way as ft
import talent_dispersion as td        # reuse prep_woba and constants

DECADES = list(range(1910, 2030, 10))
MIN_BAT_PA = 1000          # regular batter over the decade (matches batting curve)
MIN_PIT_PA = 1000          # regular pitcher: a full-time pitcher faces ~700 PA/season
#                            (~200 IP), comparable to a regular hitter's ~600, so the
#                            bar is the same. (Pitcher workloads have if anything fallen,
#                            and pitching PAs spread across deeper modern staffs, so the
#                            qualifying pitcher COUNT still differs by era -- a roster
#                            structure effect, reported via nP, not a threshold artifact.)
SCHEMES = ["shapenorm", "fixedw"]
OUT_CSV = mc.OUTPUT_DIR / "pitcher_dispersion.csv"
OUT_SUM = mc.OUTPUT_DIR / "pitcher_dispersion_summary.txt"
OUT_FIG = mc.OUTPUT_DIR / "fig_dispersion_bat_vs_pit.png"
L = []
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def P(s=""): print(s); L.append(s)


def _disp(tau, n_i, sigma2, reg_idx):
    """Noise-corrected (true) and raw SD over the regular set, PA-weighted."""
    tau_r = tau.reindex(reg_idx).dropna()
    n_r = n_i.reindex(tau_r.index)
    w = n_r.to_numpy(float); t = tau_r.to_numpy(float)
    if len(t) < 2:
        return np.nan, np.nan, len(tau_r)
    mu = np.average(t, weights=w)
    raw_var = np.average((t - mu) ** 2, weights=w)
    samp_var = np.average(sigma2 / n_r.to_numpy(float), weights=w)
    return np.sqrt(max(raw_var - samp_var, 0.0)), np.sqrt(raw_var), len(tau_r)


def decade_both(sub, ycol):
    """Fit two-way once; return batter and pitcher dispersion (true, raw, n)."""
    sub = sub[sub["iw"] != 1].dropna(subset=[ycol, "bat_age", "pit_age", "bat_id", "pit_id"]).copy()
    sub["bat_age"] = sub["bat_age"].astype(int); sub["pit_age"] = sub["pit_age"].astype(int)
    sub = sub[sub["bat_age"].between(mc.TWO_WAY_AGE_LO, mc.TWO_WAY_AGE_HI)
              & sub["pit_age"].between(mc.TWO_WAY_AGE_LO, mc.TWO_WAY_AGE_HI)]
    sub["bat_id"] = sub["bat_id"].astype(str); sub["pit_id"] = sub["pit_id"].astype(str)
    sub["season"] = sub["season"].astype(int)
    sub = mc.drop_singletons(sub, ["season", "bat_id", "pit_id", "bat_age", "pit_age"])
    if sub["bat_id"].nunique() < 50 or sub["pit_id"].nunique() < 50:
        return (np.nan,) * 6
    m = pf.feols(f"{ycol} ~ 1 | season + bat_id + pit_id + bat_age + pit_age", data=sub)
    fe = m.fixef()
    sigma2 = float(np.var(np.asarray(m.resid())))
    bkey = next(k for k in fe if "bat_id" in k); pkey = next(k for k in fe if "pit_id" in k)
    tb = pd.Series(fe[bkey]); tb.index = tb.index.astype(str)
    tp = pd.Series(fe[pkey]); tp.index = tp.index.astype(str)
    nb = sub.groupby("bat_id").size(); npi = sub.groupby("pit_id").size()
    b_true, b_raw, b_n = _disp(tb, nb, sigma2, nb[nb >= MIN_BAT_PA].index)
    p_true, p_raw, p_n = _disp(tp, npi, sigma2, npi[npi >= MIN_PIT_PA].index)
    return b_true, b_raw, b_n, p_true, p_raw, p_n


def make_figure(res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        P("  (matplotlib unavailable; CSV written, plot yourself)"); return
    sn = res[res["scheme"] == "shapenorm"].dropna(subset=["bat_true_sd", "pit_true_sd"])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(sn["decade"], sn["bat_true_sd"] * 1000, "-o", lw=2, label="batting (\u03c4_bat)")
    ax.plot(sn["decade"], sn["pit_true_sd"] * 1000, "-s", lw=2, label="pitching (\u03c4_pit)")
    for yr in [1947, 1961, 1969, 1977, 1993, 1998]:
        ax.axvline(yr, color="0.8", lw=0.8, zorder=0)
    ax.annotate("integration", (1947, ax.get_ylim()[1]), fontsize=7, rotation=90,
                va="top", ha="right", color="0.5")
    ax.set_xlabel("decade"); ax.set_ylabel("true skill SD (wOBA points, \u00d71000)")
    ax.set_title("Dispersion of batting vs pitching skill by decade")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT_FIG, dpi=200); plt.close(fig)
    P(f"  wrote {OUT_FIG}")


def main():
    OUT_SUM.parent.mkdir(parents=True, exist_ok=True)
    log(f"loading {mc.PANEL_PA}")
    df = pd.read_parquet(mc.PANEL_PA, columns=[
        "season", "bat_id", "pit_id", "iw", "bat_age", "pit_age",
        "single", "double", "triple", "hr", "walk", "hbp"])
    df["season"] = df["season"].astype(int)
    df = td.prep_woba(df)

    P("BATTING vs PITCHING SKILL DISPERSION BY DECADE")
    P("=" * 74)
    P(f"true SD of tau (run-value/PA), noise-corrected. Regular batters >= {MIN_BAT_PA:,}")
    P(f"decade PA, regular pitchers >= {MIN_PIT_PA:,} decade PA faced.")
    P("CAVEAT: pitching dispersion conflates talent with starter/reliever usage; a")
    P("rising pitching curve is not clean evidence against Gould compression.")
    rows = []
    for scheme in SCHEMES:
        ycol = f"w_{scheme}"
        P(f"\n--- {scheme} weights ---")
        P(f"  {'decade':>6} {'nB':>6} {'bat_SD':>8} {'nP':>6} {'pit_SD':>8}")
        for d in DECADES:
            sub = df[(df["season"] // 10 * 10) == d]
            if len(sub) < 5000:
                continue
            t0 = time.time()
            bt, br, bn, pt, pr, pn = decade_both(sub, ycol)
            log(f"    {d}s fit in {time.time()-t0:.0f}s")
            if bn:
                P(f"  {d:>5d}s {bn:>6d} {bt:>8.4f} {pn:>6d} {pt:>8.4f}")
                rows.append({"scheme": scheme, "decade": d, "nB": bn, "nP": pn,
                             "bat_true_sd": bt, "bat_raw_sd": br,
                             "pit_true_sd": pt, "pit_raw_sd": pr})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False, float_format="%.5f")

    sn = res[res["scheme"] == "shapenorm"].dropna(subset=["bat_true_sd", "pit_true_sd"])
    if len(sn) >= 3:
        bp = np.polyfit(sn["decade"], sn["pit_true_sd"], 1)[0]
        bb = np.polyfit(sn["decade"], sn["bat_true_sd"], 1)[0]
        rp = np.corrcoef(sn["decade"], sn["pit_true_sd"])[0, 1]
        P(f"\n--- trends (shapenorm true_SD) ---")
        P(f"  batting  slope {bb*1000:+.4f} wOBA-pts/decade")
        P(f"  pitching slope {bp*1000:+.4f} wOBA-pts/decade, corr(decade,SD) {rp:+.2f}")
        P("  (interpret pitching slope with the usage caveat above.)")
    make_figure(res)
    OUT_SUM.write_text("\n".join(L), encoding="utf-8")
    log(f"wrote {OUT_CSV}, {OUT_SUM}")


if __name__ == "__main__":
    main()
