"""
ba_skill_corr.py  (exploratory)
===============================

On a player-career basis, how well does the crude traditional statistic -- batting
average (and batting-average-allowed) -- track the clean opponent/season/age-purged
skill measure, the two-way fixed effect tau?

  hitters : corr( career BA,         career tau_bat )
  pitchers: corr( career BA allowed,  career tau_pit )

Conjecture: BA is a decent proxy for HITTER skill but a poor one for PITCHER skill,
because pitchers exert little control over balls in play (DIPS/BABIP). If so, the
same statistic validates very differently on the two sides -- a micro-data point.

tau comes from one full-panel two-way fit:
    woba_value ~ 1 | season + bat_id + pit_id + bat_age + pit_age
career BA = H / AB with H = 1B+2B+3B+HR and AB = PA - (BB + IBB + HBP + SF + SH + CI).
Reported unweighted and PA-weighted, Pearson and Spearman, for hitters and pitchers,
among players with >= MIN_PA career PA.

Run from C:\\baseball_eras\\github :  py ba_skill_corr.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import pyfixest as pf
import meso_core as mc

MIN_PA = 1000
MIN_SEASON_PA = 300          # a qualifying single season for the seasonal analysis
HIT_COLS = ["single", "double", "triple", "hr"]
NONAB_COLS = ["walk", "iw", "hbp", "sf", "sh", "xi"]   # excluded from at-bats
OUT_SUM = mc.OUTPUT_DIR / "ba_skill_corr_summary.txt"
OUT_FIG = mc.OUTPUT_DIR / "fig_ba_vs_skill.png"
L = []
def P(s=""): print(s); L.append(s)


def wcorr(x, y, w):
    """PA-weighted Pearson correlation."""
    w = np.asarray(w, float); x = np.asarray(x, float); y = np.asarray(y, float)
    mx = np.average(x, weights=w); my = np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx = np.average((x - mx) ** 2, weights=w); vy = np.average((y - my) ** 2, weights=w)
    return cov / np.sqrt(vx * vy)


def season_corr(df, who, tau, side):
    """Pool all qualifying player-SEASONS; correlate each season's BA with the
    player's CAREER tau (true skill). A season's BA tracks true skill worse the
    noisier it is -- the DIPS prediction is that this is worse for pitchers, since a
    season's BA-allowed carries BABIP luck that a career averages out."""
    g = df.groupby([who, "season"])
    pa = g.size().rename("PA")
    H = df[HIT_COLS].sum(axis=1).groupby([df[who], df["season"]]).sum().rename("H")
    nonab = df[NONAB_COLS].sum(axis=1).groupby([df[who], df["season"]]).sum().rename("nonab")
    t = pd.concat([pa, H, nonab], axis=1).reset_index()
    t["AB"] = t["PA"] - t["nonab"]
    t["ba"] = t["H"] / t["AB"].where(t["AB"] > 0, np.nan)
    t = t[t["PA"] >= MIN_SEASON_PA]
    t["tau"] = t[who].map(tau)
    t = t.dropna(subset=["ba", "tau"])
    x, y, w = t["ba"].to_numpy(), t["tau"].to_numpy(), t["PA"].to_numpy()
    pear = stats.pearsonr(x, y)[0]; spear = stats.spearmanr(x, y)[0]; wp = wcorr(x, y, w)
    P(f"\n  == {side} ({len(t):,} player-seasons, >= {MIN_SEASON_PA} PA) ==")
    P(f"    Pearson  (unweighted): {pear:+.4f}")
    P(f"    Pearson  (PA-weighted): {wp:+.4f}")
    P(f"    Spearman (rank)       : {spear:+.4f}")
    return pear


def career_ba(df, who):
    """Career BA per player (who = 'bat_id' or 'pit_id'). Returns frame indexed by id
    with H, AB, PA, ba."""
    g = df.groupby(who)
    n_pa = g.size().rename("PA")
    H = df[HIT_COLS].sum(axis=1).groupby(df[who]).sum().rename("H")
    nonab = df[NONAB_COLS].sum(axis=1).groupby(df[who]).sum().rename("nonab")
    out = pd.concat([n_pa, H, nonab], axis=1)
    out["AB"] = out["PA"] - out["nonab"]
    out["ba"] = out["H"] / out["AB"].where(out["AB"] > 0, np.nan)
    return out


def report(side, tab, tau_col, ba_col="ba"):
    t = tab.dropna(subset=[ba_col, tau_col])
    t = t[t["PA"] >= MIN_PA]
    x = t[ba_col].to_numpy(); y = t[tau_col].to_numpy(); w = t["PA"].to_numpy()
    pear = stats.pearsonr(x, y)[0]
    spear = stats.spearmanr(x, y)[0]
    wp = wcorr(x, y, w)
    P(f"\n  == {side} ({len(t):,} players, >= {MIN_PA} PA) ==")
    P(f"    Pearson  (unweighted): {pear:+.4f}")
    P(f"    Pearson  (PA-weighted): {wp:+.4f}")
    P(f"    Spearman (rank)       : {spear:+.4f}")
    P(f"    R^2 (unweighted)      : {pear**2:.4f}  -> BA explains {pear**2*100:.0f}% of skill variance")
    return t


def make_fig(hit, pit):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        P("  (matplotlib unavailable)"); return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
    for a, t, tcol, ttl in [(ax[0], hit, "tau_bat", "Hitters: BA vs skill"),
                            (ax[1], pit, "tau_pit", "Pitchers: BA allowed vs skill")]:
        a.scatter(t["ba"], t[tcol], s=6, alpha=0.25, edgecolors="none")
        a.set_xlabel("career batting average" + ("" if tcol == "tau_bat" else " allowed"))
        a.set_ylabel(tcol); a.set_title(ttl)
        r = stats.pearsonr(t["ba"], t[tcol])[0]
        a.text(0.05, 0.95, f"r = {r:+.2f}", transform=a.transAxes, va="top", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT_FIG, dpi=200); plt.close(fig)
    P(f"  wrote {OUT_FIG}")


def main():
    OUT_SUM.parent.mkdir(parents=True, exist_ok=True)
    cols = ["season", "bat_id", "pit_id", "bat_age", "pit_age", "iw", "woba_value"] \
        + HIT_COLS + NONAB_COLS + ["sh"]
    cols = list(dict.fromkeys(cols))
    P(f"loading {mc.PANEL_PA}")
    df = pd.read_parquet(mc.PANEL_PA, columns=[c for c in cols if c])
    df["season"] = df["season"].astype(int)
    for c in HIT_COLS + NONAB_COLS:
        df[c] = df[c].fillna(0)

    # sanity: league BA
    H = df[HIT_COLS].sum(axis=1).sum(); AB = len(df) - df[NONAB_COLS].sum(axis=1).sum()
    P(f"league career BA sanity: {H/AB:.4f}  (expect ~0.255-0.270)")

    # one full-panel two-way fit for career tau
    fitdf = df.dropna(subset=["woba_value", "bat_age", "pit_age", "bat_id", "pit_id"]).copy()
    fitdf = fitdf[fitdf["iw"] != 1]
    fitdf["bat_age"] = fitdf["bat_age"].astype(int); fitdf["pit_age"] = fitdf["pit_age"].astype(int)
    fitdf = fitdf[fitdf["bat_age"].between(mc.TWO_WAY_AGE_LO, mc.TWO_WAY_AGE_HI)
                  & fitdf["pit_age"].between(mc.TWO_WAY_AGE_LO, mc.TWO_WAY_AGE_HI)]
    for c in ["bat_id", "pit_id", "season"]:
        fitdf[c] = fitdf[c].astype(str)
    fitdf = mc.drop_singletons(fitdf, ["season", "bat_id", "pit_id", "bat_age", "pit_age"])
    P(f"fitting two-way on {len(fitdf):,} PAs ...")
    m = pf.feols("woba_value ~ 1 | season + bat_id + pit_id + bat_age + pit_age", data=fitdf)
    fe = m.fixef()
    bkey = next(k for k in fe if "bat_id" in k); pkey = next(k for k in fe if "pit_id" in k)
    tau_bat = pd.Series(fe[bkey]); tau_bat.index = tau_bat.index.astype(str)
    tau_pit = pd.Series(fe[pkey]); tau_pit.index = tau_pit.index.astype(str)

    # career BA tables
    df["bat_id"] = df["bat_id"].astype(str); df["pit_id"] = df["pit_id"].astype(str)
    hit = career_ba(df, "bat_id"); hit["tau_bat"] = tau_bat.reindex(hit.index)
    pit = career_ba(df, "pit_id"); pit["tau_pit"] = tau_pit.reindex(pit.index)

    P("\nCORRELATION: crude batting average vs clean two-way skill (career basis)")
    P("=" * 74)
    ht = report("HITTERS: career BA vs tau_bat", hit, "tau_bat")
    pt = report("PITCHERS: career BA-allowed vs tau_pit", pit, "tau_pit")
    P("\n  (tau_bat: higher = better hitter; tau_pit: higher = worse pitcher, so both")
    P("   correlations are positive -- a better BA goes with more batting value, and a")
    P("   higher BA-allowed goes with a worse pitcher.)")
    make_fig(ht, pt)

    # --- seasonal analysis: the DIPS asymmetry that career-aggregation hides ---
    P("\n\nSEASONAL: single-season BA vs CAREER skill (DIPS test)")
    P("=" * 74)
    hit_car = stats.pearsonr(ht["ba"], ht["tau_bat"])[0]
    pit_car = stats.pearsonr(pt["ba"], pt["tau_pit"])[0]
    hit_sea = season_corr(df, "bat_id", tau_bat, "HITTERS: season BA vs career tau_bat")
    pit_sea = season_corr(df, "pit_id", tau_pit, "PITCHERS: season BA-allowed vs career tau_pit")
    P("\n  career vs seasonal Pearson (the contrast):")
    P(f"    hitters : career {hit_car:+.3f}  ->  season {hit_sea:+.3f}   (drop {hit_car-hit_sea:.3f})")
    P(f"    pitchers: career {pit_car:+.3f}  ->  season {pit_sea:+.3f}   (drop {pit_car-pit_sea:.3f})")
    P("  DIPS prediction: equal over a career, but season BA tracks skill WORSE for")
    P("  pitchers than hitters -- the pitcher drop should exceed the hitter drop.")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        x = np.arange(2); wbar = 0.36
        ax.bar(x - wbar/2, [hit_car, pit_car], wbar, label="career", color="#1f4e8c")
        ax.bar(x + wbar/2, [hit_sea, pit_sea], wbar, label="single season", color="#c0504d")
        ax.set_xticks(x); ax.set_xticklabels(["hitters\n(BA vs skill)", "pitchers\n(BA-allowed vs skill)"])
        ax.set_ylabel("correlation with true (career) skill"); ax.set_ylim(0, 0.8)
        ax.set_title("Batting average vs skill: career vs single season")
        ax.legend(frameon=False)
        for xi, v in zip([x[0]-wbar/2, x[1]-wbar/2, x[0]+wbar/2, x[1]+wbar/2],
                         [hit_car, pit_car, hit_sea, pit_sea]):
            ax.text(xi, v+0.01, f"{v:.2f}", ha="center", fontsize=9)
        fig.tight_layout(); fig.savefig(mc.OUTPUT_DIR / "fig_ba_skill_dips.png", dpi=200); plt.close(fig)
        P(f"  wrote {mc.OUTPUT_DIR / 'fig_ba_skill_dips.png'}")
    except Exception as e:
        P(f"  (comparison figure skipped: {e})")

    OUT_SUM.write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT_SUM)


if __name__ == "__main__":
    main()
