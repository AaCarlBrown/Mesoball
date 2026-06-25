"""
cohort_menu_test.py  (exploratory)
=================================

Is the cross-era dispersion swing in batting skill a PERIOD effect (the outcome
menu compressing everyone) or a COHORT effect (more-uniform new generations)?
The decisive test holds the cohort FIXED and varies the menu: take the players
who were regulars on BOTH sides of the 1970s->80s dispersion cliff -- the same
individuals, the same talents -- and ask whether THEIR OWN age-adjusted spread
shrank when the menu narrowed. If it did, the compression is the menu, not a
new generation, because it is literally the same generation.

This is a within-player / within-cohort comparison, so it sidesteps the cross-era
level-comparison (connectedness) problem, and it converts "outcome dispersion is
menu-confounded" from an argument into a demonstration: identical talents,
repriced by a changing menu.

Reports, for windows A and B straddling the cliff: the dispersion among the FIXED
cohort (regulars in both) in each window, vs the dispersion among ALL regulars in
each window. If the fixed-cohort drop matches the all-regulars drop, the cliff is
period/menu; if the fixed cohort does NOT drop while all-regulars does, it is
cohort turnover (talent).

Reads pa_panel.parquet + wOBA_weights.csv (reuses fit_two_way helpers). pyfixest
not required (age-adjustment is the within-player delta method).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import meso_core as mc
import fit_two_way as ft

MIN_PA = 300
WIN_A = (1970, 1979)     # high-dispersion period (pre-cliff)
WIN_B = (1980, 1989)     # low-dispersion period (post-cliff)
MIN_SEAS = 2             # regular seasons required in each window to be "in the cohort"
L = []
def P(s=""): print(s); L.append(s)


def prep(df):
    df["bb_unint"] = (df["walk"] - df["iw"]).clip(lower=0)
    nonibb = df[df["iw"] != 1]
    fref, _ = ft.reference_event_mix(nonibb)
    pabys = nonibb.groupby("season").size().rename("pa").reset_index()
    w = (pd.read_csv(mc.WOBA_CSV, encoding="utf-8-sig")
         .rename(columns={"Season": "season"})[["season"] + mc.WEIGHT_COLS])
    ks, _ = ft.shapenorm_scaling(w, fref, pabys)
    df = df.merge(w, on="season", how="left").merge(ks, on="season", how="left")
    df["w_shapenorm"] = ft.woba_from_weights(
        df, df[mc.WEIGHT_COLS].to_numpy(float) * df["k_s"].to_numpy()[:, None])
    return df


def player_seasons(df):
    nb = df[df["iw"] != 1].dropna(subset=["w_shapenorm", "bat_age"])
    ps = (nb.groupby(["bat_id", "season"])
          .agg(woba=("w_shapenorm", "mean"), pa=("w_shapenorm", "size"),
               age=("bat_age", "first")).reset_index())
    ps["age"] = ps["age"].astype(int)
    return ps


def age_adjust(ps):
    """Within-player delta-method age curve; subtract age effect from each season."""
    reg = ps[ps["pa"] >= MIN_PA].copy()
    reg["w_within"] = reg["woba"] - reg.groupby("bat_id")["woba"].transform("mean")
    age_eff = reg.groupby("age")["w_within"].mean()
    ps = ps.copy()
    ps["woba_adj"] = ps["woba"] - ps["age"].map(age_eff).fillna(0.0)
    return ps


def win_means(ps, lo, hi):
    """Per-player mean age-adjusted wOBA and regular-season count in [lo,hi]."""
    w = ps[(ps["season"] >= lo) & (ps["season"] <= hi) & (ps["pa"] >= MIN_PA)]
    g = w.groupby("bat_id").agg(woba_adj=("woba_adj", "mean"), nseas=("season", "size"),
                                pa=("pa", "sum"))
    return g


def wsd(g):
    w = g["pa"].to_numpy(float); x = g["woba_adj"].to_numpy(float)
    mu = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - mu) ** 2, weights=w)))


def main():
    df = pd.read_parquet(mc.PANEL_PA, columns=[
        "season", "bat_id", "iw", "bat_age", "single", "double", "triple",
        "hr", "walk", "hbp"])
    df["season"] = df["season"].astype(int)
    df = prep(df)
    ps = age_adjust(player_seasons(df))

    A = win_means(ps, *WIN_A); B = win_means(ps, *WIN_B)
    cohort = A.index[A["nseas"] >= MIN_SEAS].intersection(B.index[B["nseas"] >= MIN_SEAS])
    sd_A_all, sd_B_all = wsd(A), wsd(B)
    sd_A_co, sd_B_co = wsd(A.loc[cohort]), wsd(B.loc[cohort])

    P("COHORT vs MENU TEST  (is the dispersion cliff period or cohort?)")
    P("=" * 66)
    P(f"window A {WIN_A}: {len(A):,} regulars;  window B {WIN_B}: {len(B):,} regulars")
    P(f"fixed cohort (regular in BOTH, >= {MIN_SEAS} seasons each): {len(cohort):,} players")
    P(f"\n  age-adjusted wOBA dispersion (SD), PA-weighted:")
    P(f"  {'group':<16s} {'win A':>8s} {'win B':>8s} {'B/A':>7s}")
    P(f"  {'all regulars':<16s} {sd_A_all:>8.4f} {sd_B_all:>8.4f} {sd_B_all/sd_A_all:>7.2f}")
    P(f"  {'fixed cohort':<16s} {sd_A_co:>8.4f} {sd_B_co:>8.4f} {sd_B_co/sd_A_co:>7.2f}")
    P("")
    r_all, r_co = sd_B_all / sd_A_all, sd_B_co / sd_A_co
    gap = r_co - r_all   # how much the fixed cohort RESISTS the population's compression
    if r_all < 0.92 and gap < 0.08:
        P("VERDICT: the FIXED cohort compresses about as much as the whole population")
        P("(cohort B/A close to population B/A) -> the cliff is a PERIOD/MENU effect. The")
        P("same talents spread less when the menu narrowed; it is not a more-uniform new")
        P("generation. Direct evidence that outcome dispersion measures the menu, not talent.")
    elif r_all < 0.92 and gap > 0.12:
        P("VERDICT: the whole population compresses but the FIXED cohort RESISTS")
        P("(cohort B/A well above population B/A) -> the cliff is COHORT turnover, a")
        P("more-uniform new generation -- a genuine talent-uniformity signal.")
    else:
        P(f"VERDICT: mixed (cohort B/A={r_co:.2f}, all B/A={r_all:.2f}, gap={gap:+.2f}); inspect.")
    (mc.OUTPUT_DIR / "cohort_menu_test_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote cohort_menu_test_summary.txt")


if __name__ == "__main__":
    main()
