"""
reconcile_dispersion.py  (exploratory)
=====================================

Why does talent_dispersion/talent_cv report ~0.068 wOBA dispersion in the 1970s
while cohort_menu_test reports ~0.035 for the same era? They differ on two axes,
and this script computes the full 2x2 on identical data so exactly one is shown
to drive the gap:

  ESTIMAND   M1 = two-way decade batter fixed effect (opponent- and age-netted)
             M2 = player decade-mean of age-adjusted season wOBA (cohort_menu way)
  POPULATION BROAD  = >= 1000 decade PA (the talent_cv bar; admits part-timers)
             STRICT = >= 3 seasons of >= 300 PA (established regulars)

All SDs are PA-weighted and raw (no noise correction; we showed it is small and
identical across methods, so it is not the driver). (M1,BROAD) reproduces the
talent_cv setup; (M2,STRICT) reproduces cohort_menu. Reading down a column
isolates the estimand; across a row isolates the population.

Reads pa_panel.parquet + wOBA_weights.csv. pyfixest (one fit per decade).
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import pyfixest as pf
import meso_core as mc
import fit_two_way as ft

L = []
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
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


def wsd(vals, wts):
    w = np.asarray(wts, float); x = np.asarray(vals, float)
    mu = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - mu) ** 2, weights=w)))


def decade_2x2(df, ycol, dec):
    sub = df[(df["season"] // 10 * 10) == dec]
    sub = sub[sub["iw"] != 1].dropna(subset=[ycol, "bat_age", "pit_age", "bat_id", "pit_id"]).copy()
    sub["bat_age"] = sub["bat_age"].astype(int); sub["pit_age"] = sub["pit_age"].astype(int)
    sub = sub[sub["bat_age"].between(mc.TWO_WAY_AGE_LO, mc.TWO_WAY_AGE_HI)
              & sub["pit_age"].between(mc.TWO_WAY_AGE_LO, mc.TWO_WAY_AGE_HI)]
    sub["bat_id"] = sub["bat_id"].astype(str); sub["pit_id"] = sub["pit_id"].astype(str)
    sub["season"] = sub["season"].astype(int)
    if len(sub) < 5000:
        return None

    # ---- player-season table (for M2 + population definitions) ----
    ps = sub.groupby(["bat_id", "season"]).agg(woba=(ycol, "mean"),
                                               pa=(ycol, "size"), age=("bat_age", "first")).reset_index()
    decpa = ps.groupby("bat_id")["pa"].sum()
    nreg = ps[ps["pa"] >= 300].groupby("bat_id").size()
    BROAD = set(decpa[decpa >= 1000].index)
    STRICT = set(nreg[nreg >= 3].index)

    # M2: age-adjusted player decade-mean (delta age curve)
    reg = ps[ps["pa"] >= 300].copy()
    reg["wd"] = reg["woba"] - reg.groupby("bat_id")["woba"].transform("mean")
    age_eff = reg.groupby("age")["wd"].mean()
    ps["woba_adj"] = ps["woba"] - ps["age"].map(age_eff).fillna(0.0)
    pm = ps.groupby("bat_id").apply(
        lambda g: pd.Series({"m": np.average(g["woba_adj"], weights=g["pa"]), "pa": g["pa"].sum()}),
        include_groups=False)

    # M1: two-way decade batter FE
    s2 = mc.drop_singletons(sub.copy(), ["season", "bat_id", "pit_id", "bat_age", "pit_age"])
    m = pf.feols(f"{ycol} ~ 1 | season + bat_id + pit_id + bat_age + pit_age", data=s2)
    fe = m.fixef(); bkey = next(k for k in fe if "bat_id" in k)
    tau = pd.Series(fe[bkey]); tau.index = tau.index.astype(str)
    fe_pa = s2.groupby("bat_id").size()

    def sd_on(measure, pop, paw):
        idx = [b for b in pop if b in measure.index and b in paw.index]
        return wsd(measure.reindex(idx).to_numpy(), paw.reindex(idx).to_numpy()), len(idx)
    out = {}
    out["M1_BROAD"], out["nB"] = sd_on(tau, BROAD, fe_pa)
    out["M1_STRICT"], out["nS"] = sd_on(tau, STRICT, fe_pa)
    out["M2_BROAD"], _ = sd_on(pm["m"], BROAD, pm["pa"])
    out["M2_STRICT"], _ = sd_on(pm["m"], STRICT, pm["pa"])
    return out


def main():
    log(f"loading {mc.PANEL_PA}")
    df = pd.read_parquet(mc.PANEL_PA, columns=[
        "season", "bat_id", "pit_id", "iw", "bat_age", "pit_age",
        "single", "double", "triple", "hr", "walk", "hbp"])
    df["season"] = df["season"].astype(int)
    df = prep(df)

    P("DISPERSION RECONCILIATION  (why talent_cv 0.068 vs cohort_menu 0.035, 1970s)")
    P("=" * 76)
    P("SD of shape-norm batting skill, PA-weighted, raw. (M1,BROAD)=talent_cv setup;")
    P("(M2,STRICT)=cohort_menu setup. Down a column = estimand; across a row = population.")
    P(f"\n  {'decade':>6s} | {'M1 two-way FE':>22s} | {'M2 season-mean':>22s}")
    P(f"  {'':>6s} | {'BROAD':>10s} {'STRICT':>10s} | {'BROAD':>10s} {'STRICT':>10s}")
    rows = []
    for dec in range(1910, 2030, 10):
        r = decade_2x2(df, "w_shapenorm", dec)
        if not r:
            continue
        P(f"  {dec:>5d}s | {r['M1_BROAD']:>10.4f} {r['M1_STRICT']:>10.4f} | "
          f"{r['M2_BROAD']:>10.4f} {r['M2_STRICT']:>10.4f}   (nB={r['nB']}, nS={r['nS']})")
        r["decade"] = dec; rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(mc.OUTPUT_DIR / "reconcile_dispersion.csv", index=False, float_format="%.5f")

    # decompose the gap on the focal 1970s (or the median decade if absent)
    foc = res[res["decade"] == 1970]
    foc = foc.iloc[0] if len(foc) else res.iloc[len(res) // 2]
    P(f"\n--- gap decomposition, {int(foc['decade'])}s ---")
    P(f"  start (M1,BROAD) = talent_cv setup : {foc['M1_BROAD']:.4f}")
    P(f"  tighten population (BROAD->STRICT) at fixed estimand: "
      f"{foc['M1_BROAD']:.4f} -> {foc['M1_STRICT']:.4f}  (population effect {foc['M1_BROAD']-foc['M1_STRICT']:+.4f})")
    P(f"  switch estimand (M1->M2) at STRICT population: "
      f"{foc['M1_STRICT']:.4f} -> {foc['M2_STRICT']:.4f}  (estimand effect {foc['M1_STRICT']-foc['M2_STRICT']:+.4f})")
    P(f"  end (M2,STRICT) = cohort_menu setup: {foc['M2_STRICT']:.4f}")
    pop_eff = abs(foc['M1_BROAD'] - foc['M1_STRICT']); est_eff = abs(foc['M1_STRICT'] - foc['M2_STRICT'])
    driver = "POPULATION (the broad >=1000-PA bar admits more-dispersed part-timers)" if pop_eff > est_eff \
        else "ESTIMAND (two-way FE vs season-mean)"
    P(f"  => the gap is driven mainly by {driver}.")
    (mc.OUTPUT_DIR / "reconcile_dispersion_summary.txt").write_text("\n".join(L), encoding="utf-8")
    log("wrote reconcile_dispersion.csv, summary")


if __name__ == "__main__":
    main()
