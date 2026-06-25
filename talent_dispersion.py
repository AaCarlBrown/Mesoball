"""
talent_dispersion.py  (exploratory)
==================================

Has the spread of batting talent compressed over the century? (Gould's claim,
on a clean measure.) The cross-era LEVEL of skill is not identified without a
bridge, but the within-era SPREAD is: within a decade, batters and pitchers are
connected, so the dispersion of the batter fixed effect is identified, in
run-value-per-PA units comparable across decades.

For each decade I fit the two-way model
    woba ~ 1 | season + bat_id + pit_id + bat_age + pit_age
on that decade's plate appearances, take the batter effect tau_hat, and recover
TRUE-skill dispersion by removing sampling noise: a raw cross-player variance of
estimated effects overstates the true variance by the mean sampling variance
(~sigma^2 / PA per batter), so
    Var(true) = Var(tau_hat) - mean(sigma^2 / n_i),
PA-weighted over regulars. SD(true) by decade is the compression curve. Run under
both shape-normalized and fixed weights so the trend cannot be blamed on weighting.

A falling SD = Gould compression: a deepening talent pool tightening the
distribution. Integration (1947) and the expansion seasons are marked.

Reads pa_panel.parquet and wOBA_weights.csv (reuses fit_two_way's weight helpers).
Requires pyfixest.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pyfixest as pf

import meso_core as mc
import fit_two_way as ft

DECADES = list(range(1910, 2030, 10))
MIN_DECADE_PA = 1000          # "regular" over the decade
SCHEMES = ["shapenorm", "fixedw"]
OUT_CSV = mc.OUTPUT_DIR / "talent_dispersion.csv"
OUT_SUM = mc.OUTPUT_DIR / "talent_dispersion_summary.txt"
L = []
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def P(s=""): print(s); L.append(s)


def prep_woba(df):
    """Compute shapenorm and fixedw wOBA on the full panel (reusing fit_two_way)."""
    df["bb_unint"] = (df["walk"] - df["iw"]).clip(lower=0)
    nonibb = df[df["iw"] != 1]
    f_ref_vec, _ = ft.reference_event_mix(nonibb)
    pa_by_season = nonibb.groupby("season").size().rename("pa").reset_index()
    w = (pd.read_csv(mc.WOBA_CSV, encoding="utf-8-sig")
         .rename(columns={"Season": "season"})[["season"] + mc.WEIGHT_COLS])
    fixed = ft.fixed_weight_vector(w, pa_by_season)
    ks, _ = ft.shapenorm_scaling(w, f_ref_vec, pa_by_season)
    df = df.merge(w, on="season", how="left").merge(ks, on="season", how="left")
    df["w_shapenorm"] = ft.woba_from_weights(
        df, df[mc.WEIGHT_COLS].to_numpy(float) * df["k_s"].to_numpy()[:, None])
    fw = np.array([fixed[c] for c in mc.WEIGHT_COLS])
    df["w_fixedw"] = ft.woba_from_weights(df, fw[None, :])
    return df


def decade_dispersion(sub, ycol):
    """Fit two-way on a decade; return (true_SD, raw_SD, n_regulars)."""
    sub = sub[sub["iw"] != 1].dropna(subset=[ycol, "bat_age", "pit_age", "bat_id", "pit_id"]).copy()
    sub["bat_age"] = sub["bat_age"].astype(int); sub["pit_age"] = sub["pit_age"].astype(int)
    sub = sub[sub["bat_age"].between(mc.TWO_WAY_AGE_LO, mc.TWO_WAY_AGE_HI)
              & sub["pit_age"].between(mc.TWO_WAY_AGE_LO, mc.TWO_WAY_AGE_HI)]
    sub["bat_id"] = sub["bat_id"].astype(str); sub["pit_id"] = sub["pit_id"].astype(str)
    sub["season"] = sub["season"].astype(int)
    sub = mc.drop_singletons(sub, ["season", "bat_id", "pit_id", "bat_age", "pit_age"])
    if sub["bat_id"].nunique() < 50:
        return np.nan, np.nan, 0
    m = pf.feols(f"{ycol} ~ 1 | season + bat_id + pit_id + bat_age + pit_age", data=sub)
    fe = m.fixef(); bkey = next(k for k in fe if "bat_id" in k)
    tau = pd.Series(fe[bkey]); tau.index = tau.index.astype(str)
    sigma2 = float(np.var(np.asarray(m.resid())))
    n_i = sub.groupby("bat_id").size()
    reg = n_i[n_i >= MIN_DECADE_PA].index
    tau_r = tau.reindex(reg).dropna(); n_r = n_i.reindex(tau_r.index)
    w = n_r.to_numpy(float); t = tau_r.to_numpy(float)
    mu = np.average(t, weights=w)
    raw_var = np.average((t - mu) ** 2, weights=w)
    samp_var = np.average(sigma2 / n_r.to_numpy(float), weights=w)
    true_var = max(raw_var - samp_var, 0.0)
    return np.sqrt(true_var), np.sqrt(raw_var), len(reg)


def main():
    OUT_SUM.parent.mkdir(parents=True, exist_ok=True)
    log(f"loading {mc.PANEL_PA}")
    df = pd.read_parquet(mc.PANEL_PA, columns=[
        "season", "bat_id", "pit_id", "iw", "bat_age", "pit_age",
        "single", "double", "triple", "hr", "walk", "hbp"])
    df["season"] = df["season"].astype(int)
    df = prep_woba(df)

    P("TALENT DISPERSION BY DECADE  (Gould compression on a clean skill measure)")
    P("=" * 74)
    P("SD of batting skill tau_bat (run-value/PA), noise-corrected, regulars "
      f">= {MIN_DECADE_PA:,} decade PA. A falling SD = compression.")
    rows = []
    for scheme in SCHEMES:
        ycol = f"w_{scheme}"
        P(f"\n--- {scheme} weights ---")
        P(f"  {'decade':>6s} {'n_reg':>6s} {'raw_SD':>8s} {'true_SD':>8s}")
        for d in DECADES:
            sub = df[(df["season"] // 10 * 10) == d]
            if len(sub) < 5000:
                continue
            t0 = time.time()
            true_sd, raw_sd, n = decade_dispersion(sub, ycol)
            log(f"    {d}s fit in {time.time()-t0:.0f}s (n_reg={n})")
            if n:
                P(f"  {d:>5d}s {n:>6d} {raw_sd:>8.4f} {true_sd:>8.4f}")
                rows.append({"scheme": scheme, "decade": d, "n_reg": n,
                             "raw_sd": raw_sd, "true_sd": true_sd})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False, float_format="%.5f")

    # trend test on the headline (shapenorm)
    sn = res[res["scheme"] == "shapenorm"].dropna()
    if len(sn) >= 3:
        b = np.polyfit(sn["decade"], sn["true_sd"], 1)[0]
        r = np.corrcoef(sn["decade"], sn["true_sd"])[0, 1]
        first, last = sn.iloc[0], sn.iloc[-1]
        P(f"\n--- trend (shapenorm true_SD) ---")
        P(f"  {int(first.decade)}s {first.true_sd:.4f}  ->  {int(last.decade)}s {last.true_sd:.4f}  "
          f"({100*(last.true_sd/first.true_sd-1):+.0f}%)")
        P(f"  slope {b*100:+.4f} wOBA per decade, corr(decade, SD) = {r:+.2f}")
        P("  Negative slope = Gould compression: the talent distribution tightened.")
        P("  Integration 1947; expansion 1961/62/69/77/93/98 (mark on the plotted curve).")
    P("\nLevel of tau_bat is NOT comparable across decades (separate within-decade")
    P("fits); only the SPREAD is. The mean-improvement companion is the chained-delta.")
    OUT_SUM.write_text("\n".join(L), encoding="utf-8")
    log(f"wrote {OUT_CSV}, {OUT_SUM}")


if __name__ == "__main__":
    main()
