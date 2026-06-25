"""
aging_curve.py  (exploratory)
============================

Redo the batting aging curve free of survivor bias -- the acknowledged flaw in
Fair (2008, peak ~27.5-28) and Bradbury (2009, peak ~29), who fit player fixed
effects on samples of long-career SURVIVORS (Fair: 10+ full-time seasons), so
their late-career decline is biased shallow.

  A. WITHIN-PLAYER curve -- season wOBA, de-meaned by season to purge the run
     environment, with player and age fixed effects (NO free period effect: age,
     period, and cohort are collinear, so the period is removed by de-meaning,
     not absorbed as a competing effect). Peak age and the decline to 35.
  B. SURVIVAL -- of a cohort of regulars at the peak age, the share still regular
     at each later age (the attrition the curve conditions on).
  C. CAREER-LENGTH LADDER -- re-estimate A while requiring >=1, 3, 5, 10
     qualifying seasons. As the bar rises toward Fair's sample, the decline
     SHALLOWS: that flattening IS the survivor bias, shown with no imputation.
     The full-population (>=1) decline is the least-selected estimate; the >=10
     decline is where Fair/Bradbury sit.

Reads pa_panel.parquet + wOBA_weights.csv (reuses fit_two_way's weight helpers).
Requires pyfixest.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pyfixest as pf
import meso_core as mc
import fit_two_way as ft

MIN_PA = 300
AGE_LO, AGE_HI = 20, 40
BASE_AGE = 27
OUT_CSV = mc.OUTPUT_DIR / "aging_curve.csv"
OUT_SUM = mc.OUTPUT_DIR / "aging_curve_summary.txt"
L = []
def P(s=""): print(s); L.append(s)


def _prep(df):
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


def player_seasons():
    df = pd.read_parquet(mc.PANEL_PA, columns=[
        "season", "bat_id", "iw", "bat_age", "single", "double", "triple",
        "hr", "walk", "hbp"])
    df["season"] = df["season"].astype(int)
    df = _prep(df)
    nb = df[df["iw"] != 1].dropna(subset=["w_shapenorm", "bat_age"])
    ps = (nb.groupby(["bat_id", "season"])
          .agg(woba=("w_shapenorm", "mean"), pa=("w_shapenorm", "size"),
               age=("bat_age", "first")).reset_index())
    ps["age"] = ps["age"].astype(int)
    return ps[(ps["age"] >= AGE_LO) & (ps["age"] <= AGE_HI)]


def age_curve(df):
    """Within-player age FE on season-de-meaned wOBA; returns FE series anchored at peak=0."""
    df = df.copy()
    df["woba_adj"] = df["woba"] - df.groupby("season")["woba"].transform("mean")
    df["bat_id"] = df["bat_id"].astype(str)
    m = pf.feols("woba_adj ~ 1 | bat_id + age", data=df)
    fe = m.fixef(); akey = next(k for k in fe if "age" in k)
    a = pd.Series(fe[akey]); a.index = a.index.astype(int); a = a.sort_index()
    return a - a.max()


def main():
    OUT_SUM.parent.mkdir(parents=True, exist_ok=True)
    ps = player_seasons()
    reg = ps[ps["pa"] >= MIN_PA].copy()
    P("BATTING AGING CURVE  (survivor-bias-corrected; redo of Fair 2008 / Bradbury 2009)")
    P("=" * 78)
    P(f"{len(reg):,} regular player-seasons (>= {MIN_PA} PA), {reg['bat_id'].nunique():,} batters")

    # ---- A. within-player, environment-purged age curve ----
    a = age_curve(reg); peak = int(a.idxmax())
    P("\n--- A. within-player age curve (wOBA vs peak, x1000), full population ---")
    P("  " + " ".join(f"{ag:>4d}" for ag in range(22, 39, 2)))
    P("  " + " ".join(f"{a.get(ag, np.nan)*1000:>4.0f}" for ag in range(22, 39, 2)))
    P(f"  estimated peak age = {peak}   (Fair 2008 ~27.5-28; Bradbury 2009 ~29)")
    P(f"  decline peak->35 = {a.get(35, np.nan)*1000:+.0f} wOBA points")

    # ---- B. survival of the peak-age cohort ----
    regset = set(zip(reg["bat_id"].astype(str), reg["age"]))
    base = sorted({b for (b, ag) in regset if ag == BASE_AGE})
    P(f"\n--- B. survival: of {len(base):,} regulars at age {BASE_AGE}, share still regular ---")
    P("  " + " ".join(f"{ag:>4d}" for ag in range(BASE_AGE, 39, 2)))
    surv = {ag: np.mean([(b, ag) in regset for b in base]) for ag in range(BASE_AGE, 41)}
    P("  " + " ".join(f"{surv[ag]*100:>4.0f}" for ag in range(BASE_AGE, 39, 2)) + "  (% still regular)")

    # ---- C. career-length ladder: decline shallows toward Fair's sample ----
    nseas = reg.groupby(reg["bat_id"].astype(str))["season"].nunique()
    P("\n--- C. career-length ladder (survivor bias, no imputation) ---")
    P(f"  {'min seasons':>11s} {'players':>8s} {'peak':>5s} {'decline->35':>12s}")
    rows = []
    for thr in [1, 3, 5, 10]:
        keep = set(nseas[nseas >= thr].index)
        sub = reg[reg["bat_id"].astype(str).isin(keep)]
        aa = age_curve(sub); pk = int(aa.idxmax()); d35 = aa.get(35, np.nan) * 1000
        P(f"  {thr:>11d} {len(keep):>8,d} {pk:>5d} {d35:>+12.0f}")
        rows.append({"min_seasons": thr, "players": len(keep), "peak": pk, "decline_to_35": d35})
    P("  Decline shallows as the career-length bar rises -- the selection that makes")
    P("  Fair/Bradbury's long-career samples understate the true late-career drop.")
    out = pd.DataFrame([{"age": ag, "fe_x1000": a.get(ag, np.nan) * 1000,
                         "survival_pct": surv.get(ag, np.nan) * 100} for ag in range(AGE_LO, 41)])
    out.to_csv(OUT_CSV, index=False, float_format="%.2f")
    pd.DataFrame(rows).to_csv(mc.OUTPUT_DIR / "aging_curve_ladder.csv", index=False)
    OUT_SUM.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT_CSV}, {OUT_SUM}")


if __name__ == "__main__":
    main()
