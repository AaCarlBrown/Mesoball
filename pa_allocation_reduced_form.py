"""
pa_allocation_reduced_form.py
=============================

Reduced-form test of the fielding-via-PA-allocation hypothesis.

Idea: playing time is allocated to TOTAL value = bat + glove. We observe
bat and playing time but not glove. At matched bat value, the EXTRA
playing time a glove-position player gets (vs a bat-first player) is the
fielding value being priced in. Tracking that premium across AGE traces
the fielding age curve; across ERA traces the era price of fielding.

Specification (OLS, player-season level):
    PA_{i,a} = beta * wOBA_{i,a-1}
             + gamma_pos                 (positional PA premium, level)
             + delta_{pos x age}         (FIELDING AGE CURVE)
             + zeta_{pos x era}          (ERA PRICE OF FIELDING)
             + season FE                 (absorb league-wide PA drift)
             + eps

We use PRIOR-season wOBA as the bat regressor to break the within-season
PA<->wOBA simultaneity (bad hitters get benched mid-year).

Read off:
  - delta_{pos x age}: for each position, how the PA premium changes with
    age relative to a reference age. If glove positions lose PA premium
    with age (their fielding erodes), delta is negative and growing for
    SS/C/CF; flat for 1B/DH.
  - zeta_{pos x era}: for each position, how the PA premium differs by era.
    The integration hypothesis predicts the glove premium is largest
    (most negative bat tolerated) in the expansion era and compresses
    afterward.

Data: retrosheet (need per-PA position to assign primary position, and
PA counts). We build a (player, season) table:
    primary_pos, age, era, pa (this season), woba_prev (last season).

Outputs:
    C:\\baseball_eras\\data\\pa_allocation_panel.parquet
    C:\\baseball_eras\\output\\pa_allocation_pos_age.csv     (delta surface)
    C:\\baseball_eras\\output\\pa_allocation_pos_era.csv      (zeta surface)
    C:\\baseball_eras\\output\\pa_allocation_reduced_form_summary.txt

Requires: pip install pyfixest
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf


PROJECT_DIR = Path(r"C:\baseball_eras")
RETRO_DIR   = Path(r"C:\overnight_effect_data\retrosheet")

PLAYS_CSV    = RETRO_DIR / "plays.csv"
GAMEINFO_CSV = RETRO_DIR / "gameinfo.csv"
BIOFILE_CSV  = RETRO_DIR / "biofile0.csv"
WOBA_CSV     = PROJECT_DIR / "wOBA_weights.csv"

OUT_PANEL = PROJECT_DIR / "data"   / "pa_allocation_panel.parquet"
OUT_PA    = PROJECT_DIR / "output" / "pa_allocation_pos_age.csv"
OUT_PE    = PROJECT_DIR / "output" / "pa_allocation_pos_era.csv"
OUT_SUM   = PROJECT_DIR / "output" / "pa_allocation_reduced_form_summary.txt"

AGE_LO, AGE_HI = 21, 40
REF_AGE = 27
REF_POS = "CORNER_OF"   # reference position (bat-first, mid-spectrum)

POS_GROUP = {
    1: "P", 2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS",
    7: "CORNER_OF", 8: "CF", 9: "CORNER_OF", 10: "DH",
}
ERAS = [
    ("pre_integration", 1910, 1946),
    ("integration",     1947, 1968),
    ("expansion_FA",    1969, 1993),
    ("steroid",         1994, 2005),
    ("modern",          2006, 2025),
]
WEIGHT_COLS = ["wBB", "wHBP", "w1B", "w2B", "w3B", "wHR"]


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
def bage(bd, s):
    by = bd//10000; bm=(bd//100)%100; bdd=bd%100
    a = s-by; after=(bm>6)|((bm==6)&(bdd>30)); return a-after.astype(int)
def bage_exact(bd, s):
    # Exact (fractional) age as of June 30 of the season, used for the age-curve spline.
    # Reconstructed: the original builder that wrote this column was lost; the June-30
    # reference matches the integer baseball-age convention in bage() so the two agree.
    bd = np.asarray(bd, dtype="int64"); s = np.asarray(s, dtype="float64")
    by = bd//10000; bm = (bd//100)%100; bdd = bd%100
    cum = np.array([0,31,59,90,120,151,181,212,243,273,304,334])  # days before month (non-leap)
    doy = cum[np.clip(bm,1,12).astype(int)-1] + np.clip(bdd,1,31)
    return (s - by) + (181.0 - doy)/365.25
def era_of(s):
    for n,lo,hi in ERAS:
        if lo<=s<=hi: return n
    return "other"


def main():
    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_PA.parent.mkdir(parents=True, exist_ok=True)

    log("reading plays.csv")
    pcols=["gid","batter","bat_f","pa","single","double","triple","hr","walk","iw","hbp"]
    plays=pd.read_csv(PLAYS_CSV, usecols=lambda c:c in pcols,
        dtype={"gid":"string","batter":"string","bat_f":"Int16","pa":"Int8",
               "single":"Int8","double":"Int8","triple":"Int8","hr":"Int8",
               "walk":"Int8","iw":"Int8","hbp":"Int8"}, low_memory=False)
    plays=plays[plays["pa"]==1]
    log(f"  {len(plays):,} PA rows")

    gi=pd.read_csv(GAMEINFO_CSV, usecols=["gid","season","gametype"],
                   dtype={"gid":"string","season":"Int32","gametype":"string"})
    gi=gi[gi["gametype"]=="regular"]
    plays=plays.merge(gi[["gid","season"]], on="gid", how="inner")
    log(f"  {len(plays):,} regular-season PA rows")

    # wOBA per PA (season weights)
    w=pd.read_csv(WOBA_CSV, encoding="utf-8-sig").rename(columns={"Season":"season"})
    plays=plays.merge(w[["season"]+WEIGHT_COLS], on="season", how="left")
    bb_u=(plays["walk"]-plays["iw"]).clip(lower=0)
    plays["wv"]=(plays["wBB"]*bb_u+plays["wHBP"]*plays["hbp"]+plays["w1B"]*plays["single"]
                 +plays["w2B"]*plays["double"]+plays["w3B"]*plays["triple"]+plays["wHR"]*plays["hr"])
    plays.loc[plays["iw"]==1,"wv"]=np.nan
    nonibb=plays[plays["iw"]!=1]

    # career-weighted primary position per player: map each PA's fielding code to its pos
    # GROUP first (so LF/RF aggregate into CORNER_OF), sum PA by group over the WHOLE career,
    # and take the modal group. This single career label is applied to every one of the
    # player's seasons -- positional identity is a career attribute, not something that
    # should be relabeled as an aging player slides off his position (a catcher's late
    # 1B/DH seasons stay charged to "catcher", where their playing-time cost belongs).
    log("career-weighted primary position per player")
    pg = plays.dropna(subset=["bat_f"]).copy()
    pg["pos"] = pg["bat_f"].map(POS_GROUP).fillna("UNK")
    cpos = pg.groupby(["batter", "pos"]).size().rename("n").reset_index()
    idx = cpos.groupby("batter")["n"].idxmax()
    primary = cpos.loc[idx, ["batter", "pos"]]      # one career-modal pos group per batter

    # player-season aggregate: pa (ex-IBB), mean wOBA
    log("player-season aggregates")
    ps=(nonibb.groupby(["batter","season"])
            .agg(pa=("wv","size"), woba=("wv","mean")).reset_index())
    ps=ps.merge(primary, on="batter", how="left")
    ps["pos"]=ps["pos"].fillna("UNK")
    ps=ps[ps["pos"]!="P"]                      # exclude career pitchers
    ps=ps[ps["pos"]!="UNK"]

    # age
    bio=pd.read_csv(BIOFILE_CSV, usecols=["id","birthdate"],
                    dtype={"id":"string","birthdate":"Int64"}).rename(
                    columns={"id":"batter","birthdate":"bd"})
    ps=ps.merge(bio, on="batter", how="left").dropna(subset=["bd"])
    ps["age"]=bage(ps["bd"].astype("int64"), ps["season"].astype("int64"))
    ps["age_exact"]=bage_exact(ps["bd"].astype("int64"), ps["season"].astype("int64"))
    ps=ps[(ps["age"]>=AGE_LO)&(ps["age"]<=AGE_HI)]
    ps["era"]=ps["season"].astype(int).map(era_of)

    # --- tau_bat: career bat level, net of season and age ---
    # Fit woba ~ 1 | batter + season + age on player-seasons, PA-weighted.
    # The batter fixed effect is the era/age-neutral career bat level.
    log("computing tau_bat (batter FE, PA-weighted, season+age controls)")
    ps_tau = ps.dropna(subset=["woba"]).copy()
    ps_tau["batter_s"] = ps_tau["batter"].astype(str)
    ps_tau["season_i"] = ps_tau["season"].astype(int)
    ps_tau["age_i"] = ps_tau["age"].astype(int)
    m_tau = pf.feols("woba ~ 1 | batter_s + season_i + age_i",
                     data=ps_tau, weights="pa")
    fe_tau = m_tau.fixef()
    bkey = next(k for k in fe_tau if "batter_s" in k)
    tau = pd.Series(fe_tau[bkey], name="tau_bat")
    tau.index = tau.index.astype(str)
    # mean-anchor
    tau = tau - tau.mean()
    ps["tau_bat"] = ps["batter"].astype(str).map(tau)
    log(f"  tau_bat computed for {ps['tau_bat'].notna().sum():,} player-seasons")

    # require a prior year of existence (drop rookie seasons) so PA is
    # predicted for established players; bat measure is tau_bat (career).
    log("requiring a prior season of existence")
    prev=ps[["batter","season"]].copy()
    prev["season"]=prev["season"]+1
    prev["had_prev"]=True
    ps=ps.merge(prev, on=["batter","season"], how="left")
    ps=ps[ps["had_prev"]==True]
    ps=ps.dropna(subset=["tau_bat"])
    log(f"  {len(ps):,} player-seasons with a prior year and tau_bat")

    # Center regressors / set reference levels
    ps["age_c"]=ps["age"]-REF_AGE
    ps["pos"]=pd.Categorical(ps["pos"],
        categories=[REF_POS]+[p for p in ["C","SS","2B","3B","CF","1B","DH"] if p!=REF_POS])
    ps["era"]=pd.Categorical(ps["era"], categories=[e[0] for e in ERAS])
    ps["season"]=ps["season"].astype(int)

    ps.to_parquet(OUT_PANEL, index=False)
    log(f"wrote {OUT_PANEL}")

    # --- Regression 1: position x age (fielding age curve) ---
    # PA ~ woba_prev + C(pos)*C(age_bucket) + season FE
    # Use age as categorical in 2-year buckets to keep it readable.
    ps["age_b"]=(ps["age"]//2*2).astype(int)
    log("fitting position x age model")
    m1=pf.feols("pa ~ tau_bat + C(pos)*C(age_b) | season", data=ps)
    c1=m1.coef().reset_index()
    c1.columns=["term","coef"]
    c1.to_csv(OUT_PA, index=False, float_format="%.3f")
    log(f"wrote {OUT_PA}")

    # --- Regression 2: position x era (era price of fielding) ---
    log("fitting position x era model (no season FE; smooth season trend)")
    ps["season_c"] = ps["season"].astype(int) - 1970
    m2=pf.feols("pa ~ tau_bat + C(pos)*C(era) + age_c + I(age_c**2) "
                "+ season_c + I(season_c**2)", data=ps)
    c2=m2.coef().reset_index()
    c2.columns=["term","coef"]
    c2.to_csv(OUT_PE, index=False, float_format="%.3f")
    log(f"wrote {OUT_PE}")

    # --- Build readable surfaces ---
    lines=[]; P=lines.append
    P("PA-allocation reduced form: fielding revealed through playing time")
    P("="*70)
    P(f"Reference position = {REF_POS}, reference age = {REF_AGE}.")
    P(f"PA regressed on tau_bat (career bat level) + position interactions.")
    P(f"beta (PA per point of prior wOBA): see below.")
    P("")
    beta=c1[c1["term"]=="tau_bat"]["coef"]
    if len(beta):
        P(f"PA per unit tau_bat (pos x age model): {beta.iloc[0]:.1f}")
        P(f"  (so +0.100 career bat level -> +{beta.iloc[0]*0.1:.0f} PA)")
    P("")

    # Reconstruct PA premium by (pos, age) relative to reference cell.
    # pyfixest term names look like 'C(pos)[T.SS]', 'C(age_b)[T.30]',
    # 'C(pos)[T.SS]:C(age_b)[T.30]'. We sum main + interaction.
    def get(term):
        r=c1[c1["term"]==term]["coef"]
        return float(r.iloc[0]) if len(r) else 0.0

    positions=["C","SS","2B","3B","CF","CORNER_OF","1B","DH"]
    age_bs=sorted(ps["age_b"].unique())
    P("=== Position PA premium by age (relative to reference pos & age) ===")
    P("Positive = more PAs than reference at matched prior-year bat.")
    P("Decline with age within a glove position = fielding value eroding.")
    P("")
    P(f"{'pos':>10s}  " + "  ".join(f"{ab:>6d}" for ab in age_bs))
    for pos in positions:
        cells=[]
        for ab in age_bs:
            pos_main = 0.0 if pos==REF_POS else get(f"C(pos)[T.{pos}]")
            age_main = get(f"C(age_b)[T.{ab}]")
            inter = 0.0 if pos==REF_POS else get(f"C(pos)[T.{pos}]:C(age_b)[T.{ab}]")
            # premium of (pos,age) over (REF_POS, ref age bucket):
            # we want pos effect at this age = pos_main + inter (age_main is common)
            val = pos_main + inter
            cells.append(f"{val:>6.0f}")
        P(f"{pos:>10s}  " + "  ".join(cells))
    P("")
    P("(Read across a row: how that position's PA premium changes with age.)")
    P("")

    # pos x era from model 2
    def get2(term):
        r=c2[c2["term"]==term]["coef"]
        return float(r.iloc[0]) if len(r) else 0.0
    P("=== Position PA premium by era (relative to reference pos) ===")
    P("More negative glove-position premium in an era = steeper fielding")
    P("price (weaker bat tolerated for the glove).")
    P("")
    era_names=[e[0] for e in ERAS]
    P(f"{'pos':>10s}  " + "  ".join(f"{e[:9]:>9s}" for e in era_names))
    for pos in positions:
        cells=[]
        for en in era_names:
            pos_main = 0.0 if pos==REF_POS else get2(f"C(pos)[T.{pos}]")
            era_main = 0.0 if en==era_names[0] else get2(f"C(era)[T.{en}]")
            inter = 0.0
            if pos!=REF_POS and en!=era_names[0]:
                inter=get2(f"C(pos)[T.{pos}]:C(era)[T.{en}]")
            val=pos_main+inter
            cells.append(f"{val:>9.0f}")
        P(f"{pos:>10s}  " + "  ".join(cells))
    P("")
    P("(Read across a row: how that position's PA premium changes by era,")
    P(" holding prior-year bat and age fixed.)")

    OUT_SUM.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {OUT_SUM}")
    print("\n".join(lines))


if __name__=="__main__":
    main()
