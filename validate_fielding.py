"""
validate_fielding.py  (exploratory)
==================================

Does the allocation residual contain fielding value (OAA) the model never saw?
Regresses the residual on OAA, controlling for baserunning, at BOTH the
player-season level and the career level (career averages out single-season
noise on both sides and is the cleaner test). Baserunning control is FanGraphs
BsR if present in fielding_external.csv, else Statcast sprint speed.

The glove is identified where it moves playing time, so the sub-full-time margin
is the sharper test. Read both the t-stat (is the signal there?) AND the R2 (how
much of the residual is fielding?).

Reads residual_by_season.parquet and fielding_external.csv. No network.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import meso_core as mc

RESID = mc.DATA_DIR / "residual_by_season.parquet"
EXT   = mc.DATA_DIR / "fielding_external.csv"
L = []
def P(s=""): print(s); L.append(s)


def ols_cluster(y, X, names, cluster=None):
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    if cluster is None:
        s2 = (e @ e) / (len(y) - X.shape[1]); V = s2 * XtX_inv
    else:
        S = pd.DataFrame(X * e[:, None]).groupby(np.asarray(cluster)).sum().to_numpy()
        V = XtX_inv @ (S.T @ S) @ XtX_inv
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / (((y - y.mean()) ** 2).sum())
    return {"b": dict(zip(names, beta)), "se": dict(zip(names, se)),
            "t": dict(zip(names, beta / se)), "r2": r2, "n": len(y)}


def zscore(s):
    return (s - s.mean()) / s.std()


def main():
    r = pd.read_parquet(RESID)
    r["batter"] = r["batter"].astype(str); r["season"] = r["season"].astype(int)
    ext = pd.read_csv(EXT)
    ext["batter"] = ext["batter"].astype(str); ext["season"] = ext["season"].astype(int)
    full_ref = float(r["pa_share"].quantile(0.99)); c = mc.HEADLINE_FRAC * full_ref

    d = r.merge(ext, on=["batter", "season"], how="inner").dropna(subset=["oaa"])
    d["sub_full"] = d["pa_share"] < c
    brun = "bsr" if ("bsr" in d.columns and d["bsr"].notna().sum() > 50) else "sprint_speed"
    d = d.dropna(subset=[brun]) if brun in d.columns else d

    P("VALIDATE FIELDING  (does the allocation residual contain OAA?)")
    P("=" * 70)
    P(f"overlap: {len(d):,} player-seasons ({d['batter'].nunique():,} players, "
      f"{int(d['season'].min())}-{int(d['season'].max())}); baserunning control = {brun}")

    def report(df, cols, label, cluster):
        df = df.dropna(subset=["resid_fts"] + cols)
        X = np.column_stack([np.ones(len(df))] + [df[col].to_numpy(float) for col in cols])
        res = ols_cluster(df["resid_fts"].to_numpy(float), X, ["const"] + cols, cluster)
        b, se, t = res["b"]["oaa_z"], res["se"]["oaa_z"], res["t"]["oaa_z"]
        ex = f"; run {res['b']['run_z']:+.3f} (t {res['t']['run_z']:+.1f})" if "run_z" in cols else ""
        P(f"  {label:<36s} OAA/+1SD={b:+.4f} (t {t:+.1f})  R2={res['r2']:.3f}  n={res['n']:,}{ex}")
        return res["r2"], t

    # ---- season level ----
    d["oaa_z"] = zscore(d["oaa"]); d["run_z"] = zscore(d[brun])
    P("\n--- season level (player-clustered) ---")
    report(d, ["oaa_z"], "all, OAA only", d["batter"].to_numpy())
    report(d, ["oaa_z", "run_z"], "all, OAA + baserunning", d["batter"].to_numpy())
    report(d[d.sub_full], ["oaa_z", "run_z"], "sub-full-time, OAA + baserunning",
           d[d.sub_full]["batter"].to_numpy())

    # ---- career level (PA-weighted; averages out season noise) ----
    def wavg(g, col):
        w = g["pa"].to_numpy(float); v = g[col].to_numpy(float); m = ~np.isnan(v)
        return np.average(v[m], weights=w[m]) if m.any() else np.nan
    car = d.groupby("batter").apply(lambda g: pd.Series({
        "resid_fts": wavg(g, "resid_fts"), "oaa": wavg(g, "oaa"),
        brun: wavg(g, brun), "pa": g["pa"].sum(),
        "sub_frac": g["sub_full"].mean()}), include_groups=False).reset_index()
    car["oaa_z"] = zscore(car["oaa"]); car["run_z"] = zscore(car[brun])
    P(f"\n--- career level ({len(car):,} players, PA-weighted) ---")
    report(car, ["oaa_z"], "career, OAA only", None)
    r2_c, t_c = report(car, ["oaa_z", "run_z"], "career, OAA + baserunning", None)
    carsub = car[car["sub_frac"] > 0.5]
    r2_cs, t_cs = report(carsub, ["oaa_z", "run_z"], "career sub-full-time, OAA + baserunning", None)

    P("")
    P(f"READ: t-stat says the fielding signal is present; R2 says how much of the")
    P(f"residual it is. Career sub-full-time R2 = {r2_cs:.2f} is the headline number for")
    P(f"'what share of value-beyond-the-bat is the glove.'")
    (mc.OUTPUT_DIR / "validate_fielding_summary.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
