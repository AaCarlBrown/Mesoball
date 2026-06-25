"""
gap_vs_imbalance.py  (exploratory)
=================================

Confirms that the mid-century inflation of two-way batter-effect dispersion
(the M1 vs M2 gap in reconcile_dispersion) is opponent-pool disruption: when the
pitching pool is unequal across teams, the two-way model cannot cleanly separate
batter from pitcher, so pitcher variation leaks into the batter effects and
inflates their spread. If the gap tracks pitching imbalance by decade -- peaking
in the integration/expansion era -- the inflation is the unbalanced pool, not a
talent change.

gap        = SD(two-way FE, strict) - SD(season-mean, strict), per decade
imbalance  = decade-mean cross-team coefficient of variation of team ERA (AL/NL)

Also prints the clean headline series: the opponent-blind season-mean dispersion
(M2, strict), which is flat across the century.

Reads reconcile_dispersion.csv and Teams.csv.
"""
import sys
import numpy as np
import pandas as pd
import meso_core as mc

RECON = sys.argv[1] if len(sys.argv) > 1 else str(mc.OUTPUT_DIR / "reconcile_dispersion.csv")
TEAMS = sys.argv[2] if len(sys.argv) > 2 else str(mc.TEAMS_CSV)


def main():
    r = pd.read_csv(RECON)
    r["gap"] = r["M1_STRICT"] - r["M2_STRICT"]

    t = pd.read_csv(TEAMS, encoding="utf-8-sig")
    t = t[t["lgID"].isin(["AL", "NL"]) & t["ERA"].notna()]
    cv = (t.groupby("yearID")["ERA"].agg(lambda s: s.std() / s.mean()))
    cv = cv.reset_index(); cv["decade"] = cv["yearID"] // 10 * 10
    imb = cv.groupby("decade")["ERA"].mean().rename("era_cv").reset_index()

    d = r.merge(imb, on="decade", how="left")
    print("decade   M2_strict(flat)    FE-gap    ERA_cv(imbalance)")
    for x in d.itertuples():
        print(f"  {x.decade}s    {x.M2_STRICT:.4f}        {x.gap:.4f}    {x.era_cv:.3f}")
    rho = d["gap"].corr(d["era_cv"])
    rho_s = d["gap"].corr(d["era_cv"], method="spearman")
    print(f"\ncorr(FE-gap, pitching imbalance): Pearson {rho:+.2f}, Spearman {rho_s:+.2f}")
    print(f"M2 (opponent-blind) dispersion: {d['M2_STRICT'].min():.4f}-{d['M2_STRICT'].max():.4f} "
          f"across decades (flat); CV across decades = {d['M2_STRICT'].std()/d['M2_STRICT'].mean():.2f}")


if __name__ == "__main__":
    main()
