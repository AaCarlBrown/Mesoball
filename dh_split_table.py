"""
dh_split_table.py  (exploratory)
===============================

Re-cast Table 2 to isolate the designated hitter. Instead of five calendar eras,
estimate the positional premia in three cells defined by whether the DH rule was
in force, so the DH effect is read WITHIN the modern period rather than across a
century of confounded change:

  pre1973        1910-1972 (no DH anywhere) .......... reference baseline
  post_noDH      1973-2021 National League ........... DH rule NOT in force
  post_DH        1973-2021 American League + 2022-25 both leagues ... DH in force

The DH rule is assigned at the team-season level from the league rule that season
(interleague years 1997-2021 are approximated by the team's league; a handful of
games run under the other league's rule). Premia are relative to a corner
outfielder within each cell (censored-share Tobit, headline censoring, player-
clustered). The DH effect on each position is post_DH - post_noDH, reported with
its standard error; because both are taken relative to the corner-OF baseline,
this difference nets out any common shift and isolates the within-spectrum
compression.

Reads pa_allocation_panel.parquet and the unified pa_team_share.parquet (for
pa_share and league). Requires scipy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import meso_core as mc

CELLS = ["pre1973", "post_noDH", "post_DH"]
COLHEAD = {"pre1973": "1910-1972", "post_noDH": "1973-2025 no-DH",
           "post_DH": "1973-2025 DH"}
MINCELL = 30
L = []
def P(s=""): print(s); L.append(s)


def cell_of(season, league):
    if season <= 1972:
        return "pre1973"
    if season >= 2022:
        return "post_DH"
    if league == "AL":
        return "post_DH"
    if league == "NL":
        return "post_noDH"
    return None


def main():
    share = pd.read_parquet(mc.SHARE_CACHE)
    share["batter"] = share["batter"].astype(str); share["season"] = share["season"].astype(int)
    SC = float(share["pa_team_mean"].iloc[0])
    ps = pd.read_parquet(mc.PANEL_ALLOC)
    ps["batter"] = ps["batter"].astype(str); ps["season"] = ps["season"].astype(int)
    ps = ps.merge(share[["batter", "season", "pa_share", "league"]],
                  on=["batter", "season"], how="inner")
    ps["cell"] = [cell_of(s, lg) for s, lg in zip(ps["season"], ps["league"].astype(str))]
    ps = ps[ps["cell"].notna()].reset_index(drop=True)

    full_ref = float(ps["pa_share"].quantile(0.99)); c = mc.HEADLINE_FRAC * full_ref
    posv = ps["pos"].astype(str).to_numpy()
    cellv = ps["cell"].to_numpy()
    age_c = ps["age_c"].to_numpy(float)
    season_c = ps["season"].to_numpy(float) - 1970.0
    y = ps["pa_share"].to_numpy(float); cl = ps["batter"].to_numpy()

    P("DH-SPLIT POSITIONAL PREMIA  (censored-share Tobit, player-clustered)")
    P("=" * 74)
    for cl_name in CELLS:
        P(f"  {cl_name:>10s}: {int((cellv==cl_name).sum()):>6,d} player-seasons")

    base = [("const", np.ones(len(ps))), ("tau_bat", ps["tau_bat"].to_numpy(float))]
    cell_main = [(f"cell_{cl_}", (cellv == cl_).astype(float)) for cl_ in CELLS[1:]]
    cell_cols, have = [], {}
    for p in mc.POS:
        for cl_ in CELLS:
            col = ((posv == p) & (cellv == cl_)).astype(float)
            if col.sum() >= MINCELL:
                cell_cols.append((f"x_{p}_{cl_}", col)); have[(p, cl_)] = True
    ctrl = [("age_c", age_c), ("age_c2", age_c ** 2),
            ("season_c", season_c), ("season_c2", season_c ** 2)]
    X, names = mc.design(base + cell_main + cell_cols + ctrl)
    res, V, cf = mc.fit_tobit(y, X, c, cl)
    P(f"  fit: {cf*100:.1f}% censored\n")

    # ---- table: premia per cell + DH effect (post_DH - post_noDH) ----
    P(f"  {'Pos':>4s} {COLHEAD['pre1973']:>11s} {COLHEAD['post_noDH']:>15s} "
      f"{COLHEAD['post_DH']:>13s}   {'DH effect (DH - noDH)':>22s}")
    def prem(p, cl_):
        if (p, cl_) not in have:
            return None, None
        return mc.lincom({f"x_{p}_{cl_}": 1.0}, names, res.x, V)
    for p in mc.POS:
        row = []
        for cl_ in CELLS:
            e, se = prem(p, cl_)
            row.append(f"{e*SC:>+6.0f}({se*SC:>3.0f})" if e is not None else f"{'--':>11s}")
        # DH effect: post_DH - post_noDH, if both present
        if (p, "post_DH") in have and (p, "post_noDH") in have:
            d, sd = mc.lincom({f"x_{p}_post_DH": 1.0, f"x_{p}_post_noDH": -1.0},
                              names, res.x, V)
            de = f"{d*SC:>+7.0f} ({sd*SC:>3.0f}, t {d/sd:>+4.1f})"
        else:
            de = f"{'--':>22s}"
        P(f"  {p:>4s} {row[0]:>11s} {row[1]:>15s} {row[2]:>13s}   {de:>22s}")

    P("\nPremia are PA-equivalent vs a corner outfielder within each cell. The DH")
    P("effect (last column) is the post-1973 DH-minus-noDH difference, netting out")
    P("the common corner-OF baseline, so it isolates within-spectrum compression.")
    P("The DH row appears only under the rule; it has no no-DH counterpart.")
    (mc.OUTPUT_DIR / "dh_split_table_summary.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
