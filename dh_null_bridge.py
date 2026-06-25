"""
dh_null_bridge.py  (exploratory robustness)
==========================================

Shows the DH "compression" was a baseline artifact, and that the clean
within-modern DH effect is null under either way of controlling for time.

Three specifications of the position-by-DH effect, all censored-share Tobit,
player-clustered, on the same allocation panel:

  S1  FULL PANEL, dh_regime:pos + season FE       -- attribution's design. The
      pos main effect pools all eras (1910-2025), so dh_regime:pos compares the
      DH regime to an all-era baseline that still holds the high-premium pre-1973
      seasons. Expected to REPRODUCE the spurious ~-17 at shortstop.
  S2  MODERN ONLY (season>=1973), dh_regime:pos + season FE  -- now the baseline
      is the modern game, so dh_regime:pos is the within-modern AL-vs-NL (and
      post-2022) contrast. Expected ~0.
  S3  THREE-CELL (post_DH - post_noDH) + season FE  -- the dh_split_table contrast
      with full season FE instead of a smooth trend. Expected ~0.

If S2 ~ S3 ~ 0 while S1 ~ -17, the apparent compression is pre-1973 contamination
of the pooled baseline, and the DH leaves the positional schedule unmoved.

Reads pa_allocation_panel.parquet + unified pa_team_share.parquet. Requires scipy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import meso_core as mc

GLOVE = mc.GLOVE
L = []
def P(s=""): print(s); L.append(s)


def dh_regime(season, league):
    s = np.asarray(season); lg = np.asarray(league)
    return (((lg == "AL") & (s >= 1973) & (s <= 2021)) | (s >= 2022)).astype(float)


def fit_dhreg(ps, SC):
    """dh_regime:pos design; returns {pos: (PA-equiv, SE, t)} for GLOVE."""
    posv = ps["pos"].astype(str).to_numpy()
    seas = ps["season"].to_numpy()
    reg = dh_regime(seas, ps["league"].astype(str).to_numpy())
    age_c = ps["age_c"].to_numpy(float)
    seasons = sorted(ps["season"].unique())
    base = [("const", np.ones(len(ps))), ("tau_bat", ps["tau_bat"].to_numpy(float)),
            ("age_c", age_c), ("age_c2", age_c ** 2)]
    pos_cols = [(f"pos_{p}", (posv == p).astype(float)) for p in mc.POS]
    dh_int = [(f"x_dh_{p}", reg * (posv == p)) for p in GLOVE]
    yr = [(f"yr_{s}", (ps["season"] == s).to_numpy(float)) for s in seasons[1:]]
    X, names = mc.design(base + pos_cols + dh_int + yr)
    res, V, cf = mc.fit_tobit(ps["pa_share"].to_numpy(float), X, ps["_c"].iloc[0],
                              ps["batter"].to_numpy())
    out = {}
    for p in GLOVE:
        e, se = mc.lincom({f"x_dh_{p}": 1.0}, names, res.x, V)
        out[p] = (e * SC, se * SC, e / se)
    return out, cf


def fit_threecell(ps, SC):
    """three-cell (pre1973/post_noDH/post_DH) with season FE; DH effect per pos."""
    posv = ps["pos"].astype(str).to_numpy()
    seas = ps["season"].to_numpy(); lg = ps["league"].astype(str).to_numpy()
    cell = np.where(seas <= 1972, "pre1973",
           np.where((seas >= 2022) | (lg == "AL"), "post_DH",
           np.where(lg == "NL", "post_noDH", "drop")))
    keep = cell != "drop"
    ps = ps[keep]; posv = posv[keep]; cell = cell[keep]; seas = seas[keep]
    age_c = ps["age_c"].to_numpy(float)
    seasons = sorted(pd.unique(seas))
    base = [("const", np.ones(len(ps))), ("tau_bat", ps["tau_bat"].to_numpy(float)),
            ("age_c", age_c), ("age_c2", age_c ** 2)]
    cell_main = [(f"cell_{c}", (cell == c).astype(float)) for c in ["post_noDH", "post_DH"]]
    cellpos = []
    for p in mc.POS:        # include DH, so it is absorbed by its own term, not the corner-OF reference
        for c in ["pre1973", "post_noDH", "post_DH"]:
            col = ((posv == p) & (cell == c)).astype(float)
            if col.sum() >= 30:
                cellpos.append((f"x_{p}_{c}", col))
    yr = [(f"yr_{s}", (seas == s).astype(float)) for s in seasons[1:]]
    X, names = mc.design(base + cell_main + cellpos + yr)
    res, V, cf = mc.fit_tobit(ps["pa_share"].to_numpy(float), X, ps["_c"].iloc[0],
                              ps["batter"].to_numpy())
    out = {}
    for p in GLOVE:
        if f"x_{p}_post_DH" in names and f"x_{p}_post_noDH" in names:
            e, se = mc.lincom({f"x_{p}_post_DH": 1.0, f"x_{p}_post_noDH": -1.0},
                              names, res.x, V)
            out[p] = (e * SC, se * SC, e / se)
        else:
            out[p] = (np.nan, np.nan, np.nan)
    return out, cf


def main():
    share = pd.read_parquet(mc.SHARE_CACHE)
    share["batter"] = share["batter"].astype(str); share["season"] = share["season"].astype(int)
    SC = float(share["pa_team_mean"].iloc[0])
    ps = pd.read_parquet(mc.PANEL_ALLOC)
    ps["batter"] = ps["batter"].astype(str); ps["season"] = ps["season"].astype(int)
    ps = ps.merge(share[["batter", "season", "pa_share", "league"]],
                  on=["batter", "season"], how="inner").reset_index(drop=True)
    ps["league"] = ps["league"].astype(str)
    full_ref = float(ps["pa_share"].quantile(0.99)); ps["_c"] = mc.HEADLINE_FRAC * full_ref

    P("DH NULL BRIDGE  (is the DH compression real, or a pre-1973 baseline artifact?)")
    P("=" * 76)
    s1, c1 = fit_dhreg(ps, SC)
    s2, c2 = fit_dhreg(ps[ps["season"] >= 1973].copy(), SC)
    s3, c3 = fit_threecell(ps, SC)

    P(f"  DH effect on each glove position, PA-equivalent (est / SE / t)")
    P(f"  {'pos':>4s} | {'S1 full-panel dh_reg':>22s} | {'S2 modern-only dh_reg':>22s} "
      f"| {'S3 three-cell +seasonFE':>24s}")
    def fmt(d, p):
        e, se, t = d[p]
        return f"{e:>+6.0f} ({se:>3.0f}, t {t:>+4.1f})" if not np.isnan(e) else f"{'--':>18s}"
    for p in GLOVE:
        P(f"  {p:>4s} | {fmt(s1,p):>22s} | {fmt(s2,p):>22s} | {fmt(s3,p):>24s}")
    P(f"\n  censor fractions: S1 {c1*100:.1f}%  S2 {c2*100:.1f}%  S3 {c3*100:.1f}%")
    P("\n  Read SS across the row: if S1 is strongly negative while S2 and S3 are ~0,")
    P("  the 'compression' was the pooled all-era baseline (pre-1973 high premia),")
    P("  not the DH. The clean within-modern contrast (S2, S3) is the one to quote.")
    (mc.OUTPUT_DIR / "dh_null_bridge_summary.txt").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
