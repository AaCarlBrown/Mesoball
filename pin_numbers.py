"""
pin_numbers.py
==============

Pins the three numbers we need before drafting, in one pass:

1. tau_bat SD  -- so the bat coefficient is reported in SD units, not the
   arbitrary "+0.100" anchor that made +299 PA look alarming.
2. Panel headline counts -- PAs, seasons, distinct batters/pitchers,
   player-seasons (for the abstract and the data section).
3. Fielding premium -> runs -> wins -- convert each position's PA premium
   (relative to corner OF) into win-equivalents via bat-equivalent wOBA.

Conversion logic:
   premium_pos (PA)  ->  bat-equivalent wOBA = premium_pos / beta
   (beta = PA per unit tau_bat; teams tolerate this much less bat for the glove)
   runs = (wOBA_equiv / wOBAScale) * PA_FULL
   wins = runs / (R per win)
This bundles the positional adjustment AND fielding into one "why a weak bat
plays here" premium, relative to a corner outfielder -- which is the object
the allocation method actually recovers.

Reads pa_allocation_panel.parquet and pa_panel.parquet. Requires pyfixest.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyfixest as pf

PROJECT_DIR = Path(r"C:\baseball_eras")
ALLOC = PROJECT_DIR / "data" / "pa_allocation_panel.parquet"
PAPANEL = PROJECT_DIR / "data" / "pa_panel.parquet"
WOBA_CSV = PROJECT_DIR / "wOBA_weights.csv"
OUT = PROJECT_DIR / "output" / "pin_numbers_summary.txt"

REF_POS = "CORNER_OF"
POSITIONS = ["C", "SS", "2B", "3B", "CF", "1B", "DH"]
PA_FULL = 600          # a full-time season of plate appearances
MIN_SEASONS = 2        # "regular": tau_bat SD computed on this population only
MIN_CAREER_PA = 300
L = []
def P(s=""): print(s); L.append(s)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # recent wOBAScale and R/W (runs per win)
    w = pd.read_csv(WOBA_CSV, encoding="utf-8-sig").rename(columns={"Season": "season"})
    wr = w.sort_values("season").iloc[-1]
    woba_scale = float(wr["wOBAScale"]); r_per_w = float(wr["R/W"])

    # ---- allocation panel: tau_bat SD + level position premia ----
    ps = pd.read_parquet(ALLOC)
    ps["pos"] = pd.Categorical(ps["pos"].astype(str),
                               categories=[REF_POS] + POSITIONS)
    ps["season"] = ps["season"].astype(int)
    # tau_bat is a player attribute; compute its SD at PLAYER level on regulars
    # (>=2 seasons, >=300 career PA), not across all noisy short-stint seasons.
    g = ps.groupby("batter").agg(tau=("tau_bat", "first"),
                                 nseas=("season", "nunique"), cpa=("pa", "sum"))
    reg = g[(g["nseas"] >= MIN_SEASONS) & (g["cpa"] >= MIN_CAREER_PA)]
    tau_sd = reg["tau"].std()
    tau_sd_all = ps.drop_duplicates("batter")["tau_bat"].std()

    m = pf.feols("pa ~ tau_bat + C(pos) | season", data=ps)
    co = m.coef()
    beta = float(co["tau_bat"])

    P("PRE-DRAFT NUMBER PINS")
    P("=" * 60)
    P(f"wOBAScale {woba_scale:.3f}, runs/win {r_per_w:.2f} (season {int(wr['season'])})")
    P("")
    P("--- 1. bat coefficient, honestly scaled ---")
    P(f"  regulars: {len(reg):,} players (>={MIN_SEASONS} seasons, >={MIN_CAREER_PA} career PA)")
    P(f"  tau_bat SD, regulars (player-level): {tau_sd:.4f} wOBA")
    P(f"  tau_bat SD, all players (player-level): {tau_sd_all:.4f} wOBA  (noisier, for ref)")
    P(f"  beta (PA per 1.0 tau_bat): {beta:.0f}")
    P(f"  => +1 SD of career bat -> +{beta*tau_sd:.0f} PA  (the number to report)")
    P("")

    P("--- 3. fielding premium -> runs -> wins (relative to corner OF) ---")
    P("  premium in PA, then bat-equivalent wOBA, runs, and wins per season:")
    P(f"  {'pos':>6s} {'PA prem':>8s} {'wOBA-eq':>8s} {'runs':>7s} {'wins':>6s}")
    rows = []
    for p in POSITIONS:
        key = f"C(pos)[T.{p}]"
        if key not in co.index:
            continue
        prem = float(co[key])
        woba_eq = prem / beta
        runs = woba_eq / woba_scale * PA_FULL
        wins = runs / r_per_w
        P(f"  {p:>6s} {prem:>8.0f} {woba_eq:>8.3f} {runs:>7.1f} {wins:>6.2f}")
        rows.append({"pos": p, "pa_premium": prem, "woba_equiv": woba_eq,
                     "runs": runs, "wins": wins})
    P(f"  (PA_FULL = {PA_FULL}; premium is positional adj + fielding combined,")
    P("   measured as bat teams forgo to play the position over a corner OF.)")
    P("")

    # ---- PA panel: headline counts ----
    pp = pd.read_parquet(PAPANEL, columns=["bat_id", "pit_id", "season"])
    P("--- 2. panel headline counts ---")
    P(f"  plate appearances:     {len(pp):,}")
    P(f"  seasons:               {pp['season'].min()}-{pp['season'].max()}")
    P(f"  distinct batters:      {pp['bat_id'].nunique():,}")
    P(f"  distinct pitchers:     {pp['pit_id'].nunique():,}")
    P(f"  batter-seasons:        {pp.groupby(['bat_id','season']).ngroups:,}")
    P(f"  allocation panel rows: {len(ps):,} player-seasons "
      f"({ps['season'].min()}-{ps['season'].max()})")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
