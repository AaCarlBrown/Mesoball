"""
coverage_diagnostic.py  (exploratory; FOUNDATIONAL)
==================================================

How complete is the Retrosheet play-by-play panel, season by season? Every
era-comparison result depends on this, because incomplete early coverage inflates
apparent dispersion and can manufacture artifacts (e.g. a cliff where coverage
jumps to complete). This compares the panel against the COMPLETE season totals in
Teams.csv (Lahman), which are box-score-sourced and effectively 100%.

  game coverage = distinct games in the panel / scheduled games (sum of team G / 2)
  PA coverage   = plate appearances in the panel / expected PA (AB+BB+HBP+SF[+SH])

Reports both by season and decade, flags the first season of sustained >=99%
game coverage (the "complete-coverage era"), and lists the worst early seasons.
Era-comparison analyses should be restricted to, or explicitly caveated outside,
the complete window.

Reads pa_panel.parquet and Teams.csv. No network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import meso_core as mc

OUT_CSV = mc.OUTPUT_DIR / "coverage_diagnostic.csv"
OUT_SUM = mc.OUTPUT_DIR / "coverage_diagnostic_summary.txt"
COMPLETE = 0.99
L = []
def P(s=""): print(s); L.append(s)


def main():
    OUT_SUM.parent.mkdir(parents=True, exist_ok=True)

    # ---- panel: games and PA per season ----
    pan = pd.read_parquet(mc.PANEL_PA, columns=["game_id", "season"])
    pan["season"] = pan["season"].astype(int)
    g_panel = pan.groupby("season")["game_id"].nunique().rename("panel_games")
    pa_panel = pan.groupby("season").size().rename("panel_pa")

    # ---- Teams.csv: complete season totals (AL/NL) ----
    t = pd.read_csv(mc.TEAMS_CSV, encoding="utf-8-sig")
    t = t[t["lgID"].isin(["AL", "NL"])]
    pa_cols = [c for c in ["AB", "BB", "HBP", "SF", "SH"] if c in t.columns]
    t["exp_pa"] = t[pa_cols].fillna(0).sum(axis=1)
    by = t.groupby("yearID").agg(teams=("teamID", "nunique"),
                                 sched_games=("G", "sum"), exp_pa=("exp_pa", "sum"))
    by["sched_games"] = by["sched_games"] / 2.0          # each game counted by both teams
    by.index = by.index.astype(int)

    cov = by.join(g_panel).join(pa_panel).dropna(subset=["panel_games"])
    cov["game_cov"] = cov["panel_games"] / cov["sched_games"]
    cov["pa_cov"] = cov["panel_pa"] / cov["exp_pa"]
    cov = cov.reset_index().rename(columns={"index": "season", "yearID": "season"})
    cov["season"] = cov["season"].astype(int)
    cov.to_csv(OUT_CSV, index=False, float_format="%.4f")

    P("RETROSHEET COVERAGE DIAGNOSTIC  (panel vs complete Lahman/Teams totals)")
    P("=" * 72)
    P(f"PA expected from columns: {'+'.join(pa_cols)}  (SH absent => PA coverage runs slightly high)")

    # ---- by decade ----
    cov["decade"] = (cov["season"] // 10 * 10)
    dec = cov.groupby("decade").agg(seasons=("season", "size"),
                                    game_cov=("game_cov", "mean"),
                                    pa_cov=("pa_cov", "mean"),
                                    min_game_cov=("game_cov", "min"))
    P("\n--- coverage by decade ---")
    P(f"  {'decade':>6s} {'seasons':>7s} {'game_cov':>9s} {'pa_cov':>8s} {'min_game':>9s}")
    for d, r in dec.iterrows():
        P(f"  {int(d):>5d}s {int(r.seasons):>7d} {r.game_cov*100:>8.1f}% {r.pa_cov*100:>7.1f}% "
          f"{r.min_game_cov*100:>8.1f}%")

    # ---- first sustained complete season ----
    cov = cov.sort_values("season").reset_index(drop=True)
    complete_start = None
    for i in range(len(cov)):
        window = cov["game_cov"].iloc[i:i + 5]
        if len(window) >= 3 and (window >= COMPLETE).all():
            complete_start = int(cov["season"].iloc[i]); break
    P(f"\n--- completeness ---")
    P(f"  first season of sustained >= {COMPLETE*100:.0f}% game coverage: {complete_start}")
    P(f"  overall: {int((cov['game_cov'] >= COMPLETE).sum())} of {len(cov)} seasons "
      f"are >= {COMPLETE*100:.0f}% game-covered")
    worst = cov.nsmallest(10, "game_cov")[["season", "game_cov", "pa_cov"]]
    P("  worst-covered seasons (game_cov):")
    for r in worst.itertuples():
        P(f"    {int(r.season)}: games {r.game_cov*100:>5.1f}%, PA {r.pa_cov*100:>5.1f}%")

    P("\nIMPLICATION: era comparisons (dispersion, aging, premia by era) should be")
    P(f"restricted to or explicitly caveated before the complete window. A coverage")
    P("cliff coinciding with a result's break (e.g. the 1970s->80s dispersion drop)")
    P("is an artifact suspect until shown to survive within the complete window.")
    OUT_SUM.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT_CSV}, {OUT_SUM}")


if __name__ == "__main__":
    main()
