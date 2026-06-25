# Mesoball

Replication code for **"Mesoball"** (working title), an omnibus cliometrics study that
treats Major League Baseball as a complete labor market and revisits a set of
baseball-economics results that were originally established on aggregate data. The
analysis runs on **15,374,332 plate appearances, 1910–2025** (Retrosheet, AL+NL,
regular season), and re-estimates each result at the level of the individual plate
appearance.

- **Author:** Aaron Brown
- **Status:** manuscript submitted
---

## What the paper does

"Meso" here means *micro-complete but flat-graph*: plate appearances are linked only
by season, hitter, pitcher, and ballpark — there is no inning, game, team, or league
hierarchy imposed. Every case study is carried by **one estimator end to end**, the
**censored-share Tobit**: the dependent variable is a player's share of his team's
plate appearances, right-censored at `c = 0.90 × full-time` (full-time ≈ the 99th
percentile share); batter-clustered standard errors via a BHHH score sandwich; linear
combinations via `lincom`.

The five case studies revisit, confirm, redefine, or reject:

1. **Gould on the .400 hitter** — has the spread of batting talent compressed?
2. **Palmer–Woolner offensive positional adjustment** — the positional premia, via
   playing-time allocation rather than offense alone.
3. **Batting aging curves** — survivor-bias-free, on the censored share.
4. **The designated hitter and moral hazard** — the GST result and its retraction.
5. **Situational performance** — the hot hand and clutch hitting.

A fan-facing online **supplement** adds ballpark factors, pitcher-handedness (platoon)
loadings, the season run-environment index μ_S, and player allocation-residual
leaderboards.

---

## Data

This repository contains **code and small reference tables only**. The large inputs
are not redistributed here:

| Input | Source | In repo? |
|---|---|---|
| Play-by-play, 1910–2025 | [Retrosheet](https://www.retrosheet.org) | no — download locally |
| `pa_panel.parquet` (15.4M PA) | built by `u_build_pa_panel.py` | no — regenerated locally |
| `leverage_panel.parquet` (15.76M rows) | built by `build_leverage_panel.py` | no — regenerated locally |
| Season wOBA weights | FanGraphs | **yes** — `wOBA_weights.csv` |
| Team season totals (AL/NL filter, Pythagenpat) | [Lahman](http://seanlahman.com) | **yes** — `Teams.csv` |

Negro Leagues are excluded for now (insufficient overlap with the modern era).
Intentional-walk PAs are excluded from the wOBA denominator (standard FanGraphs).

> **Retrosheet notice (required).** The information used here was obtained free of
> charge from and is copyrighted by Retrosheet. Interested parties may contact
> Retrosheet at https://www.retrosheet.org.

---

## Environment

Developed on Python 3.11+ (Windows, `py` launcher). Core dependencies:

```
pandas
numpy
pyarrow          # parquet I/O
scipy
pyfixest         # high-dimensional two-way fixed effects
matplotlib       # figures
pybaseball       # OPTIONAL: only for fetch_fielding_external.py (OAA validation)
```

See `requirements.txt`. Install with `pip install -r requirements.txt`.

---

## Reproducing the results

Paths are set near the top of `meso_core.py` (panel, reference tables, output dir).
Run the build once, in order, then any result script:

```text
# Build (run once)
1. u_build_pa_panel.py        ->  pa_panel.parquet         (from raw Retrosheet)
2. build_leverage_panel.py    ->  leverage_panel.parquet   (for §7 situational)
   u_build_cohort_cube.py     ->  cohort cube              (for the era cross-check)

# Then each case study reads the panel(s) and writes its CSV/figure outputs.
```

The script → case-study map below tells you which file produces each exhibit.

---

## Where each result comes from

| Case study | Primary scripts | Key outputs |
|---|---|---|
| **Talent dispersion / .400 (Gould)** | `talent_dispersion.py`, `pitcher_dispersion.py`, `dispersion_enriched.py`, `reconcile_dispersion.py`, `cohort_menu_test.py`, `gap_vs_imbalance.py`, `ba_skill_corr.py` | `talent_dispersion.csv`, `pitcher_dispersion.csv`, `dispersion_enriched.csv`, `reconcile_dispersion.csv` |
| **Two-way skill model / μ_S** | `two_way_pyfixest.py`, `two_way_shape_norm.py`, `two_way_fixed_weights.py` | `mu_three_schemes.csv`, `mu_seasonw_vs_fixedw.csv` |
| **Positional premia (Palmer–Woolner)** | `share_tobit_positional.py`, `share_tobit_aux.py`, `pa_allocation_reduced_form.py`, `validate_fielding.py`, `fetch_fielding_external.py` | `pa_allocation_pos_age.csv`, `pa_allocation_pos_era.csv`, `contact_fielding_test_coef.csv`, `expansion_fit_coef.csv` |
| **Aging curves** | `aging_curve.py`, `aging_censored.py`, `aging_effective.py`, `batting_curves_by_position.py` | aging-curve premia (feeds the supplement age-curve figure) |
| **DH / moral hazard** | `dh_hbp_moral_hazard.py`, `dh_age_curve.py`, `dh_age_distribution.py`, `dh_null_bridge.py`, `dh_split_table.py` | `dh_allocation_test_coef.csv` |
| **Situational (hot hand, clutch)** | `hot_hand.py`, `clutch.py`, `streak_clutch_tails.py`; built on `build_leverage_panel.py` (+ `check_leverage_merge.py`) | `streak_clutch_tails.txt` |
| **Ballparks (supplement)** | `ballpark_residuals.py` → `ballpark_allocation_stage2.py` | `park_profile.csv` |
| **Pitcher handedness (supplement)** | `hand_split_skill.py` → `platoon_allocation_tobit.py` | platoon loadings by decade |
| **Residual leaderboard (supplement)** | `residual_leaderboard.py` | `residual_leaderboard.csv` |
| **Coverage / units / pins** | `coverage_diagnostic.py`, `pin_numbers.py`, `eight_series_gap1.py`, `u_chained_delta_seasons.py` | `coverage_diagnostic.csv` |

A full annotated inventory of every file is in [`MANIFEST.md`](MANIFEST.md).

---

## Shared modules

- **`meso_core.py`** — imported almost everywhere (`import meso_core as mc`). Holds the
  Tobit core (`fit_tobit`, BHHH clustered SEs, `lincom`, restricted-cubic-spline basis),
  the units layer (`runs_per_win_by_season()` via Pythagenpat `runs_per_win = 2·RPG^0.713`,
  `woba_to_runs`, `runs_to_wins`), and project constants/paths.
- **`fit_two_way.py`** — the two-way (batter × pitcher) fixed-effects skill fit used by
  the dispersion scripts.
- **`talent_dispersion.py`** — doubles as an importable helper for the companion
  dispersion scripts.

---

## Conventions

- **One estimator** (censored-share Tobit) for every headline result.
- **`u_` prefix** marks published-pipeline scripts; unprefixed = working/exploratory.
- **First-person singular** in the manuscript (author preference; Cliometrica-compliant).
- **Units:** wOBA differences reported in runs, then converted to wins in parentheticals
  via each season's own Pythagenpat runs-per-win.
- **Nulls reported as fully as confirmations.**

---

## Citation

A `CITATION.cff` will be added on acceptance. Until then, please cite the manuscript
(Brown, *Mesoball*, in preparation) and, for the underlying data,
Retrosheet (notice above) and the Lahman Baseball Database.

## License

MIT
