# MANIFEST

Annotated inventory of the Mesoball repository. The **Status** column is a
recommendation for what belongs in the published GitHub repo:

- **commit** — source the repo should contain
- **commit (small ref)** — a small reference table safe to commit
- **output** — a generated artifact (regenerable by running the code; commit only if
  you want results browsable without a local run — otherwise gitignore)
- **local only** — large or raw input, do not commit (gitignored)
- **not for repo** — manuscript drafts, handoffs, and OS duplicate files

Some shared modules (`meso_core.py`, `fit_two_way.py`) are imported by the scripts but
are not present in this snapshot; they live in the working tree and should be committed.

---

## Shared modules (commit)

| File | Purpose |
|---|---|
| `meso_core.py` *(expected; not in snapshot)* | Tobit core (`fit_tobit`, clustered SEs, `lincom`, spline basis), units layer (Pythagenpat runs/win, woba→runs→wins), constants/paths. Imported by 15 scripts. |
| `fit_two_way.py` *(expected; not in snapshot)* | Two-way batter×pitcher fixed-effects skill fit. Imported by the dispersion scripts. |

## Build pipeline — `u_` prefix (commit)

| File | Purpose |
|---|---|
| `u_build_pa_panel.py` | Build `pa_panel.parquet` (15.4M PA) from raw Retrosheet — ids, dates, ages, hands, `park_id`, `woba_value`, outcome flags. The root of everything. |
| `u_build_cohort_cube.py` | Aggregate the PA panel to (pitcher birth-year, batter birth-year, season) cells for the cohort/era cross-check. |
| `u_chained_delta_seasons.py` | Reproduce the *Wrong Number* ch.19 chained-delta era method in wOBA points (cross-check on the era result). |

> Also referenced in the handoffs but **not in this snapshot:** `u_chained_delta_lahman_access.txt` (the Access-SQL replication cross-check). Commit it alongside the above if you keep the era cross-check in the repo.

## Case-study analysis scripts (commit; most are `(exploratory)` working versions)

### Talent dispersion / the .400 hitter (Gould)
| File | Purpose |
|---|---|
| `talent_dispersion.py` | Spread of batting talent (τ) by decade — Gould's compression claim on a clean measure. Also an importable helper. |
| `pitcher_dispersion.py` | Pitching-skill dispersion by decade, as a companion/contrast to batting. |
| `dispersion_enriched.py` | Does conditioning τ on park + handedness change the dispersion? (≈0.97 corr with baseline.) |
| `reconcile_dispersion.py` | Reconciles the two dispersion measures that disagreed (M1 vs M2). |
| `cohort_menu_test.py` | Period (outcome-menu) vs cohort (more-uniform generations) decomposition. |
| `gap_vs_imbalance.py` | Confirms the mid-century dispersion bump is opponent-pool disruption. |
| `ba_skill_corr.py` | How well crude batting average tracks the purged skill measure (career vs season). |

### Two-way skill model and the run environment μ_S
| File | Purpose |
|---|---|
| `two_way_pyfixest.py` | The two-way (batter×pitcher) season model via pyfixest (alternating projection). |
| `two_way_shape_norm.py` | Shape-normalized weighting scheme — the μ_S headline series. |
| `two_way_fixed_weights.py` | Single fixed-weight robustness refit. |

### Positional premia (Palmer–Woolner)
| File | Purpose |
|---|---|
| `share_tobit_positional.py` | **Headline estimator** — censored-share Tobit, all three positional specifications. |
| `share_tobit_aux.py` | The three auxiliary tests (contact, expansion, DH null) on the same Tobit. |
| `pa_allocation_reduced_form.py` | Reduced-form test of fielding-via-PA-allocation at matched bat/age. |
| `validate_fielding.py` | Does the allocation residual contain external fielding value (OAA) the model never saw? |
| `fetch_fielding_external.py` | *(run at home; needs `pybaseball`)* Pull OAA and key to Retrosheet ids for validation. |

### Aging curves
| File | Purpose |
|---|---|
| `aging_curve.py` | Survivor-bias-free batting aging curve (vs Fair 2008 / Bradbury 2009). |
| `aging_censored.py` | The censored "effective" aging curve (share observed in [0, full-time]). |
| `aging_effective.py` | Selection-corrected PA-share aging curve. |
| `batting_curves_by_position.py` | Descriptive aging mixture by position (first step). |

### DH and moral hazard
| File | Purpose |
|---|---|
| `dh_hbp_moral_hazard.py` | Does shielding a pitcher from batting raise HBP, holding batter quality fixed? |
| `dh_age_curve.py` | Does the DH flatten the batter aging curve (via PA redistribution)? |
| `dh_age_distribution.py` | Descriptive: DH PAs skew old (verifies the mechanism's premise). |
| `dh_null_bridge.py` | Shows the DH "compression" was a baseline artifact; modern effect is null. |
| `dh_split_table.py` | Recast the era table into three DH-rule cells to isolate the DH. |

### Situational — hot hand and clutch
| File | Purpose |
|---|---|
| `build_leverage_panel.py` | *(overnight)* Build `leverage_panel.parquet` — per-PA inning/outs/base/score state. |
| `check_leverage_merge.py` | Verify the leverage panel joins to the PA panel by `(game_id, pa_seq)`. |
| `hot_hand.py` | Hot hand test (permutation; cross-player spread). |
| `clutch.py` | Clutch test (raw vs opponent-purged). |
| `streak_clutch_tails.py` | *(fan-facing exhibit)* Most/least streaky and clutch hitters (≥3000 PA) vs pure noise. |

### Supplement — ballparks, handedness, leaderboard
| File | Purpose |
|---|---|
| `ballpark_residuals.py` | Stage 1 — park factors as residual means of the two-way model. |
| `ballpark_allocation_stage2.py` | Stage 2 — does park configuration change positional allocation? (clean null) |
| `hand_split_skill.py` *(present as `hand_split_skill_-_Copy.py`)* | Step 1 — per-hand EB-shrunk batting skills. **Rename: drop the `- Copy` suffix.** |
| `platoon_allocation_tobit.py` | Step 2 — platoon loading on own-hand advantage, overall skill fixed. |
| `residual_leaderboard.py` | Career allocation-residual leaderboard (no player term in the prediction). |

### Coverage / units / utilities (commit)
| File | Purpose |
|---|---|
| `coverage_diagnostic.py` | *(foundational)* Retrosheet panel completeness by season. |
| `pin_numbers.py` | Pins τ_bat SD, the wOBA→runs anchor, and Pythagenpat runs/win before drafting. |
| `eight_series_gap1.py` | Selection decomposition of the cohort signal (gap = 1). |

---

## Reference data (commit — small)

| File | Purpose |
|---|---|
| `wOBA_weights.csv` | FanGraphs season wOBA weights (1871–2026). |
| `Teams.csv` | Lahman team-season totals — AL/NL filtering and Pythagenpat RPG. |

## Generated result artifacts (output — regenerable)

These are produced by the scripts above. Commit them only if you want results browsable
without a local run; otherwise gitignore. Several feed the paper's tables/figures.

`pa_allocation_pos_age.csv`, `pa_allocation_pos_era.csv`, `contact_fielding_test_coef.csv`,
`expansion_fit_coef.csv`, `mu_three_schemes.csv`, `mu_seasonw_vs_fixedw.csv`,
`dh_allocation_test_coef.csv`, `park_profile.csv`, `residual_leaderboard.csv`,
`talent_dispersion.csv`, `reconcile_dispersion.csv`, `coverage_diagnostic.csv`,
`dispersion_enriched.csv`, `pitcher_dispersion.csv`

Text/summary outputs (logs of result scripts): `clutch_summary.txt`, `hot_hand_summary.txt`,
`ba_skill_corr_summary.txt`, `dh_hbp_moral_hazard_summary.txt`, `aging_curve_summary.txt`,
`cohort_menu_test_summary.txt`, `validate_fielding_summary.txt`, `dh_null_bridge_summary.txt`,
`dh_split_table_summary.txt`, `hand_split_skill_summary.txt`,
`platoon_allocation_tobit_summary.txt`, `share_tobit_aux_summary.txt`,
`ballpark_allocation_stage2_summary.txt`, `residual_leaderboard_summary.txt`,
`ballpark_residuals_summary.txt`, `dh_allocation_test_summary.txt`,
`streak_clutch_tails.txt`, `candidate_shocks.md`.

> **Two known regeneration flags** (from the manuscript fix list): `residual_leaderboard.csv`
> still shows first names (it was built with `usename`; rerun with `fullname`), and the
> bat-slope / positional-premia reconciliation should be settled before these CSVs are
> treated as final.

## Local only — do not commit (gitignore)

`pa_panel.parquet`, `leverage_panel.parquet`, `pa_team_share.parquet`,
`pa_allocation_panel.parquet`, the raw Retrosheet directory, and any other parquet/large
caches.

## Not for the code repo

Manuscript and process files — keep in a `/docs` folder or a separate location:
`c19_e.docx`, `Mesoball_final.docx`, `Mesoball_draft_4.docx`, `Mesoball_v2_tracked.docx`,
`Mesoball_Palmer_section.docx`, `Mesoball_omnibus_draft.md`, `Mesoball_results_draft_2.md`,
`HANDOFF.md`, `HANDOFF_draft4.md`, `HANDOFF_cleanup.md`.

OS duplicate files to delete: `wOBA_weights  Copy 2.csv`, `Teams  Copy.csv`,
`hand_split_skill_-_Copy.py` (rename to `hand_split_skill.py`),
`aging_effective_summary_-_Copy.txt`.
