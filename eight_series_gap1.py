"""
eight_series_gap1.py
--------------------
For age gap = 1 only, decompose the cohort signal into 8 series:
  younger role        (bat / pit)
  younger played next year (yes / no)
  older played prior year  (yes / no)

Vectorized version.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import time

PROJECT_DIR = Path(r"C:\baseball_eras")
PANEL = PROJECT_DIR / "data" / "pa_panel.parquet"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log("loading panel...")
panel = pd.read_parquet(PANEL, columns=[
    "season", "bat_id", "pit_id", "bat_birth_year", "pit_birth_year",
    "woba_value", "iw",
])
panel = panel[panel["iw"] != 1]
panel = panel.dropna(subset=["bat_birth_year", "pit_birth_year"])
log(f"  {len(panel):,} PAs after IBB/missing-bday filter")

# League wOBA by season
lg = panel.groupby("season")["woba_value"].mean().rename("lg_woba")

# Build (player, season) sets as DataFrames for merging (vectorized!)
log("building player-year tables...")
bat_yrs = panel[["bat_id", "season"]].drop_duplicates()
pit_yrs = panel[["pit_id", "season"]].drop_duplicates()
log(f"  {len(bat_yrs):,} (bat, season) rows, {len(pit_yrs):,} (pit, season) rows")

# Restrict to gap=1
panel["age_gap"] = (panel["bat_birth_year"] - panel["pit_birth_year"]).astype(int)
panel = panel[abs(panel["age_gap"]) == 1].copy()
log(f"  {len(panel):,} PAs at gap=1")

panel["younger_role"] = np.where(panel["age_gap"] == -1, "bat", "pit")
panel = panel.merge(lg, on="season", how="left")
panel["dev"] = panel["woba_value"] - panel["lg_woba"]

# ---- Vectorized survival flags via merges ----
log("flagging younger_next (vectorized)...")
bat_active_next = bat_yrs.copy()
bat_active_next["season"] = bat_active_next["season"] - 1
bat_active_next["bat_next_yr"] = True
panel = panel.merge(bat_active_next, on=["bat_id", "season"], how="left")
panel["bat_next_yr"] = panel["bat_next_yr"].fillna(False)

pit_active_next = pit_yrs.copy()
pit_active_next["season"] = pit_active_next["season"] - 1
pit_active_next["pit_next_yr"] = True
panel = panel.merge(pit_active_next, on=["pit_id", "season"], how="left")
panel["pit_next_yr"] = panel["pit_next_yr"].fillna(False)

log("flagging older_prev (vectorized)...")
bat_active_prev = bat_yrs.copy()
bat_active_prev["season"] = bat_active_prev["season"] + 1
bat_active_prev["bat_prev_yr"] = True
panel = panel.merge(bat_active_prev, on=["bat_id", "season"], how="left")
panel["bat_prev_yr"] = panel["bat_prev_yr"].fillna(False)

pit_active_prev = pit_yrs.copy()
pit_active_prev["season"] = pit_active_prev["season"] + 1
pit_active_prev["pit_prev_yr"] = True
panel = panel.merge(pit_active_prev, on=["pit_id", "season"], how="left")
panel["pit_prev_yr"] = panel["pit_prev_yr"].fillna(False)

log("composing survival flags by role...")
panel["younger_next"] = np.where(
    panel["younger_role"] == "bat", panel["bat_next_yr"], panel["pit_next_yr"]
)
panel["older_prev"] = np.where(
    panel["younger_role"] == "bat", panel["pit_prev_yr"], panel["bat_prev_yr"]
)

panel = panel.drop(columns=["bat_next_yr", "pit_next_yr", "bat_prev_yr", "pit_prev_yr"])

log("aggregating by 8 cells...")
print()
print("Eight-series decomposition (gap=1):")
print(f"{'younger':>10s}  {'y_next':>7s}  {'o_prev':>7s}  "
      f"{'n_pa':>11s}  {'mean_dev':>10s}  {'younger_adv':>12s}")

rows = []
for role in ["bat", "pit"]:
    for ynext in [True, False]:
        for oprev in [True, False]:
            sub = panel[
                (panel["younger_role"] == role)
                & (panel["younger_next"] == ynext)
                & (panel["older_prev"] == oprev)
            ]
            if len(sub) == 0:
                continue
            n_pa = len(sub)
            mean_dev = sub["dev"].mean()
            adv = mean_dev if role == "bat" else -mean_dev
            print(f"{role:>10s}  {str(ynext):>7s}  {str(oprev):>7s}  "
                  f"{n_pa:>11,d}  {mean_dev:>+10.5f}  {adv:>+12.5f}")
            rows.append({
                "younger_role": role,
                "younger_next_yr": ynext,
                "older_prev_yr": oprev,
                "n_pa": n_pa,
                "mean_dev": mean_dev,
                "younger_advantage": adv,
            })

print()
print("Aggregated across roles, by survival flags:")
print(f"{'y_next':>7s}  {'o_prev':>7s}  {'n_pa':>12s}  {'younger_adv':>12s}")
df = pd.DataFrame(rows)
for ynext in [True, False]:
    for oprev in [True, False]:
        s = df[(df["younger_next_yr"] == ynext) & (df["older_prev_yr"] == oprev)]
        if len(s) == 0:
            continue
        total_pa = s["n_pa"].sum()
        weighted_adv = (s["younger_advantage"] * s["n_pa"]).sum() / total_pa
        print(f"{str(ynext):>7s}  {str(oprev):>7s}  {total_pa:>12,d}  {weighted_adv:>+12.5f}")

df.to_csv(PROJECT_DIR / "output" / "eight_series_gap1.csv", index=False)
log(f"saved to {PROJECT_DIR / 'output' / 'eight_series_gap1.csv'}")
