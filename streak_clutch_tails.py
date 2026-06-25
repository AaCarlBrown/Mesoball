"""
streak_clutch_tails.py   (fan-facing exhibit)
=============================================
Among players with >= 3000 career PA (the record-eligibility threshold), who are
the ten most/least streaky and the ten most/least clutch -- and how many SDs from
the mean are they? Crucially, it reports the observed SD against the sampling-noise
SD: when the ratio is ~1, the tails are luck, not talent, and the named leaders are
the luckiest coin-flippers, not genuinely streaky/clutch players.

Streak: reuses hot_hand_player.csv (corrected lag-1 autocorrelation).
Clutch: recomputed here (opponent-purged high-minus-low, late & close).
Names:  merged from the Retrosheet biofile.

Run:  py streak_clutch_tails.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\baseball_eras")
RETRO = Path(r"C:\overnight_effect_data\retrosheet")
PA = ROOT / "data" / "pa_panel.parquet"
LEV = ROOT / "data" / "leverage_panel.parquet"
HOT = ROOT / "output" / "hot_hand_player.csv"
BIO = RETRO / "biofile0.csv"
MIN_CAREER = 3000
MIN_SIDE = 50

L = []
def P(s=""):
    print(s); L.append(s)


def names_map():
    if not BIO.exists():
        return {}
    bio = pd.read_csv(BIO, low_memory=False)
    idc = next((c for c in ["id", "playerid", "retroID", "key_retro"] if c in bio.columns), bio.columns[0])
    fulls = [c for c in ["fullname", "name", "playername"] if c in bio.columns]
    firsts = [c for c in ["first", "firstname", "nameFirst", "first_name"] if c in bio.columns]
    lasts = [c for c in ["last", "lastname", "nameLast", "last_name"] if c in bio.columns]
    if fulls:
        nm = bio[fulls[0]].astype(str)
    elif firsts and lasts:
        nm = (bio[firsts[0]].astype(str) + " " + bio[lasts[0]].astype(str))
    elif lasts:
        nm = bio[lasts[0]].astype(str)
    else:
        return {}
    return dict(zip(bio[idc].astype(str), nm))


def tails(stat: pd.DataFrame, name: str, names: dict):
    """stat has columns: batter, val (the statistic), n_eff, samp_var."""
    s = stat.copy()
    m, sd = s["val"].mean(), s["val"].std()
    samp = np.sqrt(np.average(s["samp_var"]))
    ratio = sd / samp if samp else float("nan")
    s["z"] = (s["val"] - m) / sd
    s["nm"] = s["batter"].astype(str).map(lambda b: names.get(b, b))
    P(f"\n=== {name} -- {len(s):,} players (>= {MIN_CAREER} PA) ===")
    P(f"  mean {m:+.5f}   observed SD {sd:.5f}   sampling-noise SD {samp:.5f}")
    P(f"  observed/sampling ratio {ratio:.2f}   (1.0 = pure luck; >1 = some real spread)")
    # expected tail under pure noise, by simulation
    rng = np.random.default_rng(0)
    sims = [np.sort(rng.normal(0, 1, len(s)))[[-10, -1]] for _ in range(400)]
    exp_top = np.mean([x[1] for x in sims]); exp_10th = np.mean([x[0] for x in sims])
    P(f"  under pure noise, {len(s):,} draws give max z ~ {exp_top:+.2f}, 10th ~ {exp_10th:+.2f}")
    top = s.nlargest(10, "val"); bot = s.nsmallest(10, "val")
    P(f"  TOP 10 most {name.lower()}:")
    for _, r in top.iterrows():
        P(f"    {r['nm']:<24} z={r['z']:+.2f}  stat={r['val']:+.4f}  PA={int(r['n_eff'])}")
    P(f"  top-10 mean z {top['z'].mean():+.2f}")
    P(f"  BOTTOM 10 least {name.lower()}:")
    for _, r in bot.iterrows():
        P(f"    {r['nm']:<24} z={r['z']:+.2f}  stat={r['val']:+.4f}  PA={int(r['n_eff'])}")
    P(f"  bottom-10 mean z {bot['z'].mean():+.2f}")


def main():
    names = names_map()
    P(f"names loaded: {len(names):,}")

    # --- streak from the existing hot-hand player file ---
    if HOT.exists():
        h = pd.read_csv(HOT)
        h = h[h["n"] >= MIN_CAREER].copy()
        h = h.rename(columns={"corrected": "val", "n": "n_eff"})
        h["samp_var"] = 1.0 / h["n_eff"]
        tails(h[["batter", "val", "n_eff", "samp_var"]], "Streaky", names)
    else:
        P(f"\n(skip streak: {HOT} not found -- run hot_hand.py first)")

    # --- clutch recomputed from the merged panel ---
    pa = pd.read_parquet(PA)
    if "game_id" in pa.columns and "gid" not in pa.columns:
        pa = pa.rename(columns={"game_id": "gid"})
    ren = {}
    if "batter" not in pa.columns and "bat_id" in pa.columns: ren["bat_id"] = "batter"
    if "pitcher" not in pa.columns and "pit_id" in pa.columns: ren["pit_id"] = "pitcher"
    pa = pa.rename(columns=ren).reset_index(drop=True)
    pa["pa_seq"] = pa.groupby("gid").cumcount()
    lev = pd.read_parquet(LEV)[["gid", "pa_seq", "late_close"]]
    df = pa.merge(lev, on=["gid", "pa_seq"], how="left")
    if "bat_is_pitcher" in df.columns:
        df = df[df["bat_is_pitcher"] == 0]
    df = df.dropna(subset=["late_close", "woba_value"])
    v = df["woba_value"] - df.groupby("season")["woba_value"].transform("mean")
    v = v - v.groupby(df["batter"]).transform("mean")
    df["resid"] = (v - v.groupby(df["pitcher"]).transform("mean")).to_numpy()
    df["hi"] = df["late_close"].astype(bool)
    career = df.groupby("batter").size()
    keep = career[career >= MIN_CAREER].index
    df = df[df["batter"].isin(keep)]
    hi = df[df["hi"]].groupby("batter")["resid"].agg(["mean", "size"])
    lo = df[~df["hi"]].groupby("batter")["resid"].agg(["mean", "size"])
    var = df.groupby("batter")["resid"].var()
    car = df.groupby("batter").size().rename("n_eff")
    j = hi.join(lo, lsuffix="_hi", rsuffix="_lo").join(var.rename("v")).join(car).dropna()
    j = j[(j["size_hi"] >= MIN_SIDE) & (j["size_lo"] >= MIN_SIDE)]
    j["val"] = j["mean_hi"] - j["mean_lo"]
    j["samp_var"] = j["v"] * (1.0 / j["size_hi"] + 1.0 / j["size_lo"])
    tails(j.reset_index()[["batter", "val", "n_eff", "samp_var"]], "Clutch", names)

    (ROOT / "output" / "streak_clutch_tails.txt").write_text("\n".join(L), encoding="utf-8")
    print("\nwrote streak_clutch_tails.txt")


if __name__ == "__main__":
    main()
