"""
Season-level outcome fractions from the PA panel.

For every season 1910-2025, compute the fraction of plate appearances that are
balls in play (BIP) and the fractions that are strikeouts, walks and home runs,
then report the highest- and lowest-BIP seasons.

Definitions (every PA is exactly one outcome flag in the panel):
    BIP   = single + double + triple + sh + sf + roe + fc + othout
            (batted balls fielded or landing for a non-HR hit; HR leaves the park)
    K     = k
    BB    = walk            (includes intentional; uBB = walk - iw is also shown)
    HR    = hr
    excluded from BIP: k, walk, hbp, hr, xi
    Identity check: BIP + K + BB + HBP + HR + XI == PA

Exploratory (not a GitHub pipeline script; no u_ prefix).
"""
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(r"C:\baseball_eras")
PANEL_PATH  = PROJECT_DIR / "data" / "pa_panel.parquet"
OUT_CSV     = PROJECT_DIR / "season_outcome_fractions.csv"

EVENT_COLS = ["single", "double", "triple", "hr", "walk", "iw", "hbp",
              "k", "sh", "sf", "roe", "xi", "fc", "othout"]
BIP_PARTS  = ["single", "double", "triple", "sh", "sf", "roe", "fc", "othout"]


def main():
    print(f"loading {PANEL_PATH}")
    df = pd.read_parquet(PANEL_PATH, columns=["season"] + EVENT_COLS)
    df = df[(df["season"] >= 1910) & (df["season"] <= 2025)]

    g = df.groupby("season")
    s = g[EVENT_COLS].sum()
    s["PA"] = g.size()

    s["BIP"]  = s[BIP_PARTS].sum(axis=1)
    s["uBB"]  = (s["walk"] - s["iw"]).clip(lower=0)

    # identity check
    chk = s[["k", "walk", "hbp", "hr", "xi"]].sum(axis=1) + s["BIP"]
    bad = (chk - s["PA"]).abs() > 0
    if bad.any():
        print("WARNING: BIP identity off in seasons:", list(s.index[bad]))

    for col, name in [("BIP", "bip"), ("k", "k"), ("walk", "bb"),
                      ("uBB", "ubb"), ("hr", "hr"), ("hbp", "hbp")]:
        s[name + "_frac"] = s[col] / s["PA"]

    # ---- batting-average bounds and the constant-skill (BABIP) check ----
    # AB excludes walks, HBP, sacrifices, and catcher interference.
    s["AB"]      = s["PA"] - s[["walk", "hbp", "sh", "sf", "xi"]].sum(axis=1)
    s["BA"]      = (s["single"] + s["double"] + s["triple"] + s["hr"]) / s["AB"]
    s["maxBA"]   = 1 - s["k"] / s["AB"]              # hit on every ball in play
    s["minBA"]   = s["hr"] / s["AB"]                 # out on every ball in play; HR still count
    s["rangeBA"] = s["maxBA"] - s["minBA"]           # = 1 - (K + HR)/AB
    # contact-conversion skill = non-HR hits per ball in play within at-bats
    # (ROE and FC are NOT hits, so they are excluded from the numerator)
    s["babip"]   = (s["single"] + s["double"] + s["triple"]) / (s["AB"] - s["k"] - s["hr"])
    # Exact identity: BA == minBA + babip * rangeBA
    # (BA = a HR floor + contact skill applied across the span the at-bat menu allows)

    out = s[["PA", "AB", "bip_frac", "k_frac", "bb_frac", "ubb_frac", "hr_frac",
             "hbp_frac", "BA", "maxBA", "minBA", "rangeBA", "babip"]].round(4)
    out.to_csv(OUT_CSV)
    print(f"wrote {OUT_CSV}\n")

    # decade means (sanity vs contact_fielding_test_coef.csv: 0.83 peak 1920s, 0.64 2020s)
    dec = (out.assign(decade=(out.index // 10 * 10))
              .groupby("decade")["bip_frac"].mean().round(3))
    print("decade mean BIP fraction (sanity check):")
    print(dec.to_string(), "\n")

    hi = out["bip_frac"].idxmax()
    lo = out["bip_frac"].idxmin()

    def line(yr):
        r = out.loc[yr]
        return (f"  {yr}: BIP {r.bip_frac:.3f} | K {r.k_frac:.3f} | "
                f"BB {r.bb_frac:.3f} (uBB {r.ubb_frac:.3f}) | HR {r.hr_frac:.3f}  "
                f"[PA {int(r.PA):,}]")

    print("HIGHEST balls-in-play season:")
    print(line(hi))
    print("\nLOWEST balls-in-play season:")
    print(line(lo))

    print("\nTop 5 highest BIP seasons:")
    for yr in out["bip_frac"].nlargest(5).index:
        print(line(yr))
    print("\nTop 5 lowest BIP seasons:")
    for yr in out["bip_frac"].nsmallest(5).index:
        print(line(yr))

    # ---- batting-average span, skill-constant framing (AB-based) ----
    print("\nBatting-average span (max = hit on every ball in play, "
          "min = out on every ball in play):")
    print(f"  {'yr':>4} {'AB/PA':>6} {'BA':>6} {'maxBA':>6} {'minBA':>6} "
          f"{'range':>6} {'BABIP':>6}")
    for yr in [1921, 2021]:
        if yr in s.index:
            r = s.loc[yr]
            print(f"  {yr:>4} {r.AB / r.PA:>6.3f} {r.BA:>6.3f} {r.maxBA:>6.3f} "
                  f"{r.minBA:>6.3f} {r.rangeBA:>6.3f} {r.babip:>6.3f}")
    print("\n  decomposition check  BA == minBA + BABIP * range:")
    for yr in [1921, 2021]:
        if yr in s.index:
            r = s.loc[yr]
            print(f"    {yr}: {r.minBA:.4f} + {r.babip:.4f} * {r.rangeBA:.4f} "
                  f"= {r.minBA + r.babip * r.rangeBA:.4f}   (actual BA = {r.BA:.4f})")


if __name__ == "__main__":
    main()
