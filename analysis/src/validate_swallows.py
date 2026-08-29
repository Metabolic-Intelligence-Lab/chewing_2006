"""Manual validation of swallow detection (official vs heuristic) on borderline cases.

For every borderline subject — zero official deglutitions, a single one, or an
extreme chews-per-swallow — render the mylohyoid trace with BOTH the official
phase-end+morphology deglutition events (red bands) and the simpler amplitude-
gated swallows (blue lines), plus the chewing phases (shaded). Visual review of
the contact sheets adjudicates whether absence is real or under-detection.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import official
import io_raw
import features as feat
import cohort


def borderline_table(df):
    d = cohort.adults(df).copy()
    cps = d["chews_per_swallow"]
    reason = np.where(~d["valid_swallow"], "zero_swallow",
             np.where(d["swallow_n"] <= 1, "single_swallow",
             np.where((cps < 3) | (cps > 25), "extreme_cps", "")))
    d["reason"] = reason
    return d[d["reason"] != ""][["ID", "reason", "swallow_n", "swallow_events",
                                 "chew_n", "chews_per_swallow"]].sort_values(
        ["reason", "ID"]).reset_index(drop=True)


def panel(ax, rid, official_res, recs):
    summary, extra = official_res[rid]
    fs = extra["fs"]
    mast, deg = extra["mast"], extra["deg"]
    t = np.arange(len(mast)) / fs
    ax.plot(t, mast, lw=.4, color="steelblue", alpha=.55, label="masseter")
    ax.plot(t, deg, lw=.5, color="darkorange", label="mylohyoid")
    for s, e in extra["phases"]:
        ax.axvspan(s / fs, e / fs, color="green", alpha=.05)
    # official deglutition events (red bands)
    for s, e in extra["events"]:
        ax.axvspan(s / fs, e / fs, color="red", alpha=.35)
    # heuristic amplitude-gated swallows (blue dashed lines) from the raw recording
    if rid in recs:
        r = recs[rid]
        chew = feat.channel_features(r.ch_chew, r.fs, mode="static", k_static=3.0)
        for b in feat.detect_swallows(r, chew_feat=chew):
            ax.axvline(b.start_s, color="blue", lw=.9, ls="--", alpha=.7)
    ax.set_title(f"ID {rid:03d}  off={len(extra['events'])} chews={summary['number of chews']}",
                 fontsize=8)
    ax.tick_params(labelsize=6)


def contact_sheet(ids, official_res, recs, outpath, ncol=3):
    nrow = int(np.ceil(len(ids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.3, nrow * 1.9))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, rid in zip(axes, ids):
        if rid in official_res:
            ax.axis("on")
            panel(ax, rid, official_res, recs)
    fig.suptitle("Borderline swallows — red=official event, blue dashed=amplitude-gated",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outpath, dpi=120); plt.close(fig)


def agreement(df, official_res, recs):
    """Quantify agreement official vs heuristic swallow counts across adults."""
    rows = []
    for rid in df["ID"]:
        if rid not in official_res or rid not in recs:
            continue
        off_n = len(official_res[rid][1]["events"])
        r = recs[rid]
        chew = feat.channel_features(r.ch_chew, r.fs, mode="static", k_static=3.0)
        heur_n = len(feat.detect_swallows(r, chew_feat=chew))
        rows.append(dict(ID=rid, official=off_n, heuristic=heur_n))
    return pd.DataFrame(rows)


def run(raw_dir, master_csv, outdir):
    os.makedirs(f"{outdir}/figures/validation", exist_ok=True)
    os.makedirs(f"{outdir}/tables", exist_ok=True)
    df = pd.read_csv(master_csv)
    official_res = official.analyze_all(raw_dir)
    recs = io_raw.load_all(raw_dir)
    bl = borderline_table(df)
    bl.to_csv(f"{outdir}/tables/VAL_borderline_index.csv", index=False)
    for reason in bl["reason"].unique():
        ids = bl.loc[bl.reason == reason, "ID"].tolist()
        contact_sheet(ids, official_res, recs,
                      f"{outdir}/figures/validation/borderline_{reason}.png")
    agr = agreement(cohort.adults(df), official_res, recs)
    agr.to_csv(f"{outdir}/tables/VAL_agreement.csv", index=False)
    return bl, agr


if __name__ == "__main__":
    import pathlib
    base = pathlib.Path(__file__).resolve().parents[2]
    out = pathlib.Path(__file__).resolve().parents[1] / "outputs"
    bl, agr = run(str(base / "dati_raw"), str(out / "master_dataset.csv"), str(out))
    m = agr.dropna()
    print("borderline cases:", len(bl), "by reason:", bl.reason.value_counts().to_dict())
    print("agreement official vs heuristic: r=%.3f (n=%d)" % (
        np.corrcoef(m.official, m.heuristic)[0, 1], len(m)))
    print("both-zero:", int(((m.official == 0) & (m.heuristic == 0)).sum()),
          "| official0_heur>0:", int(((m.official == 0) & (m.heuristic > 0)).sum()),
          "| official>0_heur0:", int(((m.official > 0) & (m.heuristic == 0)).sum()))
