"""Stage 5 - publication figures (focused scope).

Only the figures used by the detection + coordination manuscript are produced
here: an example two-channel trace, the pipeline schematic, single-swallow
morphology, the chew<->swallow coupling panel, and the supporting
annotation-concordance figure. Other figures are produced by their own modules
(cycles: within-meal dynamics; directionality; sensitivity; swallowing:
descriptors; qc; validate_swallows: borderline sheets). Adult primary cohort;
swallowing/coordination panels use the valid-swallow subset.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import official

sns.set_theme(style="whitegrid", context="paper")


def fig_example_trace(raw_dir, outdir, rid=11):
    res = official.analyze_all(raw_dir)
    if rid not in res:
        rid = next(iter(res))
    summ, extra = res[rid]
    fs = extra["fs"]; t = np.arange(len(extra["mast"])) / fs
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(t, extra["mast"], lw=.6, color="steelblue", label="masseter (chewing)")
    ax.plot(t, extra["deg"], lw=.6, color="darkorange", alpha=.75, label="mylohyoid (swallowing)")
    for s, e in extra["phases"]:
        ax.axvspan(s / fs, e / fs, color="green", alpha=.07)
    for s, e in extra["events"]:
        ax.axvspan(s / fs, e / fs, color="red", alpha=.35)
    ax.set_title(f"ID {rid:03d} - chewing phases (green) and gated deglutitions (red)")
    ax.set_xlabel("time [s]"); ax.set_ylabel("EMG envelope [a.u.]")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig_example_trace.png", dpi=130); plt.close(fig)


def fig_pipeline(raw_dir, outdir, rid=11):
    """Focused schematic: (1) acquire two sEMG channels, (2) detect masticatory
    phases and gate swallow events (phase-end windows + morphology filter),
    (3) derive chewing<->swallowing coordination metrics."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    res = official.analyze_all(raw_dir)
    if rid not in res:
        rid = next(iter(res))
    extra = res[rid][1]; fs = extra["fs"]
    mast, deg = extra["mast"], extra["deg"]; t = np.arange(len(mast)) / fs

    fig = plt.figure(figsize=(13.5, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.2], hspace=0.30, wspace=0.20,
                          left=0.06, right=0.97, top=0.94, bottom=0.07)
    # ---- top band: 3-stage flow ----
    axf = fig.add_subplot(gs[0, :]); axf.axis("off")
    axf.set_xlim(-3, 103); axf.set_ylim(0, 10.5)
    stages = [
        ("1. Acquisition", "Two-channel sEMG (~100 Hz)\nmasseter (chewing) +\nmylohyoid (swallowing)"),
        ("2. Detection", "Masticatory phases (masseter);\nswallow events gated on the\nmylohyoid: phase-end windows\n+ morphology filter"),
        ("3. Coordination", "Intensity coupling, envelope\nsynchrony (xcorr lag), sequence\norganisation, directionality"),
    ]
    bw, gap = 29.0, 5.5; x0 = 2.0
    for i, (head, body) in enumerate(stages):
        x = x0 + i * (bw + gap)
        axf.add_patch(FancyBboxPatch((x, 1.0), bw, 8.0, boxstyle="round,pad=0.3,rounding_size=0.6",
                                     linewidth=1.4, edgecolor="#1F3A5F", facecolor="#EAF1F8"))
        axf.text(x + bw / 2, 8.1, head, ha="center", va="center", fontsize=10,
                 fontweight="bold", color="#1F3A5F", clip_on=False)
        axf.text(x + bw / 2, 4.3, body, ha="center", va="center", fontsize=8.4, clip_on=False)
        if i < 2:
            axf.add_patch(FancyArrowPatch((x + bw, 5.0), (x + bw + gap, 5.0),
                          arrowstyle="-|>", mutation_scale=18, linewidth=1.6, color="#1F3A5F"))

    # ---- (1-2) example trace with phases + swallows ----
    ax0 = fig.add_subplot(gs[1, 0])
    ax0.plot(t, mast, lw=.5, color="steelblue", label="masseter")
    ax0.plot(t, deg, lw=.5, color="darkorange", alpha=.75, label="mylohyoid")
    for s, e in extra["phases"]:
        ax0.axvspan(s / fs, e / fs, color="green", alpha=.07)
    for s, e in extra["events"]:
        ax0.axvspan(s / fs, e / fs, color="red", alpha=.30)
    ax0.set_title("Stages 1–2 · two channels → phases (green) & gated swallows (red)", fontsize=9)
    ax0.set_xlabel("time [s]", fontsize=8); ax0.set_ylabel("EMG envelope [a.u.]", fontsize=8)
    ax0.legend(fontsize=7, loc="upper right"); ax0.tick_params(labelsize=7)

    # ---- (3) coordination parameter list ----
    ax1 = fig.add_subplot(gs[1, 1]); ax1.axis("off")
    ax1.set_title("Stage 3 · chewing↔swallowing coordination", fontsize=9)
    items = [
        ("Intensity coupling", "chew work rate ↔ swallow work rate"),
        ("Envelope synchrony", "peak cross-correlation & lag (±2 s)"),
        ("Temporal overlap", "co-activation Jaccard"),
        ("Sequence organisation", "chews/swallow, swallows/phase,\nphases-with-swallow, post-phase latency"),
        ("Oral processing", "chewing before the first swallow"),
        ("Directionality", "coherence + bivariate Granger"),
    ]
    y = 0.95
    for name, desc in items:
        ax1.text(0.02, y, name, fontsize=9, fontweight="bold", va="top", color="#1F3A5F",
                 transform=ax1.transAxes)
        ax1.text(0.04, y - 0.055, desc, fontsize=7.8, va="top", color="#333",
                 transform=ax1.transAxes)
        y -= 0.16
    fig.savefig(f"{outdir}/fig_pipeline.png", dpi=130); plt.close(fig)


def fig_chew_swallow_coupling(df, outdir):
    """Coordination axis: power coupling (chew vs swallow work rate, ρ≈0.37) and
    envelope synchrony (cross-correlation lag ≈0 s)."""
    from scipy.stats import spearmanr
    d = df[df.is_adult & df.valid_swallow]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    m = d[["chew_work_rate", "swallow_work_rate"]].dropna()
    sns.regplot(x="chew_work_rate", y="swallow_work_rate", data=m, ax=ax[0],
                scatter_kws=dict(s=22, alpha=.6), line_kws=dict(color="crimson"))
    r, p = spearmanr(m.chew_work_rate, m.swallow_work_rate)
    ax[0].set_title(f"(a) Power coupling\nchew vs swallow work rate  ρ={r:.2f}, p={p:.1g}, n={len(m)}")
    ax[0].set_xlabel("chewing work rate [mV]"); ax[0].set_ylabel("swallowing work rate [mV]")
    lag = pd.to_numeric(d["xcorr_lag_s"], errors="coerce").dropna()
    ax[1].hist(lag, bins=18, color="teal", alpha=.8)
    ax[1].axvline(0, color=".4", ls=":", lw=1)
    ax[1].axvline(lag.median(), color="crimson", ls="--", label=f"median={lag.median():.2f} s")
    ax[1].set_yscale("log")
    ax[1].set_title("(b) Envelope synchrony\nchew↔swallow cross-correlation lag (log count)")
    ax[1].set_xlabel("lag [s]  (>0: swallow follows chew)"); ax[1].set_ylabel("subjects (log)")
    ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{outdir}/fig_coupling.png", dpi=130); plt.close(fig)


def fig_swallow_morphology(outdir):
    """Event-level swallow morphology distributions (reads swallow_events.csv)."""
    ev_path = pathlib.Path(outdir).parent / "swallow_events.csv"
    if not ev_path.exists():
        return
    ev = pd.read_csv(ev_path)
    cols = [("duration_s", "swallow duration [s]", False),
            ("rise_decay_ratio", "rise/decay ratio (log)", True),
            ("fwhm_s", "FWHM [s]", False), ("rfd", "rate of rise [a.u./s] (log)", True),
            ("n_subpeaks", "sub-peaks (piecemeal)", False),
            ("peak_norm", "peak (norm. %ref)", False)]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, (c, lab, logx) in zip(axes.ravel(), cols):
        if c in ev.columns:
            x = ev[c].dropna()
            if logx:
                x = x[x > 0]
                lo, hi = np.log10(x.min()), np.log10(x.max())
                ax.hist(x, bins=np.logspace(lo, hi, 20), color="darkorange", alpha=.8)
                ax.set_xscale("log")
            else:
                ax.hist(x, bins=20, color="darkorange", alpha=.8)
            ax.set_title(lab); ax.set_ylabel("events")
    fig.suptitle(f"Single-swallow morphology ({len(ev)} events, {ev.ID.nunique()} subjects)")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig_swallow_morphology.png", dpi=130); plt.close(fig)


def fig_annotation_concordance(outdir):
    """Supporting concordance with the manual protocol annotations: (a) detected
    vs annotated swallow counts per recording; (b) per-subject detected-event −
    nearest-annotation time offset (operator latency). Reads the tables written by
    annotation_concordance.py."""
    tables = pathlib.Path(outdir).parent / "tables"
    sub_p = tables / "ANN_concordance_subject.csv"
    if not sub_p.exists():
        return
    d = pd.read_csv(sub_p)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    # (a) count scatter with identity line
    mx = int(max(d.n_annotated.max(), d.n_detected.max(), 1)) + 1
    ax[0].plot([0, mx], [0, mx], ls=":", color=".5", lw=1, label="identity")
    ax[0].scatter(d.n_annotated, d.n_detected, s=30, alpha=.6, color="slateblue",
                  edgecolor="white", linewidth=.4)
    ax[0].set_xlim(-0.5, mx); ax[0].set_ylim(-0.5, mx)
    ax[0].set_xlabel("annotated swallows (manual marks)")
    ax[0].set_ylabel("detected swallows (gated)")
    ax[0].set_title("(a) Detected vs annotated counts\n(detector is conservative → below identity)")
    ax[0].legend(fontsize=8, loc="upper left")
    # (b) per-subject median offset histogram
    off = pd.to_numeric(d.median_offset_s, errors="coerce").dropna()
    ax[1].hist(off, bins=18, color="darkorange", alpha=.8)
    ax[1].axvline(0, color=".4", ls=":", lw=1)
    if len(off):
        ax[1].axvline(off.median(), color="crimson", ls="--",
                      label=f"median={off.median():.1f} s")
        ax[1].legend(fontsize=8)
    ax[1].set_xlabel("detected − nearest annotation [s]  (<0: detector leads press)")
    ax[1].set_ylabel("recordings")
    ax[1].set_title("(b) Timing offset (operator latency)")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig_annotation_concordance.png", dpi=130); plt.close(fig)


def run(raw_dir, master_csv, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(master_csv)
    fig_example_trace(raw_dir, outdir)
    fig_pipeline(raw_dir, outdir)
    fig_swallow_morphology(outdir)
    fig_chew_swallow_coupling(df, outdir)
    fig_annotation_concordance(outdir)


if __name__ == "__main__":
    base = pathlib.Path(__file__).resolve().parents[2]
    out = pathlib.Path(__file__).resolve().parents[1] / "outputs"
    run(str(base / "dati_raw"), str(out / "master_dataset.csv"), str(out / "figures"))
    print("figures done")
