"""Descriptive deep-dive of the swallowing (mylohyoid) channel.

Adults with at least one gated swallow. Summarises the swallow descriptors and
their distributions. The body-composition / strength correlates, the self-
reported dysphagia comparison and the sex contrast were removed: the focused
manuscript is about the detection pipeline and chewing<->swallowing
coordination, not clinical/anthropometric associations.
"""
from __future__ import annotations

import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import cohort

sns.set_theme(style="whitegrid", context="paper")

SWALLOW_FEATURES = ["swallow_n", "swallow_rate_per_min", "swallow_work",
                    "swallow_work_rate", "swallow_dur_mean", "swallow_peak_mean",
                    "inter_swallow_interval_s", "swallow_active_frac",
                    "chews_per_swallow"]


def describe(df):
    d = cohort.adults_valid_swallow(df)
    return d[SWALLOW_FEATURES].describe().T[["count", "mean", "50%", "std", "min", "max"]]


def make_figures(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    d = cohort.adults_valid_swallow(df)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, c in zip(axes.ravel(), ["swallow_n", "swallow_rate_per_min",
                                    "inter_swallow_interval_s", "swallow_dur_mean",
                                    "swallow_work_rate", "chews_per_swallow"]):
        ax.hist(d[c].dropna(), bins=16, color="darkorange", alpha=.8)
        ax.set_title(c); ax.set_ylabel("subjects")
    fig.suptitle("Swallowing descriptors (adults, valid-swallow)")
    fig.tight_layout(); fig.savefig(f"{outdir}/fig_swallow_distributions.png", dpi=130); plt.close(fig)


def run(master_csv, outdir):
    os.makedirs(f"{outdir}/tables", exist_ok=True)
    df = pd.read_csv(master_csv)
    describe(df).to_csv(f"{outdir}/tables/SW_describe.csv")
    make_figures(df, f"{outdir}/figures")


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).resolve().parents[1] / "outputs"
    run(str(out / "master_dataset.csv"), str(out))
    df = pd.read_csv(out / "master_dataset.csv")
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("=== Swallow descriptors (adults, valid-swallow) ===")
    print(describe(df).round(3).to_string())
