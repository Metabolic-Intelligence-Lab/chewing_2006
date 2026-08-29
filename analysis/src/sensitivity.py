"""Sensitivity of swallow detection to the official algorithm's parameters.

One-at-a-time sweeps around the default values. For each setting we recompute
deglutition detection across all subjects and report:
  - total events, zero-swallow fraction, median events/subject
  - rank stability of per-subject counts vs the default (Spearman)
  - robustness of the headline clinical result (dysphagia -> longer inter-swallow
    interval): Mann-Whitney p and Cliff's delta under each setting

Phase-related parameters (min_pause_s, adaptive_percentile) also change the
masticatory phases (and thus the phase-end allowed windows), so phases are
recomputed for those sweeps.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

import official
import cohort
from stats import _yes

# default deglutition parameters (mirror official.identify_deglutition_events)
DEG_DEFAULTS = dict(
    env_filter_size=151, env_peak_height=8, env_prominence=5, env_peak_distance=80,
    border_fraction=0.10, mast_filter_size=51, adaptive_percentile=25,
    min_sustained_s=1.0, min_delay_after_start_s=5.0, phase_start_buffer_s=3.0,
    use_phase_end_windows=True, pre_phase_end_s=1.0, post_phase_end_s=8.0,
    mast_relax_factor=1.50, max_derivative_sign_changes_per_s=35.0,
    max_derivative_activity_ratio=4.00, min_deg_amplitude=10,
    min_duration_s=0.15, max_duration_s=5.0, fine_filter_size=7,
)
PHASE_DEFAULTS = dict(min_pause_s=3.0, min_phase_duration_s=3.0, adaptive_percentile=25)

# one-at-a-time grids (default value included)
GRID = {
    "post_phase_end_s": [4.0, 6.0, 8.0, 10.0, 12.0],
    "pre_phase_end_s": [0.5, 1.0, 2.0],
    "min_delay_after_start_s": [3.0, 5.0, 7.0],
    "env_prominence": [3, 5, 8],
    "min_deg_amplitude": [8, 10, 15],
    "max_derivative_sign_changes_per_s": [25.0, 35.0, 45.0],
    "max_derivative_activity_ratio": [3.0, 4.0, 5.0],
    "mast_relax_factor": [1.25, 1.50, 2.0],
}
PHASE_GRID = {
    "min_pause_s": [2.0, 3.0, 4.0],
    "adaptive_percentile": [20, 25, 30],
}


def _load(raw_dir):
    """Pre-load adjusted signals per subject (avoids re-reading files)."""
    sig = {}
    for fn in sorted(os.listdir(raw_dir)):
        if not fn.lower().endswith(".txt"):
            continue
        fid = official.parse_id(fn)
        if fid is None:
            continue
        arr = official.load_and_preprocess_data(os.path.join(raw_dir, fn))
        if arr.size == 0:
            continue
        sig[int(fid)] = (official.adjust_signal(arr[:, 0]),
                         official.adjust_signal(arr[:, 1]))
    return sig


def detect(mast, deg, deg_kw, phase_kw):
    fs = official.SAMPLING_RATE
    _, _, thr = official.compute_start_and_threshold(
        mast, fs, percentile=phase_kw["adaptive_percentile"])
    phases = official.identify_mastication_phases(
        mast, thr, fs, min_pause_s=phase_kw["min_pause_s"],
        min_phase_duration_s=phase_kw["min_phase_duration_s"])
    events, *_ = official.identify_deglutition_events(deg, mast, phases, fs, **deg_kw)
    return [s for s, _ in events]


def counts_for(sig, deg_kw, phase_kw):
    out = {}
    for rid, (mast, deg) in sig.items():
        starts = detect(mast, deg, deg_kw, phase_kw)
        isi = float(np.diff(starts).mean() / official.SAMPLING_RATE) if len(starts) > 1 else np.nan
        out[rid] = (len(starts), isi)
    return out


def summarise(counts, base, adults, dysph):
    ids = list(counts)
    n = np.array([counts[i][0] for i in ids])
    n_base = np.array([base[i][0] for i in ids])
    mask = (n_base > 0) | (n > 0)
    rho = spearmanr(n[mask], n_base[mask]).statistic if mask.sum() > 3 else np.nan
    # dysphagia robustness on adults with >=1 swallow under this setting
    adf = pd.DataFrame({"ID": ids,
                        "isi": [counts[i][1] for i in ids],
                        "n": n}).merge(adults[["ID"]], on="ID", how="inner")
    adf = adf.merge(dysph, on="ID", how="left")
    # Drop subjects with no dysphagia status (unmatched by the left merge):
    # `NaN == False` is False, which would silently dump them into the non-
    # dysphagia group and unbalance it. Filter on the boolean mask directly.
    adf = adf[adf.dys.notna()].copy()
    adf["dys"] = adf.dys.astype(bool)
    a = adf[(adf.n > 0) & adf.dys]["isi"].dropna()
    b = adf[(adf.n > 0) & ~adf.dys]["isi"].dropna()
    if len(a) >= 3 and len(b) >= 3:
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        cliffs = 2 * U / (len(a) * len(b)) - 1
    else:
        p, cliffs = np.nan, np.nan
    return dict(total_events=int(n.sum()),
                zero_swallow_frac=float((n == 0).mean()),
                median_events=float(np.median(n)),
                rank_corr_vs_default=float(rho),
                dysphagia_isi_p=float(p) if p == p else np.nan,
                dysphagia_isi_cliffs=float(cliffs) if cliffs == cliffs else np.nan)


def run(raw_dir, master_csv, outdir):
    os.makedirs(f"{outdir}/tables", exist_ok=True)
    os.makedirs(f"{outdir}/figures", exist_ok=True)
    sig = _load(raw_dir)
    master = pd.read_csv(master_csv)
    adults = cohort.adults(master)
    dysph = adults[["ID", "dysphagia"]].copy()
    dysph["dys"] = _yes(dysph["dysphagia"])
    base = counts_for(sig, dict(DEG_DEFAULTS), dict(PHASE_DEFAULTS))

    rows = [dict(parameter="(default)", value="-", **summarise(base, base, adults, dysph))]
    for param, values in GRID.items():
        for v in values:
            deg_kw = dict(DEG_DEFAULTS); deg_kw[param] = v
            c = counts_for(sig, deg_kw, dict(PHASE_DEFAULTS))
            rows.append(dict(parameter=param, value=v, **summarise(c, base, adults, dysph)))
    for param, values in PHASE_GRID.items():
        for v in values:
            ph_kw = dict(PHASE_DEFAULTS); ph_kw[param] = v
            c = counts_for(sig, dict(DEG_DEFAULTS), ph_kw)
            rows.append(dict(parameter=param, value=v, **summarise(c, base, adults, dysph)))
    res = pd.DataFrame(rows)
    res.to_csv(f"{outdir}/tables/SENS_swallow_params.csv", index=False)
    _figure(res, f"{outdir}/figures/fig_sensitivity.png")
    return res


def _figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = res[res.parameter == "(default)"].iloc[0]
    d = res[res.parameter != "(default)"].copy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # left: total events per setting, grouped by parameter
    labels = [f"{r.parameter}={r.value}" for _, r in d.iterrows()]
    ax[0].barh(range(len(d)), d.total_events, color="steelblue")
    ax[0].axvline(base.total_events, color="crimson", ls="--", label=f"default={int(base.total_events)}")
    ax[0].set_yticks(range(len(d))); ax[0].set_yticklabels(labels, fontsize=6)
    ax[0].set_xlabel("total swallow events"); ax[0].set_title("Event count vs parameter"); ax[0].legend(fontsize=7)
    # right: rank corr vs default and dysphagia p
    ax[1].scatter(d.rank_corr_vs_default, range(len(d)), c="seagreen", s=18, label="rank corr vs default")
    ax[1].set_yticks(range(len(d))); ax[1].set_yticklabels(labels, fontsize=6)
    ax[1].axvline(1.0, color="grey", ls=":"); ax[1].set_xlim(0, 1.05)
    ax[1].set_xlabel("Spearman rank corr vs default"); ax[1].set_title("Per-subject count stability")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    import pathlib
    base = pathlib.Path(__file__).resolve().parents[2]
    out = pathlib.Path(__file__).resolve().parents[1] / "outputs"
    res = run(str(base / "dati_raw"), str(out / "master_dataset.csv"), str(out))
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(res.round(3).to_string(index=False))
