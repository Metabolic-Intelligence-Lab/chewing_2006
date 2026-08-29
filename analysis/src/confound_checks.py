"""Construct-validity confound checks for the OMPI.

(1) Composition-vs-strength dissociation: because the EMG amplitude is uncalibrated
    and subcutaneous fat attenuates sEMG, an amplitude-based index may track body
    COMPOSITION (lean/fat ratio) rather than muscular FORCE. We test this by the
    pattern of associations: OMPI vs muscle-mass % (a fat-entangled ratio) vs
    muscle index (absolute mass/height²) vs handgrip (mechanical strength), plus
    partial correlations to see whether muscle or fat is the primary correlate.

(2) Zero-swallow selection bias: swallowing analyses run on the valid-swallow
    subset; if isolability correlates with physiology this biases swallow–body
    correlations. We test whether valid- vs zero-swallow subjects differ on body
    composition / strength.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, pearsonr, mannwhitneyu

import cohort
import statutil


def _ompi_core(master_csv, cycles_csv):
    m = cohort.adults(pd.read_csv(master_csv))
    chpk = pd.read_csv(cycles_csv).groupby("ID")["peak"].mean().rename("chew_peak")
    m = m.merge(chpk, on="ID", how="left")
    s = pd.to_numeric(m["chew_peak"], errors="coerce")
    m["OMPI"] = (s - s.mean()) / s.std()
    return m


def _partial(d, x, y, ctrl):
    d = d[[x, y, ctrl]].dropna()
    if len(d) < 10:
        return np.nan, np.nan, len(d)
    R = {c: rankdata(d[c]) for c in (x, y, ctrl)}

    def res(a):
        A = np.c_[np.ones(len(d)), R[ctrl]]
        b, *_ = np.linalg.lstsq(A, R[a], rcond=None)
        return R[a] - A @ b
    r, _ = pearsonr(res(x), res(y))
    # p-value from the partial-correlation t with the CORRECT residual df
    # = n - 2 - k (here k = 1 covariate partialled out). pearsonr's own p uses
    # df = n - 2 and is anti-conservative for a partial correlation.
    p = statutil.partial_corr_p(float(r), len(d), 1)
    return float(r), p, len(d)


def dissociation(m):
    rows = []
    for t, kind in [("muscle_pct", "ratio (fat-entangled)"),
                    ("muscle_index", "absolute mass"),
                    ("muscle_mass_kg", "absolute mass"),
                    ("handgrip_max", "mechanical strength"),
                    ("fat_pct", "adiposity"), ("visceral_fat", "adiposity"),
                    ("bmi", "adiposity"), ("vo2max", "fitness")]:
        x = m[["OMPI", t]].dropna()
        if len(x) > 8:
            r, p = spearmanr(x["OMPI"], x[t])
            rows.append(dict(target=t, interpretation=kind, n=len(x),
                             r=round(float(r), 3), p=round(float(p), 4)))
    diss = pd.DataFrame(rows)
    # CAVEAT - structural collinearity: in BIA, muscle% + fat% (+ a small bone/
    # water share) are compositional and sum to ~constant, so muscle_pct and
    # fat_pct are near-perfectly negatively dependent. Partialling one out of the
    # other (e.g. "muscle_pct | fat_pct" or "fat_pct | muscle_pct") is therefore
    # ill-conditioned and almost tautological: the residuals carry little
    # independent information. These reciprocal partials must NOT be read as
    # evidence that one composition variable adds signal beyond the other.
    parts = []
    for x, y, c in [("OMPI", "muscle_pct", "fat_pct"), ("OMPI", "muscle_pct", "bmi"),
                    ("OMPI", "muscle_pct", "visceral_fat"), ("OMPI", "fat_pct", "muscle_pct")]:
        r, p, n = _partial(m, x, y, c)
        parts.append(dict(relation=f"{y} | {c}", partial_r=round(r, 3),
                          p=round(p, 4) if p == p else np.nan, n=n))
    return diss, pd.DataFrame(parts)


def selection_bias(m):
    m = m.copy()
    m["valid"] = m["swallow_n"] > 0
    rows = []
    for t in ["muscle_pct", "muscle_index", "fat_pct", "bmi", "visceral_fat",
              "age", "handgrip_max", "chew_peak", "vo2max"]:
        a = pd.to_numeric(m.loc[m.valid, t], errors="coerce").dropna()
        b = pd.to_numeric(m.loc[~m.valid, t], errors="coerce").dropna()
        if len(a) >= 4 and len(b) >= 4:
            U, p = mannwhitneyu(a, b)
            rows.append(dict(variable=t, valid_med=round(float(a.median()), 2),
                             zero_med=round(float(b.median()), 2),
                             n_valid=len(a), n_zero=len(b), p=round(float(p), 4)))
    return pd.DataFrame(rows)


def run(master_csv, cycles_csv, outdir):
    os.makedirs(f"{outdir}/tables", exist_ok=True)
    m = _ompi_core(master_csv, cycles_csv)
    diss, parts = dissociation(m)
    diss.to_csv(f"{outdir}/tables/CONF_dissociation.csv", index=False)
    parts.to_csv(f"{outdir}/tables/CONF_partials.csv", index=False)
    selection_bias(m).to_csv(f"{outdir}/tables/CONF_selection_bias.csv", index=False)
    return diss, parts, selection_bias(m)


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).resolve().parents[1] / "outputs"
    diss, parts, sb = run(str(out / "master_dataset.csv"), str(out / "chew_cycles.csv"), str(out))
    pd.set_option("display.width", 200)
    print("=== Composition-vs-strength dissociation (OMPI ~ ...) ===")
    print(diss.to_string(index=False))
    print("\n=== Partial correlations ===")
    print(parts.to_string(index=False))
    print("\n=== Zero-swallow selection bias (valid vs zero) ===")
    print(sb.to_string(index=False))
