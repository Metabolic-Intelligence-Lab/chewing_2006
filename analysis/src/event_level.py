"""Event-level (single-swallow) mixed-effects analysis.

Uses the ~276 swallow events with a random intercept per subject (statsmodels
MixedLM), giving more power than per-subject aggregates and modelling within-meal
variability. Adults only; clinical / body-composition predictors come from the
master dataset. Continuous predictors and outcomes are z-scored so fixed-effect
coefficients are standardised (per-SD) and comparable.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import cohort
import swallow_morphology as morph
from stats import _yes

OUTCOMES = ["peak_norm", "area_norm", "rfd_norm", "duration_s",
            "rise_decay_ratio", "n_subpeaks", "sign_changes_per_s"]


def build_event_dataset(raw_dir, master_csv):
    master = pd.read_csv(master_csv)
    master = cohort.adults(master)            # adults only
    ev = morph.build_event_table(raw_dir)
    ev = ev[ev.ID.isin(master.ID)].copy()
    cov = master[["ID", "age", "sex", "muscle_pct", "handgrip_max", "bmi",
                  "dysphagia"]].copy()
    cov["dysphagia_bin"] = _yes(cov["dysphagia"]).astype(int)
    cov["sex_m"] = (cov["sex"] == "Uomo").astype(int)
    df = ev.merge(cov, on="ID", how="left")
    return df


def _z(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)


def fit_mixed(df, outcome, predictor, covars=("age", "sex_m")):
    """MixedLM: z(outcome) ~ predictor + covars + (1|ID). Returns the fixed
    effect of `predictor`. Continuous vars are z-scored."""
    d = df.copy()
    # Drop any covariate that coincides with the predictor so the predictor does
    # not appear twice in the model formula (dict.fromkeys below only dedups the
    # column selection, not the RHS terms built afterwards). The caller already
    # handles the `age` case, but this guards every predictor generically.
    covars = tuple(c for c in covars if c != predictor)
    cols = [outcome, predictor, "ID"] + list(covars)
    d = d[[c for c in dict.fromkeys(cols)]].dropna()
    if d["ID"].nunique() < 8 or len(d) < 20:
        return None
    d["_y"] = _z(d[outcome])
    terms = []
    for v in [predictor] + list(covars):
        if v == "sex_m":
            terms.append("sex_m")
        elif d[v].nunique() <= 2:
            terms.append(v)
        else:
            d[v + "_z"] = _z(d[v]); terms.append(v + "_z")
    rhs = " + ".join(terms)
    pred_term = predictor if (predictor == "sex_m" or d[predictor].nunique() <= 2) else predictor + "_z"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = None
        for method in ("lbfgs", "powell", "cg"):
            try:
                fit = smf.mixedlm(f"_y ~ {rhs}", d, groups=d["ID"]).fit(method=method)
                # accept only a converged, non-degenerate fit. Be conservative:
                # treat a fit as converged ONLY if it explicitly reports so; a
                # missing `converged` attribute is treated as NOT converged.
                if getattr(fit, "converged", False) and pred_term in fit.bse.index \
                        and np.isfinite(fit.bse[pred_term]) and fit.bse[pred_term] < 100:
                    m = fit
                    break
            except Exception:
                continue
    if m is None or pred_term not in m.params.index:
        return None
    return dict(outcome=outcome, predictor=predictor,
                n_events=len(d), n_subj=d["ID"].nunique(),
                beta=float(m.params[pred_term]), se=float(m.bse[pred_term]),
                p=float(m.pvalues[pred_term]))


def within_meal_dynamics(df):
    """Proper within-subject use of event-level data: does swallow morphology
    change with swallow order across the meal? (fatigue / progression)."""
    d = df.sort_values(["ID", "start_s"]).copy()
    d["order"] = d.groupby("ID").cumcount() + 1
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for out in ["peak_norm", "area_norm", "duration_s", "rfd_norm", "n_subpeaks"]:
            dd = d[[out, "order", "ID"]].dropna()
            if dd["ID"].nunique() < 8:
                continue
            dd["_y"] = _z(dd[out]); dd["order_z"] = _z(dd["order"])
            try:
                m = smf.mixedlm("_y ~ order_z", dd, groups=dd["ID"]).fit(method="lbfgs")
                rows.append(dict(outcome=out, n_events=len(dd), n_subj=dd["ID"].nunique(),
                                 beta_per_sd=float(m.params["order_z"]),
                                 p=float(m.pvalues["order_z"])))
            except Exception:
                continue
    return pd.DataFrame(rows)


def run(raw_dir, master_csv, outdir):
    """Focused scope: only the within-meal swallow-morphology dynamics are
    produced. The clinical / body-composition event-level mixed models were
    removed together with the rest of the clinical battery; fit_mixed() remains
    defined for reuse."""
    import os
    os.makedirs(f"{outdir}/tables", exist_ok=True)
    df = build_event_dataset(raw_dir, master_csv)
    df.to_csv(f"{outdir}/swallow_events_adults.csv", index=False)
    wm = within_meal_dynamics(df)
    wm.to_csv(f"{outdir}/tables/EV_within_meal.csv", index=False)
    return df, wm


if __name__ == "__main__":
    import pathlib
    base = pathlib.Path(__file__).resolve().parents[2]
    out = pathlib.Path(__file__).resolve().parents[1] / "outputs"
    df, wm = run(str(base / "dati_raw"), str(out / "master_dataset.csv"), str(out))
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("event-level adults:", len(df), "events,", df.ID.nunique(), "subjects")
    print("\n=== Within-meal swallow-morphology dynamics (standardised beta per SD) ===")
    print(wm.round(3).to_string(index=False))
