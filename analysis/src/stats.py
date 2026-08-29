"""Stage 4 - statistical analysis across the four focuses.

A - chewing<->swallowing coordination (within-signal correlations)
B - body composition & strength (EMG features vs BIA/handgrip/VO2), raw + age/sex-adjusted
C - clinical correlates (group comparisons, FDR-corrected)
D - age/sex effects

Conventions: Spearman correlations (non-normal features), Mann-Whitney for
group tests, Benjamini-Hochberg FDR within each test family, bootstrap CIs for r.
Primary cohort = adults (>=18); swallowing/coordination metrics restricted to
valid-swallow subjects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import statutil

RNG = np.random.default_rng(20260609)

CHEW_FEATURES = ["chew_n", "chew_tcyc_s", "chew_active_s", "chew_work",
                 "chew_work_rate", "chew_rate_hz", "chew_phases"]
SWALLOW_FEATURES = ["swallow_n", "swallow_work_rate", "swallow_active_s"]
COORD_FEATURES = ["chews_per_swallow", "swallows_per_phase", "xcorr_peak",
                  "xcorr_lag_s", "coactivation_jaccard",
                  "post_phase_swallow_latency_s", "frac_phases_with_swallow"]
BODY_VARS = ["handgrip_max", "muscle_index", "muscle_pct", "muscle_mass_kg",
             "fat_pct", "visceral_fat", "bmi", "vo2max", "resting_metab"]


# ---------- helpers ----------
def spearman_ci(x, y, n_boot=2000):
    m = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
    if len(m) < 6:
        return dict(n=len(m), r=np.nan, p=np.nan, lo=np.nan, hi=np.nan)
    a, b = m.iloc[:, 0].to_numpy(), m.iloc[:, 1].to_numpy()
    r, p = stats.spearmanr(a, b)
    idx = np.arange(len(a))
    boots = []
    for _ in range(n_boot):
        s = RNG.choice(idx, len(idx), replace=True)
        if np.std(a[s]) == 0 or np.std(b[s]) == 0:
            continue
        boots.append(stats.spearmanr(a[s], b[s]).statistic)
    lo, hi = (np.nanpercentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return dict(n=len(m), r=float(r), p=float(p), lo=float(lo), hi=float(hi))


def partial_spearman(df, x, y, covars=("age",), sex_col="sex"):
    """Spearman of x,y after regressing out covariates (rank-residualisation).
    Sex is dummy-coded if present in covars list as 'sex'. Covariates equal to
    x or y are skipped (e.g. age-vs-age)."""
    num_covars = [c for c in covars if c not in ("sex", x, y)]
    use_sex = "sex" in covars and sex_col in df.columns and sex_col not in (x, y)
    cols = [x, y] + num_covars
    d = df[cols].copy()
    if use_sex:
        d["_sex"] = (df[sex_col] == "Uomo").astype(float)
    d = d.dropna()
    if len(d) < 8:
        return dict(n=len(d), r=np.nan, p=np.nan)
    cov_cols = num_covars + (["_sex"] if use_sex else [])
    if not cov_cols:
        r, p = stats.spearmanr(d[x], d[y])
        return dict(n=len(d), r=float(r), p=float(p))
    Z = np.column_stack([np.ones(len(d))] + [stats.rankdata(d[c]) for c in cov_cols])

    def resid(v):
        rv = stats.rankdata(d[v])
        beta, *_ = np.linalg.lstsq(Z, rv, rcond=None)
        return rv - Z @ beta
    rx, ry = resid(x), resid(y)
    r = float(stats.pearsonr(rx, ry)[0])
    # p from the partial-correlation t with df = n - 2 - k (k covariates
    # partialled out). pearsonr's own p uses df = n - 2 and is anti-conservative.
    p = statutil.partial_corr_p(r, len(d), len(cov_cols))
    return dict(n=len(d), r=r, p=p)


def mannwhitney(df, feature, group_col, pos_label):
    g = df[group_col]
    a = df.loc[g == pos_label, feature].dropna()
    b = df.loc[(g.notna()) & (g != pos_label), feature].dropna()
    if len(a) < 5 or len(b) < 5:
        return dict(n_pos=len(a), n_neg=len(b), U=np.nan, p=np.nan,
                    med_pos=a.median() if len(a) else np.nan,
                    med_neg=b.median() if len(b) else np.nan, cliffs=np.nan)
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    cliffs = 2 * U / (len(a) * len(b)) - 1   # Cliff's delta effect size
    return dict(n_pos=len(a), n_neg=len(b), U=float(U), p=float(p),
                med_pos=float(a.median()), med_neg=float(b.median()),
                cliffs=float(cliffs))


def add_fdr(df, pcol="p", out="p_fdr"):
    df = df.copy()
    mask = df[pcol].notna()
    df[out] = np.nan
    if mask.sum() > 0:
        df.loc[mask, out] = multipletests(df.loc[mask, pcol], method="fdr_bh")[1]
    return df


# ---------- focus analyses ----------
def focus_A_coordination(df):
    d = df[df.is_adult & df.valid_swallow]
    pairs = [("chew_work_rate", "swallow_work_rate"), ("chew_n", "swallow_n"),
             ("chew_active_s", "swallow_active_s"), ("chew_work_rate", "swallow_n"),
             ("chew_n", "post_phase_swallow_latency_s")]
    rows = [dict(x=x, y=y, **spearman_ci(d[x], d[y])) for x, y in pairs]
    res = add_fdr(pd.DataFrame(rows))
    desc = d[COORD_FEATURES].describe().T[["count", "mean", "50%", "std", "min", "max"]]
    return res, desc


def focus_B_body(df):
    d = df[df.is_adult]
    feats = CHEW_FEATURES + ["chews_per_swallow", "swallow_work_rate"]
    rows = []
    for f in feats:
        for v in BODY_VARS:
            sub = d[d.valid_swallow] if f in SWALLOW_FEATURES + ["chews_per_swallow"] else d
            raw = spearman_ci(sub[f], sub[v])
            adj = partial_spearman(sub, f, v, covars=("age", "sex"))
            rows.append(dict(feature=f, body_var=v, n=raw["n"], r=raw["r"],
                             ci_lo=raw["lo"], ci_hi=raw["hi"], p=raw["p"],
                             r_adj=adj["r"], p_adj=adj["p"]))
    res = add_fdr(pd.DataFrame(rows))
    res = add_fdr(res, pcol="p_adj", out="p_adj_fdr")
    return res


def _yes(s):
    """Positive answer matcher robust to 'Si'/'Sì' and free text variants."""
    return s.astype(str).str.normalize("NFKD").str.encode("ascii", "ignore") \
            .str.decode("ascii").str.lower().str.startswith("si")


def focus_C_clinical(df):
    groups = [("chew_pain_bin", "chew_pain", lambda s: (s > 0)),
              ("dysphagia_bin", "dysphagia", _yes),
              ("night_byte_bin", "night_byte", _yes),
              ("reflux_bin", "reflux", _yes),
              ("occlusion_bin", "occlusion_problem", _yes),
              ("prosthesis_bin", "dental_prosthesis", _yes),
              ("headache_bin", "headache", _yes),
              ("chew_asym_bin", "chew_side", _yes)]
    d = df[df.is_adult].copy()
    feats = CHEW_FEATURES + ["chews_per_swallow", "swallow_work_rate",
                             "frac_phases_with_swallow"]
    rows = []
    for binname, col, fn in groups:
        if col not in d.columns:
            continue
        d[binname] = fn(d[col])
        for f in feats:
            sub = d[d.valid_swallow] if f in SWALLOW_FEATURES + ["chews_per_swallow", "frac_phases_with_swallow"] else d
            r = mannwhitney(sub.assign(**{binname: d[binname]}), f, binname, True)
            rows.append(dict(group=col, feature=f, **r))
    res = pd.DataFrame(rows)
    # FDR within each clinical-group family
    out = [add_fdr(g) for _, g in res.groupby("group", sort=False)]
    return pd.concat(out, ignore_index=True)


def focus_D_age_sex(df):
    d = df[df.is_adult]
    feats = CHEW_FEATURES + COORD_FEATURES + ["swallow_work_rate", "swallow_n"]
    age_rows, sex_rows = [], []
    for f in feats:
        sub = d[d.valid_swallow] if (f in SWALLOW_FEATURES + COORD_FEATURES) else d
        age_rows.append(dict(feature=f, **spearman_ci(sub[f], sub["age"])))
        sex_rows.append(dict(feature=f, **mannwhitney(sub, f, "sex", "Uomo")))
    age_res = add_fdr(pd.DataFrame(age_rows))
    sex_res = add_fdr(pd.DataFrame(sex_rows))
    return age_res, sex_res


def correlation_matrix(df):
    """Coordination-focused Spearman matrix: chewing + swallowing + coordination
    features only (the body-composition / clinical axes are out of scope now)."""
    d = df[df.is_adult & df.valid_swallow]
    cols = CHEW_FEATURES + SWALLOW_FEATURES + ["chews_per_swallow",
            "swallows_per_phase", "xcorr_peak", "xcorr_lag_s",
            "coactivation_jaccard", "post_phase_swallow_latency_s",
            "frac_phases_with_swallow"]
    cols = [c for c in cols if c in d.columns]
    return d[cols].corr(method="spearman")


def run(master_csv, outdir):
    """Focused scope: only the chewing<->swallowing COORDINATION statistics
    (former Focus A) are produced. The body-composition (B), clinical-group (C)
    and age/sex (D) batteries were removed to concentrate the manuscript on
    detection + coordination; their helper functions remain defined for reuse by
    other modules but are no longer run here."""
    import os
    os.makedirs(f"{outdir}/tables", exist_ok=True)
    df = pd.read_csv(master_csv)
    A_res, A_desc = focus_A_coordination(df)
    cmat = correlation_matrix(df)

    A_res.to_csv(f"{outdir}/tables/A_coordination_corr.csv", index=False)
    A_desc.to_csv(f"{outdir}/tables/A_coordination_describe.csv")
    cmat.to_csv(f"{outdir}/tables/correlation_matrix.csv")
    return dict(A=A_res, A_desc=A_desc, cmat=cmat, df=df)


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).resolve().parents[1] / "outputs"
    r = run(str(out / "master_dataset.csv"), str(out))
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== Focus A chew<->swallow coordination correlations ===")
    print(r["A"][["x", "y", "n", "r", "p", "p_fdr"]].to_string(index=False))
