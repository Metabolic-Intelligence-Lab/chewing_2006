"""Cycle-level (single-chew) extraction and within-sequence / rhythm analysis.

The official pipeline collapses each recording to per-subject scalars; here we
recover every individual masticatory cycle (~5300 chews) so within-subject
structure can be modelled at native resolution:
  - cycle table: amplitude, duration, area, rise/decay, inter-cycle interval,
    instantaneous rate, order within phase and within meal
  - within-sequence dynamics: do amplitude / duration / rate change along the
    chewing sequence? (bolus reduction, slowing, fatigue) via mixed-effects
  - rhythm & spectrum per subject: chewing fundamental frequency, cycle-to-cycle
    variability (CV of inter-cycle interval), spectral entropy, rate drift
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy.signal import welch
import statsmodels.formula.api as smf

import official
import segmentation
from statutil import _z

FS = official.SAMPLING_RATE


def extract_cycles(raw_dir, official_res=None) -> pd.DataFrame:
    res = official_res or official.analyze_all(raw_dir)
    rows = []
    for rid, (summary, extra) in res.items():
        mast = extra["mast"]; fs = extra["fs"]
        g_order = 0
        for ph_i, (ps, pe) in enumerate(extra["phases"]):
            seg = mast[ps:pe]
            # ICI is measured within a phase only: reset so the first cycle of
            # each phase has no (spurious) interval back to the previous phase.
            prev_global_start = None
            c_in_phase = 0
            phase_starts = []
            # contiguous above-threshold runs = individual chews (shared gate)
            for a, b, s in segmentation.segment_runs(seg):
                pk_i = int(np.argmax(s))
                dur = (b - a) / fs
                g_start = (ps + a) / fs
                ici = (g_start - prev_global_start) if prev_global_start is not None else np.nan
                rows.append(dict(
                    ID=int(rid), phase_idx=ph_i, cycle_in_phase=c_in_phase,
                    global_order=g_order, t_start_s=g_start,
                    duration_s=dur, peak=float(s.max()), mean_amp=float(s.mean()),
                    area=float(np.trapezoid(s) / fs),
                    rise_time_s=pk_i / fs, decay_time_s=(len(s) - 1 - pk_i) / fs,
                    inter_cycle_interval_s=ici,
                    inst_rate_hz=(1.0 / ici) if (ici and ici == ici and ici > 0) else np.nan,
                ))
                phase_starts.append(g_start)
                c_in_phase += 1; g_order += 1
                prev_global_start = g_start
    df = pd.DataFrame(rows)
    # normalised position within phase (0=first .. 1=last)
    df["phase_n"] = df.groupby(["ID", "phase_idx"])["cycle_in_phase"].transform("max") + 1
    df["pos_in_phase"] = df["cycle_in_phase"] / df["phase_n"].clip(lower=1)
    df["meal_n"] = df.groupby("ID")["global_order"].transform("max") + 1
    df["pos_in_meal"] = df["global_order"] / df["meal_n"].clip(lower=1)
    return df


# ---------- within-sequence dynamics ----------
def within_sequence(cycles, adults_ids):
    """Mixed-effects: cycle feature ~ position-in-meal, random intercept per
    subject (+ phase). Slope sign reveals bolus reduction (amp/dur down) and
    slowing (rate down) across the sequence."""
    d = cycles[cycles.ID.isin(adults_ids)].copy()
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for out in ["peak", "duration_s", "area", "inst_rate_hz", "rise_time_s"]:
            dd = d[[out, "pos_in_meal", "ID"]].dropna()
            if dd.ID.nunique() < 10 or len(dd) < 100:
                continue
            dd["_y"] = _z(dd[out]); dd["pos"] = _z(dd["pos_in_meal"])
            try:
                m = smf.mixedlm("_y ~ pos", dd, groups=dd["ID"]).fit(method="lbfgs")
                rows.append(dict(outcome=out, n_cycles=len(dd), n_subj=dd.ID.nunique(),
                                 beta_per_sd=float(m.params["pos"]),
                                 p=float(m.pvalues["pos"])))
            except Exception:
                continue
    return pd.DataFrame(rows)


# ---------- rhythm & spectrum per subject ----------
def spectral_entropy(x, fs, band=(0.3, 4.0)):
    nper = min(len(x), 512)
    if nper < 32:
        return np.nan, np.nan
    f, p = welch(x - x.mean(), fs=fs, nperseg=nper)
    sel = (f >= band[0]) & (f <= band[1])
    if sel.sum() < 3 or p[sel].sum() <= 0:
        return np.nan, np.nan
    pf = p[sel] / p[sel].sum()
    ent = float(-np.sum(pf * np.log(pf + 1e-12)) / np.log(len(pf)))
    dom = float(f[sel][np.argmax(p[sel])])
    return dom, ent


def rhythm_table(cycles, official_res):
    rows = []
    for rid, (summary, extra) in official_res.items():
        c = cycles[cycles.ID == rid]
        ici = c["inter_cycle_interval_s"].dropna()
        ici = ici[(ici > 0) & (ici < 5)]            # plausible chew intervals
        rate = c["inst_rate_hz"].dropna()
        dom, ent = spectral_entropy(extra["mast"], extra["fs"])
        # rate drift: first third vs last third of the meal (slowing if <0)
        cc = c.sort_values("global_order")
        n = len(cc)
        drift = np.nan
        if n >= 9:
            k = n // 3
            r1 = cc["inst_rate_hz"].iloc[:k].dropna().median()
            r3 = cc["inst_rate_hz"].iloc[-k:].dropna().median()
            if r1 == r1 and r3 == r3:
                drift = float(r3 - r1)
        rows.append(dict(
            ID=int(rid),
            chew_freq_hz=float(1.0 / ici.median()) if len(ici) else np.nan,
            ici_cv=float(ici.std() / ici.mean()) if len(ici) > 2 else np.nan,
            cycle_amp_cv=float(c["peak"].std() / c["peak"].mean()) if len(c) > 2 else np.nan,
            spectral_dom_hz=dom, spectral_entropy=ent,
            rate_drift_hz=drift, n_cycles=int(len(c)),
        ))
    return pd.DataFrame(rows)


def within_sequence_by_group(cycles, master_csv):
    """Does the WITHIN-MEAL trajectory differ by clinical group? Mixed-effects
    z(feature) ~ pos * group + (1|ID); the interaction (pos:group) is the
    difference in within-meal slope between group+ and group−. FDR within feature."""
    import cohort
    from stats import _yes, add_fdr
    m = cohort.adults(pd.read_csv(master_csv))
    c = cycles[cycles.ID.isin(m.ID)].copy()
    c["pos"] = _z(c["pos_in_meal"])
    groups = {"occlusion": _yes(m.occlusion_problem), "dysphagia": _yes(m.dysphagia),
              "sex_M": (m.sex == "Uomo"), "chew_pain": (m.chew_pain > 0),
              "bruxism": _yes(m.night_byte), "reflux": _yes(m.reflux),
              "smoker": _yes(m.smoker)}
    gmap = {k: dict(zip(m.ID, v.astype(float))) for k, v in groups.items()}
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for gname, gd in gmap.items():
            c["g"] = c.ID.map(gd)
            for f in ["duration_s", "rise_time_s", "inst_rate_hz", "peak"]:
                d = c.dropna(subset=[f, "g", "pos"]).copy()
                if d.ID.nunique() < 12:
                    continue
                d["_y"] = _z(d[f])
                try:
                    mm = smf.mixedlm("_y ~ pos * g", d, groups=d["ID"]).fit(method="lbfgs")
                    rows.append(dict(group=gname, cycle_feature=f,
                                     interaction_beta=round(float(mm.params.get("pos:g", np.nan)), 3),
                                     p=round(float(mm.pvalues.get("pos:g", np.nan)), 4),
                                     n_subj=d.ID.nunique(), n_cyc=len(d)))
                except Exception:
                    continue
    res = pd.DataFrame(rows)
    out = [add_fdr(g) for _, g in res.groupby("cycle_feature", sort=False)]
    return pd.concat(out, ignore_index=True) if out else res


def _figure_group(cycles, master_csv, outdir, wsg=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import cohort
    m = cohort.adults(pd.read_csv(master_csv))
    if wsg is None:
        wsg = within_sequence_by_group(cycles, master_csv)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) interaction heatmap: ALL groups x cycle features (dynamic-axis screen)
    flab = {"rise_time_s": "rise time", "duration_s": "duration",
            "inst_rate_hz": "inst. rate", "peak": "peak"}
    fcols = [c for c in ["rise_time_s", "duration_s", "inst_rate_hz", "peak"]
             if c in set(wsg.cycle_feature)]
    piv = wsg.pivot(index="group", columns="cycle_feature", values="interaction_beta")[fcols]
    sig = wsg.pivot(index="group", columns="cycle_feature", values="p_fdr")[fcols]
    piv = piv.reindex(piv.abs().max(axis=1).sort_values(ascending=False).index)
    sig = sig.reindex(piv.index)
    annot = sig.map(lambda v: "*" if (pd.notna(v) and v < 0.1) else "")
    sns.heatmap(piv, cmap="RdBu_r", center=0, vmin=-0.12, vmax=0.12, ax=ax[0],
                annot=annot.values, fmt="", annot_kws={"color": "black", "fontsize": 12},
                cbar_kws={"label": "within-meal slope difference\n(group+ − group−), β/SD", "shrink": .8},
                linewidths=.4, linecolor="white")
    ax[0].set_xticklabels([flab.get(c, c) for c in fcols], rotation=0, fontsize=8)
    ax[0].set_title("(a) Which groups shift the within-meal trajectory\n(pos×group interaction; * = FDR<0.1)",
                    fontsize=9)
    ax[0].set_xlabel(""); ax[0].set_ylabel("")

    # (b) the strongest example: chewing-pain rise-time trajectory
    c = cycles[cycles.ID.isin(m.ID)].copy()
    c["pain"] = c.ID.map(dict(zip(m.ID, (m.chew_pain > 0))))
    c["bin"] = (c["pos_in_meal"] * 10).clip(0, 9).astype(int)
    for lab, col, sel in [("chewing pain", "crimson", c.pain == True),
                          ("no pain", "steelblue", c.pain == False)]:
        gb = c[sel].groupby("bin")["rise_time_s"].mean()
        ax[1].plot(gb.index, gb.values, "o-", color=col, label=lab, ms=4)
    ax[1].set_xlabel("position in meal (deciles)"); ax[1].set_ylabel("chew rise time [s]")
    ax[1].set_title("(b) Example: rise-time trajectory by chewing pain\n(steepest, FDR-significant)", fontsize=9)
    ax[1].legend(fontsize=8)
    fig.suptitle("Within-meal cycle dynamics by clinical group — pain and reflux reshape the "
                 "trajectory; occlusion does not", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{outdir}/figures/fig_cycle_group.png", dpi=130); plt.close(fig)


def _figure(cycles, rhythm, adults_ids, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = cycles[cycles.ID.isin(adults_ids)].copy()
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    # within-meal trajectory: binned mean duration & rate vs position
    d["bin"] = (d["pos_in_meal"] * 10).clip(0, 9).astype(int)
    gb = d.groupby("bin")
    ax[0].plot(gb["duration_s"].mean(), "o-", color="steelblue", label="duration [s]")
    ax2 = ax[0].twinx()
    ax2.plot(gb["inst_rate_hz"].mean(), "s--", color="darkorange", label="rate [Hz]")
    ax[0].set_xlabel("position in meal (deciles)"); ax[0].set_ylabel("chew duration [s]", color="steelblue")
    ax2.set_ylabel("chew rate [Hz]", color="darkorange")
    ax[0].set_title("Within-meal cycle dynamics")
    ax[1].hist(rhythm.chew_freq_hz.dropna(), bins=20, color="purple", alpha=.8)
    ax[1].set_title("Chewing fundamental frequency"); ax[1].set_xlabel("Hz")
    ax[2].hist(rhythm.ici_cv.dropna(), bins=20, color="seagreen", alpha=.8)
    ax[2].set_title("Cycle-to-cycle variability (ICI CV)"); ax[2].set_xlabel("CV")
    fig.tight_layout(); fig.savefig(f"{outdir}/figures/fig_cycle_dynamics.png", dpi=130); plt.close(fig)


def run(raw_dir, master_csv, outdir):
    import cohort
    os.makedirs(f"{outdir}/tables", exist_ok=True)
    os.makedirs(f"{outdir}/figures", exist_ok=True)
    res = official.analyze_all(raw_dir)
    cycles = extract_cycles(raw_dir, official_res=res)
    cycles.to_csv(f"{outdir}/chew_cycles.csv", index=False)
    rhythm = rhythm_table(cycles, res)
    rhythm.to_csv(f"{outdir}/tables/CYC_rhythm.csv", index=False)
    master = pd.read_csv(master_csv)
    adults_ids = cohort.adults(master).ID.tolist()
    ws = within_sequence(cycles, adults_ids)
    ws.to_csv(f"{outdir}/tables/CYC_within_sequence.csv", index=False)
    wsg = within_sequence_by_group(cycles, master_csv)
    wsg.to_csv(f"{outdir}/tables/CYC_group_interaction.csv", index=False)
    _figure(cycles, rhythm, adults_ids, outdir)
    _figure_group(cycles, master_csv, outdir, wsg)
    return cycles, rhythm, ws


if __name__ == "__main__":
    import pathlib
    import cohort
    base = pathlib.Path(__file__).resolve().parents[2]
    out = pathlib.Path(__file__).resolve().parents[1] / "outputs"
    cycles, rhythm, ws = run(str(base / "dati_raw"), str(out / "master_dataset.csv"), str(out))
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("cycles:", len(cycles), "| subjects:", cycles.ID.nunique())
    print("\nrhythm medians:")
    print(rhythm[["chew_freq_hz", "ici_cv", "cycle_amp_cv", "spectral_dom_hz",
                  "spectral_entropy", "rate_drift_hz"]].median().round(3).to_string())
    print("\n=== Within-sequence dynamics (mixed; beta/SD per position-in-meal) ===")
    print(ws.round(4).to_string(index=False))
