"""Stage 0 - data-quality control (official-algorithm basis).

Per-recording QC: sampling frequency / duration (variable fs -> real time used
only for QC; the official pipeline assumes 100 Hz), ADC clipping, cross-channel
contamination, age distribution, zero-deglutition subjects, and validation of
the ported official features against database_finale_completo.xlsx.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import io_raw
import official

ADC_MAX = 1023


def _safe_max(a):
    """max() of an array, NaN if empty (avoids ValueError on n_samples==0)."""
    a = np.asarray(a, dtype=float)
    return float(a.max()) if a.size else np.nan


def _safe_xcorr(a, b):
    """Pearson corr between two channels; NaN if empty or zero-variance
    (np.corrcoef would emit a RuntimeWarning and return NaN)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size != a.size:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])
AGE_COL = "QUESTIONARI_Inserisci la tua\xa0età (in anni):"


def build_qc_table(recs, official_res, db):
    age = db[AGE_COL] if AGE_COL in db.columns else pd.Series(dtype=float)
    rows = []
    for rid, r in recs.items():
        ev = len(official_res[rid][1]["events"]) if rid in official_res else np.nan
        summ = official_res[rid][0] if rid in official_res else {}
        rows.append(dict(
            ID=rid, fs_hz=r.fs, duration_s=r.duration_s, n_samples=r.n_samples,
            clip_chew=int(_safe_max(r.ch_chew) >= ADC_MAX) if r.ch_chew.size else 0,
            clip_deg=int(_safe_max(r.ch_deg) >= ADC_MAX) if r.ch_deg.size else 0,
            xcorr_ch1_ch2=_safe_xcorr(r.ch_chew, r.ch_deg),
            chews_official=summ.get("number of chews", np.nan),
            deglutitions_official=summ.get("number of deglutition", np.nan),
            deg_events_official=ev,
            age=age.get(rid, np.nan),
        ))
    df = pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)
    df["minor"] = df["age"] < 18
    df["zero_swallow"] = df["deg_events_official"] == 0
    return df


def validate_against_file(official_res, db):
    """Confirm the port reproduces the official file (should be ~100% exact)."""
    cols = ["number of chews", "work", "work rate", "Cycle Time",
            "number of deglutition events", "number of deglutition"]
    rows = []
    for c in cols:
        if c not in db.columns:
            continue
        a, b = [], []
        for rid, (summ, _) in official_res.items():
            if rid in db.index and not pd.isna(db.loc[rid, c]):
                a.append(summ[c]); b.append(float(db.loc[rid, c]))
        a, b = np.array(a, float), np.array(b, float)
        if len(a) > 2:
            rows.append(dict(feature=c, n=len(a),
                             r=float(np.corrcoef(a, b)[0, 1]),
                             exact_match_pct=float(np.isclose(a, b, rtol=1e-3, atol=1e-3).mean() * 100)))
    return pd.DataFrame(rows)


def make_figures(df, recs, official_res, outdir):
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].hist(df.fs_hz, bins=20, color="steelblue"); ax[0].set_title("Sampling frequency [Hz]")
    ax[1].hist(df.duration_s, bins=20, color="indianred"); ax[1].set_title("Recording duration [s]")
    fig.tight_layout(); fig.savefig(f"{outdir}/qc_fs_duration.png", dpi=120); plt.close(fig)

    # example official-annotated trace
    rid = int(df.ID.iloc[0])
    if rid in official_res:
        summ, extra = official_res[rid]
        fs = extra["fs"]; t = np.arange(len(extra["mast"])) / fs
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.plot(t, extra["mast"], lw=.6, color="steelblue", label="masseter (chew)")
        ax.plot(t, extra["deg"], lw=.6, color="darkorange", alpha=.8, label="mylohyoid (swallow)")
        for s, e in extra["phases"]:
            ax.axvspan(s / fs, e / fs, color="green", alpha=.06)
        for s, e in extra["events"]:
            ax.axvspan(s / fs, e / fs, color="red", alpha=.35)
        ax.set_title(f"ID {rid:03d} - official phases (green) & deglutitions (red)")
        ax.set_xlabel("time [s]"); ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout(); fig.savefig(f"{outdir}/qc_example_trace.png", dpi=120); plt.close(fig)


def write_report(df, val, path):
    L = []
    A = L.append
    A("# QC report - Chewing/Swallowing dataset (official basis)\n")
    A(f"- Recordings (raw): **{len(df)}**")
    A(f"- Sampling frequency [Hz]: min {df.fs_hz.min():.1f}, median {df.fs_hz.median():.1f}, max {df.fs_hz.max():.1f} (official pipeline assumes 100 Hz)")
    A(f"- Duration [s]: min {df.duration_s.min():.0f}, median {df.duration_s.median():.0f}, max {df.duration_s.max():.0f}")
    A(f"- ADC clipping (>= {ADC_MAX}): masseter {int(df.clip_chew.sum())}, mylohyoid {int(df.clip_deg.sum())}")
    A(f"- Cross-channel envelope corr: median {df.xcorr_ch1_ch2.median():.2f} (IQR {df.xcorr_ch1_ch2.quantile(.25):.2f}-{df.xcorr_ch1_ch2.quantile(.75):.2f})")
    A("")
    A("## Port validation vs database_finale_completo.xlsx")
    for _, r in val.iterrows():
        A(f"- {r['feature']}: r={r['r']:.4f}, exact-match={r['exact_match_pct']:.1f}% (n={int(r['n'])})")
    A("")
    A("## Inclusion-relevant flags")
    A(f"- Minors (<18): **{int(df.minor.sum())}** -> excluded from analysis")
    A(f"- Zero official deglutitions: **{int(df.zero_swallow.sum())}** "
      f"(IDs: {sorted(df.loc[df.zero_swallow, 'ID'].tolist())})")
    with open(path, "w") as f:
        f.write("\n".join(L))


def run(raw_dir, db_path, outdir):
    recs = io_raw.load_all(raw_dir)
    official_res = official.analyze_all(raw_dir)
    db = pd.read_excel(db_path).dropna(subset=["ID"])
    db["ID"] = db["ID"].astype(int)
    db_idx = db.set_index("ID")
    df = build_qc_table(recs, official_res, db_idx)
    val = validate_against_file(official_res, db_idx)
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(f"{outdir}/qc_table.csv", index=False)
    val.to_csv(f"{outdir}/qc_port_validation.csv", index=False)
    make_figures(df, recs, official_res, f"{outdir}/figures")
    write_report(df, val, f"{outdir}/qc_report.md")
    return df, val


if __name__ == "__main__":
    import pathlib
    base = pathlib.Path(__file__).resolve().parents[2]
    out = pathlib.Path(__file__).resolve().parents[1] / "outputs"
    run(str(base / "dati_raw"), str(base / "database_finale_completo.xlsx"), str(out))
    print("QC done")
