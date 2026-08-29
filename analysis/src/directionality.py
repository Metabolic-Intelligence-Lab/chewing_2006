"""Frequency-domain coupling and temporal directionality between the masseter
(chewing) and mylohyoid (swallowing) envelopes.

- Magnitude-squared coherence in the chewing band (0.5-3 Hz): how rhythmically
  coupled the two channels are.
- Granger causality (bivariate VAR on downsampled, differenced envelopes):
  does masseter activity predict mylohyoid activity (chew->swallow) more than
  the reverse? Per-subject, aggregated across the cohort.

Exploratory: Granger on physiological envelopes indicates temporal precedence,
not mechanism.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy.signal import coherence, decimate
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR

import official

FS = official.SAMPLING_RATE
DS = 5                      # decimate 100 -> 20 Hz for Granger/coherence
FS_DS = FS // DS
CHEW_BAND = (0.5, 3.0)
MAXLAG = 5                  # up to ~0.25 s at 20 Hz


def band_coherence(mast, deg, fs=FS_DS, band=CHEW_BAND):
    nper = min(len(mast), 256)
    if nper < 32:
        return np.nan
    f, Cxy = coherence(mast, deg, fs=fs, nperseg=nper)
    sel = (f >= band[0]) & (f <= band[1])
    return float(np.mean(Cxy[sel])) if sel.any() else np.nan


def _stationary(x):
    x = np.asarray(x, float)
    x = np.diff(x)                       # difference to remove drift
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def select_lag(data, maxlag=MAXLAG):
    """Pick a single VAR order by BIC on the bivariate system. Reporting the
    Granger test at one principled lag avoids the false-positive inflation of
    taking the minimum p-value across all lags."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            lag = int(VAR(np.asarray(data, float)).select_order(maxlag).bic)
        except Exception:
            lag = 1
    return max(1, lag)


def granger_pair(driver, target, lag):
    """Granger test for driver -> target (target's past + driver's past vs
    target's past only) at a single, pre-selected lag. Returns (F, p)."""
    data = np.column_stack([target, driver])     # col0=target, col1=driver
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = grangercausalitytests(data, maxlag=[lag], verbose=False)
        except Exception:
            return np.nan, np.nan
    ft = res[lag][0]["ssr_ftest"]
    return float(ft[0]), float(ft[1])


def analyze(mast, deg):
    m = decimate(mast, DS, ftype="fir") if len(mast) > 3 * DS else mast[::DS]
    d = decimate(deg, DS, ftype="fir") if len(deg) > 3 * DS else deg[::DS]
    n = min(len(m), len(d))
    m, d = m[:n], d[:n]
    coh = band_coherence(m, d)
    ms, ds = _stationary(m), _stationary(d)
    n2 = min(len(ms), len(ds))
    if n2 < 30:
        return dict(coherence_chewband=coh, gc_chew2swallow_p=np.nan,
                    gc_swallow2chew_p=np.nan, gc_net_chew_drives=np.nan)
    ms, ds = ms[:n2], ds[:n2]
    lag = select_lag(np.column_stack([ds, ms]))       # one BIC-selected order
    F_cs, p_cs = granger_pair(driver=ms, target=ds, lag=lag)   # chew -> swallow
    F_sc, p_sc = granger_pair(driver=ds, target=ms, lag=lag)   # swallow -> chew
    # net directionality: positive => chewing drives swallowing more
    net = (-np.log10(p_cs + 1e-12)) - (-np.log10(p_sc + 1e-12))
    return dict(coherence_chewband=coh, gc_lag=int(lag),
                gc_chew2swallow_F=F_cs, gc_chew2swallow_p=p_cs,
                gc_swallow2chew_F=F_sc, gc_swallow2chew_p=p_sc,
                gc_net_chew_drives=float(net))


def run(raw_dir, outdir, official_res=None):
    os.makedirs(f"{outdir}/tables", exist_ok=True)
    os.makedirs(f"{outdir}/figures", exist_ok=True)
    res = official_res or official.analyze_all(raw_dir)
    rows = []
    for rid, (summary, extra) in res.items():
        r = analyze(extra["mast"], extra["deg"])
        r["ID"] = int(rid)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f"{outdir}/tables/DIR_directionality.csv", index=False)
    _figure(df, f"{outdir}/figures/fig_directionality.png")
    return df


def _figure(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].hist(df.coherence_chewband.dropna(), bins=20, color="purple", alpha=.8)
    ax[0].set_title("Chew-band coherence (0.5-3 Hz)"); ax[0].set_xlabel("magnitude-squared coherence")
    sig_cs = (df.gc_chew2swallow_p < 0.05).mean() * 100
    sig_sc = (df.gc_swallow2chew_p < 0.05).mean() * 100
    ax[1].bar(["chew→swallow", "swallow→chew"], [sig_cs, sig_sc],
              color=["steelblue", "darkorange"])
    ax[1].set_ylabel("% subjects with significant Granger (p<0.05)")
    ax[1].set_title("Directional coupling prevalence")
    ax[2].hist(df.gc_net_chew_drives.dropna(), bins=20, color="seagreen", alpha=.8)
    ax[2].axvline(0, color="k", ls="--")
    ax[2].set_title("Net directionality\n(>0: chewing drives swallowing)")
    ax[2].set_xlabel("Δ -log10(p)  [chew→sw  −  sw→chew]")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    import pathlib
    base = pathlib.Path(__file__).resolve().parents[2]
    out = pathlib.Path(__file__).resolve().parents[1] / "outputs"
    df = run(str(base / "dati_raw"), str(out))
    pd.set_option("display.width", 200)
    print("subjects:", len(df))
    print("chew-band coherence: median %.2f (IQR %.2f-%.2f)" % (
        df.coherence_chewband.median(), df.coherence_chewband.quantile(.25),
        df.coherence_chewband.quantile(.75)))
    print("Granger significant (p<0.05): chew→swallow %.0f%% | swallow→chew %.0f%%" % (
        (df.gc_chew2swallow_p < 0.05).mean() * 100, (df.gc_swallow2chew_p < 0.05).mean() * 100))
    print("net directionality (>0 chew drives): median %.2f, %% positive %.0f%%" % (
        df.gc_net_chew_drives.median(), (df.gc_net_chew_drives > 0).mean() * 100))
