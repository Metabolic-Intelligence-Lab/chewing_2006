"""Per-event swallow morphology + per-subject amplitude normalisation.

For every official deglutition event on the mylohyoid channel we extract the
shape of the single swallow (rise/decay, symmetry, FWHM, rate-of-rise/RFD,
sub-peaks => piecemeal swallowing, smoothness). Amplitudes are normalised to
the subject's own channel reference (95th percentile of the channel envelope),
a %MVC-like normalisation that removes per-subject/per-electrode gain so that
intensity features become comparable across subjects.

Outputs:
  - event-level table (one row per swallow, ~276 rows) for mixed-models
  - per-subject aggregates (mean + CV of key morphology) to merge into master
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

import official

FS = official.SAMPLING_RATE
EPS = 1e-9
# Physical floor on the per-subject reference: the envelope is in the same a.u.
# as the chewing/swallow signal where active runs are gated at THR=1. A 95th
# percentile below this means the channel is essentially flat/disconnected; we
# clamp so a degenerate reference cannot inflate every *_norm feature.
REF_FLOOR = 1.0


def channel_reference(sig: np.ndarray) -> float:
    """Per-subject within-channel amplitude reference: 95th percentile of the
    active (non-zero) envelope, floored at REF_FLOOR. Removes gain for
    %reference normalisation while protecting against near-flat channels."""
    active = sig[sig > 0]
    if len(active) < 10:
        active = sig
    ref = float(np.percentile(active, 95)) if len(active) else 1.0
    return max(ref, REF_FLOOR)


def event_features(deg: np.ndarray, s: int, e: int, fs: float,
                   deg_ref: float) -> dict:
    seg = np.asarray(deg[s:e], dtype=float)
    n = len(seg)
    if n < 3:
        return {}
    dur = (n - 1) / fs
    pk_i = int(np.argmax(seg))
    peak = float(seg[pk_i])
    mean_amp = float(seg.mean())
    area = float(np.trapezoid(seg) / fs)                 # a.u.*s
    rise_t = pk_i / fs
    decay_t = (n - 1 - pk_i) / fs
    tmin = 0.05                                          # floor: avoid div-by-~0 blow-ups
    rise_tf, decay_tf = max(rise_t, tmin), max(decay_t, tmin)
    # FWHM (seconds above half-peak)
    half = peak / 2.0
    above = np.where(seg >= half)[0]
    fwhm = (above[-1] - above[0]) / fs if len(above) > 1 else dur
    # rate of force development: peak / rise time, and max instantaneous slope
    rfd = peak / rise_tf
    # max_slope: smooth (50 ms median) before differentiating so the slope
    # tracks the rising contraction, not a single-sample envelope spike.
    seg_f = median_filter(seg, size=max(3, int(round(0.05 * fs))))
    max_slope = float(np.max(np.diff(seg_f)) * fs) if n > 1 else np.nan
    # sub-peaks within the burst (piecemeal swallowing): smooth to suppress
    # envelope ripple, require prominence and a minimum inter-peak distance
    seg_s = median_filter(seg, size=max(3, int(0.07 * fs)))
    sub, _ = find_peaks(seg_s, height=peak * 0.4, prominence=peak * 0.20,
                        distance=max(1, int(0.25 * fs)))
    n_subpeaks = max(1, int(len(sub)))
    # smoothness: derivative sign-changes per second (jitter; lower=smoother)
    d = np.diff(seg)
    sc = np.sign(d); sc = sc[sc != 0]
    sign_changes_per_s = float(np.sum(sc[1:] != sc[:-1]) / dur) if (len(sc) > 1 and dur > 0) else 0.0
    return dict(
        duration_s=dur, peak=peak, mean_amp=mean_amp, area=area,
        rise_time_s=rise_t, decay_time_s=decay_t,
        rise_decay_ratio=rise_tf / decay_tf,
        time_to_peak_s=rise_t, fwhm_s=fwhm,
        rfd=rfd, max_slope=max_slope,
        area_to_peak=area / (peak + EPS),
        n_subpeaks=n_subpeaks, sign_changes_per_s=sign_changes_per_s,
        # normalised (within-channel %reference)
        peak_norm=peak / (deg_ref + EPS),
        mean_amp_norm=mean_amp / (deg_ref + EPS),
        area_norm=area / (deg_ref + EPS),
        rfd_norm=rfd / (deg_ref + EPS),
    )


def build_event_table(raw_dir: str, official_res=None) -> pd.DataFrame:
    res = official_res or official.analyze_all(raw_dir)
    rows = []
    for rid, (summary, extra) in res.items():
        deg = extra["deg"]; fs = extra["fs"]
        deg_ref = channel_reference(deg)
        # flag subjects whose reference is degenerate (channel near-flat: raw
        # 95th pctile at/below the floor) so downstream *_norm features can be
        # treated with caution / excluded.
        active = deg[deg > 0]
        raw_ref = float(np.percentile(active, 95)) if len(active) else 0.0
        ref_degenerate = bool(raw_ref <= REF_FLOOR)
        for k, (s, e) in enumerate(extra["events"]):
            f = event_features(deg, s, e, fs, deg_ref)
            if not f:
                continue
            f.update(ID=int(rid), event_idx=k, start_s=s / fs,
                     deg_ref=deg_ref, ref_degenerate=ref_degenerate,
                     n_events=len(extra["events"]))
            rows.append(f)
    return pd.DataFrame(rows)


# morphology features aggregated per subject (mean + coefficient of variation)
MORPH = ["duration_s", "peak", "area", "rise_time_s", "decay_time_s",
         "rise_decay_ratio", "fwhm_s", "rfd", "n_subpeaks",
         "sign_changes_per_s", "peak_norm", "area_norm", "rfd_norm"]


def subject_aggregates(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["ID"])
    g = events.groupby("ID")
    out = {}
    for f in MORPH:
        out[f"sw_{f}_mean"] = g[f].mean()
    # coefficients of variation (rhythm/amplitude regularity) for selected
    for f in ["peak", "duration_s"]:
        out[f"sw_{f}_cv"] = g[f].std() / (g[f].mean().abs() + EPS)
    agg = pd.DataFrame(out)
    # piecemeal-swallow fraction (>=2 sub-peaks)
    agg["sw_piecemeal_frac"] = g["n_subpeaks"].apply(lambda x: float((x >= 2).mean()))
    return agg.reset_index()


def chewing_reference_table(raw_dir: str, official_res=None) -> pd.DataFrame:
    """Per-subject masseter reference (for chewing-amplitude normalisation)."""
    res = official_res or official.analyze_all(raw_dir)
    rows = []
    for rid, (summary, extra) in res.items():
        rows.append(dict(ID=int(rid),
                         chew_ref=channel_reference(extra["mast"]),
                         deg_ref=channel_reference(extra["deg"])))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import pathlib
    base = pathlib.Path(__file__).resolve().parents[2]
    out = pathlib.Path(__file__).resolve().parents[1] / "outputs"
    out.mkdir(exist_ok=True)
    ev = build_event_table(str(base / "dati_raw"))
    ev.to_csv(out / "swallow_events.csv", index=False)
    agg = subject_aggregates(ev)
    print("event-level rows:", len(ev), "| subjects with events:", ev.ID.nunique())
    pd.set_option("display.width", 200, "display.max_columns", 40)
    print("\nevent feature medians:")
    print(ev[["duration_s", "rise_time_s", "decay_time_s", "rise_decay_ratio",
              "fwhm_s", "rfd", "n_subpeaks", "peak_norm", "area_norm"]].median().round(3).to_string())
    print("\nper-subject aggregate columns:", list(agg.columns))
