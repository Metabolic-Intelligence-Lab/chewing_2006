"""Faithful port of the lab's official pipeline (analisi_chewing.py).

Reproduces the official masticatory phase / deglutition detection and feature
metrics so the dataset can be regenerated and validated end-to-end against
database_finale_completo.xlsx. Fixed SAMPLING_RATE=100 and integer signals as
in the original. Beyond the original, analyze_single_id() also returns the
adjusted signals, phases and events so coordination metrics can be derived.

Deglutition detection (key): mylohyoid peaks are accepted only inside phase-end
windows (-1 s .. +8 s after each chewing phase) and must be morphologically
smooth (few derivative sign-changes/s; low derivative-activity ratio) to reject
chewing contamination.
"""
from __future__ import annotations

import os
import re

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

SAMPLING_RATE = 100


# ---------- IO ----------
def load_and_preprocess_data(file_path):
    rows = []
    with open(file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != 3 or parts[0].strip().lower() == "data":
                continue
            vals = parts[2].strip().split()
            if len(vals) < 2:
                continue
            try:
                rows.append([int(float(vals[0])), int(float(vals[1]))])
            except ValueError:
                continue
    return np.array(rows, dtype=float)


def adjust_signal(signal, sampling_rate=SAMPLING_RATE, baseline_seconds=0.5):
    signal = np.asarray(signal, dtype=float)
    if len(signal) == 0:
        return signal
    n_off = max(1, min(int(baseline_seconds * sampling_rate), len(signal)))
    out = signal - np.mean(signal[:n_off])
    out[out < 0] = 0
    return out


def parse_id(file_name):
    block = os.path.splitext(os.path.basename(file_name))[0].split("_")[0]
    m = re.search(r"\d+", block)
    return f"{int(m.group()):03d}" if m else None


# ---------- start / threshold ----------
def find_analysis_start(mast, mast_threshold, sampling_rate=SAMPLING_RATE,
                        min_sustained_s=1.0, mast_filter_size=51):
    sm = median_filter(mast, size=mast_filter_size)
    min_sus = int(min_sustained_s * sampling_rate)
    above = sm >= mast_threshold
    count = 0
    for i, a in enumerate(above):
        if a:
            count += 1
            if count >= min_sus:
                return max(0, i - min_sus + 1)
        else:
            count = 0
    peaks, _ = find_peaks(mast, height=10)
    return int(peaks[0]) if len(peaks) else 0


def compute_start_and_threshold(mast, sampling_rate=SAMPLING_RATE, mast_filter_size=51,
                                percentile=25, min_sustained_s=1.0, min_active=5,
                                min_delay_after_start_s=5.0):
    sm = median_filter(mast, size=mast_filter_size)
    peaks, _ = find_peaks(mast, height=10)
    first_peak = int(peaks[0]) if len(peaks) else 0
    rough = sm[first_peak:]
    rough = rough[rough > min_active]
    thresh_rough = float(np.percentile(rough, percentile)) if len(rough) else 35.0
    analysis_start = find_analysis_start(mast, thresh_rough, sampling_rate,
                                         min_sustained_s, mast_filter_size)
    active = sm[analysis_start:]
    active = active[active > min_active]
    mast_threshold = float(np.percentile(active, percentile)) if len(active) else thresh_rough
    deg_start = analysis_start + int(min_delay_after_start_s * sampling_rate)
    return analysis_start, deg_start, mast_threshold


# ---------- phases ----------
def identify_mastication_phases(mast, mast_threshold, sampling_rate=SAMPLING_RATE,
                                mast_filter_size=51, min_pause_s=3.0, min_phase_duration_s=3.0):
    mast = np.asarray(mast, dtype=float)
    if len(mast) == 0:
        return []
    sm = median_filter(mast, size=mast_filter_size)
    min_pause = int(min_pause_s * sampling_rate)
    min_phase = int(min_phase_duration_s * sampling_rate)
    above = sm >= mast_threshold
    phases, in_phase, phase_start, below = [], False, 0, 0
    for i, a in enumerate(above):
        if a:
            if not in_phase:
                in_phase, phase_start = True, i
            below = 0
        elif in_phase:
            below += 1
            if below >= min_pause:
                end = i - below
                if (end - phase_start) >= min_phase:
                    phases.append((phase_start, end))
                in_phase, below = False, 0
    if in_phase:
        end = len(mast) - 1
        if (end - phase_start) >= min_phase:
            phases.append((phase_start, end))
    return phases


# ---------- zones ----------
def build_excluded_zones(deg_start, phases, sampling_rate=SAMPLING_RATE, phase_start_buffer_s=3.0):
    excluded = [(0, deg_start)]
    buf = int(phase_start_buffer_s * sampling_rate)
    for ps, pe in phases:
        ee = min(ps + buf, pe)
        if ee > ps:
            excluded.append((ps, ee))
    return excluded


def build_allowed_zones(phases, signal_length, sampling_rate=SAMPLING_RATE,
                        pre_phase_end_s=1.0, post_phase_end_s=8.0):
    pre, post = int(pre_phase_end_s * sampling_rate), int(post_phase_end_s * sampling_rate)
    allowed = []
    for ps, pe in phases:
        s, e = max(0, pe - pre), min(signal_length - 1, pe + post)
        if e > s:
            allowed.append((s, e))
    return allowed


def _in(idx, zones):
    return any(s <= idx < e for s, e in zones)


def _in_incl(idx, zones):
    return len(zones) == 0 or any(s <= idx <= e for s, e in zones)


def _overlaps(start, end, zones):
    return any(start < ze and end > zs for zs, ze in zones)


def _overlaps_allowed(start, end, zones):
    return len(zones) == 0 or any(start < ze and end > zs for zs, ze in zones)


# ---------- deglutition ----------
def identify_deglutition_events(deg, mast, phases, sampling_rate=SAMPLING_RATE,
                                env_filter_size=151, env_peak_height=8, env_prominence=5,
                                env_peak_distance=80, border_fraction=0.10, mast_filter_size=51,
                                adaptive_percentile=25, min_sustained_s=1.0,
                                min_delay_after_start_s=5.0, phase_start_buffer_s=3.0,
                                use_phase_end_windows=True, pre_phase_end_s=1.0,
                                post_phase_end_s=8.0, mast_relax_factor=1.50,
                                max_derivative_sign_changes_per_s=35.0,
                                max_derivative_activity_ratio=4.00, min_deg_amplitude=10,
                                min_duration_s=0.15, max_duration_s=5.0, fine_filter_size=7):
    deg = np.asarray(deg, dtype=float)
    mast = np.asarray(mast, dtype=float)
    analysis_start, deg_start, mast_threshold = compute_start_and_threshold(
        mast, sampling_rate, mast_filter_size, adaptive_percentile,
        min_sustained_s, min_delay_after_start_s=min_delay_after_start_s)
    deg_env = median_filter(deg, size=env_filter_size)
    sm_mast = median_filter(mast, size=mast_filter_size)
    deg_fine = median_filter(deg, size=fine_filter_size)
    excluded = build_excluded_zones(deg_start, phases or [], sampling_rate, phase_start_buffer_s)
    allowed = (build_allowed_zones(phases or [], len(deg), sampling_rate,
                                   pre_phase_end_s, post_phase_end_s)
               if use_phase_end_windows else [])
    peaks, props = find_peaks(deg_env, height=env_peak_height, distance=env_peak_distance,
                              prominence=env_prominence)
    if len(peaks) == 0:
        return [], mast_threshold, analysis_start, deg_start
    mask = peaks >= deg_start
    peaks = peaks[mask]
    if len(peaks) == 0:
        return [], mast_threshold, analysis_start, deg_start
    valleys = []
    for i in range(len(peaks) - 1):
        seg = deg_env[peaks[i]:peaks[i + 1]]
        valleys.append(int(np.argmin(seg)) + peaks[i])
    events = []
    for j, pk in enumerate(peaks):
        lb = (max(deg_start, pk - int(2.0 * sampling_rate)) if j == 0 else valleys[j - 1])
        rb = (min(len(deg) - 1, pk + int(2.0 * sampling_rate)) if j == len(peaks) - 1 else valleys[j])
        half = deg_env[pk] * border_fraction
        left = pk
        while left > lb and deg_env[left] > half:
            left -= 1
        right = pk
        while right < rb and deg_env[right] > half:
            right += 1
        if right - left < int(min_duration_s * sampling_rate):
            left = max(deg_start, pk - int(0.3 * sampling_rate))
            right = min(len(deg) - 1, pk + int(0.5 * sampling_rate))
        dur = (right - left) / sampling_rate
        max_amp = float(deg[left:right].max()) if right > left else 0.0
        if _in(pk, excluded) or _overlaps(left, right, excluded):
            continue
        if use_phase_end_windows and not (_in_incl(pk, allowed) or _overlaps_allowed(left, right, allowed)):
            continue
        if not (min_duration_s <= dur <= max_duration_s):
            continue
        seg = deg_fine[left:right]
        d_seg = np.diff(seg)
        if len(d_seg) == 0:
            continue
        sc = np.sign(d_seg)
        sc = sc[sc != 0]
        sc_per_s = float(np.sum(sc[1:] != sc[:-1]) / dur) if len(sc) > 1 else 0.0
        mean_amp = float(np.mean(seg))
        dar = float(np.mean(np.abs(d_seg)) / mean_amp) if mean_amp > 0 else 0.0
        deriv_ok = sc_per_s <= max_derivative_sign_changes_per_s and dar <= max_derivative_activity_ratio
        local_mast = float(sm_mast[left:right].mean())
        mast_ok = local_mast < mast_threshold * mast_relax_factor or _in_incl(pk, allowed)
        if (max_amp >= min_deg_amplitude) and deriv_ok and mast_ok:
            events.append((left, right))
    return events, mast_threshold, analysis_start, deg_start


# ---------- metrics ----------
def segment_active_parts(signal, threshold=1, min_len=6, min_area=100, sample_period=0.01):
    signal = np.asarray(signal, dtype=float)
    segments, start = [], None
    for i, v in enumerate(signal):
        if v >= threshold and start is None:
            start = i
        elif v < threshold and start is not None:
            segments.append(signal[start:i]); start = None
    if start is not None:
        segments.append(signal[start:])
    valid = [s for s in segments if len(s) > min_len and np.sum(s) > min_area]
    work, total_time, times = 0.0, 0.0, []
    for seg in valid:
        n = len(seg)
        if n <= 1:
            continue
        mean_volt = (np.sum(seg) / n) * 0.001
        t = sample_period * (n - 1)
        work += mean_volt * t
        total_time += t
        times.append(t)
    return valid, times, work, total_time


def compute_metrics_on_events(signal, events, sampling_rate=SAMPLING_RATE,
                              threshold=1, min_len=6, min_area=100):
    signal = np.asarray(signal, dtype=float)
    sp = 1.0 / sampling_rate
    cnt, tt, tw = 0, 0.0, 0.0
    for s, e in events:
        segs, _, work, active = segment_active_parts(signal[s:e], threshold, min_len, min_area, sp)
        cnt += len(segs); tt += active; tw += work
    return dict(count_total=int(cnt), active_time_s=round(tt, 6),
                mean_cycle_time_s=round(tt / cnt, 6) if cnt else 0.0,
                work_total=round(tw, 6), work_rate=round(tw / tt, 6) if tt > 0 else 0.0)


def total_events_time(events, sampling_rate=SAMPLING_RATE):
    return sum(e - s for s, e in events) / sampling_rate if events else 0.0


# ---------- single-subject ----------
def analyze_file(file_path, file_id, sampling_rate=SAMPLING_RATE):
    arr = load_and_preprocess_data(file_path)
    if arr.size == 0:
        return None
    mast = adjust_signal(arr[:, 0], sampling_rate)
    deg = adjust_signal(arr[:, 1], sampling_rate)
    analysis_start, deg_start, mast_threshold = compute_start_and_threshold(mast, sampling_rate)
    phases = identify_mastication_phases(mast, mast_threshold, sampling_rate)
    events, mast_threshold, analysis_start, deg_start = identify_deglutition_events(
        deg, mast, phases, sampling_rate=sampling_rate)
    msum = compute_metrics_on_events(mast, phases, sampling_rate)
    dsum = compute_metrics_on_events(deg, events, sampling_rate)
    ct = round(total_events_time(phases, sampling_rate), 6)
    dt = round(total_events_time(events, sampling_rate), 6)
    summary = {
        "ID": int(file_id),
        "number of chewing events": len(phases),
        "chewing time_s": ct,
        "number of chews": msum["count_total"],
        "work": msum["work_total"],
        "work rate": msum["work_rate"],
        "Cycle Time": msum["mean_cycle_time_s"],
        "number of deglutition events": len(events),
        "deglutition time_s": dt,
        "number of deglutition": dsum["count_total"],
        "deglutition work": dsum["work_total"],
        "deglutition work rate": dsum["work_rate"],
        "deglutition cycle time": dsum["mean_cycle_time_s"],
    }
    extra = dict(mast=mast, deg=deg, phases=phases, events=events,
                 mast_threshold=mast_threshold, analysis_start=analysis_start,
                 deg_start=deg_start, fs=sampling_rate)
    return summary, extra


def analyze_all(raw_dir, sampling_rate=SAMPLING_RATE):
    out = {}
    for fn in sorted(os.listdir(raw_dir)):
        if not fn.lower().endswith(".txt"):
            continue
        fid = parse_id(fn)
        if fid is None:
            continue
        res = analyze_file(os.path.join(raw_dir, fn), fid, sampling_rate)
        if res is not None:
            out[int(fid)] = res
    return out


if __name__ == "__main__":
    import pathlib
    import pandas as pd
    base = pathlib.Path(__file__).resolve().parents[2]
    res = analyze_all(str(base / "dati_raw"))
    rows = [r[0] for r in res.values()]
    df = pd.DataFrame(rows).sort_values("ID")
    out = pathlib.Path(__file__).resolve().parents[1] / "outputs"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "official_features.csv", index=False)
    print("official_features.csv", df.shape)
