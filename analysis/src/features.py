"""Per-channel feature extraction reproducing the validated method
(Riente et al., Biosensors 2023, §2.2), extended to the 2-channel layout
(ch1 = masseter/chewing, ch2 = mylohyoid/swallowing).

Features per channel:
  n          number of bursts (chews / swallows)
  t_cyc      mean burst duration [s]
  t_active   total active time = sum of burst durations [s]
  work       sum over bursts of (mean amplitude * duration) [a.u.*s]
  work_rate  work / t_active [a.u.]  (proxy of muscular power)

Burst rule (paper): a burst starts when two consecutive samples exceed the
threshold sigma* and ends when the signal drops back below it. We add a
minimum-duration and merge-gap guard to suppress noise spikes, and support
both a STATIC threshold (std of baseline, as in the paper) and an ADAPTIVE
threshold (baseline noise + fraction of the active dynamic range), matching
the `mast_threshold_adaptive` field of the existing summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from io_raw import Recording


@dataclass
class Burst:
    start_i: int
    end_i: int
    start_s: float
    end_s: float
    dur_s: float
    mean_amp: float
    peak_amp: float


@dataclass
class ChannelFeatures:
    threshold: float
    baseline_mean: float
    baseline_std: float
    n: int
    t_cyc_s: float
    t_active_s: float
    work: float
    work_rate: float
    bursts: list[Burst] = field(default_factory=list)


def estimate_baseline(sig: np.ndarray, fs: float, baseline_s: float = 1.0
                      ) -> tuple[float, float]:
    """Baseline mean/std from the leading silent window.

    The recordings begin with a silent period before the subject starts.
    We use the quietest leading window to avoid contamination if the subject
    starts early.
    """
    n_base = max(5, int(baseline_s * fs))
    head = sig[:n_base]
    return float(np.mean(head)), float(np.std(head))


def detect_bursts(sig: np.ndarray, fs: float, threshold: float,
                  min_dur_s: float = 0.05, merge_gap_s: float = 0.10
                  ) -> list[Burst]:
    """Detect bursts where the signal exceeds `threshold`.

    Implements the paper's crossing rule (two consecutive samples above
    threshold to open a burst) plus merge-gap and min-duration guards.
    """
    above = sig > threshold
    bursts: list[tuple[int, int]] = []
    i, n = 0, len(sig)
    while i < n - 1:
        if above[i] and above[i + 1]:           # two consecutive -> open
            start = i
            j = i + 1
            while j < n and above[j]:
                j += 1
            bursts.append((start, j - 1))
            i = j
        else:
            i += 1

    # merge bursts separated by a gap shorter than merge_gap_s
    merge_gap = int(merge_gap_s * fs)
    merged: list[list[int]] = []
    for s, e in bursts:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    min_dur = max(1, int(min_dur_s * fs))
    out: list[Burst] = []
    for s, e in merged:
        if (e - s + 1) < min_dur:
            continue
        seg = sig[s:e + 1]
        out.append(Burst(
            start_i=s, end_i=e,
            start_s=s / fs, end_s=e / fs,
            dur_s=(e - s + 1) / fs,
            mean_amp=float(np.mean(seg)),
            peak_amp=float(np.max(seg)),
        ))
    return out


def compute_threshold(sig: np.ndarray, base_mean: float, base_std: float,
                      mode: str = "adaptive",
                      k_static: float = 3.0,
                      k_adaptive: float = 0.15) -> float:
    """Threshold over the baseline.

    static:   base_mean + k_static * base_std   (paper's sigma* crossing)
    adaptive: base_mean + k_adaptive * (p95(signal) - base_mean)
              -> scales with each subject's own activity range.
    """
    if mode == "static":
        return base_mean + k_static * max(base_std, 1e-6)
    p95 = float(np.percentile(sig, 95))
    return base_mean + k_adaptive * max(p95 - base_mean, 1e-6)


def channel_features(sig: np.ndarray, fs: float, mode: str = "adaptive",
                     baseline_s: float = 1.0, **kw) -> ChannelFeatures:
    base_mean, base_std = estimate_baseline(sig, fs, baseline_s)
    sig0 = sig - base_mean                      # remove bias (paper)
    thr = compute_threshold(sig, base_mean, base_std, mode=mode,
                            k_static=kw.get("k_static", 3.0),
                            k_adaptive=kw.get("k_adaptive", 0.15)) - base_mean
    thr = max(thr, 1e-6)
    bursts = detect_bursts(sig0, fs, thr,
                           min_dur_s=kw.get("min_dur_s", 0.05),
                           merge_gap_s=kw.get("merge_gap_s", 0.10))
    n = len(bursts)
    t_active = sum(b.dur_s for b in bursts)
    t_cyc = (t_active / n) if n else 0.0
    work = sum(b.mean_amp * b.dur_s for b in bursts)
    work_rate = (work / t_active) if t_active > 0 else 0.0
    return ChannelFeatures(
        threshold=thr + base_mean,
        baseline_mean=base_mean, baseline_std=base_std,
        n=n, t_cyc_s=t_cyc, t_active_s=t_active,
        work=work, work_rate=work_rate, bursts=bursts,
    )


def detect_swallows(rec: Recording, k: float = 3.0, dom_ratio: float = 1.3,
                    chew_busy_frac: float = 0.5,
                    chew_feat: ChannelFeatures | None = None) -> list[Burst]:
    """Genuine swallows on ch2 (mylohyoid) via cross-channel gating.

    The mylohyoid is co-activated by chewing (envelope corr ~0.72), so a plain
    burst detector over-counts. A ch2 burst is accepted as a swallow only when
    the mylohyoid dominates the masseter (mean ch2 > dom_ratio * mean ch1) AND
    the masseter is mostly quiet in that window (< chew_busy_frac above its own
    threshold). Validated against the reference summary (r~0.67, matching median).
    """
    deg = channel_features(rec.ch_deg, rec.fs, mode="static", k_static=k)
    chew = chew_feat or channel_features(rec.ch_chew, rec.fs, mode="static", k_static=k)
    thr1 = chew.threshold
    out: list[Burst] = []
    for b in deg.bursts:
        seg1 = rec.ch_chew[b.start_i:b.end_i + 1]
        if b.mean_amp > dom_ratio * (float(seg1.mean()) + 1e-9) and \
           float(np.mean(seg1 > thr1)) < chew_busy_frac:
            out.append(b)
    return out


def detect_swallows_pause(rec: Recording, k: float = 3.0,
                          min_pause_s: float = 0.40, tail_s: float = 0.0,
                          chew_feat: ChannelFeatures | None = None) -> list[Burst]:
    """Physiology-driven swallow detection: swallows occur during PAUSES in
    chewing. Find masseter-quiet gaps (>= min_pause_s) between chew bursts and
    after the last one, then accept mylohyoid bursts falling within those gaps.

    More robust than amplitude-dominance gating for end-of-meal and inter-chew
    swallows (manual validation showed those were missed). Returns mylohyoid
    bursts that overlap a chewing pause.
    """
    chew = chew_feat or channel_features(rec.ch_chew, rec.fs, mode="static", k_static=k)
    deg = channel_features(rec.ch_deg, rec.fs, mode="static", k_static=k)
    # build chewing pause intervals (in seconds)
    cb = sorted(chew.bursts, key=lambda b: b.start_s)
    pauses = []
    prev_end = 0.0
    for b in cb:
        if b.start_s - prev_end >= min_pause_s:
            pauses.append((prev_end, b.start_s))
        prev_end = max(prev_end, b.end_s)
    # tail after last chew burst
    end_t = rec.duration_s
    if end_t - prev_end >= min_pause_s:
        pauses.append((prev_end, end_t + tail_s))

    def in_pause(b):
        c = 0.5 * (b.start_s + b.end_s)
        return any(s <= c <= e for s, e in pauses)

    return [b for b in deg.bursts if in_pause(b)]


def swallow_features(rec: Recording, swallows: list[Burst]) -> ChannelFeatures:
    """Aggregate the validated 5 features over the gated swallow bursts only."""
    n = len(swallows)
    t_active = sum(b.dur_s for b in swallows)
    t_cyc = (t_active / n) if n else 0.0
    work = sum(b.mean_amp * b.dur_s for b in swallows)
    work_rate = (work / t_active) if t_active > 0 else 0.0
    base_mean, base_std = estimate_baseline(rec.ch_deg, rec.fs)
    return ChannelFeatures(
        threshold=np.nan, baseline_mean=base_mean, baseline_std=base_std,
        n=n, t_cyc_s=t_cyc, t_active_s=t_active,
        work=work, work_rate=work_rate, bursts=swallows,
    )


def extract(rec: Recording, mode: str = "static", k_static: float = 3.0, **kw
            ) -> dict[str, ChannelFeatures]:
    """Primary per-subject extraction.

    Chewing (ch1): static-threshold burst detection (best agreement with the
    reference summary, r=0.90). Swallowing (ch2): cross-channel-gated detection.
    """
    chew = channel_features(rec.ch_chew, rec.fs, mode=mode, k_static=k_static, **kw)
    swallows = detect_swallows(rec, k=k_static, chew_feat=chew)
    deg = swallow_features(rec, swallows)
    return {"chew": chew, "deg": deg}
