"""Stage 2 - chewing<->swallowing coordination metrics (novel layer).

These quantify how the two functions relate in time and intensity, which was
impossible in the prior single-function (L/R masseter) device. Swallow-dependent
metrics return NaN when no gated swallow was detected (handled downstream:
zero-swallow subjects are dropped from swallowing/coordination analyses).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import correlate, correlation_lags

from io_raw import Recording
from features import Burst, ChannelFeatures, channel_features
from scipy.ndimage import median_filter as _medfilt

# --- Physiological timing windows -------------------------------------------
# A deglutition normally follows the end of a chewing phase: the swallow can
# start slightly before the phase end is annotated (chew/swallow overlap) and
# up to several seconds after the phase ends, while the bolus is cleared.
SWALLOW_BEFORE_PHASE_S = 1.0   # swallow may start up to 1 s before phase end
SWALLOW_AFTER_PHASE_S = 8.0    # ...and up to 8 s after phase end
# Max lag (s) searched for the envelope cross-correlation peak.
XCORR_MAX_LAG_S = 2.0
# Window (s) within which a swallow is considered "preceded by" a chew burst.
CHEW_SWALLOW_WINDOW_S = 2.0


def coordination_official(extra: dict) -> dict:
    """Coordination metrics from the OFFICIAL phases/events.

    extra holds: mast, deg (adjusted signals), phases [(s,e)], events [(s,e)],
    fs. Swallow-dependent metrics return NaN when there is no deglutition event.
    """
    mast, deg = extra["mast"], extra["deg"]
    phases, events = extra["phases"], extra["events"]
    fs = extra["fs"]
    dur = len(mast) / fs
    n_ph, n_ev = len(phases), len(events)

    # envelope cross-correlation (peak coef + lag), |lag|<=XCORR_MAX_LAG_S.
    # The normalisation and correlation_lags() assume the two signals have the
    # same length, so truncate both to the common length n first and compute
    # std on the arrays actually used.
    n = min(len(mast), len(deg))
    assert n > 0, "empty signal in coordination_official"
    mast_c, deg_c = mast[:n], deg[:n]
    a = mast_c - mast_c.mean()
    b = deg_c - deg_c.mean()
    denom = np.std(mast_c) * np.std(deg_c) * n
    if denom > 0:
        xc = correlate(b, a, mode="full") / denom
        lags = correlation_lags(len(b), len(a), mode="full") / fs
        sel = np.abs(lags) <= XCORR_MAX_LAG_S
        xc, lags = xc[sel], lags[sel]
        i = int(np.argmax(xc))
        xpeak, xlag = float(xc[i]), float(lags[i])
    else:
        xpeak = xlag = np.nan

    # co-activation (Jaccard) of phase vs event masks
    mp = np.zeros(len(mast), dtype=bool)
    for s, e in phases:
        mp[s:e] = True
    md = np.zeros(len(deg), dtype=bool)
    for s, e in events:
        md[s:e] = True
    inter, union = np.sum(mp & md), np.sum(mp | md)

    # post-phase swallow latency: for each event, time from the end of the
    # preceding chewing phase to the event start (events live in phase-end windows)
    phase_ends = np.array([e for _, e in phases]) / fs
    lats = []
    for s, _ in events:
        t = s / fs
        prev = phase_ends[phase_ends <= t]
        if len(prev):
            lat = t - prev.max()
            # cap the latency: a swallow farther than SWALLOW_AFTER_PHASE_S from
            # the preceding phase end is not plausibly that phase's swallow, so
            # discard it instead of producing an arbitrarily large latency.
            if lat <= SWALLOW_AFTER_PHASE_S:
                lats.append(lat)
    # fraction of phases followed by >=1 swallow within the physiological window
    followed = 0
    ev_starts = np.array([s for s, _ in events]) / fs
    for _, e in phases:
        te = e / fs
        if np.any((ev_starts >= te - SWALLOW_BEFORE_PHASE_S)
                  & (ev_starts <= te + SWALLOW_AFTER_PHASE_S)):
            followed += 1

    # oral-processing organisation: time chewing before the first swallow and
    # number of chews preceding it (bolus-formation effort before deglutition)
    oral_proc = np.nan
    chews_before = np.nan
    if events and phases:
        import official
        first_ev = min(s for s, _ in events)
        first_ph = min(s for s, _ in phases)
        oral_proc = (first_ev - first_ph) / fs
        segs, _, _, _ = official.segment_active_parts(
            mast[:first_ev], sample_period=1.0 / fs)
        chews_before = len(segs)

    return dict(
        # chews_per_swallow := (total chew count) / (total swallow count).
        # Unified definition shared with coordination_metrics() below; here the
        # counts are the official chew segments (n_chews) and swallow events
        # (n_ev), there they are the detected chew/swallow bursts.
        chews_per_swallow=(extra.get("n_chews", np.nan) / n_ev) if n_ev else np.nan,
        swallows_per_phase=(n_ev / n_ph) if n_ph else np.nan,
        xcorr_peak=xpeak, xcorr_lag_s=xlag,
        coactivation_jaccard=float(inter / union) if union else np.nan,
        post_phase_swallow_latency_s=float(np.median(lats)) if lats else np.nan,
        post_phase_latency_cv=(float(np.std(lats) / (np.mean(lats) + 1e-9))
                               if len(lats) > 1 else np.nan),
        frac_phases_with_swallow=(followed / n_ph) if n_ph else np.nan,
        oral_processing_time_s=oral_proc,
        chews_before_first_swallow=chews_before,
    )


def _active_mask(sig: np.ndarray, feats: ChannelFeatures) -> np.ndarray:
    m = np.zeros(len(sig), dtype=bool)
    for b in feats.bursts:
        m[b.start_i:b.end_i + 1] = True
    return m


def envelope_xcorr(rec: Recording, max_lag_s: float = XCORR_MAX_LAG_S):
    """Normalised cross-correlation of the two envelopes.

    Returns (peak_coef, lag_s): lag>0 means mylohyoid (swallow) follows
    masseter (chew). Peak searched within +/- max_lag_s.
    """
    a = rec.ch_chew - rec.ch_chew.mean()
    b = rec.ch_deg - rec.ch_deg.mean()
    denom = (np.std(rec.ch_chew) * np.std(rec.ch_deg) * len(a))
    if denom == 0:
        return np.nan, np.nan
    xc = correlate(b, a, mode="full") / denom
    lags = correlation_lags(len(b), len(a), mode="full") / rec.fs
    sel = np.abs(lags) <= max_lag_s
    xc, lags = xc[sel], lags[sel]
    i = int(np.argmax(xc))
    return float(xc[i]), float(lags[i])


def coactivation(rec: Recording, chew: ChannelFeatures,
                 deg_bursts: list[Burst]) -> dict:
    """Temporal overlap of the two active masks (Jaccard) and co-active fraction."""
    mc = _active_mask(rec.ch_chew, chew)
    md = np.zeros(len(rec.ch_deg), dtype=bool)
    for b in deg_bursts:
        md[b.start_i:b.end_i + 1] = True
    inter = np.sum(mc & md)
    union = np.sum(mc | md)
    return dict(
        coactivation_jaccard=float(inter / union) if union else np.nan,
        frac_chew_coactive=float(inter / mc.sum()) if mc.sum() else np.nan,
    )


def chew_swallow_timing(chew: ChannelFeatures, swallows: list[Burst],
                        window_s: float = CHEW_SWALLOW_WINDOW_S) -> dict:
    """Temporal coupling: latency from the preceding chew burst end to each
    swallow, fraction of swallows preceded by chewing within `window_s`, and
    mean number of chew bursts per inter-swallow run (chewing run length)."""
    chew_ends = np.array([b.end_s for b in chew.bursts])
    chew_starts = np.array([b.start_s for b in chew.bursts])
    if len(swallows) == 0:
        return dict(chew_swallow_latency_s=np.nan,
                    frac_swallow_after_chew=np.nan,
                    chews_per_run=np.nan)
    lats, preceded = [], 0
    for s in swallows:
        prev = chew_ends[chew_ends <= s.start_s]
        if len(prev):
            lat = s.start_s - prev.max()
            lats.append(lat)
            if lat <= window_s:
                preceded += 1
    # chews per run = chew bursts between consecutive swallows
    sw_times = sorted(b.start_s for b in swallows)
    bounds = [0.0] + sw_times
    runs = [np.sum((chew_starts > bounds[i]) & (chew_starts <= bounds[i + 1]))
            for i in range(len(sw_times))]
    return dict(
        chew_swallow_latency_s=float(np.median(lats)) if lats else np.nan,
        frac_swallow_after_chew=float(preceded / len(swallows)),
        chews_per_run=float(np.mean(runs)) if runs else np.nan,
    )


def coordination_metrics(rec: Recording, chew: ChannelFeatures,
                         deg: ChannelFeatures) -> dict:
    swallows = deg.bursts
    n_sw = len(swallows)
    peak, lag = envelope_xcorr(rec)
    out = dict(
        # chews_per_swallow := (total chew count) / (total swallow count), the
        # same definition used in coordination_official(); counts here are the
        # detected chew bursts (chew.n) and swallow bursts (n_sw).
        chews_per_swallow=(chew.n / n_sw) if n_sw else np.nan,
        work_ratio_chew_deg=(chew.work / deg.work) if deg.work > 0 else np.nan,
        active_time_ratio_chew_deg=(chew.t_active_s / deg.t_active_s)
                                    if deg.t_active_s > 0 else np.nan,
        xcorr_peak=peak,
        xcorr_lag_s=lag,
    )
    out.update(coactivation(rec, chew, swallows))
    out.update(chew_swallow_timing(chew, swallows))
    return out
