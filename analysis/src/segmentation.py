"""Shared active-run segmentation (single source of truth).

The official pipeline defines an "active part" of the baseline-clipped envelope
as a contiguous above-threshold run that is long enough and has enough area
(see official.segment_active_parts). All modules call `segment_runs` so the
*same* run population is used everywhere.

A "run" here is one above-threshold excursion. Note this is deliberately
coarser than per-peak detection: a single run may contain several peaks, so
per-peak modules legitimately find more events than per-run (cycle) modules.
"""
from __future__ import annotations

import numpy as np

# Canonical constants (match official.segment_active_parts defaults).
THR = 1.0          # baseline-clipped envelope: active = strictly above THR
MIN_LEN = 6        # minimum run length in samples
MIN_AREA = 100     # minimum run area (sum of samples)


def find_runs(seg, thr: float = THR):
    """Return [(start, end), ...] half-open index ranges of contiguous runs
    where `seg > thr`. Mirrors official.segment_active_parts boundary logic."""
    seg = np.asarray(seg, dtype=float)
    runs, start = [], None
    for i, v in enumerate(seg):
        if v >= thr and start is None:
            start = i
        elif v < thr and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(seg)))
    return runs


def segment_runs(seg, thr: float = THR, min_len: int = MIN_LEN,
                 min_area: float = MIN_AREA):
    """Yield (start, end, sub_array) for runs passing the length AND area gate.

    Gate is `len > min_len and sum > min_area`, identical to
    official.segment_active_parts (`valid = [s for s in segments if
    len(s) > min_len and np.sum(s) > min_area]`).
    """
    seg = np.asarray(seg, dtype=float)
    out = []
    for a, b in find_runs(seg, thr):
        s = seg[a:b]
        if len(s) > min_len and float(np.sum(s)) > min_area:
            out.append((a, b, s))
    return out
