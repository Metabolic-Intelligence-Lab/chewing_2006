"""Ingestion of raw 2-channel sEMG traces.

File format (dati_raw/NNN.txt):
    Data,Orario,Dato
    2025-10-15,22:16:34,<ch1> <ch2>
where ch1 = masseter (chewing) and ch2 = mylohyoid/submental (swallowing).
Samples carry only 1-second timestamp resolution (~96-97 samples/s), so the
time axis is reconstructed uniformly from the estimated sampling frequency.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# masseter = chewing, mylohyoid = swallowing/deglutition
CH_CHEW = 0
CH_DEG = 1


@dataclass
class Recording:
    rec_id: int               # numeric ID (e.g. 1 for 001.txt)
    ch_chew: np.ndarray       # channel 1 envelope (masseter)
    ch_deg: np.ndarray        # channel 2 envelope (mylohyoid)
    fs: float                 # estimated sampling frequency [Hz]
    t: np.ndarray             # time axis [s], uniform, starting at 0
    duration_s: float
    n_samples: int
    timestamps: pd.Series     # raw per-sample datetime (1 s resolution)

    @property
    def dt(self) -> float:
        return 1.0 / self.fs


def _parse_id(filename: str) -> int:
    base = os.path.basename(filename)
    m = re.match(r"0*(\d+)", base)
    if m is None:
        raise ValueError(f"cannot parse numeric recording ID from filename: {base!r}")
    return int(m.group(1))


def estimate_fs(timestamps: pd.Series, n_samples: int) -> float:
    """Estimate sampling frequency from the span of 1-second timestamps.

    Using span (last-first) rather than count-per-second avoids edge effects
    from partially filled first/last seconds.
    """
    if n_samples <= 1 or len(timestamps) == 0:
        return float(n_samples)  # degenerate: nothing to span over
    span = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
    if span <= 0:
        return float(n_samples)  # degenerate: all in one second
    return (n_samples - 1) / span


def load_recording(path: str) -> Recording:
    df = pd.read_csv(path)
    # Combine Data + Orario into a datetime (1 s resolution)
    ts = pd.to_datetime(df["Data"].astype(str) + " " + df["Orario"].astype(str),
                        errors="coerce")
    dato = df["Dato"].astype(str).str.strip().str.split(r"\s+", expand=True)
    # guard: rows with a single value never yield a 2nd column -> NaN ch2,
    # so they are dropped by `good` (consistent with official.load_and_preprocess
    # _data, which skips rows where len(vals) < 2). If NO row has a 2nd value,
    # the split produces a single column and dato[1] would KeyError.
    if dato.shape[1] < 2:
        ch1 = np.empty(0, dtype=float)
        ch2 = np.empty(0, dtype=float)
        ts = ts.iloc[:0]
    else:
        ch1 = pd.to_numeric(dato[0], errors="coerce").to_numpy(dtype=float)
        ch2 = pd.to_numeric(dato[1], errors="coerce").to_numpy(dtype=float)
        # drop rows with NaN channels or unparseable timestamps (NaT in head/
        # tail would otherwise drive estimate_fs to NaN)
        good = ~(np.isnan(ch1) | np.isnan(ch2) | ts.isna().to_numpy())
        ch1, ch2, ts = ch1[good], ch2[good], ts[good].reset_index(drop=True)
    n = len(ch1)
    fs = estimate_fs(ts, n)
    t = np.arange(n) / fs
    return Recording(
        rec_id=_parse_id(path),
        ch_chew=ch1,
        ch_deg=ch2,
        fs=fs,
        t=t,
        duration_s=t[-1] if n else 0.0,
        n_samples=n,
        timestamps=ts,
    )


def list_recordings(raw_dir: str) -> list[str]:
    files = [f for f in os.listdir(raw_dir) if re.match(r"\d+\.txt$", f)]
    return [os.path.join(raw_dir, f) for f in sorted(files)]


def load_all(raw_dir: str) -> dict[int, Recording]:
    return {(_parse_id(p)): load_recording(p) for p in list_recordings(raw_dir)}
