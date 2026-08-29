"""Shared statistical helpers (single source of truth): z-scores, FDR, effect sizes,
bootstrap CIs. Import from here instead of re-implementing per module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def zscore(s, ddof: int = 0):
    """Z-score a 1-D series, NaN-safe. Population SD (ddof=0) by default,
    matching the convention used across the OMPI/PSI composites."""
    s = pd.to_numeric(pd.Series(s), errors="coerce")
    sd = s.std(ddof=ddof)
    return (s - s.mean()) / (sd + 1e-9)


# Backwards-compatible alias (modules historically called the helper `_z`).
_z = zscore


def partial_corr_p(r: float, n: int, k: int):
    """Two-sided p-value for a partial correlation coefficient `r` computed on
    `n` observations after partialling out `k` covariates.

    The correct residual degrees of freedom are n - 2 - k. Using df = n - 2
    (as a plain pearsonr on the residuals does) ignores the covariates and is
    anti-conservative.
    """
    df = n - 2 - k
    if df <= 0 or not np.isfinite(r) or abs(r) >= 1.0:
        return np.nan
    t = r * np.sqrt(df / (1.0 - r * r))
    return float(2.0 * stats.t.sf(abs(t), df))
