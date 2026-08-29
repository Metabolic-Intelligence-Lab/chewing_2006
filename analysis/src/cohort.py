"""Single source of truth for the analysis cohort.

Minors are excluded from ALL analyses (decision: adults only). Swallowing /
coordination analyses additionally require at least one detected swallow.
"""
from __future__ import annotations

import pandas as pd

ADULT_MIN_AGE = 18


def adults(df: pd.DataFrame) -> pd.DataFrame:
    """Adult analysis cohort (age >= 18)."""
    return df[df["age"] >= ADULT_MIN_AGE].copy()


def adults_valid_swallow(df: pd.DataFrame) -> pd.DataFrame:
    """Adults with at least one detected swallow (swallowing/coordination)."""
    d = adults(df)
    return d[d["valid_swallow"]].copy()
