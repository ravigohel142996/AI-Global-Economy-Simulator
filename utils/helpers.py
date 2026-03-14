"""
Utility helpers shared across the codebase.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fmt_billions(value: float, decimals: int = 2) -> str:
    """Format a large USD value in billions with a ``$B`` suffix."""
    return f"${value / 1e9:,.{decimals}f}B"


def fmt_trillions(value: float, decimals: int = 2) -> str:
    """Format a large USD value in trillions with a ``$T`` suffix."""
    return f"${value / 1e12:,.{decimals}f}T"


def fmt_percent(value: float, decimals: int = 1) -> str:
    """Format a float as a percentage string."""
    return f"{value:.{decimals}f}%"


def normalize_series(series: pd.Series) -> pd.Series:
    """Min-max normalize a pandas Series to [0, 1]."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Divide *a* by *b*, returning *default* if *b* is zero."""
    return a / b if b != 0 else default


def delta_arrow(value: float, threshold: float = 0.0) -> str:
    """Return an up/down arrow based on whether *value* exceeds *threshold*."""
    return "▲" if value > threshold else "▼"
