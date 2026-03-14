"""
Global Metrics.

Computes aggregate indicators for the simulated world economy,
summarising each simulation round into a single row.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_global_metrics(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Compute round-level global economic metrics.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel output from ``EconomyEngine.run`` containing a ``round`` column.

    Returns
    -------
    pd.DataFrame
        One row per round with columns:
        - round
        - total_world_gdp
        - avg_inflation
        - total_trade_volume          (sum of absolute trade balances)
        - avg_stability
        - avg_gdp_growth
        - n_high_inflation_countries  (inflation > 7 %)
        - n_recession_risk_countries  (growth < 0 %)
    """
    grouped = panel_df.groupby("round")

    metrics = grouped.agg(
        total_world_gdp=("gdp", "sum"),
        avg_inflation=("inflation_rate", "mean"),
        avg_stability=("stability_index", "mean"),
        avg_gdp_growth=("gdp_growth_rate", "mean"),
    ).reset_index()

    metrics["total_trade_volume"] = (
        grouped["trade_balance"].apply(lambda s: s.abs().sum()).values
    )
    metrics["n_high_inflation_countries"] = (
        grouped.apply(lambda g: (g["inflation_rate"] > 7.0).sum(), include_groups=False).values
    )
    metrics["n_recession_risk_countries"] = (
        grouped.apply(lambda g: (g["gdp_growth_rate"] < 0).sum(), include_groups=False).values
    )

    return metrics


def compute_latest_snapshot_metrics(df: pd.DataFrame) -> dict[str, float]:
    """Compute headline metrics for the latest simulation snapshot *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Single-round country data (last round of the simulation).

    Returns
    -------
    dict with keys: world_gdp, avg_inflation, trade_volume, stability_index
    """
    return {
        "world_gdp":       float(df["gdp"].sum()),
        "avg_inflation":   float(df["inflation_rate"].mean()),
        "trade_volume":    float(df["trade_balance"].abs().sum()),
        "stability_index": float(df["stability_index"].mean()),
    }
