"""
Economic Analysis.

Country-level analytics: trend extraction, ranking, risk scoring, and
cross-country comparisons.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def get_country_history(
    panel_df: pd.DataFrame, country_name: str
) -> pd.DataFrame:
    """Return the time series for a single country.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Full simulation panel.
    country_name : str
        Name of the country to extract.
    """
    return (
        panel_df[panel_df["country_name"] == country_name]
        .sort_values("round")
        .reset_index(drop=True)
    )


def rank_countries(
    df: pd.DataFrame,
    by: str = "gdp",
    top_n: int = 10,
    ascending: bool = False,
) -> pd.DataFrame:
    """Return a ranked subset of countries from the latest snapshot *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Single-round snapshot (most recent round).
    by : str
        Column to rank by.
    top_n : int
        Number of countries to return.
    ascending : bool
        Sort direction.
    """
    return (
        df.sort_values(by, ascending=ascending)
        .head(top_n)[["country_name", by]]
        .reset_index(drop=True)
    )


def compute_economic_health_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a composite economic health score for each country.

    Score formula (all normalised to 0–1):
        health = 0.35 * norm(gdp_growth)
               + 0.25 * norm(stability)
               + 0.20 * (1 – norm(inflation))
               + 0.20 * norm(trade_balance_positive)

    Parameters
    ----------
    df : pd.DataFrame
        Single-round snapshot.

    Returns
    -------
    pd.DataFrame
        Input df with an additional ``health_score`` column.
    """
    result = df.copy()

    def _normalize(series: pd.Series) -> pd.Series:
        rng = series.max() - series.min()
        return (series - series.min()) / (rng if rng != 0 else 1.0)

    norm_growth    = _normalize(result["gdp_growth_rate"].clip(-10, 15))
    norm_stability = _normalize(result["stability_index"])
    norm_inflation = _normalize(result["inflation_rate"])
    norm_trade     = _normalize(result["trade_balance"])

    result["health_score"] = (
        0.35 * norm_growth
        + 0.25 * norm_stability
        + 0.20 * (1 - norm_inflation)
        + 0.20 * norm_trade
    ).clip(0.0, 1.0)

    return result


def growth_forecast_summary(
    panel_df: pd.DataFrame, n_future: int = 5
) -> pd.DataFrame:
    """Simple linear extrapolation of GDP growth per country.

    Projects the next *n_future* rounds using the slope from the
    simulated history.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Full panel from the simulation engine.
    n_future : int
        Number of rounds to project.
    """
    records: list[dict] = []
    max_round = int(panel_df["round"].max())

    for country, grp in panel_df.groupby("country_name"):
        grp = grp.sort_values("round")
        x = grp["round"].values.astype(float)
        y = grp["gdp_growth_rate"].values

        # Fit simple linear trend
        if len(x) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
        else:
            slope, intercept = 0.0, y[-1] if len(y) else 0.0

        for f in range(1, n_future + 1):
            future_round = max_round + f
            forecast = float(np.clip(slope * future_round + intercept, -10, 15))
            records.append(
                {
                    "country_name":    country,
                    "round":           future_round,
                    "gdp_growth_forecast": forecast,
                    "is_forecast":     True,
                }
            )

    return pd.DataFrame(records)
