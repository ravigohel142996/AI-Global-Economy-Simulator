"""
Economic dataset builder.

Wraps the country generator and adds cross-sectional and time-series
helper functions so that models and the simulation engine have a clean
data interface.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import RANDOM_SEED
from data.country_generator import generate_countries


def build_base_dataset(
    n_countries: int = 30,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Return the baseline cross-sectional dataset of synthetic countries.

    Parameters
    ----------
    n_countries:
        Number of countries to include.
    seed:
        Random seed passed to the generator.
    """
    return generate_countries(n=n_countries, seed=seed)


def build_time_series(
    base_df: pd.DataFrame,
    n_rounds: int = 10,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Expand the base dataset into a panel dataset with *n_rounds* time steps.

    Each row corresponds to (country, round).  The economic variables evolve
    according to simple stochastic difference equations that are later
    overridden by the simulation engine.  This function is mainly used to
    generate ML training data.

    Parameters
    ----------
    base_df:
        Cross-sectional baseline from ``build_base_dataset``.
    n_rounds:
        Number of simulation periods to generate.
    seed:
        Random seed.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    for rnd in range(n_rounds):
        snapshot = base_df.copy()
        snapshot["round"] = rnd

        noise_gdp        = rng.normal(0, 0.01, len(snapshot))
        noise_inflation  = rng.normal(0, 0.3, len(snapshot))
        noise_energy     = rng.normal(0, 0.02, len(snapshot))

        snapshot["gdp"]              = snapshot["gdp"] * (1 + noise_gdp + snapshot["gdp_growth_rate"] / 100)
        snapshot["inflation_rate"]   = (snapshot["inflation_rate"] + noise_inflation).clip(0.1, 30.0)
        snapshot["energy_production"]= snapshot["energy_production"] * (1 + noise_energy)
        snapshot["trade_balance"]    = snapshot["trade_balance"] + rng.normal(0, 5e9, len(snapshot))

        records.append(snapshot)
        # Roll the snapshot forward for the next round
        base_df = snapshot.copy()

    return pd.concat(records, ignore_index=True)


def prepare_ml_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Extract feature matrices and target vectors for ML models.

    Returns
    -------
    features : pd.DataFrame
        The full feature set.
    targets : dict
        ``gdp_growth``, ``inflation_rate``, ``recession_risk`` series.
    """
    features = df[
        [
            "population",
            "energy_production",
            "technology_index",
            "trade_balance",
            "energy_price",
            "inflation_rate",
            "gdp_growth_rate",
            "stability_index",
        ]
    ].copy()

    # Derived recession risk label: high inflation + low growth + low stability
    recession_risk = (
        (df["inflation_rate"] > 6.0).astype(float) * 0.4
        + (df["gdp_growth_rate"] < 1.0).astype(float) * 0.4
        + ((1 - df["stability_index"]) * 0.2)
    ).clip(0.0, 1.0)

    targets: dict[str, pd.Series] = {
        "gdp_growth":    df["gdp_growth_rate"],
        "inflation_rate": df["inflation_rate"],
        "recession_risk": recession_risk,
    }

    return features, targets
