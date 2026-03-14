"""
Resource Dynamics.

Models how energy resources are produced, consumed, and traded between
countries.  Energy affects both inflation and GDP growth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DEFAULT_ENERGY_PRICE, RANDOM_SEED


class ResourceDynamics:
    """Tracks energy resource flows across the simulated economy.

    Parameters
    ----------
    energy_price:
        Initial global energy price.
    seed:
        Random seed.
    """

    def __init__(
        self,
        energy_price: float = DEFAULT_ENERGY_PRICE,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.base_energy_price = energy_price
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def update(self, df: pd.DataFrame, round_num: int) -> pd.DataFrame:
        """Compute energy metrics for *df* at *round_num*.

        Adds or updates:
        - ``energy_demand``:      estimated demand (GWh)
        - ``energy_surplus``:     production − demand
        - ``energy_price``:       current world price
        - ``energy_dependency``:  demand / production (>1 = net importer)

        Parameters
        ----------
        df : pd.DataFrame
            Country snapshot for the current round.
        round_num : int
            Used to introduce cyclical price shocks.
        """
        df = df.copy()
        n = len(df)

        # Demand grows with GDP and population
        demand_growth = (
            np.log1p(df["gdp"] / 1e12) * 1e5
            + df["population"] / 1e7
            + self.rng.normal(0, 0.5, n)
        ).clip(lower=0)
        df["energy_demand"] = demand_growth

        df["energy_surplus"] = df["energy_production"] - df["energy_demand"]

        # World price reacts to aggregate supply/demand
        aggregate_surplus = df["energy_surplus"].sum()
        price_change = -aggregate_surplus / (df["energy_production"].sum() + 1e-9) * 0.1
        cycle_effect = np.sin(round_num * np.pi / 6) * 5.0   # seasonal cycle
        df["energy_price"] = float(
            (self.base_energy_price * (1 + price_change) + cycle_effect).clip(10.0, 300.0)
        )

        df["energy_dependency"] = (df["energy_demand"] / df["energy_production"].clip(lower=1)).clip(lower=0)

        return df

    # ------------------------------------------------------------------
    def global_energy_summary(self, panel_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate energy metrics across all rounds.

        Parameters
        ----------
        panel_df : pd.DataFrame
            Output of ``EconomyEngine.run`` (contains ``round`` column).
        """
        summary = (
            panel_df.groupby("round")
            .agg(
                total_production=("energy_production", "sum"),
                avg_energy_price=("energy_price", "mean"),
            )
            .reset_index()
        )
        return summary
