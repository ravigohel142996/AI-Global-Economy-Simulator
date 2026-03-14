"""
Economy Engine.

Implements a multi-round economic simulation where each country's
macro-economic variables are updated according to a set of difference
equations.  The engine produces a panel DataFrame (round × country).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DEFAULT_ENERGY_PRICE,
    DEFAULT_INFLATION_PRESSURE,
    DEFAULT_SIMULATION_ROUNDS,
    DEFAULT_TECH_GROWTH,
    DEFAULT_TRADE_OPENNESS,
    RANDOM_SEED,
)


class EconomyEngine:
    """Runs the global economic simulation.

    Parameters
    ----------
    energy_price:
        Global baseline energy price (USD / barrel-equivalent).
    tech_growth:
        Technology improvement factor per round (fraction, e.g. 0.03 = 3 %).
    trade_openness:
        0–1 scalar controlling how strongly trade flows affect GDP.
    inflation_pressure:
        0–1 scalar representing external inflationary shocks.
    n_rounds:
        Number of simulation rounds.
    seed:
        Random seed for stochastic terms.
    """

    def __init__(
        self,
        energy_price: float = DEFAULT_ENERGY_PRICE,
        tech_growth: float = DEFAULT_TECH_GROWTH,
        trade_openness: float = DEFAULT_TRADE_OPENNESS,
        inflation_pressure: float = DEFAULT_INFLATION_PRESSURE,
        n_rounds: int = DEFAULT_SIMULATION_ROUNDS,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.energy_price       = energy_price
        self.tech_growth        = tech_growth
        self.trade_openness     = trade_openness
        self.inflation_pressure = inflation_pressure
        self.n_rounds           = n_rounds
        self.rng                = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def run(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Simulate the economy forward from *base_df*.

        Parameters
        ----------
        base_df : pd.DataFrame
            Cross-sectional country data (one row per country).

        Returns
        -------
        pd.DataFrame
            Panel data with columns ``round`` and ``country_name`` plus
            all economic variables.
        """
        snapshots: list[pd.DataFrame] = []
        current = base_df.copy()
        current["round"] = 0
        current["energy_price"] = self.energy_price
        snapshots.append(current.copy())

        for rnd in range(1, self.n_rounds + 1):
            current = self._step(current, rnd)
            snapshots.append(current.copy())

        return pd.concat(snapshots, ignore_index=True)

    # ------------------------------------------------------------------
    def _step(self, df: pd.DataFrame, rnd: int) -> pd.DataFrame:
        """Advance the economy by one round."""
        n = len(df)
        rng = self.rng

        # ── Technology improves each round ────────────────────────────
        df = df.copy()
        df["technology_index"] = (
            df["technology_index"] * (1 + self.tech_growth)
        ).clip(0.0, 1.0)

        # ── Energy price shock (global) ────────────────────────────────
        energy_shock = float(rng.normal(0, 0.05))
        df["energy_price"] = float(
            max(10.0, min(300.0, self.energy_price * (1 + energy_shock)))
        )

        # ── GDP growth ────────────────────────────────────────────────
        # Growth is driven by technology, trade openness, and stability
        base_growth = (
            df["technology_index"] * 3.0
            + df["stability_index"] * 2.0
            + df["trade_balance"].clip(lower=0) / 1e12 * self.trade_openness
            + rng.normal(0, 0.5, n)
        )
        df["gdp_growth_rate"] = base_growth.clip(-10.0, 15.0)
        df["gdp"] = df["gdp"] * (1 + df["gdp_growth_rate"] / 100)
        df["gdp_per_capita"] = df["gdp"] / df["population"]

        # ── Inflation dynamics ────────────────────────────────────────
        energy_inflation_push = (df["energy_price"] - self.energy_price) / self.energy_price * 2.0
        df["inflation_rate"] = (
            df["inflation_rate"]
            + energy_inflation_push
            + self.inflation_pressure * rng.uniform(0, 1, n)
            - df["technology_index"] * 0.5          # tech dampens inflation
            + rng.normal(0, 0.3, n)
        ).clip(0.1, 30.0)

        # ── Trade balance ─────────────────────────────────────────────
        trade_noise = rng.normal(0, 5e9, n)
        df["trade_balance"] = (
            df["trade_balance"]
            + trade_noise
            + df["technology_index"] * 10e9 * self.trade_openness
        )

        # ── Energy production ─────────────────────────────────────────
        energy_growth = rng.normal(0.01, 0.02, n)
        df["energy_production"] = (df["energy_production"] * (1 + energy_growth)).clip(lower=0)

        # ── Stability dynamics ────────────────────────────────────────
        stability_change = (
            -0.05 * (df["inflation_rate"] > 8).astype(float)
            + 0.02 * (df["gdp_growth_rate"] > 2).astype(float)
            + rng.normal(0, 0.02, n)
        )
        df["stability_index"] = (df["stability_index"] + stability_change).clip(0.05, 1.0)

        df["round"] = rnd
        return df
