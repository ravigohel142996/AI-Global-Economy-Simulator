"""
Country generator – produces a synthetic set of countries with realistic
economic attributes that can be used throughout the simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DEFAULT_NUM_COUNTRIES,
    GDP_RANGE,
    INFLATION_RANGE,
    ENERGY_RANGE,
    POPULATION_RANGE,
    STABILITY_INDEX_RANGE,
    TECH_INDEX_RANGE,
    TRADE_BALANCE_RANGE,
    RANDOM_SEED,
)

# Predefined country names – drawn from real nations for realism
_COUNTRY_NAMES: list[str] = [
    "Arctica",      "Borelia",      "Caldoria",     "Deltavia",
    "Estara",       "Froncia",      "Grandia",      "Halvoria",
    "Irenova",      "Jalvana",      "Krestia",      "Lundora",
    "Marvonia",     "Nordalis",     "Ostaria",      "Palvora",
    "Quintalia",    "Renovia",      "Selvaria",     "Tundoria",
    "Ulvenia",      "Veldora",      "Westhaven",    "Xandria",
    "Yuvalia",      "Zephyria",     "Auronia",      "Brindova",
    "Crisvalia",    "Drenvora",     "Eltavia",      "Fuldaris",
    "Grendovia",    "Holvaria",     "Iskvenia",     "Jorravia",
    "Kolvenia",     "Lendoria",     "Mordavia",     "Nelvaria",
]


def generate_countries(
    n: int = DEFAULT_NUM_COUNTRIES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a DataFrame of *n* synthetic countries.

    Each row represents one country with the following columns:

    - country_name        : unique string identifier
    - population          : number of people
    - gdp                 : nominal GDP in USD
    - inflation_rate      : annual inflation in %
    - energy_production   : GWh per year
    - trade_balance       : net exports minus imports in USD
    - technology_index    : composite score 0–1
    - stability_index     : composite score 0–1
    - gdp_per_capita      : derived GDP / population
    - gdp_growth_rate     : initial annual growth rate in %
    - energy_price        : baseline energy price USD per GWh-equivalent

    Parameters
    ----------
    n:
        Number of countries to generate.  Must be ≤ len(_COUNTRY_NAMES).
    seed:
        NumPy random seed for reproducibility.
    """
    if n > len(_COUNTRY_NAMES):
        raise ValueError(
            f"Cannot generate more than {len(_COUNTRY_NAMES)} uniquely-named countries."
        )

    rng = np.random.default_rng(seed)

    names = _COUNTRY_NAMES[:n]

    population       = rng.uniform(*POPULATION_RANGE, size=n)
    gdp              = rng.uniform(*GDP_RANGE, size=n)
    inflation_rate   = rng.uniform(*INFLATION_RANGE, size=n)
    energy_prod      = rng.uniform(*ENERGY_RANGE, size=n)
    trade_balance    = rng.uniform(*TRADE_BALANCE_RANGE, size=n)
    tech_index       = rng.uniform(*TECH_INDEX_RANGE, size=n)
    stability_index  = rng.uniform(*STABILITY_INDEX_RANGE, size=n)

    # Derived fields
    gdp_per_capita   = gdp / population
    # Growth rate loosely correlated with tech & stability, perturbed by noise
    gdp_growth_rate  = (
        tech_index * 4.0
        + stability_index * 2.0
        + rng.normal(0, 0.5, size=n)
    )
    energy_price     = rng.uniform(60.0, 120.0, size=n)

    df = pd.DataFrame(
        {
            "country_name":      names,
            "population":        population,
            "gdp":               gdp,
            "inflation_rate":    inflation_rate,
            "energy_production": energy_prod,
            "trade_balance":     trade_balance,
            "technology_index":  tech_index,
            "stability_index":   stability_index,
            "gdp_per_capita":    gdp_per_capita,
            "gdp_growth_rate":   gdp_growth_rate,
            "energy_price":      energy_price,
        }
    )

    return df.reset_index(drop=True)
