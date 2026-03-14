"""
Global configuration for the AI Global Economy Simulator.
Centralizes all constants, default parameters, and thresholds.
"""

from __future__ import annotations

# ── Simulation defaults ────────────────────────────────────────────────────────
DEFAULT_NUM_COUNTRIES: int = 30
DEFAULT_SIMULATION_ROUNDS: int = 10
DEFAULT_ENERGY_PRICE: float = 80.0          # USD per barrel equivalent
DEFAULT_TECH_GROWTH: float = 0.03           # 3 % per round
DEFAULT_TRADE_OPENNESS: float = 0.6         # 0–1 scale
DEFAULT_INFLATION_PRESSURE: float = 0.5     # 0–1 scale

# ── Country parameter ranges ───────────────────────────────────────────────────
POPULATION_RANGE: tuple[float, float] = (1e6, 1.4e9)       # persons
GDP_RANGE: tuple[float, float] = (1e10, 25e12)              # USD
INFLATION_RANGE: tuple[float, float] = (0.5, 15.0)         # percent
ENERGY_RANGE: tuple[float, float] = (1e3, 5e6)             # GWh
TRADE_BALANCE_RANGE: tuple[float, float] = (-200e9, 300e9) # USD
TECH_INDEX_RANGE: tuple[float, float] = (0.1, 1.0)
STABILITY_INDEX_RANGE: tuple[float, float] = (0.1, 1.0)

# ── Model hyper-parameters ─────────────────────────────────────────────────────
GDP_MODEL_N_ESTIMATORS: int = 100
GDP_MODEL_MAX_DEPTH: int = 5
INFLATION_MODEL_N_ESTIMATORS: int = 100
RECESSION_MODEL_N_ESTIMATORS: int = 100

# ── Economic thresholds ────────────────────────────────────────────────────────
RECESSION_THRESHOLD: float = 0.5           # probability above which = high risk
HIGH_INFLATION_THRESHOLD: float = 7.0      # percent

# ── Trade network ──────────────────────────────────────────────────────────────
MIN_TRADE_EDGES_PER_COUNTRY: int = 2
MAX_TRADE_EDGES_PER_COUNTRY: int = 8

# ── UI colours ──────────────────────────────────────────────────────────────────
COLOUR_POSITIVE = "#2ECC71"
COLOUR_NEGATIVE = "#E74C3C"
COLOUR_NEUTRAL  = "#3498DB"
COLOUR_WARNING  = "#F39C12"

# ── Random seed (reproducibility) ──────────────────────────────────────────────
RANDOM_SEED: int = 42
