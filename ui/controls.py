"""
Sidebar controls for the AI Global Economy Simulator.

Returns a ``SimulationParams`` dataclass with all user-selected settings.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from config import (
    DEFAULT_ENERGY_PRICE,
    DEFAULT_INFLATION_PRESSURE,
    DEFAULT_NUM_COUNTRIES,
    DEFAULT_SIMULATION_ROUNDS,
    DEFAULT_TECH_GROWTH,
    DEFAULT_TRADE_OPENNESS,
)


@dataclass
class SimulationParams:
    """Holds all user-configurable simulation parameters."""
    n_countries:        int
    energy_price:       float
    tech_growth:        float
    trade_openness:     float
    inflation_pressure: float
    n_rounds:           int
    run_simulation:     bool


def render_sidebar() -> SimulationParams:
    """Render the sidebar UI and return the selected parameters."""
    st.sidebar.image(
        "https://img.icons8.com/fluency/96/000000/world-map.png",
        width=80,
    )
    st.sidebar.title("⚙️ Simulation Controls")
    st.sidebar.markdown("---")

    st.sidebar.subheader("🌍 World Parameters")
    n_countries = st.sidebar.slider(
        "Number of Countries",
        min_value=10,
        max_value=40,
        value=DEFAULT_NUM_COUNTRIES,
        step=1,
        help="How many synthetic countries to generate.",
    )

    n_rounds = st.sidebar.slider(
        "Simulation Rounds",
        min_value=5,
        max_value=30,
        value=DEFAULT_SIMULATION_ROUNDS,
        step=1,
        help="Number of economic time-steps to simulate.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Economic Levers")

    energy_price = st.sidebar.slider(
        "Energy Price (USD)",
        min_value=20.0,
        max_value=200.0,
        value=DEFAULT_ENERGY_PRICE,
        step=5.0,
        help="Global baseline energy price per barrel-equivalent.",
    )

    tech_growth = st.sidebar.slider(
        "Technology Growth Rate",
        min_value=0.0,
        max_value=0.10,
        value=DEFAULT_TECH_GROWTH,
        step=0.005,
        format="%.3f",
        help="Annual technology improvement factor.",
    )

    trade_openness = st.sidebar.slider(
        "Trade Openness",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TRADE_OPENNESS,
        step=0.05,
        help="How strongly trade affects economic growth (0 = closed, 1 = fully open).",
    )

    inflation_pressure = st.sidebar.slider(
        "Inflation Pressure",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_INFLATION_PRESSURE,
        step=0.05,
        help="External inflationary shock intensity.",
    )

    st.sidebar.markdown("---")
    run_simulation = st.sidebar.button(
        "🚀 Run Economic Simulation",
        use_container_width=True,
        type="primary",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("AI Global Economy Simulator v1.0")

    return SimulationParams(
        n_countries=n_countries,
        energy_price=energy_price,
        tech_growth=tech_growth,
        trade_openness=trade_openness,
        inflation_pressure=inflation_pressure,
        n_rounds=n_rounds,
        run_simulation=run_simulation,
    )
