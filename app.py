"""
AI Global Economy Simulator
============================
Entry-point for the Streamlit application.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

# ── Page configuration (must be first Streamlit call) ──────────────────────────
st.set_page_config(
    page_title="AI Global Economy Simulator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project imports ─────────────────────────────────────────────────────────────
from config import RANDOM_SEED
from data.economic_dataset import (
    build_base_dataset,
    build_time_series,
    prepare_ml_features,
)
from models.gdp_forecast_model import GDPForecastModel
from models.inflation_predictor import InflationPredictor
from models.recession_risk_model import RecessionRiskModel
from simulation.economy_engine import EconomyEngine
from simulation.trade_network import TradeNetwork
from simulation.resource_dynamics import ResourceDynamics
from ui.controls import render_sidebar
from ui.dashboard import (
    render_country_analysis,
    render_global_forecast,
    render_overview,
    render_raw_data,
    render_trade_network,
    render_world_map,
)


# ── Session-state key ───────────────────────────────────────────────────────────
_SS_KEY = "simulation_state"


def _run_simulation(params) -> dict:
    """Execute the full simulation + ML pipeline and return results."""
    with st.spinner("🔄 Generating countries…"):
        base_df = build_base_dataset(n_countries=params.n_countries, seed=RANDOM_SEED)

    engine = EconomyEngine(
        energy_price=params.energy_price,
        tech_growth=params.tech_growth,
        trade_openness=params.trade_openness,
        inflation_pressure=params.inflation_pressure,
        n_rounds=params.n_rounds,
        seed=RANDOM_SEED,
    )

    with st.spinner("⚙️ Running economic simulation…"):
        panel_df = engine.run(base_df)

    # Resource dynamics enrichment
    resource = ResourceDynamics(energy_price=params.energy_price)
    latest_round = int(panel_df["round"].max())
    latest_df = panel_df[panel_df["round"] == latest_round].copy()
    latest_df = resource.update(latest_df, latest_round)

    # Build trade network on latest snapshot
    with st.spinner("🕸️ Building trade network…"):
        trade_net = TradeNetwork(seed=RANDOM_SEED)
        trade_net.build(latest_df)

    # ── ML training ───────────────────────────────────────────────────────────
    with st.spinner("🤖 Training ML models…"):
        train_df = build_time_series(base_df, n_rounds=max(params.n_rounds, 20), seed=RANDOM_SEED)
        features, targets = prepare_ml_features(train_df)

        gdp_model       = GDPForecastModel()
        inflation_model = InflationPredictor()
        recession_model = RecessionRiskModel()

        gdp_model.train(features, targets["gdp_growth"])
        inflation_model.train(features, targets["inflation_rate"])
        recession_model.train(features, targets["recession_risk"])

    # Enrich latest_df with ML predictions
    latest_df = gdp_model.predict_df(latest_df)
    latest_df = inflation_model.predict_df(latest_df)
    latest_df = recession_model.predict_df(latest_df)

    return {
        "panel_df":       panel_df,
        "latest_df":      latest_df,
        "trade_network":  trade_net,
        "gdp_model":      gdp_model,
        "inflation_model":inflation_model,
        "recession_model":recession_model,
        "params":         params,
    }


def main() -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🌍 AI Global Economy Simulator")
    st.markdown(
        "_A research-grade synthetic economy simulation platform with machine learning forecasting._"
    )
    st.markdown("---")

    # ── Sidebar controls ──────────────────────────────────────────────────────
    params = render_sidebar()

    # ── Trigger simulation on button press or first load ──────────────────────
    if params.run_simulation or _SS_KEY not in st.session_state:
        st.session_state[_SS_KEY] = _run_simulation(params)

    state = st.session_state[_SS_KEY]

    panel_df        = state["panel_df"]
    latest_df       = state["latest_df"]
    trade_net       = state["trade_network"]
    gdp_model       = state["gdp_model"]
    inflation_model = state["inflation_model"]
    recession_model = state["recession_model"]

    # ── Navigation tabs ───────────────────────────────────────────────────────
    tab_overview, tab_country, tab_forecast, tab_network, tab_map, tab_data = st.tabs(
        [
            "🌐 Overview",
            "🔍 Country Analysis",
            "🔮 Forecast",
            "🕸️ Trade Network",
            "🗺️ World Map",
            "📋 Raw Data",
        ]
    )

    with tab_overview:
        render_overview(panel_df, latest_df)

    with tab_country:
        render_country_analysis(panel_df, latest_df, latest_df)

    with tab_forecast:
        render_global_forecast(panel_df, latest_df, gdp_model, inflation_model, recession_model)

    with tab_network:
        render_trade_network(trade_net)

    with tab_map:
        render_world_map(latest_df)

    with tab_data:
        render_raw_data(latest_df)


if __name__ == "__main__":
    main()
