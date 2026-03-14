"""
Main Streamlit dashboard for the AI Global Economy Simulator.

This module wires together all sub-components and defines the page layout.
It is imported by ``app.py``.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from analytics.economic_analysis import (
    compute_economic_health_score,
    get_country_history,
    growth_forecast_summary,
    rank_countries,
)
from analytics.global_metrics import (
    compute_global_metrics,
    compute_latest_snapshot_metrics,
)
from ui.charts import (
    country_radar,
    country_time_series,
    feature_importance_bar,
    gdp_growth_line,
    gdp_ranking_bar,
    global_gdp_area,
    inflation_histogram,
    inflation_trend_line,
    recession_risk_bar,
    scatter_gdp_stability,
    trade_balance_bar,
)
from ui.trade_network_viz import plot_trade_network
from ui.world_map import world_bubble_map
from utils.helpers import fmt_billions, fmt_percent, fmt_trillions


# ── Helpers ────────────────────────────────────────────────────────────────────

def _metric_card(label: str, value: str, delta: str = "", delta_color: str = "normal") -> None:
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


# ── Section renderers ──────────────────────────────────────────────────────────

def render_overview(panel_df: pd.DataFrame, latest_df: pd.DataFrame) -> None:
    """Section 1 – Global Economy Overview."""
    st.header("🌐 Global Economy Overview")

    metrics   = compute_latest_snapshot_metrics(latest_df)
    global_ts = compute_global_metrics(panel_df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric_card("🏦 World GDP",       fmt_trillions(metrics["world_gdp"]))
    with col2:
        _metric_card("📈 Avg Inflation",   fmt_percent(metrics["avg_inflation"]))
    with col3:
        _metric_card("💱 Trade Volume",    fmt_trillions(metrics["trade_volume"]))
    with col4:
        _metric_card("⚖️ Stability Index", f"{metrics['stability_index']:.3f}")

    st.plotly_chart(global_gdp_area(global_ts), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(gdp_ranking_bar(latest_df), use_container_width=True)
    with col_b:
        st.plotly_chart(inflation_histogram(latest_df), use_container_width=True)

    st.plotly_chart(scatter_gdp_stability(latest_df), use_container_width=True)


def render_country_analysis(
    panel_df: pd.DataFrame,
    latest_df: pd.DataFrame,
    recession_df: pd.DataFrame,
) -> None:
    """Section 2 – Country Economic Analysis."""
    st.header("🔍 Country Economic Analysis")

    countries = sorted(latest_df["country_name"].unique().tolist())
    selected  = st.selectbox("Select a Country", countries)

    country_latest  = latest_df[latest_df["country_name"] == selected].iloc[0]
    country_history = get_country_history(panel_df, selected)

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric_card("💰 GDP", fmt_trillions(country_latest["gdp"]))
    with col2:
        _metric_card("📈 Growth", fmt_percent(country_latest["gdp_growth_rate"]))
    with col3:
        _metric_card("🔥 Inflation", fmt_percent(country_latest["inflation_rate"]))
    with col4:
        rec_row = recession_df[recession_df["country_name"] == selected]
        rec_prob = rec_row["recession_probability"].values[0] if len(rec_row) else 0.0
        _metric_card("⚠️ Recession Risk", fmt_percent(rec_prob * 100))

    # Charts
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.plotly_chart(country_time_series(country_history), use_container_width=True)
    with col_b:
        st.plotly_chart(country_radar(country_latest), use_container_width=True)


def render_global_forecast(
    panel_df: pd.DataFrame,
    latest_df: pd.DataFrame,
    gdp_model,
    inflation_model,
    recession_model,
) -> None:
    """Section 3 – Global Economic Forecast."""
    st.header("🔮 Global Economic Forecast")

    forecast_df = growth_forecast_summary(panel_df)

    # Top 10 countries by current GDP for the growth chart
    top_countries = latest_df.nlargest(10, "gdp")["country_name"].tolist()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("GDP Growth Projections (Historical + Forecast)")
        combined = pd.concat(
            [
                panel_df[panel_df["country_name"].isin(top_countries)][
                    ["country_name", "round", "gdp_growth_rate"]
                ].assign(is_forecast=False),
                forecast_df[forecast_df["country_name"].isin(top_countries)].rename(
                    columns={"gdp_growth_forecast": "gdp_growth_rate"}
                ),
            ],
            ignore_index=True,
        )
        st.plotly_chart(
            gdp_growth_line(combined, countries=top_countries),
            use_container_width=True,
        )
    with col_b:
        st.subheader("Inflation Trend")
        st.plotly_chart(
            inflation_trend_line(panel_df, countries=top_countries),
            use_container_width=True,
        )

    # Recession risk bar
    st.subheader("Recession Risk Distribution")
    st.plotly_chart(recession_risk_bar(latest_df if "recession_probability" in latest_df.columns
                                       else _add_recession_col(latest_df, recession_model)),
                    use_container_width=True)

    # ML Model insights
    st.subheader("🤖 ML Model Insights")
    c1, c2, c3 = st.columns(3)
    with c1:
        if gdp_model.feature_importances_ is not None:
            st.plotly_chart(
                feature_importance_bar(gdp_model.feature_importances_, "GDP Model – Feature Importance"),
                use_container_width=True,
            )
    with c2:
        if inflation_model.feature_importances_ is not None:
            st.plotly_chart(
                feature_importance_bar(inflation_model.feature_importances_, "Inflation Model – Feature Importance"),
                use_container_width=True,
            )
    with c3:
        if recession_model.feature_importances_ is not None:
            st.plotly_chart(
                feature_importance_bar(recession_model.feature_importances_, "Recession Model – Feature Importance"),
                use_container_width=True,
            )


def _add_recession_col(df: pd.DataFrame, model) -> pd.DataFrame:
    """Safely add recession_probability to df if not present."""
    try:
        return model.predict_df(df)
    except Exception:
        df = df.copy()
        df["recession_probability"] = 0.0
        return df


def render_trade_network(trade_network) -> None:
    """Section 4 – Trade Network Visualisation."""
    st.header("🕸️ Global Trade Network")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.plotly_chart(plot_trade_network(trade_network), use_container_width=True)
    with col_b:
        st.subheader("Most Influential Economies")
        influence_df = trade_network.get_most_influential(top_n=10)
        influence_df["influence"] = influence_df["influence"].map(lambda v: f"{v:.4f}")
        st.dataframe(influence_df, use_container_width=True)

    st.subheader("Trade Volume by Country")
    trade_metrics = trade_network.get_trade_metrics()
    st.dataframe(
        trade_metrics[["country_name", "export_volume", "import_volume", "total_trade_volume", "trade_degree"]]
        .head(15),
        use_container_width=True,
    )


def render_world_map(latest_df: pd.DataFrame) -> None:
    """Section 5 – Global Economy Map."""
    st.header("🗺️ Global Economy Map")

    map_options = {
        "GDP Growth Rate": "gdp_growth_rate",
        "Inflation Rate":  "inflation_rate",
        "Stability Index": "stability_index",
        "GDP":             "gdp",
        "Trade Balance":   "trade_balance",
    }
    col_choice, _ = st.columns([2, 3])
    with col_choice:
        selected_metric = st.selectbox("Colour map by", list(map_options.keys()))

    col_by = map_options[selected_metric]
    fig = world_bubble_map(
        latest_df,
        color_by=col_by,
        size_by="gdp",
        title=f"🌍 World Map – {selected_metric}",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rankings table
    st.subheader(f"🏆 Country Rankings by {selected_metric}")
    ranked = rank_countries(latest_df, by=col_by, top_n=15)
    st.dataframe(ranked, use_container_width=True)


def render_raw_data(latest_df: pd.DataFrame) -> None:
    """Optional section – Raw data explorer."""
    st.header("📋 Raw Simulation Data")
    health_df = compute_economic_health_score(latest_df)
    cols_to_show = [
        "country_name", "gdp", "gdp_growth_rate", "inflation_rate",
        "trade_balance", "stability_index", "technology_index",
        "energy_production", "health_score",
    ]
    display_cols = [c for c in cols_to_show if c in health_df.columns]
    st.dataframe(
        health_df[display_cols].sort_values("gdp", ascending=False),
        use_container_width=True,
    )
