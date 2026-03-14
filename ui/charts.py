"""
Plotly chart builders for the AI Global Economy Simulator dashboard.

All functions return ``plotly.graph_objects.Figure`` objects so they can
be passed directly to ``st.plotly_chart``.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Colour palette ─────────────────────────────────────────────────────────────
_PRIMARY   = "#2E86AB"
_SECONDARY = "#A23B72"
_SUCCESS   = "#2ECC71"
_WARNING   = "#F39C12"
_DANGER    = "#E74C3C"


def gdp_growth_line(panel_df: pd.DataFrame, countries: list[str] | None = None) -> go.Figure:
    """Line chart of GDP growth rate over simulation rounds."""
    df = panel_df.copy()
    if countries:
        df = df[df["country_name"].isin(countries)]

    fig = px.line(
        df,
        x="round",
        y="gdp_growth_rate",
        color="country_name",
        title="📈 GDP Growth Rate by Country",
        labels={"round": "Simulation Round", "gdp_growth_rate": "GDP Growth (%)", "country_name": "Country"},
        template="plotly_dark",
    )
    fig.update_layout(legend_title_text="Country", hovermode="x unified")
    return fig


def inflation_trend_line(panel_df: pd.DataFrame, countries: list[str] | None = None) -> go.Figure:
    """Line chart of inflation rate over simulation rounds."""
    df = panel_df.copy()
    if countries:
        df = df[df["country_name"].isin(countries)]

    fig = px.line(
        df,
        x="round",
        y="inflation_rate",
        color="country_name",
        title="📉 Inflation Rate Trend",
        labels={"round": "Simulation Round", "inflation_rate": "Inflation (%)", "country_name": "Country"},
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_layout(legend_title_text="Country", hovermode="x unified")
    return fig


def recession_risk_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing recession probability per country."""
    df_sorted = df.sort_values("recession_probability", ascending=True).tail(20)

    colours = [
        _DANGER if p >= 0.5 else (_WARNING if p >= 0.3 else _SUCCESS)
        for p in df_sorted["recession_probability"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df_sorted["recession_probability"],
            y=df_sorted["country_name"],
            orientation="h",
            marker_color=colours,
            text=[f"{p:.1%}" for p in df_sorted["recession_probability"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="⚠️ Recession Risk Probability by Country",
        xaxis_title="Recession Probability",
        yaxis_title="Country",
        template="plotly_dark",
        xaxis=dict(tickformat=".0%", range=[0, 1.1]),
    )
    return fig


def gdp_ranking_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of top-*n* economies by GDP."""
    df_sorted = df.nlargest(top_n, "gdp").sort_values("gdp")

    fig = go.Figure(
        go.Bar(
            x=df_sorted["gdp"] / 1e12,
            y=df_sorted["country_name"],
            orientation="h",
            marker_color=_PRIMARY,
            text=[f"${v/1e12:.1f}T" for v in df_sorted["gdp"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"🏆 Top {top_n} Economies by GDP",
        xaxis_title="GDP (Trillions USD)",
        yaxis_title="Country",
        template="plotly_dark",
    )
    return fig


def trade_balance_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Bar chart of trade balance (positive = surplus, negative = deficit)."""
    df_sorted = df.reindex(df["trade_balance"].abs().nlargest(top_n).index)
    colours = [_SUCCESS if v >= 0 else _DANGER for v in df_sorted["trade_balance"]]

    fig = go.Figure(
        go.Bar(
            x=df_sorted["country_name"],
            y=df_sorted["trade_balance"] / 1e9,
            marker_color=colours,
            text=[f"${v/1e9:.0f}B" for v in df_sorted["trade_balance"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="⚖️ Trade Balance by Country",
        xaxis_title="Country",
        yaxis_title="Trade Balance (Billions USD)",
        template="plotly_dark",
    )
    return fig


def global_gdp_area(global_metrics_df: pd.DataFrame) -> go.Figure:
    """Area chart of total world GDP across simulation rounds."""
    fig = go.Figure(
        go.Scatter(
            x=global_metrics_df["round"],
            y=global_metrics_df["total_world_gdp"] / 1e12,
            fill="tozeroy",
            mode="lines+markers",
            line=dict(color=_PRIMARY, width=2),
            fillcolor="rgba(46,134,171,0.3)",
            name="World GDP",
        )
    )
    fig.update_layout(
        title="🌍 Total World GDP Over Time",
        xaxis_title="Simulation Round",
        yaxis_title="World GDP (Trillions USD)",
        template="plotly_dark",
    )
    return fig


def inflation_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of current inflation rates across all countries."""
    fig = px.histogram(
        df,
        x="inflation_rate",
        nbins=20,
        title="📊 Inflation Rate Distribution",
        labels={"inflation_rate": "Inflation (%)"},
        template="plotly_dark",
        color_discrete_sequence=[_WARNING],
    )
    fig.add_vline(x=7.0, line_dash="dash", line_color=_DANGER, annotation_text="High inflation threshold")
    return fig


def scatter_gdp_stability(df: pd.DataFrame) -> go.Figure:
    """Scatter plot: GDP growth vs stability index, bubble size = GDP."""
    fig = px.scatter(
        df,
        x="stability_index",
        y="gdp_growth_rate",
        size="gdp",
        color="inflation_rate",
        hover_name="country_name",
        title="🔵 GDP Growth vs Stability (bubble = GDP size)",
        labels={
            "stability_index":  "Stability Index",
            "gdp_growth_rate":  "GDP Growth (%)",
            "inflation_rate":   "Inflation (%)",
        },
        template="plotly_dark",
        color_continuous_scale="RdYlGn_r",
    )
    return fig


def country_radar(df_row: pd.Series) -> go.Figure:
    """Radar chart showing economic profile of a single country."""
    from utils.helpers import normalize_series

    # We normalise globally so the chart is meaningful
    categories = [
        "technology_index",
        "stability_index",
        "gdp_growth_rate",
        "trade_balance",
        "energy_production",
    ]
    labels = [
        "Technology",
        "Stability",
        "GDP Growth",
        "Trade Balance",
        "Energy",
    ]

    # Scalar values – use raw normalised position vs max possible
    values = [
        float(df_row.get("technology_index", 0)),
        float(df_row.get("stability_index", 0)),
        max(0.0, min(1.0, (float(df_row.get("gdp_growth_rate", 0)) + 10) / 25)),
        max(0.0, min(1.0, (float(df_row.get("trade_balance", 0)) + 200e9) / 500e9)),
        max(0.0, min(1.0, float(df_row.get("energy_production", 0)) / 5e6)),
    ]

    fig = go.Figure(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            line=dict(color=_PRIMARY),
            fillcolor="rgba(46,134,171,0.3)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"🕸️ Economic Profile – {df_row.get('country_name', '')}",
        template="plotly_dark",
    )
    return fig


def country_time_series(history_df: pd.DataFrame) -> go.Figure:
    """Four-panel time-series chart for a single country's key indicators."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "GDP Growth (%)", "Inflation (%)",
            "Trade Balance ($B)", "Stability Index",
        ),
    )

    rounds = history_df["round"]

    fig.add_trace(go.Scatter(x=rounds, y=history_df["gdp_growth_rate"],
                             mode="lines+markers", name="GDP Growth",
                             line=dict(color=_SUCCESS)), row=1, col=1)
    fig.add_trace(go.Scatter(x=rounds, y=history_df["inflation_rate"],
                             mode="lines+markers", name="Inflation",
                             line=dict(color=_WARNING)), row=1, col=2)
    fig.add_trace(go.Scatter(x=rounds, y=history_df["trade_balance"] / 1e9,
                             mode="lines+markers", name="Trade Balance",
                             line=dict(color=_PRIMARY)), row=2, col=1)
    fig.add_trace(go.Scatter(x=rounds, y=history_df["stability_index"],
                             mode="lines+markers", name="Stability",
                             line=dict(color=_SECONDARY)), row=2, col=2)

    fig.update_layout(
        title=f"📊 Economic History – {history_df['country_name'].iloc[0]}",
        template="plotly_dark",
        showlegend=False,
        height=500,
    )
    return fig


def feature_importance_bar(importances: pd.Series, title: str = "Feature Importance") -> go.Figure:
    """Horizontal bar chart for ML model feature importances."""
    fig = go.Figure(
        go.Bar(
            x=importances.values,
            y=importances.index.tolist(),
            orientation="h",
            marker_color=_PRIMARY,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_dark",
        yaxis=dict(autorange="reversed"),
    )
    return fig
