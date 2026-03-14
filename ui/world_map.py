"""
World Map visualisation using Plotly choropleth.

The simulator uses synthetic country names, so the map uses a custom
bubble/scatter approach plotted on a blank geo base rather than
matching ISO codes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Assign pseudo-coordinates to each synthetic country so they spread
# across the globe in a visually plausible way.
_COUNTRY_COORDS: dict[str, tuple[float, float]] = {
    "Arctica":    (71.0,  25.0),
    "Borelia":    (60.0,  15.0),
    "Caldoria":   (45.0,  -3.0),
    "Deltavia":   (52.0,  20.0),
    "Estara":     (38.0,  22.0),
    "Froncia":    (46.0,   2.0),
    "Grandia":    (51.0, -0.5),
    "Halvoria":   (55.0,  10.0),
    "Irenova":    (42.0,  14.0),
    "Jalvana":    (36.0,  36.0),
    "Krestia":    (50.0,  30.0),
    "Lundora":    (59.0,  18.0),
    "Marvonia":   (40.0,  -8.0),
    "Nordalis":   (64.0,  26.0),
    "Ostaria":    (47.0,  14.0),
    "Palvora":    (33.0,  44.0),
    "Quintalia":  (41.0,  12.0),
    "Renovia":    (44.0,  26.0),
    "Selvaria":   (37.0,  -6.0),
    "Tundoria":   (67.0,  68.0),
    "Ulvenia":    (54.0,  25.0),
    "Veldora":    (48.0,   8.0),
    "Westhaven":  (53.0,  -7.0),
    "Xandria":    (30.0,  31.0),
    "Yuvalia":    (35.0,  34.0),
    "Zephyria":   (28.0,  57.0),
    "Auronia":    (43.0,  28.0),
    "Brindova":   (56.0,  22.0),
    "Crisvalia":  (40.0,  22.0),
    "Drenvora":   (46.0,  15.0),
    # Extended set
    "Eltavia":    (49.0,  32.0),
    "Fuldaris":   (51.0,  11.0),
    "Grendovia":  (57.0,   9.0),
    "Holvaria":   (45.0,  19.0),
    "Iskvenia":   (62.0,  27.0),
    "Jorravia":   (42.0,  44.0),
    "Kolvenia":   (39.0,  16.0),
    "Lendoria":   (48.0,  21.0),
    "Mordavia":   (44.0,  27.0),
    "Nelvaria":   (36.0,  30.0),
}


def _add_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Attach lat/lon to a country DataFrame."""
    df = df.copy()
    df["lat"] = df["country_name"].map(lambda c: _COUNTRY_COORDS.get(c, (0.0, 0.0))[0])
    df["lon"] = df["country_name"].map(lambda c: _COUNTRY_COORDS.get(c, (0.0, 0.0))[1])
    return df


def world_bubble_map(
    df: pd.DataFrame,
    color_by: str = "gdp_growth_rate",
    size_by: str = "gdp",
    title: str = "🌍 Global Economy Map",
) -> go.Figure:
    """Render a bubble map of all synthetic countries.

    Parameters
    ----------
    df : pd.DataFrame
        Latest snapshot of country data.
    color_by : str
        Column used to colour bubbles.
    size_by : str
        Column used to size bubbles.
    title : str
        Chart title.
    """
    df_geo = _add_coords(df)

    # Normalise size to a sensible pixel range
    size_raw = df_geo[size_by].clip(lower=0)
    size_norm = (size_raw / size_raw.max() * 40 + 5).fillna(5)

    hover_cols = [
        "country_name", "gdp", "gdp_growth_rate",
        "inflation_rate", "stability_index", "trade_balance",
    ]

    fig = px.scatter_geo(
        df_geo,
        lat="lat",
        lon="lon",
        color=color_by,
        size=size_norm,
        hover_name="country_name",
        hover_data={
            col: True for col in hover_cols
            if col in df_geo.columns and col not in ("lat", "lon", "country_name")
        },
        title=title,
        template="plotly_dark",
        color_continuous_scale="RdYlGn",
        projection="natural earth",
    )

    fig.update_layout(
        geo=dict(
            showland=True, landcolor="rgba(50,50,50,0.8)",
            showocean=True, oceancolor="rgba(20,30,60,0.9)",
            showcoastlines=True, coastlinecolor="rgba(100,100,100,0.5)",
            showframe=False,
        ),
        coloraxis_colorbar=dict(title=color_by.replace("_", " ").title()),
        height=520,
    )
    return fig
