"""
Trade Network visualisation using NetworkX + Plotly.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from simulation.trade_network import TradeNetwork


def plot_trade_network(trade_network: TradeNetwork, max_edges: int = 80) -> go.Figure:
    """Render the trade network as an interactive Plotly force-layout graph.

    Parameters
    ----------
    trade_network : TradeNetwork
        A built ``TradeNetwork`` instance.
    max_edges : int
        Cap on the number of edges shown (heaviest edges are kept).
    """
    G = trade_network.graph
    if len(G.nodes) == 0:
        return go.Figure()

    # Use spring layout for positioning
    pos = nx.spring_layout(G, seed=42, k=1.5 / max(1, len(G.nodes) ** 0.5))

    # Rank edges by weight, keep top *max_edges*
    edges_sorted = sorted(
        G.edges(data=True),
        key=lambda e: e[2].get("trade_volume", 0),
        reverse=True,
    )[:max_edges]

    # Build edge traces
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_weights: list[float] = []

    for u, v, data in edges_sorted:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_weights.append(data.get("trade_volume", 0))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="rgba(100,160,220,0.4)"),
        hoverinfo="none",
        name="Trade Link",
    )

    # Node trace
    node_x   = [pos[n][0] for n in G.nodes]
    node_y   = [pos[n][1] for n in G.nodes]
    node_text = list(G.nodes)

    # Size nodes by total trade volume (degree-weighted)
    degrees  = dict(G.degree())
    node_size = [5 + degrees.get(n, 1) * 3 for n in G.nodes]

    # Colour nodes by PageRank influence
    pagerank = nx.pagerank(G, weight="trade_volume")
    node_colour = [pagerank.get(n, 0) for n in G.nodes]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=8, color="white"),
        marker=dict(
            size=node_size,
            color=node_colour,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Trade Influence", thickness=12),
            line=dict(width=1, color="white"),
        ),
        hovertext=[
            f"{n}<br>Trade Degree: {degrees.get(n, 0)}<br>Influence: {pagerank.get(n, 0):.4f}"
            for n in G.nodes
        ],
        name="Country",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="🕸️ Global Trade Network",
        showlegend=False,
        hovermode="closest",
        template="plotly_dark",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=560,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig
