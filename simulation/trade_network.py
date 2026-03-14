"""
Trade Network.

Builds and analyses a directed trade graph using NetworkX.
Nodes represent countries; weighted directed edges represent bilateral
trade volume.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from config import (
    MAX_TRADE_EDGES_PER_COUNTRY,
    MIN_TRADE_EDGES_PER_COUNTRY,
    RANDOM_SEED,
)


class TradeNetwork:
    """Manages the global trade network.

    Parameters
    ----------
    seed:
        Random seed for edge generation.
    """

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def build(self, df: pd.DataFrame) -> "TradeNetwork":
        """Construct the trade graph from the latest country snapshot.

        Each country node stores all country attributes as metadata.
        Edges are assigned a ``trade_volume`` weight proportional to the
        combined economic size of the two endpoints.

        Parameters
        ----------
        df : pd.DataFrame
            Country-level data (one row per country).
        """
        self.graph.clear()

        # Add nodes
        for _, row in df.iterrows():
            self.graph.add_node(
                row["country_name"],
                **row.to_dict(),
            )

        countries = list(df["country_name"])
        gdp_map   = dict(zip(df["country_name"], df["gdp"]))
        n         = len(countries)

        # Add directed edges using a stochastic preferential-attachment rule
        for i, src in enumerate(countries):
            n_edges = int(
                self.rng.integers(
                    MIN_TRADE_EDGES_PER_COUNTRY,
                    MAX_TRADE_EDGES_PER_COUNTRY + 1,
                )
            )
            # Candidates exclude self
            candidates = [c for j, c in enumerate(countries) if j != i]
            # Weight selection by GDP (larger economies attract more trade)
            weights = np.array([gdp_map[c] for c in candidates], dtype=float)
            weights /= weights.sum()
            chosen = self.rng.choice(
                candidates,
                size=min(n_edges, len(candidates)),
                replace=False,
                p=weights,
            )
            for dst in chosen:
                vol = (gdp_map[src] + gdp_map[dst]) * self.rng.uniform(0.001, 0.01)
                self.graph.add_edge(src, dst, trade_volume=vol)

        return self

    # ------------------------------------------------------------------
    def get_trade_metrics(self) -> pd.DataFrame:
        """Return a DataFrame with per-country trade network metrics."""
        records: list[dict[str, Any]] = []
        for node in self.graph.nodes:
            in_vol  = sum(
                self.graph[u][node]["trade_volume"]
                for u in self.graph.predecessors(node)
            )
            out_vol = sum(
                self.graph[node][v]["trade_volume"]
                for v in self.graph.successors(node)
            )
            records.append(
                {
                    "country_name":        node,
                    "import_volume":       in_vol,
                    "export_volume":       out_vol,
                    "total_trade_volume":  in_vol + out_vol,
                    "trade_degree":        self.graph.degree(node),
                    "economic_influence":  nx.pagerank(self.graph, weight="trade_volume").get(node, 0),
                }
            )
        return pd.DataFrame(records).sort_values("total_trade_volume", ascending=False)

    # ------------------------------------------------------------------
    def get_most_influential(self, top_n: int = 10) -> pd.DataFrame:
        """Return the *top_n* most economically influential countries."""
        pagerank = nx.pagerank(self.graph, weight="trade_volume")
        df = (
            pd.DataFrame.from_dict(pagerank, orient="index", columns=["influence"])
            .sort_values("influence", ascending=False)
            .head(top_n)
            .reset_index()
            .rename(columns={"index": "country_name"})
        )
        return df
