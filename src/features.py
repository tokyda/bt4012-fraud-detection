"""Feature engineering for the BT4012 Elliptic Bitcoin fraud-detection task.

The transaction graph is partitioned by time step (0 edges cross the train/test
boundary), so all graph-derived features are inherently *within time step*. We
build, per transaction (node):

  * degree features   -- in/out/total degree + ratios, computed over the full
                         edge list (so "background" nodes that are neither in
                         train nor test still count toward a node's degree).
  * neighbor aggregates -- mean/std of a block of local features over a node's
                         1-hop predecessors and successors. Only neighbors that
                         carry features (train/test nodes) contribute values;
                         background neighbors are skipped (NaN -> ignored).

Features are returned as a single frame keyed by txId, ready to merge onto the
raw feature table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Local (per-transaction) feature block to build neighbor aggregates over.
# In the Elliptic dataset the first ~93 features are local tx attributes and the
# remainder are the dataset's own 1-hop aggregates; re-aggregating the local
# block adds the most signal while keeping width bounded.
LOCAL_FEATS = [f"feat_{i}" for i in range(1, 94)]
ALL_FEATS = [f"feat_{i}" for i in range(1, 166)]


def load_raw(data_dir: str | Path):
    """Load train, test and edge list. Returns (train, test, edges)."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    edges = pd.read_csv(data_dir / "txs_edgelist.csv")
    return train, test, edges


def _degree_features(edges: pd.DataFrame) -> pd.DataFrame:
    """In/out/total degree and ratios per node, over the full edge list."""
    out_deg = edges.groupby("txId1").size().rename("out_degree")
    in_deg = edges.groupby("txId2").size().rename("in_degree")
    deg = pd.concat([out_deg, in_deg], axis=1).fillna(0.0)
    deg["total_degree"] = deg["in_degree"] + deg["out_degree"]
    # +1 smoothing so leaves/sources don't divide by zero
    deg["in_out_ratio"] = (deg["in_degree"] + 1) / (deg["out_degree"] + 1)
    deg["in_frac"] = deg["in_degree"] / deg["total_degree"].replace(0, np.nan)
    deg.index.name = "txId"
    return deg.reset_index()


def _neighbor_aggregates(
    edges: pd.DataFrame, feats: pd.DataFrame, agg_cols: list[str]
) -> pd.DataFrame:
    """Mean/std of ``agg_cols`` over each node's predecessors and successors.

    ``feats`` must be indexed by txId and contain ``agg_cols``.
    """
    agg = {c: ["mean", "std"] for c in agg_cols}

    # Successor aggregates: for source node txId1, aggregate features of txId2.
    succ_feat = edges.merge(
        feats[agg_cols], left_on="txId2", right_index=True, how="left"
    )
    succ = succ_feat.groupby("txId1")[agg_cols].agg(["mean", "std"]).copy()
    succ.columns = [f"succ_{c}_{stat}" for c, stat in succ.columns]
    succ.index.name = "txId"

    # Predecessor aggregates: for dest node txId2, aggregate features of txId1.
    pred_feat = edges.merge(
        feats[agg_cols], left_on="txId1", right_index=True, how="left"
    )
    pred = pred_feat.groupby("txId2")[agg_cols].agg(["mean", "std"]).copy()
    pred.columns = [f"pred_{c}_{stat}" for c, stat in pred.columns]
    pred.index.name = "txId"

    # concat (not iterative join) to avoid frame fragmentation
    return pd.concat([succ, pred], axis=1).reset_index().copy()


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    edges: pd.DataFrame,
    agg_cols: list[str] = LOCAL_FEATS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) with graph features merged on, keyed consistently.

    Degree and neighbor-aggregate features are computed once over the union of
    train+test nodes (features) and the full edge list, then merged back.
    """
    # Feature table for all labeled/testable nodes, indexed by txId.
    cols = ["txId"] + ALL_FEATS
    feats = pd.concat([train[cols], test[cols]], ignore_index=True).set_index("txId")

    deg = _degree_features(edges)
    nbr = _neighbor_aggregates(edges, feats, agg_cols)
    graph_feats = deg.merge(nbr, on="txId", how="outer")

    def _merge(df: pd.DataFrame) -> pd.DataFrame:
        out = df.merge(graph_feats, on="txId", how="left")
        # Nodes with no incoming/outgoing edges get 0 degree, not NaN.
        for c in ["in_degree", "out_degree", "total_degree"]:
            out[c] = out[c].fillna(0.0)
        return out.copy()  # de-fragment after the wide merge

    return _merge(train), _merge(test)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns: raw feats + engineered graph feats (not id/label)."""
    exclude = {"txId", "index", "time_step", "label"}
    return [c for c in df.columns if c not in exclude]
