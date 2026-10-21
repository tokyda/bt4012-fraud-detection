"""Beyond-1-hop graph-structure features for the BT4012 task.

The dataset's own feat_94..165 are already 1-hop aggregates, so 1-hop features
add little. These features capture structure the dataset does *not* encode:

  * structural embeddings  -- DeepWalk-as-matrix-factorization: sparse truncated
                              SVD of a short random-walk co-occurrence matrix on
                              the UNDIRECTED graph (the directed graph is a DAG,
                              so directed walks die at sinks). numpy-2 safe,
                              scipy-only, deterministic.
  * pagerank               -- global flow/authority position (directed graph).
  * clustering coefficient -- local density (undirected).
  * hop2_reach             -- # distinct nodes within 2 undirected hops.

All are per-node, keyed by txId, computed once over the full graph (train+test+
background nodes), then merged onto the feature frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import networkx as nx

SEED = 42


def build_structural_features(
    edges: pd.DataFrame, emb_dim: int = 32, n_powers: int = 3
) -> pd.DataFrame:
    # Contiguous node indexing over all nodes that appear in the edge list.
    nodes = pd.Index(pd.unique(pd.concat([edges["txId1"], edges["txId2"]])))
    idx = pd.Series(np.arange(len(nodes)), index=nodes)
    n = len(nodes)
    r = idx.loc[edges["txId1"]].values
    c = idx.loc[edges["txId2"]].values

    # Directed adjacency (binary) and its undirected, binarized symmetrization.
    A_dir = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    A = (A_dir + A_dir.T)
    A.data[:] = 1.0

    # Row-normalized transition P over the undirected graph.
    deg = np.asarray(A.sum(1)).ravel()
    deg_inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    P = sp.diags(deg_inv) @ A

    # Short random-walk co-occurrence S = P + P^2 + ... + P^n_powers
    # (stays sparse at avg degree ~2.3). Captures structure out to n_powers hops.
    S = P.copy()
    Pk = P
    for _ in range(n_powers - 1):
        Pk = Pk @ P
        S = S + Pk
    S = S.tocsr()

    # Truncated SVD -> embeddings (left singular vectors scaled by singular values).
    k = min(emb_dim, min(S.shape) - 2)
    U, sigma, _ = svds(S, k=k, random_state=SEED)
    emb = U * sigma  # (n, k)
    out = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    out.insert(0, "txId", nodes.values)

    # PageRank on the directed graph.
    G_dir = nx.from_scipy_sparse_array(A_dir, create_using=nx.DiGraph)
    pr = nx.pagerank(G_dir, alpha=0.85)
    out["pagerank"] = [pr.get(i, 0.0) for i in range(n)]

    # Clustering coefficient on the undirected graph.
    G_und = nx.from_scipy_sparse_array(A, create_using=nx.Graph)
    clus = nx.clustering(G_und)
    out["clustering"] = [clus.get(i, 0.0) for i in range(n)]

    # Distinct nodes within 2 undirected hops.
    reach = (A + A @ A).tocsr()
    reach.setdiag(0)
    reach.eliminate_zeros()
    out["hop2_reach"] = np.asarray((reach > 0).sum(1)).ravel()

    return out


def struct_columns(df: pd.DataFrame) -> list[str]:
    """Names of the structural-feature columns (everything except txId)."""
    return [c for c in df.columns if c != "txId"]
