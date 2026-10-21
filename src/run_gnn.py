"""Train the GraphSAGE GNN, blend with the current best tree submission, and
write candidates for public-LB evaluation.

Run:  PYTHONPATH=src .venv/bin/python src/run_gnn.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from features import ALL_FEATS, load_raw
from gnn import build_norm_adj, node_index, train_gnn
from model import rank_average, write_submission

BEST = Path("submissions/cand_struct_lgb_xgb.csv")  # current best tree submission (0.95137)
VAL_START = 31


def main():
    t0 = time.time()
    train, test, edges = load_raw("data/raw")
    print(f"train={train.shape} test={test.shape} edges={edges.shape}")

    idx = node_index(edges)
    n = len(idx)
    adj = build_norm_adj(edges, idx)
    print(f"graph nodes={n}  adj nnz={adj._nnz()}")

    # Node feature matrix over ALL graph nodes; background nodes -> 0 (= mean after scaling).
    scaler = StandardScaler()
    both = pd.concat([train[["txId"] + ALL_FEATS], test[["txId"] + ALL_FEATS]], ignore_index=True)
    feats_scaled = scaler.fit_transform(both[ALL_FEATS].values)
    X = np.zeros((n, len(ALL_FEATS)), dtype=np.float32)
    rows = idx.loc[both["txId"]].values
    X[rows] = feats_scaled
    X = torch.tensor(X, dtype=torch.float32)

    # Labels + temporal masks aligned to node rows.
    y = np.full(n, np.nan, dtype=float)
    lab = train[train["label"].notna()]
    y[idx.loc[lab["txId"]].values] = lab["label"].astype(int).values
    tstep = np.full(n, -1)
    tstep[idx.loc[train["txId"]].values] = train["time_step"].values

    train_mask = (~np.isnan(y)) & (tstep < VAL_START)
    val_mask = (~np.isnan(y)) & (tstep >= VAL_START) & (tstep <= 35)
    print(f"GNN train nodes={train_mask.sum()}  val nodes={val_mask.sum()}")

    prob, best_auc = train_gnn(X, adj, y, train_mask, val_mask)
    print(f"GNN holdout AUC (saturated sanity) = {best_auc:.4f}")

    # Test-node probabilities, in test.csv order.
    test_rows = idx.loc[test["txId"]].values
    p_gnn = prob[test_rows]
    write_submission(test, p_gnn, "submissions/cand_gnn_only.csv")

    # Blend with current best tree submission (align on index).
    best = pd.read_csv(BEST)
    merged = test[["index"]].merge(best, on="index", how="left")
    p_blend = rank_average([p_gnn, merged["target"].values])
    sub = write_submission(test, p_blend, "submissions/cand_blend_trees_gnn.csv")

    rho_gnn = spearmanr(merged["target"].values, p_gnn).statistic
    rho_blend = spearmanr(merged["target"].values, p_blend).statistic
    print(f"\nGNN-only vs best tree:   Spearman={rho_gnn:.4f}")
    print(f"blend vs best tree:      Spearman={rho_blend:.4f}  "
          f"({'worth a slot' if rho_blend < 0.995 else 'near-duplicate'})")
    print(f"wrote cand_gnn_only.csv and cand_blend_trees_gnn.csv  rows={len(sub)}")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
