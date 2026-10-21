"""Stronger GNN swing: deeper residual SAGE on richer inputs (raw + structural).

Feeds the GNN raw 165 features + the 35 structural features (embeddings,
pagerank, clustering, hop2) so it has global graph position, and uses the
deeper SAGEDeep model. Produces a GNN-only submission and light-weight blends
with the current best tree submission.

Run:  PYTHONPATH=src .venv/bin/python src/run_gnn2.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.preprocessing import StandardScaler

from features import ALL_FEATS, load_raw
from gnn import SAGEDeep, build_norm_adj, node_index, train_gnn
from graph_struct import build_structural_features, struct_columns
from model import write_submission

BEST = Path("submissions/cand_struct_lgb_xgb.csv")  # 0.95137
CACHE = Path("data/processed/struct_feats.pkl")
VAL_START = 31


def main():
    t0 = time.time()
    train, test, edges = load_raw("data/raw")
    idx = node_index(edges)
    n = len(idx)
    adj = build_norm_adj(edges, idx)

    sf = pd.read_pickle(CACHE) if CACHE.exists() else build_structural_features(edges)
    scols = struct_columns(sf)

    # Build node feature matrix: raw (train/test) + structural (all nodes); background raw -> 0.
    both = pd.concat([train[["txId"] + ALL_FEATS], test[["txId"] + ALL_FEATS]], ignore_index=True)
    raw = np.zeros((n, len(ALL_FEATS)), dtype=np.float32)
    raw[idx.loc[both["txId"]].values] = StandardScaler().fit_transform(both[ALL_FEATS].values)
    st = np.zeros((n, len(scols)), dtype=np.float32)
    st[idx.loc[sf["txId"]].values] = StandardScaler().fit_transform(sf[scols].values)
    X = torch.tensor(np.hstack([raw, st]), dtype=torch.float32)
    print(f"GNN input features = {X.shape[1]} (raw {len(ALL_FEATS)} + struct {len(scols)})")

    y = np.full(n, np.nan)
    lab = train[train["label"].notna()]
    y[idx.loc[lab["txId"]].values] = lab["label"].astype(int).values
    tstep = np.full(n, -1)
    tstep[idx.loc[train["txId"]].values] = train["time_step"].values
    train_mask = (~np.isnan(y)) & (tstep < VAL_START)
    val_mask = (~np.isnan(y)) & (tstep >= VAL_START) & (tstep <= 35)

    model = SAGEDeep(X.shape[1], hidden=256, n_layers=3, dropout=0.4)
    prob, auc = train_gnn(X, adj, y, train_mask, val_mask, lr=0.005, weight_decay=1e-4,
                          max_epochs=500, patience=50, model=model)
    print(f"GNN(deep,rich) holdout AUC = {auc:.4f}")

    p_gnn = prob[idx.loc[test["txId"]].values]
    write_submission(test, p_gnn, "submissions/cand_gnn2_only.csv")

    best = test[["index"]].merge(pd.read_csv(BEST), on="index", how="left")["target"].values
    rt, rg = rankdata(best) / len(best), rankdata(p_gnn) / len(p_gnn)
    for w in [0.85, 0.80]:
        blend = w * rt + (1 - w) * rg
        name = f"cand_blend_t{int(w*100)}_gnn2.csv"
        write_submission(test, blend, f"submissions/{name}")
        print(f"  {name}: rho_vs_tree={spearmanr(best, blend).statistic:.4f}")
    print(f"GNN2-only vs tree: Spearman={spearmanr(best, p_gnn).statistic:.4f}")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
