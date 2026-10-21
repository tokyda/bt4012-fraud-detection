"""A/B test: does adding beyond-1-hop structural features help the LGB+XGB blend?

Compares the current architecture (raw + 1-hop graph features) against the same
plus structural embeddings / pagerank / clustering / hop2. Produces a candidate
submission and reports its rank-correlation vs the current public-LB best
(cand_blend_lgb_xgb.csv, 0.95026) so we only spend a submission slot if it is
meaningfully different.

Run:  PYTHONPATH=src .venv/bin/python src/run_struct.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from features import build_features, feature_columns, load_raw
from graph_struct import build_structural_features, struct_columns
from model import (
    fit_final_lightgbm,
    labeled,
    rank_average,
    temporal_split,
    train_lightgbm,
    write_submission,
)
from experiments import fit_xgb

CACHE = Path("data/processed/struct_feats.pkl")
BEST = Path("submissions/cand_blend_lgb_xgb.csv")  # current public-LB best (0.95026)


def blend_preds(train, test, feat_cols):
    """LGB+XGB rank-average, refit on all labeled train."""
    tr, va = temporal_split(train)
    m = train_lightgbm(tr, va, feat_cols)  # for tuned iteration count
    lgb_f = fit_final_lightgbm(train, feat_cols, m.best_iteration)
    xgb_f = fit_xgb(train, feat_cols, weighted=False)
    p_test = rank_average([lgb_f.predict(test[feat_cols]),
                           xgb_f.predict_proba(test[feat_cols].fillna(0))[:, 1]])
    # holdout sanity (saturated)
    p_va = rank_average([m.predict(va[feat_cols], num_iteration=m.best_iteration),
                         fit_xgb(tr, feat_cols, weighted=False).predict_proba(va[feat_cols].fillna(0))[:, 1]])
    auc = roc_auc_score(va["label"].astype(int).values, p_va)
    return p_test, auc


def main():
    t0 = time.time()
    train, test, edges = load_raw("data/raw")
    train, test = build_features(train, test, edges)
    base_cols = feature_columns(train)

    sf = pd.read_pickle(CACHE) if CACHE.exists() else build_structural_features(edges)
    scols = struct_columns(sf)
    train = train.merge(sf, on="txId", how="left")
    test = test.merge(sf, on="txId", how="left")
    full_cols = base_cols + scols
    print(f"base features={len(base_cols)}  +structural={len(scols)}  total={len(full_cols)}")

    p_base, auc_base = blend_preds(train, test, base_cols)
    p_full, auc_full = blend_preds(train, test, full_cols)
    print(f"\nHoldout AUC (saturated, sanity only):  base={auc_base:.4f}  +struct={auc_full:.4f}")

    sub = write_submission(test, p_full, "submissions/cand_struct_lgb_xgb.csv")
    print(f"wrote submissions/cand_struct_lgb_xgb.csv  rows={len(sub)}  range=[{p_full.min():.3f},{p_full.max():.3f}]")

    # how different is this from the current public-LB best?
    if BEST.exists():
        best = pd.read_csv(BEST).sort_values("index")["target"].values
        rho = spearmanr(best, pd.read_csv("submissions/cand_struct_lgb_xgb.csv").sort_values("index")["target"].values).statistic
        print(f"Spearman vs current best (0.95026): {rho:.4f}  "
              f"({'different enough to test' if rho < 0.995 else 'near-duplicate, maybe skip'})")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
