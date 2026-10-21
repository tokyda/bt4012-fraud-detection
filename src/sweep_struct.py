"""Sweep structural-feature knobs (embedding dim, walk depth) feeding the LGB+XGB blend.

For each (emb_dim, n_powers) config: build structural features, merge, train the
blend refit on all labeled train, write a candidate submission, and report the
saturated-holdout AUC (sanity only) plus Spearman rank-corr vs the current
public-LB best (0.95137) and between candidates (skip near-duplicates).

Run:  PYTHONPATH=src .venv/bin/python src/sweep_struct.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from experiments import fit_xgb
from features import build_features, feature_columns, load_raw
from graph_struct import build_structural_features, struct_columns
from model import fit_final_lightgbm, rank_average, temporal_split, train_lightgbm, write_submission

CONFIGS = [(64, 3), (128, 3), (64, 4)]
BEST = Path("submissions/cand_struct_lgb_xgb.csv")  # current best, 0.95137


def blend(train, test, cols):
    tr, va = temporal_split(train)
    m = train_lightgbm(tr, va, cols)
    lgb_f = fit_final_lightgbm(train, cols, m.best_iteration)
    xgb_f = fit_xgb(train, cols, weighted=False)
    p_test = rank_average([lgb_f.predict(test[cols]),
                           xgb_f.predict_proba(test[cols].fillna(0))[:, 1]])
    p_va = rank_average([m.predict(va[cols], num_iteration=m.best_iteration),
                         fit_xgb(tr, cols, weighted=False).predict_proba(va[cols].fillna(0))[:, 1]])
    return p_test, roc_auc_score(va["label"].astype(int).values, p_va)


def main():
    t0 = time.time()
    train, test, edges = load_raw("data/raw")
    train, test = build_features(train, test, edges)
    base_cols = feature_columns(train)

    best = pd.read_csv(BEST).sort_values("index")["target"].values if BEST.exists() else None
    preds = {}
    for dim, powers in CONFIGS:
        sf = build_structural_features(edges, emb_dim=dim, n_powers=powers)
        scols = struct_columns(sf)
        tr = train.merge(sf, on="txId", how="left")
        te = test.merge(sf, on="txId", how="left")
        p, auc = blend(tr, te, base_cols + scols)
        name = f"cand_struct_d{dim}_p{powers}"
        write_submission(te, p, f"submissions/{name}.csv")
        rho = spearmanr(best, p).statistic if best is not None else float("nan")
        preds[name] = p
        print(f"{name:26s} holdout={auc:.4f}  rho_vs_best={rho:.4f}  "
              f"range=[{p.min():.3f},{p.max():.3f}]  -> submissions/{name}.csv")

    print("\nPairwise Spearman between candidates:")
    names = list(preds)
    M = pd.DataFrame({n: pd.Series(preds[n]) for n in names}).corr(method="spearman")
    print(M.round(4))
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
