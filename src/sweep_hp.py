"""Principled hyperparameter sweep for the LGB+XGB blend on raw+graph+structural.

The temporal holdout is saturated and anti-correlated with the LB for model
selection, so we do NOT tune to it. Instead we build a few configs varying along
*regularization / robustness* axes (more trees at lower lr, stronger L1/L2 and
subsampling, varied depth) plus blend weights, and report rank-correlation vs the
current best so only distinct candidates spend an LB slot. Final decision = public LB.

Run:  PYTHONPATH=src .venv/bin/python src/sweep_hp.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

from features import build_features, feature_columns, load_raw
from graph_struct import build_structural_features, struct_columns
from model import labeled, temporal_split, write_submission

BEST = Path("submissions/cand_struct_lgb_xgb.csv")  # 0.95137
CACHE = Path("data/processed/struct_feats.pkl")
SEED = 42


def spw(y):
    return float((y == 0).sum()) / max((y == 1).sum(), 1)


# Regularization-oriented configs (fixed tree counts -- no early stop on the useless holdout).
LGB_CFG = {
    "lgb_base": dict(learning_rate=0.02, num_leaves=64, min_child_samples=50,
                     feature_fraction=0.7, bagging_fraction=0.7, n=150),
    "lgb_reg": dict(learning_rate=0.01, num_leaves=48, min_child_samples=100,
                    feature_fraction=0.6, bagging_fraction=0.7, lambda_l1=1.0,
                    lambda_l2=1.0, n=600),
    "lgb_heavyreg": dict(learning_rate=0.005, num_leaves=32, min_child_samples=200,
                         feature_fraction=0.6, bagging_fraction=0.6, lambda_l1=2.0,
                         lambda_l2=2.0, n=1500),
}
XGB_CFG = {
    "xgb_base": dict(n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.7,
                     colsample_bytree=0.7),
    "xgb_reg": dict(n_estimators=800, learning_rate=0.02, max_depth=4, subsample=0.7,
                    colsample_bytree=0.6, reg_lambda=3.0, min_child_weight=5),
}


def fit_lgb(lab, cols, cfg):
    y = lab["label"].astype(int).values
    p = {k: v for k, v in cfg.items() if k != "n"}
    p.update(objective="binary", metric="auc", bagging_freq=1, scale_pos_weight=spw(y),
             seed=SEED, verbosity=-1)
    return lgb.train(p, lgb.Dataset(lab[cols], y), num_boost_round=cfg["n"])


def fit_xgb(lab, cols, cfg):
    y = lab["label"].astype(int).values
    clf = xgb.XGBClassifier(**cfg, scale_pos_weight=spw(y), eval_metric="auc",
                            random_state=SEED, n_jobs=-1, tree_method="hist")
    return clf.fit(lab[cols].fillna(0), y)


def wrank(ps, ws):
    rs = [rankdata(p) / len(p) for p in ps]
    return np.average(rs, axis=0, weights=ws)


def main():
    t0 = time.time()
    train, test, edges = load_raw("data/raw")
    train, test = build_features(train, test, edges)
    base_cols = feature_columns(train)  # raw+graph, before merging structural
    sf = pd.read_pickle(CACHE) if CACHE.exists() else build_structural_features(edges)
    train = train.merge(sf, on="txId", how="left")
    test = test.merge(sf, on="txId", how="left")
    cols = base_cols + struct_columns(sf)
    lab = labeled(train)
    tr, va = temporal_split(train)
    yva = va["label"].astype(int).values

    # Test predictions + saturated-holdout sanity for each base learner.
    P, Pva = {}, {}
    for name, cfg in LGB_CFG.items():
        P[name] = fit_lgb(lab, cols, cfg).predict(test[cols])
        Pva[name] = fit_lgb(tr, cols, cfg).predict(va[cols])
    for name, cfg in XGB_CFG.items():
        P[name] = fit_xgb(lab, cols, cfg).predict_proba(test[cols].fillna(0))[:, 1]
        Pva[name] = fit_xgb(tr, cols, cfg).predict_proba(va[cols].fillna(0))[:, 1]
    print("holdout AUC (saturated sanity):")
    for n in P:
        print(f"  {n:12s} {roc_auc_score(yva, Pva[n]):.4f}")

    best = test[["index"]].merge(pd.read_csv(BEST), on="index", how="left")["target"].values
    cands = {
        "cand_hp_reg": wrank([P["lgb_reg"], P["xgb_reg"]], [1, 1]),
        "cand_hp_heavyreg": wrank([P["lgb_heavyreg"], P["xgb_reg"]], [1, 1]),
        "cand_hp_lgbheavy": wrank([P["lgb_reg"], P["xgb_reg"]], [0.65, 0.35]),
        "cand_hp_xgbheavy": wrank([P["lgb_reg"], P["xgb_reg"]], [0.35, 0.65]),
    }
    print("\ncandidate            rho_vs_best")
    for name, p in cands.items():
        write_submission(test, p, f"submissions/{name}.csv")
        print(f"  {name:18s} {spearmanr(best, p).statistic:.4f}  -> submissions/{name}.csv")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
