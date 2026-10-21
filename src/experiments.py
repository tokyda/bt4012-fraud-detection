"""Next-round candidate generation + an offline test-budget guard.

Because the temporal holdout is saturated (~0.99) it cannot rank these
candidates; the public LB is the only real yardstick and is rate-limited. So we
(a) build a few high-prior variants, and (b) report each candidate's Spearman
rank-correlation with the current primary blend on the *test* predictions. A
candidate that correlates ~0.999 with the incumbent is not worth a submission
slot; a meaningfully different one is.

Run:  PYTHONPATH=src .venv/bin/python src/experiments.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import RandomForestClassifier

from features import build_features, feature_columns, load_raw
from model import SEED, _scale_pos_weight, labeled, rank_average, write_submission

GAMMA = 0.10  # mild recency weighting: w(t) = exp(GAMMA * (t - t_max))


def recency_weights(time_step: pd.Series) -> np.ndarray:
    t_max = time_step.max()
    return np.exp(GAMMA * (time_step.values - t_max))


def fit_lgb(train, feat_cols, num_rounds, weighted):
    import lightgbm as lgb
    lab = labeled(train)
    y = lab["label"].astype(int).values
    w = recency_weights(lab["time_step"]) if weighted else None
    params = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=64,
                  min_child_samples=50, feature_fraction=0.7, bagging_fraction=0.7,
                  bagging_freq=1, scale_pos_weight=_scale_pos_weight(y), seed=SEED, verbosity=-1)
    return lgb.train(params, lgb.Dataset(lab[feat_cols], y, weight=w),
                     num_boost_round=max(num_rounds, 100))


def fit_xgb(train, feat_cols, weighted):
    import xgboost as xgb
    lab = labeled(train)
    y = lab["label"].astype(int).values
    w = recency_weights(lab["time_step"]) if weighted else None
    clf = xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.7,
        colsample_bytree=0.7, scale_pos_weight=_scale_pos_weight(y),
        eval_metric="auc", random_state=SEED, n_jobs=-1, tree_method="hist",
    )
    clf.fit(lab[feat_cols].fillna(0.0), y, sample_weight=w)
    return clf


def fit_rf(train, feat_cols, weighted):
    lab = labeled(train)
    y = lab["label"].astype(int).values
    w = recency_weights(lab["time_step"]) if weighted else None
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                n_jobs=-1, random_state=SEED)
    rf.fit(lab[feat_cols].fillna(0.0), y, sample_weight=w)
    return rf


def main():
    train, test, edges = load_raw("data/raw")
    train, test = build_features(train, test, edges)
    feat_cols = feature_columns(train)

    # need a tuned iteration count for lgb; reuse the saturated-holdout best_iter=~7 -> floor 100
    p = {}
    p["lgb"] = fit_lgb(train, feat_cols, 100, weighted=False).predict(test[feat_cols])
    p["lgb_w"] = fit_lgb(train, feat_cols, 100, weighted=True).predict(test[feat_cols])
    p["xgb"] = fit_xgb(train, feat_cols, weighted=False).predict_proba(test[feat_cols].fillna(0))[:, 1]
    p["xgb_w"] = fit_xgb(train, feat_cols, weighted=True).predict_proba(test[feat_cols].fillna(0))[:, 1]
    p["rf"] = fit_rf(train, feat_cols, weighted=False).predict_proba(test[feat_cols].fillna(0))[:, 1]

    # incumbent primary blend = rank-avg(lgb, rf)
    incumbent = rank_average([p["lgb"], p["rf"]])

    candidates = {
        "blend_lgb_xgb_rf": rank_average([p["lgb"], p["xgb"], p["rf"]]),
        "blend_lgb_xgb": rank_average([p["lgb"], p["xgb"]]),
        "wblend_lgb_rf": rank_average([p["lgb_w"], p["rf"]]),
        "wblend_lgb_xgb_rf": rank_average([p["lgb_w"], p["xgb_w"], p["rf"]]),
    }

    print(f"{'candidate':24s} {'spearman_vs_incumbent':>22s}  {'range'}")
    for name, pr in candidates.items():
        rho = spearmanr(incumbent, pr).statistic
        write_submission(test, pr, f"submissions/cand_{name}.csv")
        print(f"{name:24s} {rho:>22.4f}  [{pr.min():.3f},{pr.max():.3f}]  -> submissions/cand_{name}.csv")

    # also report how different the base learners are from each other on test
    print("\nTest-prediction Spearman between base learners:")
    names = ["lgb", "lgb_w", "xgb", "rf"]
    M = pd.DataFrame({n: rankdata(p[n]) for n in names}).corr(method="spearman")
    print(M.round(3))


if __name__ == "__main__":
    main()
