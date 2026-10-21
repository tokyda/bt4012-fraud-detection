"""End-to-end dev pipeline: load -> features -> temporal validation -> submission.

Run from the project root:  PYTHONPATH=src python src/run_pipeline.py
This is the development harness; the graded deliverable is the notebook, which
mirrors these steps with narrative.
"""
from __future__ import annotations

import time
from pathlib import Path

from sklearn.metrics import roc_auc_score

from features import ALL_FEATS, build_features, feature_columns, load_raw
from model import (
    auc_of,
    fit_final_lightgbm,
    fit_final_rf,
    rank_average,
    temporal_split,
    train_baselines,
    train_lightgbm,
    write_submission,
)

DATA = Path("data/raw")
SUB = Path("submissions/submission.csv")
SUB_ALT = Path("submissions/submission_rf.csv")


def main() -> None:
    t0 = time.time()
    print("Loading raw data ...")
    train, test, edges = load_raw(DATA)
    print(f"  train={train.shape} test={test.shape} edges={edges.shape}")

    print("Building graph features ...")
    train, test = build_features(train, test, edges)
    feat_cols = feature_columns(train)
    graph_cols = [c for c in feat_cols if c not in ALL_FEATS]
    print(f"  features: {len(feat_cols)} (raw={len(ALL_FEATS)}, graph={len(graph_cols)})")

    tr, va = temporal_split(train)
    print(f"  temporal split: train={tr.shape[0]} (steps<31)  valid={va.shape[0]} (31-35)")

    # ---- Model exploration on the temporal holdout (for the report) ----
    print("\nTemporal-holdout AUC (steps 31-35):")
    base = train_baselines(tr, ALL_FEATS)
    for name, m in base.items():
        print(f"  {name:22s} (raw)        = {auc_of(m, va, ALL_FEATS, 'skl'):.4f}")

    lgb_raw = train_lightgbm(tr, va, ALL_FEATS)
    print(f"  lightgbm (raw)             = {auc_of(lgb_raw, va, ALL_FEATS, 'lgb'):.4f}  (iter={lgb_raw.best_iteration})")
    lgb_all = train_lightgbm(tr, va, feat_cols)
    print(f"  lightgbm (raw+graph)       = {auc_of(lgb_all, va, feat_cols, 'lgb'):.4f}  (iter={lgb_all.best_iteration})")

    base_all = train_baselines(tr, feat_cols)
    rf_all = base_all["random_forest"]
    p_lgb = lgb_all.predict(va[feat_cols], num_iteration=lgb_all.best_iteration)
    p_rf = rf_all.predict_proba(va[feat_cols].fillna(0.0))[:, 1]
    y = va["label"].astype(int).values
    print(f"  random_forest (raw+graph)  = {roc_auc_score(y, p_rf):.4f}")
    print(f"  BLEND rf+lgb (raw+graph)   = {roc_auc_score(y, rank_average([p_lgb, p_rf])):.4f}")

    # ---- Final models: refit on ALL labeled train, predict test ----
    print("\nRefitting final models on all labeled train ...")
    final_lgb = fit_final_lightgbm(train, feat_cols, lgb_all.best_iteration)
    final_rf = fit_final_rf(train, feat_cols)
    p_test_lgb = final_lgb.predict(test[feat_cols])
    p_test_rf = final_rf.predict_proba(test[feat_cols].fillna(0.0))[:, 1]

    blend = rank_average([p_test_lgb, p_test_rf])
    sub = write_submission(test, blend, SUB)
    write_submission(test, p_test_rf, SUB_ALT)
    print(f"  primary  (blend): {SUB}  rows={len(sub)}  range=[{blend.min():.4f},{blend.max():.4f}]")
    print(f"  alt (pure RF)   : {SUB_ALT}")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
