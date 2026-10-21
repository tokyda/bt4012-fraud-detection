"""Model training, temporal validation and submission generation for BT4012.

Because the test set is strictly future time steps, we validate on a *temporal*
holdout (train on the earliest labeled steps, validate on the latest) rather
than a random split, which would leak future information and overstate AUC.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
VAL_START_STEP = 31  # train on time_step < 31, validate on 31..35


def labeled(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known 0/1 label."""
    return df[df["label"].notna()].copy()


def temporal_split(train: pd.DataFrame, val_start: int = VAL_START_STEP):
    """Split labeled train rows into (early) train and (late) validation folds."""
    lab = labeled(train)
    tr = lab[lab["time_step"] < val_start]
    va = lab[lab["time_step"] >= val_start]
    return tr, va


def _scale_pos_weight(y: np.ndarray) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return neg / max(pos, 1.0)


def train_lightgbm(tr: pd.DataFrame, va: pd.DataFrame, feat_cols: list[str]):
    """Train LightGBM with early stopping on the temporal validation AUC."""
    import lightgbm as lgb

    Xtr, ytr = tr[feat_cols], tr["label"].astype(int).values
    Xva, yva = va[feat_cols], va["label"].astype(int).values

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.02,
        num_leaves=64,
        min_child_samples=50,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=1,
        scale_pos_weight=_scale_pos_weight(ytr),
        seed=SEED,
        verbosity=-1,
    )
    dtr = lgb.Dataset(Xtr, label=ytr)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    model = lgb.train(
        params,
        dtr,
        num_boost_round=3000,
        valid_sets=[dtr, dva],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    return model


def train_baselines(tr: pd.DataFrame, feat_cols: list[str]):
    """Logistic regression and random forest baselines (for the report)."""
    Xtr, ytr = tr[feat_cols], tr["label"].astype(int).values
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    ).fit(Xtr.fillna(0.0), ytr)
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    ).fit(Xtr.fillna(0.0), ytr)
    return {"logreg": logreg, "random_forest": rf}


def auc_of(model, df: pd.DataFrame, feat_cols: list[str], kind: str) -> float:
    y = df["label"].astype(int).values
    if kind == "lgb":
        p = model.predict(df[feat_cols], num_iteration=model.best_iteration)
    else:
        p = model.predict_proba(df[feat_cols].fillna(0.0))[:, 1]
    return roc_auc_score(y, p)


def write_submission(
    test: pd.DataFrame, proba: np.ndarray, path: str | Path
) -> pd.DataFrame:
    """Write an index,target submission and return it."""
    sub = pd.DataFrame({"index": test["index"].values, "target": proba})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return sub


def rank_average(probas: list[np.ndarray]) -> np.ndarray:
    """Blend scores by averaging their normalized ranks (AUC-friendly)."""
    from scipy.stats import rankdata

    return np.mean([rankdata(p) / len(p) for p in probas], axis=0)


def fit_final_lightgbm(train: pd.DataFrame, feat_cols: list[str], num_rounds: int):
    """Refit LightGBM on all labeled rows at a fixed round count (no holdout)."""
    import lightgbm as lgb

    lab = labeled(train)
    y = lab["label"].astype(int).values
    params = dict(
        objective="binary", metric="auc", learning_rate=0.02, num_leaves=64,
        min_child_samples=50, feature_fraction=0.7, bagging_fraction=0.7,
        bagging_freq=1, scale_pos_weight=_scale_pos_weight(y), seed=SEED, verbosity=-1,
    )
    return lgb.train(params, lgb.Dataset(lab[feat_cols], label=y),
                     num_boost_round=max(num_rounds, 100))


def fit_final_rf(train: pd.DataFrame, feat_cols: list[str]):
    """Refit a class-balanced random forest on all labeled rows."""
    lab = labeled(train)
    y = lab["label"].astype(int).values
    rf = RandomForestClassifier(
        n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=SEED
    )
    return rf.fit(lab[feat_cols].fillna(0.0), y)
