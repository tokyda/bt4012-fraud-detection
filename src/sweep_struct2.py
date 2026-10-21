"""Round 2 of structural-feature tuning, driven by the dim-sweep result (more dims hurt).

Produces three candidates:
  * dim=16, depth=3   -- lower-dim embeddings (does going lower help further?)
  * dim=8,  depth=3   -- even lower
  * topo-only         -- pagerank + clustering + hop2 only, NO embeddings
                         (isolates whether the embeddings add anything at all)

Run:  PYTHONPATH=src .venv/bin/python src/sweep_struct2.py
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

BEST = Path("submissions/cand_struct_lgb_xgb.csv")  # dim=32 depth=3, current best 0.95137
EMB_PREFIX = "emb_"


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

    # Build dim=16 once; derive topo-only by dropping emb_* columns.
    sf16 = build_structural_features(edges, emb_dim=16, n_powers=3)
    sf8 = build_structural_features(edges, emb_dim=8, n_powers=3)
    topo_cols = [c for c in struct_columns(sf16) if not c.startswith(EMB_PREFIX)]

    jobs = {
        "cand_struct_d16_p3": (sf16, struct_columns(sf16)),
        "cand_struct_d8_p3": (sf8, struct_columns(sf8)),
        "cand_struct_topo_only": (sf16, topo_cols),
    }
    for name, (sf, scols) in jobs.items():
        tr = train.merge(sf, on="txId", how="left")
        te = test.merge(sf, on="txId", how="left")
        p, auc = blend(tr, te, base_cols + scols)
        write_submission(te, p, f"submissions/{name}.csv")
        rho = spearmanr(best, p).statistic if best is not None else float("nan")
        print(f"{name:26s} (+{len(scols):2d} feats)  holdout={auc:.4f}  rho_vs_best={rho:.4f}  "
              f"-> submissions/{name}.csv")

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
