# BT4012 — Fraudulent Bitcoin Transaction Detection (e1408825)

Classifies Bitcoin transactions in the **Elliptic** dataset as illicit vs. licit and predicts
`P(illicit)` for the test set, scored by **ROC-AUC**. Final public-LB score: **0.9514**.

The model is a **rank-average blend of LightGBM + XGBoost** on the 165 raw features plus engineered
**graph-structure features** (degree + 1-hop aggregates, and beyond-1-hop structural embeddings via
sparse SVD, PageRank, clustering, 2-hop reach), validated on a **temporal holdout** and selected on
the Kaggle public leaderboard (the test set is strictly future time periods).

## Deliverable

- **`notebooks/e1408825.ipynb`** — the end-to-end notebook: raw data → features → temporal
  validation → final model → `submission.csv`. Reproduces the top submission exactly.

## Reproduce

```bash
# 1. Python env (Python 3.11)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS only: LightGBM/XGBoost need the OpenMP runtime
#   brew install libomp

# 2. Place the competition data in data/raw/
#    train.csv, test.csv, txs_edgelist.csv, sample_submission.csv

# 3. Run the notebook end-to-end -> writes submissions/submission.csv
jupyter nbconvert --to notebook --execute --inplace notebooks/e1408825.ipynb
```

## Repository layout

```
notebooks/e1408825.ipynb   final, self-contained pipeline (the graded deliverable)
reports/e1408825.md        report + figures/
requirements.txt           pinned dependencies
src/                       development harness (exploration, not required to run the notebook):
  features.py              raw + 1-hop graph features
  graph_struct.py          beyond-1-hop structural features (SVD embeddings, pagerank, ...)
  model.py                 temporal split, training, blending, submission helpers
  build_notebook.py        assembles the deliverable notebook
  experiments.py sweep_*.py run_*.py gnn.py   explored candidates incl. the (rejected) GNN
data/                      (git-ignored) raw + processed data
submissions/               (git-ignored) generated prediction CSVs
```

The `src/` scripts document the full exploration, including approaches that were tested and
**rejected** (GraphSAGE GNN, heavier regularization, recency weighting) — see the report for the
reasoning. The notebook is the single source of truth for the final model.

## Notes

- Validation is **temporal** (train on early time steps, validate on the latest), never random —
  the test set is future periods.
- The transaction graph is **partitioned by time step** (no train↔test edges), so graph features
  are within-time-step and no label propagation into the future is possible.
