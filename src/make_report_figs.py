"""Generate the two figures used in the report (saved as PNGs under reports/figures/)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Fig 1: illicit rate + labeled volume by time step (the temporal challenge).
train = pd.read_csv("data/raw/train.csv", usecols=["time_step", "label"])
lab = train[train["label"].notna()]
by = lab.groupby("time_step")["label"].agg(["mean", "size"])
fig, ax1 = plt.subplots(figsize=(6.2, 3.0))
ax1.bar(by.index, by["size"], color="#B9C6E8", label="labeled count")
ax1.set_xlabel("time step"); ax1.set_ylabel("labeled transactions", color="#4C72B0")
ax2 = ax1.twinx()
ax2.plot(by.index, by["mean"], color="#C44E52", marker="o", ms=3, label="illicit rate")
ax2.set_ylabel("illicit rate", color="#C44E52")
ax1.axvspan(35.5, 35.5, color="k")
fig.tight_layout(); fig.savefig(OUT / "illicit_by_time.png", dpi=150); plt.close(fig)

# Fig 2: public-LB AUC progression across the model-development journey.
labels = ["Competition\nbaseline", "Random\nForest", "LightGBM", "+XGBoost\nblend",
          "+structural\nfeatures"]
auc = [0.9212, 0.9390, 0.9443, 0.9503, 0.9514]
colors = ["#8C8C8C", "#55A868", "#4C72B0", "#4C72B0", "#C44E52"]
fig, ax = plt.subplots(figsize=(6.2, 3.0))
bars = ax.bar(range(len(auc)), auc, color=colors)
ax.set_xticks(range(len(auc))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0.90, 0.96); ax.set_ylabel("public-LB ROC-AUC")
for b, a in zip(bars, auc):
    ax.text(b.get_x() + b.get_width() / 2, a + 0.0015, f"{a:.3f}", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "lb_progression.png", dpi=150); plt.close(fig)
print("wrote", list(OUT.glob("*.png")))
