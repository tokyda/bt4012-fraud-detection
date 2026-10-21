"""Minimal inductive GraphSAGE in pure PyTorch (no torch-geometric) for BT4012.

Mean-aggregation message passing is a sparse mat-mul, so we implement SAGE
directly: one new dependency (torch), numpy-2 safe. The graph is built over ALL
nodes (train+test+background); background nodes get zero features. Because the
graph is block-diagonal by time step, a full-graph forward pass keeps message
passing within each time step, and the shared weights make the model inductive
(weights learned on early steps apply to future test subgraphs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

SEED = 42


def node_index(edges: pd.DataFrame) -> pd.Series:
    """Map each txId that appears in the edge list to a contiguous row id."""
    nodes = pd.Index(pd.unique(pd.concat([edges["txId1"], edges["txId2"]])))
    return pd.Series(np.arange(len(nodes)), index=nodes)


def build_norm_adj(edges: pd.DataFrame, idx: pd.Series) -> torch.Tensor:
    """Row-normalized UNDIRECTED adjacency (neighbor-mean operator) as torch sparse."""
    n = len(idx)
    r = idx.loc[edges["txId1"]].values
    c = idx.loc[edges["txId2"]].values
    A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    A = A + A.T
    A.data[:] = 1.0  # binarize (collapse multi-edges), undirected
    deg = np.asarray(A.sum(1)).ravel()
    deg_inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    A = sp.diags(deg_inv) @ A  # D^-1 A  -> mean of neighbours
    A = A.tocoo()
    indices = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    values = torch.tensor(A.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


class SAGEConv(nn.Module):
    """h' = W · [h ‖ mean_{u∈N(v)} h_u]."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(2 * in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        neigh = torch.sparse.mm(adj, x)
        return self.lin(torch.cat([x, neigh], dim=1))


class SAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.dropout(F.relu(self.c1(x, adj)), self.dropout, self.training)
        h = F.dropout(F.relu(self.c2(h, adj)), self.dropout, self.training)
        return self.out(h).squeeze(-1)


class SAGEDeep(nn.Module):
    """Stronger variant: input projection + N residual SAGE blocks with LayerNorm."""

    def __init__(self, in_dim: int, hidden: int = 256, n_layers: int = 3, dropout: float = 0.4):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(SAGEConv(hidden, hidden) for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_layers))
        self.out = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.inp(x))
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.dropout(F.relu(norm(conv(h, adj))), self.dropout, self.training)
        return self.out(h).squeeze(-1)


def train_gnn(
    X: torch.Tensor,
    adj: torch.Tensor,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    hidden: int = 128,
    dropout: float = 0.3,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    max_epochs: int = 400,
    patience: int = 40,
    model: nn.Module | None = None,
) -> tuple[np.ndarray, float]:
    """Train with temporal early stopping; return (per-node probabilities, best val AUC)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    y_t = torch.tensor(np.nan_to_num(y), dtype=torch.float32)
    tr = torch.tensor(train_mask)
    ytr = y[train_mask]
    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    if model is None:
        model = SAGE(X.shape[1], hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc, best_prob, wait = -1.0, None, 0
    yval = y[val_mask]
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        logits = model(X, adj)
        loss = loss_fn(logits[tr], y_t[tr])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(X, adj)).numpy()
        auc = roc_auc_score(yval, prob[val_mask])
        if auc > best_auc:
            best_auc, best_prob, wait = auc, prob, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best_prob, best_auc
