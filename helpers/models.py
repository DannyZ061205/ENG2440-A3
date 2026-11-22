"""A small bond-aware message-passing network for graph-level binary classification.

Each atom and each bond is described by integer category codes, so the first thing the model does is look
every code up in a trainable embedding table and add the results. That keeps categories as categories: two
elements are simply two different vectors, with no ordering implied by their codes.

Message passing uses GINEConv, which adds the bond embedding to the neighbor's atom embedding before the
update, so bond features take part in every message rather than being ignored. After three rounds the atom
vectors are averaged into one molecule vector and a small head turns it into a single logit.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool

from helpers.features import ATOM_VOCAB, BOND_VOCAB


class CategoryEncoder(nn.Module):
    """Sum of one embedding table per categorical column."""

    def __init__(self, vocab_sizes, dim):
        super().__init__()
        self.tables = nn.ModuleList([nn.Embedding(size, dim) for size in vocab_sizes])
        for table in self.tables:
            nn.init.xavier_uniform_(table.weight)

    def forward(self, codes):
        return sum(table(codes[:, i]) for i, table in enumerate(self.tables))


def _mlp(dim):
    return nn.Sequential(nn.Linear(dim, 2 * dim), nn.BatchNorm1d(2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))


class MolecularGNN(nn.Module):
    def __init__(self, hidden=128, layers=3, dropout=0.3):
        super().__init__()
        self.atom_encoder = CategoryEncoder(ATOM_VOCAB, hidden)
        self.bond_encoders = nn.ModuleList([CategoryEncoder(BOND_VOCAB, hidden) for _ in range(layers)])
        self.convs = nn.ModuleList([GINEConv(_mlp(hidden), train_eps=True) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden // 2, 1))

    def forward(self, data):
        h = self.atom_encoder(data.x)
        for conv, norm, bond_encoder in zip(self.convs, self.norms, self.bond_encoders):
            message = conv(h, data.edge_index, bond_encoder(data.edge_attr))
            h = h + self.dropout(torch.relu(norm(message)))      # residual: keeps early layers reachable
        return self.head(global_mean_pool(h, data.batch)).view(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_table(model):
    import pandas as pd
    groups = {"atom embeddings": "atom_encoder", "bond embeddings": "bond_encoders",
              "message passing": "convs", "batch norm": "norms", "classification head": "head"}
    rows = [{"part": label, "parameters": sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix))}
            for label, prefix in groups.items()]
    rows.append({"part": "total", "parameters": count_parameters(model)})
    return pd.DataFrame(rows)
