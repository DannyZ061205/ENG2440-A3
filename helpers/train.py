"""Training the graph network: one run per seed, model selection on validation only.

Nothing in this file reads a test subset. The epoch that is kept is the one with the best validation
ROC-AUC; the decision threshold is chosen afterwards, also on validation.
"""
import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from helpers.models import MolecularGNN, count_parameters


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)      # MPS lacks deterministic kernels for scatter-add


def loaders(dataset, parts, batch_size, seed=0):
    """Training loader shuffled with its own generator; evaluation loaders in a fixed order."""
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader([dataset[int(i)] for i in parts[0]], batch_size=batch_size, shuffle=True,
                       generator=generator, drop_last=False)
    val, test = (DataLoader([dataset[int(i)] for i in part], batch_size=batch_size, shuffle=False)
                 for part in parts[1:])
    return train, val, test


def positive_weight(labels):
    """pos_weight = negatives / positives, estimated on the training subset only."""
    labels = np.asarray(labels)
    return float((labels == 0).sum() / max((labels == 1).sum(), 1))


@torch.no_grad()
def predict(model, loader, device):
    """Penetration probabilities and labels, in loader order."""
    model.eval()
    scores, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        scores.append(torch.sigmoid(model(batch)).cpu())
        labels.append(batch.y.view(-1).cpu())
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def train_run(dataset, parts, config, seed, device, verbose=False):
    """Train one model; return it with its history and the epoch chosen on validation."""
    from sklearn.metrics import roc_auc_score

    set_seed(seed)
    train_loader, val_loader, test_loader = loaders(dataset, parts, config["batch_size"], seed=seed)
    train_labels = [int(dataset[int(i)].y.view(-1)[0]) for i in parts[0]]

    model = MolecularGNN(hidden=config["hidden"], layers=config["layers"],
                         dropout=config["dropout"]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config["epochs"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight(train_labels), device=device))

    history, best = [], {"val_roc_auc": -np.inf, "epoch": -1, "state": None}
    started = time.time()
    for epoch in range(config["epochs"]):
        model.train()
        total, seen = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(batch), batch.y.view(-1))
            loss.backward()
            optimiser.step()
            total += loss.item() * batch.num_graphs
            seen += batch.num_graphs
        scheduler.step()

        val_scores, val_labels = predict(model, val_loader, device)
        with torch.no_grad():                                   # validation loss, same criterion
            val_logits = torch.logit(torch.tensor(val_scores).clamp(1e-6, 1 - 1e-6))
            val_loss = loss_fn(val_logits.to(device), torch.tensor(val_labels, dtype=torch.float32,
                                                                   device=device)).item()
        row = {"epoch": epoch, "train_loss": total / seen, "val_loss": val_loss,
               "val_roc_auc": roc_auc_score(val_labels, val_scores)}
        history.append(row)

        if row["val_roc_auc"] > best["val_roc_auc"]:
            best = {"val_roc_auc": row["val_roc_auc"], "epoch": epoch,
                    "state": copy.deepcopy(model.state_dict())}
        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  train {row['train_loss']:.4f}  val {row['val_loss']:.4f}  "
                  f"val ROC-AUC {row['val_roc_auc']:.3f}")
        if epoch - best["epoch"] >= config["patience"]:
            break

    model.load_state_dict(best["state"])
    return {
        "model": model,
        "history": history,
        "best_epoch": best["epoch"],
        "epochs_run": len(history),
        "parameters": count_parameters(model),
        "seconds": time.time() - started,
        "loaders": {"train": train_loader, "val": val_loader, "test": test_loader},
    }
