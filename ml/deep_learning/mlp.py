"""A small multilayer perceptron for "will this session end early?", as a scikit-learn estimator.

Being a scikit-learn estimator (fit / predict_proba / get_params) means it drops into the
same Pipeline, cross-validation, evaluation cards and tuning as every other model:

    Pipeline([("prep", make_preprocessor()), ("model", TorchMLPClassifier())])

The PyTorch pieces, in the order they are used:
    Tensor       n-dimensional array that can live on a GPU and records operations for autograd
    Dataset      indexable source of (features, label) pairs
    DataLoader   draws shuffled mini-batches from a Dataset
    nn.Module    the network: layers + forward()
    loss         BCEWithLogitsLoss: binary cross-entropy on raw logits (numerically stable)
    backward()   backpropagation: gradients of the loss w.r.t. every weight
    optimizer    AdamW: an adaptive gradient step, with decoupled weight decay (L2-like)
    epoch        one pass over the training data; batch = one optimiser step
    validation   a held-out slice scored after every epoch, for early stopping
    device       "cuda" if an NVIDIA GPU is available, else "cpu"
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from torch import nn
from torch.utils.data import DataLoader, Dataset


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class TabularDataset(Dataset):
    """Features and labels as float32 tensors. Data is small, so it is materialised once."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


class MLP(nn.Module):
    """input -> [Linear -> ReLU -> Dropout] x len(hidden) -> Linear -> 1 logit.

    Each Linear computes h = W x + b (a matrix multiplication). ReLU (max(0, ·)) is the
    non-linearity: without it, stacked Linear layers collapse into a single linear map.
    Dropout randomly zeroes activations during training only, a regulariser."""

    def __init__(self, n_inputs: int, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.2):
        super().__init__()
        layers, width = [], n_inputs
        for size in hidden:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (batch,) logits


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """MLP binary classifier with mini-batch training and early stopping.

    Early stopping holds out the LAST ``validation_fraction`` of the rows it is given. Our
    splits keep rows in time order, so that slice is the most recent period: the network is
    stopped on the future it has not seen, consistent with the rest of the project.
    """

    def __init__(
        self,
        hidden: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 15,
        validation_fraction: float = 0.15,
        device: str = "auto",
        random_state: int = 0,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.device = device
        self.random_state = random_state

    @staticmethod
    def _as_array(X) -> np.ndarray:
        return (X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)).astype(np.float32)

    def fit(self, X, y):
        X, y = self._as_array(X), np.asarray(y, dtype=np.float32)
        if np.isnan(X).any():
            raise ValueError("TorchMLPClassifier needs imputed features (put it after a preprocessor)")
        torch.manual_seed(self.random_state)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.device_ = resolve_device(self.device)

        n_val = max(1, int(len(X) * self.validation_fraction))
        train_ds, val_ds = TabularDataset(X[:-n_val], y[:-n_val]), TabularDataset(X[-n_val:], y[-n_val:])
        loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.random_state),
                            num_workers=0)  # no worker processes: safe under gevent, data is tiny

        self.model_ = MLP(X.shape[1], tuple(self.hidden), self.dropout).to(self.device_)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        X_val, y_val = val_ds.X.to(self.device_), val_ds.y.to(self.device_)

        best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
        self.history_ = []
        started = time.perf_counter()
        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()                       # dropout on
            running, seen = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device_), yb.to(self.device_)
                optimizer.zero_grad()                 # gradients accumulate otherwise
                loss = loss_fn(self.model_(xb), yb)   # forward pass
                loss.backward()                       # backpropagation
                optimizer.step()                      # weight update
                running += loss.item() * len(yb)
                seen += len(yb)

            self.model_.eval()                        # dropout off
            with torch.no_grad():                     # no graph needed for evaluation
                val_loss = float(loss_fn(self.model_(X_val), y_val))
            self.history_.append({"epoch": epoch, "train_loss": running / seen, "val_loss": val_loss})

            if val_loss < best_loss - 1e-5:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = copy.deepcopy(self.model_.state_dict())
            else:
                stale += 1
                if stale >= self.patience:
                    break

        self.model_.load_state_dict(best_state)       # roll back to the best validation epoch
        self.best_epoch_ = best_epoch
        self.n_epochs_ = len(self.history_)
        self.fit_seconds_ = time.perf_counter() - started
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = torch.as_tensor(self._as_array(X), device=self.device_)
        self.model_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.model_(X)).cpu().numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model_.parameters())
