"""Data loading and preprocessing pipeline for the breast cancer task.

Implements the preprocessing protocol described in
assets/Unconstrained_Optimization_Comparison.md (Section 2):

- stratified train / validation / test split,
- feature standardization fitted on the training split ONLY,
- the same transformation applied to validation and test splits,
- support for repeating the experiment over multiple random seeds.

The label convention of scikit-learn's ``load_breast_cancer`` is kept:
    y = 0 -> malignant, y = 1 -> benign.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.datasets import load_breast_cancer


@dataclass
class Dataset:
    """Container for one train/val/test split of the dataset."""

    X_train: np.ndarray  # (N_train, 30), standardized
    y_train: np.ndarray  # (N_train,) in {0, 1}
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    # Indices into the raw dataset, kept for the reproducibility checklist
    # (record split indices, see Section 7 of the project spec).
    train_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    val_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    test_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    # Standardization statistics fitted on the training split only.
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    @property
    def input_dim(self) -> int:
        return self.X_train.shape[1]


def _stratified_split_indices(
    y: np.ndarray,
    val_frac: float,
    test_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) with per-class proportions kept.

    Each class is shuffled independently and then split into val / test /
    train chunks so that every split preserves the class balance.
    """
    train_parts, val_parts, test_parts = [], [], []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)

        n_val = int(round(len(cls_idx) * val_frac))
        n_test = int(round(len(cls_idx) * test_frac))

        val_parts.append(cls_idx[:n_val])
        test_parts.append(cls_idx[n_val:n_val + n_test])
        train_parts.append(cls_idx[n_val + n_test:])

    train_idx = np.sort(np.concatenate(train_parts))
    val_idx = np.sort(np.concatenate(val_parts))
    test_idx = np.sort(np.concatenate(test_parts))
    return train_idx, val_idx, test_idx


def load_and_split(
    seed: int = 0,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
) -> Dataset:
    """Load the Wisconsin breast cancer dataset and build one split.

    Args:
        seed: Random seed controlling both the split and nothing else
            (weight initialization uses its own seed in ``src/model.py``).
        val_frac: Fraction of samples assigned to the validation split.
        test_frac: Fraction of samples assigned to the test split.

    Returns:
        A ``Dataset`` with standardized features. The standardizer is
        fitted on the training split only.
    """
    X, y = load_breast_cancer(return_X_y=True)
    y = y.astype(np.int64)

    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = _stratified_split_indices(
        y, val_frac=val_frac, test_frac=test_frac, rng=rng
    )

    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_val_raw, y_val = X[val_idx], y[val_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]

    # Fit standardization on the training split only, then reuse it.
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # guard against degenerate features

    return Dataset(
        X_train=(X_train_raw - mean) / std,
        y_train=y_train,
        X_val=(X_val_raw - mean) / std,
        y_val=y_val,
        X_test=(X_test_raw - mean) / std,
        y_test=y_test,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        mean=mean,
        std=std,
    )
