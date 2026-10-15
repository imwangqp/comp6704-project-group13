"""Training driver: plugs optimizers into the MLP classification pipeline.

The data/model side is complete. The optimization side is an extension
point: OPTIMIZER_REGISTRY maps optimizer names to factories. The four
methods (Newton-CG, L-BFGS, Nelder-Mead, CMA-ES) live in the separate
``optimizers/`` package and are wired in through
``optimizers.make_optimizer``; until they are implemented, a plain
full-batch gradient descent baseline is provided so the whole pipeline
(data -> model -> optimize -> evaluate) can run end to end.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from optimizers import make_optimizer
from optimizers.base import BaseOptimizer, OptimizerResult
from src.data import Dataset, load_and_split
from src.model import MLP, LogisticRegression, make_model

# scikit-learn label convention: 0 = malignant, 1 = benign.
MALIGNANT_CLASS = 0


# ----------------------------------------------------------------------
# Placeholder optimizer: full-batch gradient descent.
# Kept only so the pipeline runs before the four real methods land.
# Replace its uses with the optimizers from the ``optimizers`` package.
# ----------------------------------------------------------------------
class GradientDescentBaseline(BaseOptimizer):
    """Simple full-batch GD with a fixed step size (NOT part of the study)."""

    def __init__(self, lr: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr

    def optimize(self) -> OptimizerResult:
        start = time.perf_counter()
        theta = self.x0.copy()
        loss = self.objective(theta)
        for _ in range(self.max_iter):
            theta -= self.lr * self.grad(theta)
            new_loss = self.objective(theta)
            if abs(loss - new_loss) < self.tol:
                loss = new_loss
                break
            loss = new_loss
        return self._pack_result(
            theta, loss, success=True,
            message="gradient descent baseline", wall_time=time.perf_counter() - start,
        )


def build_optimizer(name: str, objective, grad, hess_vec, x0, max_iter: int, tol: float):
    """Factory hook — the single place where optimizers get selected.

    The four study methods are instantiated from the ``optimizers``
    package (see optimizers/__init__.py::make_optimizer). Add or swap
    entries here without touching the training loop below.
    """
    if name == "gd":
        return GradientDescentBaseline(
            objective=objective, grad=grad, hess_vec=hess_vec,
            x0=x0, max_iter=max_iter, tol=tol,
        )
    return make_optimizer(
        name, objective=objective, grad=grad, hess_vec=hess_vec,
        x0=x0, max_iter=max_iter, tol=tol,
    )


# ----------------------------------------------------------------------
# Evaluation metrics (Section 5 of the spec)
# ----------------------------------------------------------------------
@dataclass
class Metrics:
    accuracy: float
    f1: float
    roc_auc: float
    recall_malignant: float

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "recall_malignant": self.recall_malignant,
        }


def evaluate(model: MLP, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> Metrics:
    """Predictive performance of parameters ``theta`` on one split."""
    model.set_flat_params(theta)
    proba = model.predict_proba(X)
    pred = (proba >= 0.5).astype(np.int64)

    # Malignant recall: sensitivity for class 0 (malignant = positive).
    malignant_mask = y == MALIGNANT_CLASS
    recall_malignant = float((pred[malignant_mask] == MALIGNANT_CLASS).mean())

    return Metrics(
        accuracy=float(accuracy_score(y, pred)),
        f1=float(f1_score(y, pred)),
        roc_auc=float(roc_auc_score(y, proba)),
        recall_malignant=recall_malignant,
    )


# ----------------------------------------------------------------------
# One training run: data -> model -> optimizer -> test evaluation
# ----------------------------------------------------------------------
@dataclass
class RunReport:
    model: str
    seed: int
    optimizer: str
    result: OptimizerResult
    val_metrics: Metrics
    test_metrics: Metrics


def run_training(
    data: Dataset,
    optimizer_name: str = "gd",
    model_name: str = "mlp",
    seed: int = 0,
    init_seed: int | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> RunReport:
    """Train ``model_name`` on ``data`` with the selected optimizer.

    Args:
        data: dataset split from ``src.data.load_and_split``.
        optimizer_name: one of the optimizer names ("gd" baseline, or
            "newton-cg" / "lbfgs" / "nelder-mead" / "cma-es").
        model_name: "mlp" (Experiment B, 30->2->1, 65 params) or
            "logreg" (Experiment A, convex baseline, 31 params). The same
            optimizers run on either model unchanged.
        seed: weight-initialization seed (defaults to ``seed`` if
            ``init_seed`` is None) — kept separate from the split seed so
            robustness over "multiple random seeds and starting points"
            (Section 5) can be measured along either axis.
    """
    init_seed = seed if init_seed is None else init_seed
    model = make_model(model_name, input_dim=data.input_dim, seed=init_seed)
    x0 = model.get_flat_params()

    # Full-batch objective — deterministic, required by the spec (Section 4).
    objective = model.make_objective(data.X_train, data.y_train)
    grad = model.make_grad(data.X_train, data.y_train)
    hess_vec = model.make_hess_vec(data.X_train, data.y_train)

    optimizer = build_optimizer(
        optimizer_name, objective=objective, grad=grad, hess_vec=hess_vec,
        x0=x0, max_iter=max_iter, tol=tol,
    )
    result = optimizer.optimize()

    val_metrics = evaluate(model, result.theta, data.X_val, data.y_val)
    test_metrics = evaluate(model, result.theta, data.X_test, data.y_test)
    return RunReport(
        model=model_name, seed=seed, optimizer=optimizer_name, result=result,
        val_metrics=val_metrics, test_metrics=test_metrics,
    )
