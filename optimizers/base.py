"""Base classes and shared bookkeeping for all optimizers.

Every optimizer sees the MLP as an unconstrained real-valued objective:

    objective(theta)          -> float          (always available)
    grad(theta)               -> (d,) array     (None for derivative-free)
    hess_vec(theta, v)        -> (d,) array     (None except Newton-CG)

Subclasses implement :meth:`BaseOptimizer.optimize` and must only touch
the parameters through the callables given to ``__init__`` so that
objective/gradient evaluation counts are tracked uniformly (Section 5
of the project spec requires reporting objective-function evaluations).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OptimizerResult:
    """Uniform result record returned by every optimizer."""

    theta: np.ndarray                 # best parameters found
    final_loss: float                 # loss at ``theta``
    best_loss: float                  # best loss seen during the run
    loss_history: list[float] = field(default_factory=list)
    # Evaluation counters required by Section 5 of the spec.
    n_objective_evals: int = 0
    n_gradient_evals: int = 0
    n_hessian_vec_evals: int = 0
    wall_time: float = 0.0            # seconds, filled in by optimize()
    success: bool = False
    message: str = ""


class BaseOptimizer(ABC):
    """Common bookkeeping + abstract interface for the four methods.

    Args:
        objective: f(theta) -> float.
        grad: f'(theta) -> (d,) array; None for derivative-free methods.
        hess_vec: Hessian-vector product f''(theta) v -> (d,) array.
        x0: starting point theta_0.
        max_iter: iteration / evaluation budget (meaning is method-specific
            but must be documented in each subclass).
        tol: convergence tolerance (meaning is method-specific).
    """

    def __init__(
        self,
        objective,
        x0: np.ndarray,
        grad=None,
        hess_vec=None,
        max_iter: int = 200,
        tol: float = 1e-6,
    ):
        self.objective_fn = objective
        self.grad_fn = grad
        self.hess_vec_fn = hess_vec
        self.x0 = np.asarray(x0, dtype=float).ravel()
        self.max_iter = max_iter
        self.tol = tol

        # Bookkeeping state, updated via the wrapped callables below.
        self.n_objective_evals = 0
        self.n_gradient_evals = 0
        self.n_hessian_vec_evals = 0
        self.loss_history: list[float] = []
        self.best_loss = float("inf")
        self.best_theta = self.x0.copy()

    # Wrapped callables — optimizers MUST call these instead of the raw
    # functions so eval counters and the loss history stay consistent.
    def objective(self, theta: np.ndarray) -> float:
        self.n_objective_evals += 1
        value = float(self.objective_fn(theta))
        self.loss_history.append(value)
        if value < self.best_loss:  # keep the best point ever visited
            self.best_loss = value
            self.best_theta = np.asarray(theta, dtype=float).copy()
        return value

    def grad(self, theta: np.ndarray) -> np.ndarray:
        if self.grad_fn is None:
            raise ValueError(
                f"{type(self).__name__} requested a gradient but none was provided."
            )
        self.n_gradient_evals += 1
        return np.asarray(self.grad_fn(theta), dtype=float)

    def hess_vec(self, theta: np.ndarray, v: np.ndarray) -> np.ndarray:
        if self.hess_vec_fn is None:
            raise ValueError(
                f"{type(self).__name__} requested a Hessian-vector product "
                "but none was provided."
            )
        self.n_hessian_vec_evals += 1
        return np.asarray(self.hess_vec_fn(theta, v), dtype=float)

    @abstractmethod
    def optimize(self) -> OptimizerResult:
        """Run the optimization and return an :class:`OptimizerResult`."""
        raise NotImplementedError

    def _pack_result(self, theta: np.ndarray, loss: float, success: bool, message: str,
                     wall_time: float) -> OptimizerResult:
        return OptimizerResult(
            theta=np.asarray(theta, dtype=float),
            final_loss=float(loss),
            best_loss=float(self.best_loss),
            loss_history=list(self.loss_history),
            n_objective_evals=self.n_objective_evals,
            n_gradient_evals=self.n_gradient_evals,
            n_hessian_vec_evals=self.n_hessian_vec_evals,
            wall_time=wall_time,
            success=success,
            message=message,
        )
