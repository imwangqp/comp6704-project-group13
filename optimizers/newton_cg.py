"""Newton-CG: outer Newton loop and Armijo backtracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .base import BaseOptimizer, OptimizerResult

# ABLATION TOGGLE: leave exactly one assignment uncommented
DIRECTION_METHOD = "newton_cg"       #original matrix-free truncated CG
# DIRECTION_METHOD = "full_newton_fd"    #full finite-difference Hessian + direct solve
HESSIAN_EPS = 1e-5

# LINE SEARCH ABLATION: leave exactly one assignment uncommented
STEP_METHOD = "armijo"     #original backtracking
# STEP_METHOD = "unit"         #always take the full proposed displacement (step=1)


@dataclass
class NewtonCGResult(OptimizerResult):
    # Each entry is an objective evaluation, including line-search trials.
    evaluation_history: list[dict] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    final_gradient_norm: float = 0.0
    outer_updates: int = 0
    n_hessian_evals: int = 0
    n_linear_solves: int = 0
    singular_fallbacks: int = 0
    downhill_fallbacks: int = 0


class NewtonCGOptimizer(BaseOptimizer):
    """Compute a Newton direction, choose a step length, then update theta."""

    def __init__(
        self,
        *,
        cg_rtol: float = 0.010506253,
        grad_tol: float | None = None,
        max_cg_iter: int = 2,
        armijo_c: float = 0.45842318,
        backtracking_factor: float = 0.62983795,
        **kwargs,
    ):
       
        super().__init__(**kwargs)
        self.cg_rtol = cg_rtol
        self.grad_tol = self.tol if grad_tol is None else grad_tol
        self.max_cg_iter = len(self.x0) if max_cg_iter is None else max_cg_iter
        self.armijo_c = armijo_c
        self.backtracking_factor = backtracking_factor


    def objective(self, theta):
        value = super().objective(theta)
        if hasattr(self, '_start_time'):
            self.evaluation_history.append({
                'elapsed_seconds': time.perf_counter() - self._start_time,
                'loss': value, 'best_loss': self.best_loss,
                'nfe': self.n_objective_evals,
                'nge_direct': self.n_gradient_evals,
                'nhvp': self.n_hessian_vec_evals,
                'n_hessian_evals': self.n_hessian_evals,
                'n_linear_solves': self.n_linear_solves,
            })
        return value

    def optimize(self) -> OptimizerResult:
        start_time = self._start_time = time.perf_counter()
        self.evaluation_history = []
        self.n_hessian_evals = 0
        self.n_linear_solves = 0
        self.singular_fallbacks = 0
        self.downhill_fallbacks = 0
        if DIRECTION_METHOD not in ("newton_cg", "full_newton_fd"):
            raise ValueError("Unknown DIRECTION_METHOD")
        if STEP_METHOD not in ("armijo", "unit"):
            raise ValueError("Unknown STEP_METHOD")
        outer_updates = 0
        theta = self.x0.copy()
        loss = self.objective(theta)
        gradient = self.grad(theta)

        success = False
        message = "Maximum iterations reached"

        for iteration in range(self.max_iter + 1):
            if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
                message = "Non-finite loss or gradient"
                break

            #check stationarity
            if np.linalg.norm(gradient) <= self.grad_tol:
                success = True
                message = "Gradient tolerance reached"
                break
            if iteration == self.max_iter:
                break

            if DIRECTION_METHOD == "full_newton_fd":
                direction = self._full_newton_direction(theta, gradient)
            else:
                direction = self._cg_direction(theta, gradient)

            if not np.all(np.isfinite(direction)):
                message = "Direction solver returned a non-finite direction"
                break

            #a nonconvex Hessian may give an uphill direction
            #fall back to steepest descent
            if np.dot(gradient, direction) >= 0:
                self.downhill_fallbacks += 1
                direction = -gradient

            if STEP_METHOD == "armijo":
                step_size, next_loss = self._line_search(
                    theta, loss, gradient, direction
                )
            else:
                step_size = 1.0
                next_loss = self.objective(theta + direction)
            if step_size is None:
                message = "Armijo line search failed"
                break

            
            outer_updates += 1
            theta = theta + step_size * direction
            loss = next_loss
            gradient = self.grad(theta)

        result = self._pack_result(
            theta,
            loss,
            success=success,
            message=message,
            wall_time=time.perf_counter() - start_time,
        )

        return NewtonCGResult(
            **vars(result), evaluation_history=self.evaluation_history,
            settings=dict(direction_method=DIRECTION_METHOD, step_method=STEP_METHOD,
                          fixed_step=1.0 if STEP_METHOD == "unit" else None,
                          line_search_parameters_active=STEP_METHOD == "armijo",
                          hessian_eps=HESSIAN_EPS if DIRECTION_METHOD == "full_newton_fd" else None,
                          cg_parameters_active=DIRECTION_METHOD == "newton_cg",
                          cg_rtol=self.cg_rtol, grad_tol=self.grad_tol,
                          max_cg_iter=self.max_cg_iter, armijo_c=self.armijo_c,
                          backtracking_factor=self.backtracking_factor,
                          max_iter=self.max_iter, tol=self.tol),
            final_gradient_norm=float(np.linalg.norm(gradient)),
            outer_updates=outer_updates,
            n_hessian_evals=self.n_hessian_evals,
            n_linear_solves=self.n_linear_solves,
            singular_fallbacks=self.singular_fallbacks,
            downhill_fallbacks=self.downhill_fallbacks,
        )

    def _full_hessian(self, theta):
        self.n_hessian_evals += 1
        n = len(theta)
        H = np.empty((n, n))
        for j in range(n):
            shift = np.zeros_like(theta)
            shift[j] = HESSIAN_EPS
            H[:, j] = (self.grad(theta + shift) - self.grad(theta - shift)) / (2 * HESSIAN_EPS)
        return (H + H.T) / 2  

    def _full_newton_direction(self, theta, gradient):
        H = self._full_hessian(theta)
        if not np.all(np.isfinite(H)):
            return np.full_like(gradient, np.nan)
        self.n_linear_solves += 1
        try:
            return np.linalg.solve(H, -gradient)  #never explicitly invert h
        except np.linalg.LinAlgError:
            #a singular Hessian has no unique Newton solution
            self.singular_fallbacks += 1
            return -gradient

    def _cg_direction(self, theta: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        max_cg_iter = self.max_cg_iter
        cg_tol = self.cg_rtol * np.linalg.norm(gradient)

        p = np.zeros_like(theta)
        r = -gradient
        d = r.copy()

        for iteration in range(max_cg_iter):
            if np.linalg.norm(r) <= cg_tol:
                break

            # theta remains fixed throughout this loop.
            Hd = self.hess_vec(theta, d)
            curvature = d @ Hd

            # stop if this direction does not have positive curvature
            if curvature <= 0:
                if iteration == 0:
                    return -gradient
                return p

            r_squared = r @ r
            alpha = r_squared / curvature

            p = p + alpha * d
            r_new = r - alpha * Hd

            if np.linalg.norm(r_new) <= cg_tol:
                break

            beta = (r_new @ r_new) / r_squared
            d = r_new + beta * d
            r = r_new

        return p

    def _line_search(self, theta, loss, gradient, direction):
        step_size = 1.0
        shrink_factor = self.backtracking_factor
        sufficient_decrease = self.armijo_c
        max_trials = 50
        slope = np.dot(gradient, direction)

        for _ in range(max_trials):
            trial_theta = theta + step_size * direction
            trial_loss = self.objective(trial_theta)
            threshold = loss + sufficient_decrease * step_size * slope

            if np.isfinite(trial_loss) and trial_loss <= threshold:
                return step_size, trial_loss

            step_size *= shrink_factor

        return None, loss
