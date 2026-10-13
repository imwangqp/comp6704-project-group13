"""L-BFGS optimizer — quasi-Newton method (Section 3.2 of the spec).

Research question: can approximate curvature achieve a better
efficiency-convergence trade-off?

TODO (implementation goes here):
    - Keep the last ``m`` pairs (s_k = theta_{k+1} - theta_k,
      y_k = grad_{k+1} - grad_k) and apply the two-loop recursion to get
      the search direction without forming the inverse Hessian.
    - Add a line search (e.g. strong Wolfe) along the direction using
      ``self.objective`` and ``self.grad``.
    - Alternative shortcut for the report baseline:
      scipy.optimize.minimize(fun=self.objective, x0=self.x0,
                              jac=self.grad, method="L-BFGS-B", ...)

Counters to respect: self.objective / self.grad already increment
n_objective_evals / n_gradient_evals.
"""

from __future__ import annotations

import numpy as np

from .base import BaseOptimizer, OptimizerResult


class LBFGSOptimizer(BaseOptimizer):
    """Limited-memory BFGS with curvature-aware, Hessian-free updates."""

    def optimize(self) -> OptimizerResult:
        raise NotImplementedError(
            "L-BFGS is not implemented yet — fill in optimizers/lbfgs.py."
        )
