"""Newton-CG optimizer — second-order method (Section 3.1 of the spec).

Research question: does curvature information improve convergence enough
to offset its cost?

TODO (implementation goes here):
    - Approximately solve for the Newton direction using conjugate
      gradients, requiring only Hessian-VECTOR products (never form the
      Hessian). Use ``self.hess_vec(theta, v)``.
    - Typical inner loop: CG on  H(theta) d = -grad(theta)  with a
      tolerance proportional to ||grad|| (Eisenstat-Walker), then a
      backtracking line search along d using ``self.objective``.
    - Alternative shortcut for the report baseline:
      scipy.optimize.minimize(fun=self.objective, x0=self.x0,
                              jac=self.grad, hessp=self.hess_vec,
                              method="Newton-CG", ...)

Counters to respect: self.objective / self.grad / self.hess_vec already
increment n_objective_evals / n_gradient_evals / n_hessian_vec_evals.
"""

from __future__ import annotations

import numpy as np

from .base import BaseOptimizer, OptimizerResult


class NewtonCGOptimizer(BaseOptimizer):
    """Curvature-informed Newton direction computed with conjugate gradients."""

    def optimize(self) -> OptimizerResult:
        raise NotImplementedError(
            "Newton-CG is not implemented yet — fill in optimizers/newton_cg.py."
        )
