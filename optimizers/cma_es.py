"""CMA-ES optimizer — evolutionary, population-based method (Section 3.4).

Research question: does population-based exploration improve robustness
on a nonconvex objective?

TODO (implementation goes here):
    - Sample a population from N(mean, sigma^2 * C), rank candidates by
      ``self.objective``, and update mean, step size sigma, and covariance
      C via the standard CMA update equations (rank-one + rank-mu).
    - Initialize the search distribution around a starting point
      comparable to the other methods (x0 from the shared init protocol).
    - The ``cma`` Python package (pypi.org/project/cma) may be used as a
      shortcut for the report baseline:
          import cma
          es = cma.CMAEvolutionStrategy(self.x0, sigma0)
          while not es.stop():
              solutions = es.ask()
              es.tell(solutions, [self.objective(s) for s in solutions])

Note: ``self.grad_fn`` is intentionally unused in this method.
"""

from __future__ import annotations

import numpy as np

from .base import BaseOptimizer, OptimizerResult


class CMAESOptimizer(BaseOptimizer):
    """Adaptive multivariate-normal sampling with covariance updates."""

    def optimize(self) -> OptimizerResult:
        raise NotImplementedError(
            "CMA-ES is not implemented yet — fill in optimizers/cma_es.py."
        )
