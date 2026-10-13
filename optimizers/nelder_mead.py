"""Nelder-Mead optimizer — derivative-free direct search (Section 3.3).

Research question: how competitive is local search when derivative
information is unavailable?

TODO (implementation goes here):
    - Maintain a simplex of d+1 candidate parameter vectors (d = 65 here,
      so expect many objective evaluations — that is part of the point).
    - Implement reflection / expansion / contraction / shrink steps using
      ONLY ``self.objective`` values.
    - Use standard coefficients, e.g. reflect 1.0, expand 2.0,
      contract 0.5, shrink 0.5.
    - Alternative shortcut for the report baseline:
      scipy.optimize.minimize(fun=self.objective, x0=self.x0,
                              method="Nelder-Mead", ...)

Note: ``self.grad_fn`` is intentionally unused in this method.
"""

from __future__ import annotations

import numpy as np

from .base import BaseOptimizer, OptimizerResult


class NelderMeadOptimizer(BaseOptimizer):
    """Simplex-based local search using objective values only."""

    def optimize(self) -> OptimizerResult:
        raise NotImplementedError(
            "Nelder-Mead is not implemented yet — fill in optimizers/nelder_mead.py."
        )
