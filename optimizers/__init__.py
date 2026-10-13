"""Optimizer package for the COMP6704 unconstrained-optimization project.

The four paradigms compared in assets/Unconstrained_Optimization_Comparison.md:

    - newton_cg.NewtonCGOptimizer       (second-order,        Section 3.1)
    - lbfgs.LBFGSOptimizer              (quasi-Newton,        Section 3.2)
    - nelder_mead.NelderMeadOptimizer   (derivative-free,     Section 3.3)
    - cma_es.CMAESOptimizer             (evolutionary,        Section 3.4)

These are imported into src/train.py's OPTIMIZER_REGISTRY. Each class
inherits optimizers.base.BaseOptimizer, which wraps the MLP objective so
objective/gradient/Hessian-vector evaluation counts are tracked for the
efficiency report (Section 5).
"""

from __future__ import annotations

import importlib

from .base import BaseOptimizer, OptimizerResult

__all__ = ["BaseOptimizer", "OptimizerResult", "OPTIMIZER_CLASSES", "make_optimizer"]

# name -> (module, class) inside this package
OPTIMIZER_CLASSES = {
    "newton-cg": ("newton_cg", "NewtonCGOptimizer"),
    "lbfgs": ("lbfgs", "LBFGSOptimizer"),
    "nelder-mead": ("nelder_mead", "NelderMeadOptimizer"),
    "cma-es": ("cma_es", "CMAESOptimizer"),
}


def make_optimizer(name: str, **kwargs) -> BaseOptimizer:
    """Instantiate an optimizer by name.

    Raises NotImplementedError (with a pointer to the stub file) until
    the corresponding method is implemented.
    """
    if name not in OPTIMIZER_CLASSES:
        raise KeyError(f"Unknown optimizer '{name}'. Valid: {sorted(OPTIMIZER_CLASSES)}")
    module_name, class_name = OPTIMIZER_CLASSES[name]
    module = importlib.import_module(f".{module_name}", __name__)
    cls = getattr(module, class_name)
    return cls(**kwargs)
