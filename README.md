# COMP6704 Project — Unconstrained Optimization Comparison

Binary classification on the Wisconsin Breast Cancer dataset, comparing
four optimization paradigms — Newton-CG, L-BFGS, Nelder-Mead, and CMA-ES —
on two models (Section 4 of the spec):

- **Experiment A (convex baseline)**: logistic regression, 31 parameters
  (30 weights + intercept).
- **Experiment B (nonconvex model)**: small MLP, 30 → 2 → 1, 65 parameters.

See `assets/Unconstrained_Optimization_Comparison.md` for the full task
specification.

---

## 1. Environment Setup

Python **3.10+** is recommended.

Required packages:

| Package        | Version   | Used for                                        |
| -------------- | --------- | ----------------------------------------------- |
| `numpy`        | any modern | MLP forward/backprop, all numerics             |
| `scipy`        | ≥ 1.10    | `expit` (stable sigmoid); optional shortcut baselines (`scipy.optimize.minimize`) |
| `scikit-learn` | ≥ 1.3     | `load_breast_cancer` dataset, metrics (accuracy, F1, ROC-AUC) |
| `cma`          | optional  | Only if you take the `cma` package shortcut in `optimizers/cma_es.py` |

```bash
pip install numpy scipy scikit-learn
pip install cma        # optional, for the CMA-ES shortcut
```

---

## 2. Code Structure

```
comp6704-project-group13/
├── assets/
│   └── Unconstrained_Optimization_Comparison.md   # task specification
├── src/                          # data + model + training (COMPLETE)
│   ├── data.py                   # load, stratified split, standardization
│   ├── model.py                  # logreg (Exp A) + 30->2->1 MLP (Exp B),
│   │                             #   both exposed as flat-vector objectives
│   └── train.py                  # optimizer hook, metrics, one training run
├── optimizers/                   # the four methods (TO IMPLEMENT)
│   ├── base.py                   # BaseOptimizer + OptimizerResult (shared)
│   ├── newton_cg.py              # TODO: Section 3.1
│   ├── lbfgs.py                  # TODO: Section 3.2
│   ├── nelder_mead.py            # TODO: Section 3.3
│   └── cma_es.py                 # TODO: Section 3.4
├── run_experiment.py             # entry point: seeds x optimizers
└── README.md
```

### What each module provides

- **`src/data.py`** — `load_and_split(seed, val_frac, test_frac)` returns a
  `Dataset` with stratified train/val/test splits and standardization fitted
  on the training split only. Split indices are recorded for
  reproducibility.

- **`src/model.py`** — two models, both exposing the objective exactly as
  an unconstrained optimizer needs it (`make_model("logreg" | "mlp", ...)`
  builds either):

  | Model | Experiment | Params | `hess_vec` |
  | --- | --- | --- | --- |
  | `LogisticRegression` | A (convex baseline) | 31 | analytic: `H = (1/N) Xᵀ diag(p(1-p)) X` |
  | `MLP` (30 → 2 → 1) | B (nonconvex) | 65 | finite differences (2 gradient evals per call) |

  | Callback | Signature | Used by |
  | --- | --- | --- |
  | `make_objective(X, y)` | `f(theta) -> float` | all four methods |
  | `make_grad(X, y)` | `grad(theta) -> (d,)` | Newton-CG, L-BFGS |
  | `make_hess_vec(X, y)` | `hess_vec(theta, v) -> (d,)` | Newton-CG only |

  The loss is a numerically stable logits-based binary cross-entropy,
  evaluated full-batch (deterministic objective — required by the
  derivative-free methods). Both models share the same flat-parameter
  interface, so every optimizer runs on either model unchanged.

- **`src/train.py`** — glue code. `build_optimizer(name, ...)` is the single
  place where an optimizer is selected; `run_training(..., model_name=...)`
  runs data → model → optimize → evaluate for either model and returns a
  `RunReport` with `OptimizerResult` (loss history,
  objective/gradient/Hessian-vector evaluation counts, wall-clock time)
  plus val/test metrics (accuracy, F1, ROC-AUC, malignant-class recall).

- **`optimizers/base.py`** — `BaseOptimizer` wraps the raw objective and
  automatically counts objective/gradient/Hessian-vector evaluations and
  tracks the best point visited. Every optimizer **must** call
  `self.objective(...)`, `self.grad(...)`, `self.hess_vec(...)` instead of
  the raw callables so these counters stay correct (Section 5 of the spec).

  A uniform result is returned as:

  ```python
  OptimizerResult(
      theta, final_loss, best_loss, loss_history,
      n_objective_evals, n_gradient_evals, n_hessian_vec_evals,
      wall_time, success, message,
  )
  ```

- A placeholder **`GradientDescentBaseline`** (`name="gd"`) lives in
  `src/train.py`. It is **not** part of the study; it only exists so the
  pipeline can run end-to-end before the four methods are implemented.

---

## 3. Where to Add Optimizer Code

All optimizer implementations go in the `optimizers/` folder — one file per
method, one class per file. **Do not modify `src/train.py` or `src/model.py`**
to add an optimizer; the interface is already wired.

Each stub follows this pattern:

```python
# optimizers/newton_cg.py
from .base import BaseOptimizer, OptimizerResult

class NewtonCGOptimizer(BaseOptimizer):
    def optimize(self) -> OptimizerResult:
        raise NotImplementedError  # <-- replace this with your implementation
```

Implementation checklist:

1. **Open the stub file** (`newton_cg.py`, `lbfgs.py`, `nelder_mead.py`,
   or `cma_es.py`). Its docstring lists the method-specific TODO and a
   suggested approach.
2. **Fill in `optimize()`** using only the wrapped callables provided by
   `BaseOptimizer`:
   - `self.objective(theta)` — scalar loss (increments eval counter),
   - `self.grad(theta)` — flat gradient (Newton-CG, L-BFGS),
   - `self.hess_vec(theta, v)` — Hessian-vector product (Newton-CG only),
   - starting point `self.x0`, budget `self.max_iter`, tolerance `self.tol`.
3. **Return a result** by calling the provided helper:

   ```python
   return self._pack_result(theta, loss, success=True,
                            message="...", wall_time=elapsed)
   ```

   It fills in the eval counters, loss history, and best point seen.

   Shortcut: you may implement a method via SciPy / the `cma` package, but
   pass the wrapped `self.objective` / `self.grad` / `self.hess_vec` into
   them (e.g. `scipy.optimize.minimize(fun=self.objective, x0=self.x0,
   jac=self.grad, method="L-BFGS-B")`) so evaluation counts remain
   meaningful. If SciPy consumes evaluations internally, read them back
   from its result object into the counters before packing.

4. **Nothing else to register** — `optimizers/__init__.py` already maps the
   names and `make_optimizer("newton-cg" | "lbfgs" | "nelder-mead" |
   "cma-es", ...)` imports the class by name.

Notes per method (full details in each stub's docstring):

| Method | Uses | Key requirement |
| --- | --- | --- |
| Newton-CG | `objective`, `grad`, `hess_vec` | CG with Hessian-vector products only, never form the Hessian |
| L-BFGS | `objective`, `grad` | two-loop recursion over last m `(s, y)` pairs + line search |
| Nelder-Mead | `objective` only | d+1 = 66-point simplex; reflect/expand/contract/shrink |
| CMA-ES | `objective` only | population sampling; init distribution around `self.x0` |

---

## 4. How to Run

Run experiments (multiple seeds × multiple optimizers):

```bash
# Baseline smoke test (works today, before any optimizer is implemented)
python run_experiment.py --models logreg mlp --optimizers gd --seeds 0 1 2

# Once the four methods are implemented (both experiments, Section 4):
python run_experiment.py --models logreg mlp \
    --optimizers newton-cg lbfgs nelder-mead cma-es --seeds 0 1 2

# Custom budgets / init points
python run_experiment.py --models mlp --optimizers lbfgs cma-es \
    --seeds 0 1 2 --init-seeds 0 1 2 --max-iter 500 --tol 1e-8
```

Arguments:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--optimizers` | `gd` | names: `gd`, `newton-cg`, `lbfgs`, `nelder-mead`, `cma-es` |
| `--models` | `mlp` | `logreg` (Experiment A) and/or `mlp` (Experiment B) |
| `--seeds` | `0 1 2` | data-split seeds |
| `--init-seeds` | = seeds | separate weight-initialization seeds (robustness axis) |
| `--max-iter` | `200` | iteration / evaluation budget passed to each optimizer |
| `--tol` | `1e-6` | convergence tolerance |

Example output (baseline run):

```
model   optimizer    seed  final loss  obj evals  grad evals  hvp evals  time (s)  test acc  test F1     AUC  mal recall
------------------------------------------------------------------------------------------------------------------------
logreg  gd              0    0.069137        201         200          0     0.030    0.9646   0.9722  0.9926      0.9286
mlp     gd              0    0.049226        201         200          0     0.034    0.9558   0.9655  0.9966      0.9048
logreg  gd              1    0.036892        201         200          0     0.007    0.9646   0.9722  0.9946      0.9286
mlp     gd              1    0.021295        201         200          0     0.020    0.9646   0.9718  0.9950      0.9524
```

Note for Experiment A: the unregularized logistic-regression objective is
nearly separable on this dataset, so the Hessian becomes near-singular at
the optimum (min eigenvalue ~1e-5). Newton-CG may therefore need damping
or a line search to be stable — a good discussion point for the report.

Not-yet-implemented methods print a pointer to their stub file instead of
crashing the run.

### Using the pipeline programmatically

```python
from src.data import load_and_split
from src.train import run_training

data = load_and_split(seed=0)
report = run_training(data, optimizer_name="lbfgs", model_name="logreg",
                      seed=0, max_iter=300)
print(report.result.final_loss, report.test_metrics.as_dict())
print(report.result.loss_history)      # loss vs. iteration for plots
print(report.result.n_objective_evals) # loss vs. objective evals for plots
```

---

## 5. Reproducibility Checklist

Per Section 7 of the spec, record for every run: split indices (kept in
`Dataset.train_idx / val_idx / test_idx`), random seeds (`--seeds`,
`--init-seeds`), software versions, optimizer hyperparameters
(`max_iter`, `tol`, method-specific constants), termination criteria
(from `OptimizerResult.message`), and evaluation counts (from
`OptimizerResult`).
