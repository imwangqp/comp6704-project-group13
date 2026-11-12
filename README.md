# COMP6704 Project — Unconstrained Optimization Comparison

Binary classification on the Wisconsin Breast Cancer dataset, comparing
four optimization paradigms — Newton-CG, L-BFGS, Nelder-Mead, and CMA-ES —
on two models (Section 4 of the spec):

- **Experiment A (convex baseline)**: logistic regression, 31 parameters
  (30 weights + intercept).
- **Experiment B (nonconvex model)**: small MLP, 30 → 2 → 1, 65 parameters.

All four optimizers are fully implemented in `optimizers/` and run
end-to-end through `run_experiment.py`. See
`assets/Unconstrained_Optimization_Comparison.md` for the task
specification and `assets/Experiment_Control_and_Metrics.md` for the
experimental-control protocol. Example result figures are included in
`assets/` (`fig1_loss_vs_iteration.png`, `fig2_loss_vs_evals.png`,
`fig3_test_metrics.png`).

---

## 1. Environment Setup

Python **3.10+** (tested on 3.12). CPU only; the dataset (569 × 30) ships
with scikit-learn — nothing to download.

```bash
pip install -r requirements.txt
```

| Package | Version | Used for |
| --- | --- | --- |
| `numpy` | ≥ 1.24 | MLP forward/backprop, all four optimizers, all numerics |
| `scipy` | ≥ 1.10 | `expit` (stable sigmoid), `optimize.line_search` (L-BFGS) |
| `scikit-learn` | ≥ 1.3 | `load_breast_cancer` dataset, metrics (accuracy, F1, ROC-AUC) |

---

## 2. Repository Structure

```
comp6704-project-group13/
├── assets/
│   ├── Unconstrained_Optimization_Comparison.md   # task specification
│   ├── Experiment_Control_and_Metrics.md          # experimental controls & metrics plan
│   └── fig1..fig3 .png                            # example result figures
├── src/                          # data + model + training driver
│   ├── data.py                   # load, stratified split, standardization (train-only fit)
│   ├── model.py                  # logreg (Exp A) + 30->2->1 MLP (Exp B) as flat-vector objectives
│   └── train.py                  # optimizer factory, metrics, one training run
├── optimizers/                   # the four study methods (all implemented)
│   ├── __init__.py               # name -> class registry (make_optimizer)
│   ├── base.py                   # BaseOptimizer + OptimizerResult: eval counters, best-point tracking
│   ├── newton_cg.py              # Newton-CG (Section 3.1)
│   ├── lbfgs.py                  # L-BFGS (Section 3.2)
│   ├── nelder_mead.py            # Nelder-Mead (Section 3.3)
│   └── cma_es.py                 # CMA-ES (Section 3.4)
├── run_experiment.py             # entry point: models × optimizers × seeds
├── requirements.txt
└── README.md
```

### Pipeline at a glance

`load_and_split(seed)` → `make_model(...)` → flat `objective / grad /
hess_vec` → `build_optimizer(name)` → `optimize()` → val/test metrics.

- **`src/data.py`** — `load_and_split(seed, val_frac, test_frac)` returns a
  `Dataset` with a stratified train/val/test split (70/15/15) and
  standardization fitted on the training split only. Split indices are
  recorded for reproducibility.

- **`src/model.py`** — two models, both exposing the objective exactly as
  an unconstrained optimizer needs it (`make_model("logreg" | "mlp", ...)`
  builds either):

  | Model | Experiment | Params | `hess_vec` |
  | --- | --- | --- | --- |
  | `LogisticRegression` | A (convex baseline) | 31 | analytic: `H = (1/N) Xᵀ diag(p(1-p)) X` |
  | `MLP` (30 → 2 → 1) | B (nonconvex) | 65 | finite differences (2 gradient evals per call) |

  The loss is a numerically stable logits-based binary cross-entropy,
  evaluated full-batch (deterministic objective — required by the
  derivative-free methods). Both models share the same flat-parameter
  interface, so every optimizer runs on either model unchanged.

- **`optimizers/base.py`** — every optimizer calls the wrapped
  `self.objective / self.grad / self.hess_vec`, so objective/gradient/
  Hessian-vector evaluation counts, the loss history, and the best point
  visited are tracked uniformly (Section 5 of the spec). All four methods
  return the same `OptimizerResult`
  (`theta, final_loss, best_loss, loss_history, n_objective_evals,
  n_gradient_evals, n_hessian_vec_evals, wall_time, success, message`).

---

## 3. The Four Implementations

| Method | File / class | Information used | `max_iter` means | Algorithm & typical termination (reported in `message`) |
| --- | --- | --- | --- | --- |
| Newton-CG | `newton_cg.py::NewtonCGOptimizer` | objective, grad, hess_vec | outer Newton updates | matrix-free truncated CG on Hessian-vector products (Hessian never formed); nonconvex-curvature and downhill fallbacks; Armijo backtracking line search. Stops on gradient tolerance or iteration budget. |
| L-BFGS | `lbfgs.py::LBFGSOptimizer` | objective, grad | iterations | two-loop recursion over the last `m=10` `(s, y)` pairs; Wolfe line search. Stops on gradient or function tolerance. |
| Nelder-Mead | `nelder_mead.py::NelderMeadOptimizer` | objective only | simplex iterations | adaptive (Gao–Han) coefficients, initial simplex scale 2.0, reflect/expand/contract/shrink; optional NFE / wall-clock caps. Stops when simplex size and objective spread satisfy `xatol`/`fatol`. |
| CMA-ES | `cma_es.py::CMAESOptimizer` | objective only | generations (λ = 4 + ⌊3 ln d⌋ offspring each) | standard rank-one + rank-mu covariance update. Stops when σ·√λmax(C) ≤ `tol` ("distribution collapsed") or the generation budget is exhausted. |

Notes for reading the results:

- Nelder-Mead and CMA-ES are derivative-free, so `grad evals` and
  `hvp evals` stay at 0 for them — expected, not a bug.
- The unregularized logistic-regression objective is nearly separable on
  this dataset, so its Hessian becomes near-singular at the optimum
  (min eigenvalue ~1e-5). Newton-CG handles this via its line search and
  fallbacks — a good discussion point for the report.
- `gd` (in `src/train.py`) is a plain full-batch gradient-descent baseline
  used to smoke-test the pipeline; it is not part of the study.

---

## 4. How to Run

Main experiment — both models × all four optimizers × 3 seeds:

```bash
python run_experiment.py --models logreg mlp \
    --optimizers newton-cg lbfgs nelder-mead cma-es --seeds 0 1 2
```

Quick checks and custom budgets:

```bash
# Single-optimizer smoke test
python run_experiment.py --models mlp --optimizers lbfgs --seeds 0

# Baseline pipeline test (gradient-descent placeholder)
python run_experiment.py --models logreg mlp --optimizers gd --seeds 0 1 2

# Custom budgets / separate weight-init seeds (robustness axis)
python run_experiment.py --models mlp --optimizers lbfgs cma-es \
    --seeds 0 1 2 --init-seeds 0 1 2 --max-iter 500 --tol 1e-8
```

Arguments:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--optimizers` | `gd` | names: `newton-cg`, `lbfgs`, `nelder-mead`, `cma-es`, and `gd` (baseline) |
| `--models` | `mlp` | `logreg` (Experiment A) and/or `mlp` (Experiment B) |
| `--seeds` | `0 1 2` | data-split seeds |
| `--init-seeds` | = seeds | separate weight-initialization seeds |
| `--max-iter` | `200` | iteration / evaluation budget passed to each optimizer (meaning per method — see Section 3) |
| `--tol` | `1e-6` | convergence tolerance (method-specific — see Section 3) |

Example output (`python run_experiment.py --models mlp --optimizers newton-cg lbfgs nelder-mead cma-es --seeds 0`):

```
model   optimizer    seed  final loss  obj evals  grad evals  hvp evals  time (s)  test acc  test F1     AUC  mal recall
------------------------------------------------------------------------------------------------------------------------
mlp     newton-cg       0    0.000001        137         113        224     0.093    0.9469   0.9565  0.9977      0.9762
mlp     lbfgs           0    0.009811         44          42          0     0.015    0.9558   0.9645  0.9839      0.9524
mlp     nelder-mead     0    0.143171        305           0          0     0.033    0.9558   0.9655  0.9816      0.9048
mlp     cma-es          0    0.008941       3201           0          0     2.137    0.9381   0.9510  0.9634      0.9048
```

Columns: `final loss` is the training loss at the returned parameters;
`obj/grad/hvp evals` are the objective-function, gradient, and
Hessian-vector evaluation counts required by Section 5 of the spec;
the last four columns are test metrics computed on the untouched test
split. Wall-clock times vary by machine.

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

