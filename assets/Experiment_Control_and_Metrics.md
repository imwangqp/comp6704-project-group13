**Experimental Controls & Evaluation Plan**

Unconstrained Optimization: Newton-CG vs. L-BFGS vs. Nelder–Mead vs. CMA-ES

Purpose. Define a reproducible, fair comparison of four optimization paradigms on the Breast Cancer Wisconsin (Diagnostic) classification task. Separate optimization quality from predictive generalization and account for unequal per-iteration computational costs.

# 1\. Task and Shared Experimental Conditions

| Control                | Recommended setting / protocol                                                                                                                           |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset                | Breast Cancer Wisconsin (Diagnostic); 569 samples, 30 numerical features; binary benign/malignant classification.                                        |
| Data partition         | Stratified train/validation/test split = 70% / 15% / 15%; fix split per seed; keep test untouched until final reporting.                                 |
| Preprocessing          | Fit StandardScaler on training data only; transform validation/test using training statistics.                                                           |
| Models                 | Experiment A: logistic regression (31 parameters). Experiment B: MLP 30–2–1 (65 parameters), fixed activation (e.g., tanh).                              |
| Training objective     | Identical full-batch binary cross-entropy for all algorithms; no regularization in the primary experiment.                                               |
| Initialization         | Share identical initial parameter vector per seed for single-point methods; set CMA-ES initial mean to same vector and separately specify initial sigma. |
| Numerics & environment | Float64; same machine, CPU/GPU configuration, software versions, and objective implementation.                                                           |
| Replications           | 10 prespecified random seeds; use paired seeds and identical data splits across methods.                                                                 |
| Stopping & budget      | Common wall-clock cap per run (pilot: 60 s); common loss target; log method-specific termination reasons and all evaluation calls.                       |

# 2\. Method-Specific Hyperparameters

| Optimizer   | Tune / disclose                                                           | Starting point (not a universal optimum)            |
| ----------- | ------------------------------------------------------------------------- | --------------------------------------------------- |
| Newton-CG   | xtol; line-search behavior; HVP implementation; maximum iterations        | xtol = 1e-5; exact or automatic-differentiation HVP |
| L-BFGS      | history size (maxcor), gtol, ftol, line search                            | maxcor = 10; gtol = 1e-5                            |
| Nelder–Mead | initial simplex scale, adaptive option, xatol, fatol                      | adaptive = True; xatol = 1e-5                       |
| CMA-ES      | initial step size sigma0, population size, tolerance, covariance settings | sigma0 = 0.1; default population size               |

Tuning rule: allocate a comparable, prespecified validation/tuning budget to every method. Choose settings without inspecting test performance. Report both tuned settings and tuning cost. Method-specific hyperparameters should not be forced to equal numerical values.

# 3\. Compute Accounting and Fairness

- Primary comparison: best-so-far training loss versus elapsed wall-clock time, under the same time cap.
- Secondary comparison: best-so-far training loss versus number of objective function evaluations (NFE). NFE alone is not compute-equivalent across methods.
- Log gradient evaluations (NGE), Hessian-vector products (NHVP), and optional peak memory separately; include line-search and internal solver calls.
- Define target loss in advance using pilot/reference runs, not post hoc to favor a method. Report "not reached" for unsuccessful runs.
- For CMA-ES, count every candidate evaluation; evaluate and record its best-so-far candidate. Do not equate a generation with an iteration of another algorithm.
- Use a common logging interval where possible; include preprocessing and compilation/warm-up policy explicitly in timing methodology.

# 4\. Required Metrics

| Dimension                  | Metric                              | Reporting convention                                           |
| -------------------------- | ----------------------------------- | -------------------------------------------------------------- |
| Convergence                | Final training loss at fixed budget | Mean ± SD across seeds; lower is better                        |
| Convergence                | Time to target loss                 | Seconds; report non-reaching runs as censored/not reached      |
| Efficiency                 | Wall-clock time and NFE             | Time to target or fixed-budget outcomes; count all calls       |
| Efficiency                 | NGE / NHVP                          | Separate columns; not applicable where unused                  |
| Optimization quality       | Success rate                        | Fraction of seeds reaching prespecified target                 |
| Robustness                 | Variability across seeds            | Mean ± SD and distribution of final loss                       |
| Generalization             | Test ROC-AUC                        | Evaluate held-out test once per selected run; higher is better |
| Generalization (secondary) | Accuracy, F1, malignant recall      | Use validation-set threshold selection; report test results    |

Optional diagnostics: gradient norm at final solution (with extra evaluation cost disclosed), reference optimality gap for convex logistic regression, peak memory, and sensitivity to initial simplex / sigma0.

# 5\. Required Figures and Tables

| Deliverable                     | Axes / content                                                    | Research question                                    |
| ------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------- |
| Figure 1: Convergence           | Elapsed seconds vs. best-so-far training BCE                      | Which method reduces loss fastest in real time?      |
| Figure 2: Evaluation efficiency | NFE vs. best-so-far training BCE                                  | How many objective calls are needed?                 |
| Figure 3: Final loss            | Optimizer vs. loss distribution at common time cap                | Which finds the best solution, and how consistently? |
| Figure 4: Robustness            | Per-seed final-loss box/violin plot                               | How sensitive is each method to randomness?          |
| Figure 5: Generalization        | Optimizer vs. held-out ROC-AUC, mean ± SD                         | Does lower training loss improve prediction?         |
| Table 1: Summary                | Final loss; time-to-target; NFE; NGE; NHVP; success rate; ROC-AUC | One-page comparison across methods                   |

# 6\. Research Questions and Interpretation

- RQ1 — Convergence: Which method attains the lowest objective value under a fixed compute budget?
- RQ2 — Efficiency: Does exploiting curvature justify the extra derivative computation?
- RQ3 — Robustness: How sensitive are single-point and population methods to initialization and stochasticity?
- RQ4 — Generalization: Do optimization gains translate into better held-out classification performance?

Interpretation caveats. Logistic regression gives a convex baseline; the small MLP is nonconvex. Derivative-free methods may struggle as dimensionality grows. A lower training BCE does not guarantee a higher ROC-AUC. These experiments benchmark optimizers, not clinical diagnostic readiness.

# 7\. Suggested Reporting Template

| Method      | Loss ↓ | Time to target ↓ | NFE ↓ | Success ↑ | ROC-AUC ↑ |
| ----------- | ------ | ---------------- | ----- | --------- | --------- |
| Newton-CG   | TBD    | TBD              | TBD   | TBD       | TBD       |
| L-BFGS      | TBD    | TBD              | TBD   | TBD       | TBD       |
| Nelder–Mead | TBD    | TBD              | TBD   | TBD       | TBD       |
| CMA-ES      | TBD    | TBD              | TBD   | TBD       | TBD       |

Dataset reference: <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>