Unconstrained Optimization

A Comparative Study of Four Optimization Paradigms

# 1\. Task Definition

Application scenario. Predict whether a breast tumor is malignant or benign using quantitative features extracted from digitized fine-needle aspirate images. This is a supervised binary-classification task intended as an optimization benchmark, not a clinically validated diagnostic system.

Input: x ∈ R³⁰ (30 numeric cell-nucleus features). Target: y ∈ {0,1} (benign or malignant). Prediction: pθ(x) = sigmoid(fθ(x)).

Optimization objective: minimize L(θ) = −(1/N) Σᵢ \[yᵢ log pθ(xᵢ) + (1−yᵢ) log(1−pθ(xᵢ))\], over θ ∈ Rᵈ. There are no explicit parameter constraints; this is unconstrained optimization. Numerically stable implementations should use logits-based binary cross-entropy.

# 2\. Open Dataset

Dataset: Breast Cancer Wisconsin (Diagnostic), hosted by the UCI Machine Learning Repository and included in scikit-learn as load_breast_cancer.

| Samples        | 569                                                                         |
| -------------- | --------------------------------------------------------------------------- |
| Input features | 30 real-valued attributes derived from cell nuclei                          |
| Classes        | 212 malignant; 357 benign                                                   |
| Data type      | Tabular, binary classification                                              |
| Source         | <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic> |
| Python access  | sklearn.datasets.load_breast_cancer                                         |

Suggested preprocessing: stratified train/validation/test split; fit feature standardization on the training split only; apply the same transformation to validation and test sets. Repeat experiments with multiple random seeds.

# 3\. Selected Optimization Methods

The comparison follows a conceptual progression: exact/implicit curvature → approximated curvature → derivative-free local search → population-based exploration. This is a methodological narrative, not a strict historical chronology.

## 3.1 Newton-CG — Second-Order Optimization

Uses gradient and Hessian–vector products to approximately solve for a curvature-informed Newton direction with conjugate gradients. It avoids explicitly inverting the Hessian but can require substantial computation per iteration.

Research question: Does curvature information improve convergence enough to offset its cost?

## 3.2 L-BFGS — Quasi-Newton Optimization

Constructs a limited-memory approximation to inverse-Hessian action from recent parameter and gradient differences. It offers curvature-aware updates without forming the full Hessian.

Research question: Can approximate curvature achieve a better efficiency–convergence trade-off?

## 3.3 Nelder–Mead — Derivative-Free Direct Search

Maintains a simplex of candidate parameter vectors and uses reflection, expansion, contraction, and shrink steps based only on objective values. It is simple but typically scales poorly with parameter dimension.

Research question: How competitive is local search when derivative information is unavailable?

## 3.4 CMA-ES — Evolutionary Optimization

Samples a population from an adaptive multivariate normal distribution, updating its mean, step size, and covariance based on candidate fitness. It enables broader stochastic exploration but can be expensive in higher dimensions.

Research question: Does population-based exploration improve robustness on a nonconvex objective?

# 4\. Experimental Design

Experiment A — Convex baseline: Logistic regression with 31 trainable parameters (30 coefficients plus intercept). Compare convergence and evaluation costs on a relatively well-understood objective.

Experiment B — Nonconvex model: A small MLP with architecture 30 → 2 → 1, nonlinear hidden activation, and 65 trainable parameters including biases. Compare optimizer sensitivity to initialization and nonconvexity.

Keep the data split, model, full-batch loss, feature scaling, initialization protocol, and stopping rules consistent. For population methods, initialize the search distribution around a comparable starting point. Tune each optimizer using a documented validation protocol; do not assume that equal iteration counts imply equal computational effort.

# 5\. Evaluation Metrics

- Optimization quality: final training binary cross-entropy and best objective reached.
- Convergence: loss versus elapsed time and objective-function evaluations; additionally track gradient/Hessian–vector evaluations where applicable.
- Efficiency: wall-clock time, peak memory, and total computational budget.
- Robustness: mean and standard deviation over multiple random seeds and starting points.
- Predictive performance: test accuracy, F1 score, ROC-AUC, and malignant-class recall, reported separately from training-loss optimization.

# 6\. Expected Discussion and Limitations

Compare the benefits of curvature exploitation, the cost of gradient-free evaluations, and the trade-off between local refinement and population exploration. No optimizer should be assumed to win before the experiments. The dataset is small, and findings may not generalize to high-dimensional neural networks. Nelder–Mead and CMA-ES may require much larger evaluation budgets than gradient-based approaches. A low training loss does not guarantee better test-set classification.

# 7\. References and Reproducibility

UCI dataset: <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>

Scikit-learn dataset loader: <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html>

SciPy optimize.minimize (Newton-CG, L-BFGS-B without bounds, Nelder–Mead): <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>

CMA-ES Python package: <https://pypi.org/project/cma/>

Reproducibility checklist: record split indices, random seeds, software versions, optimizer hyperparameters, termination criteria, and evaluation counts.