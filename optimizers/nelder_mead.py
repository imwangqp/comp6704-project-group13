from __future__ import annotations

import time

import numpy as np

from .base import BaseOptimizer, OptimizerResult


class _BudgetExhausted(Exception):
    """Internal signal: a hard budget (NFE or wall-clock) has been hit."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class NelderMeadOptimizer(BaseOptimizer):
    """Simplex-based local search using objective values only.

    Args:
        initial_simplex_scale: relative size of the initial simplex.  Vertex
            ``i + 1`` is ``x0`` displaced along coordinate ``i`` by
            ``initial_simplex_scale * max(1, |x0_i|)``.  The absolute floor
            of 1 keeps the simplex non-degenerate when a parameter starts
            at zero (e.g. all bias terms in this project).  The default
            ``2.0`` is the value selected on the project's validation
            seeds; see ``results/nelder_mead/analysis.md``.
        adaptive: use Gao-Han dimension-dependent coefficients when True.
        xatol: absolute simplex-size tolerance (defaults to ``tol``).
        fatol: absolute objective-spread tolerance (defaults to ``tol``).
        max_nfev: optional hard cap on objective-function evaluations,
            counting the ``d + 1`` initial simplex evaluations.
        time_limit: optional wall-clock cap in seconds.
    """

    def __init__(
        self,
        initial_simplex_scale: float = 2.0,
        adaptive: bool = True,
        xatol: float | None = None,
        fatol: float | None = None,
        max_nfev: int | None = None,
        time_limit: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if initial_simplex_scale <= 0:
            raise ValueError("initial_simplex_scale must be positive.")
        if max_nfev is not None and max_nfev < 1:
            raise ValueError("max_nfev must be >= 1 when provided.")
        if time_limit is not None and time_limit <= 0:
            raise ValueError("time_limit must be positive when provided.")

        self.initial_simplex_scale = float(initial_simplex_scale)
        self.adaptive = bool(adaptive)
        self.xatol = float(self.tol if xatol is None else xatol)
        self.fatol = float(self.tol if fatol is None else fatol)
        self.max_nfev = None if max_nfev is None else int(max_nfev)
        self.time_limit = None if time_limit is None else float(time_limit)

        # Per-evaluation traces filled by the overridden objective().
        self.time_history: list[float] = []
        self.nfev_history: list[int] = []
        self.best_history: list[float] = []
        # Records (NFE, time, theta) each time the best objective improves,
        # so that a solution at any intermediate target loss can be recovered
        # without storing every candidate point.
        self.best_theta_history: list[np.ndarray] = []
        self.best_theta_nfev: list[int] = []
        self.best_theta_time: list[float] = []
        self._start_time: float | None = None

    # ------------------------------------------------------------------
    # Instrumented objective wrapper
    # ------------------------------------------------------------------
    def objective(self, theta: np.ndarray) -> float:
        """Evaluate f(theta) and record (time, NFE, best-so-far)."""
        previous_best = self.best_loss
        value = super().objective(theta)
        if self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
            self.time_history.append(elapsed)
            self.nfev_history.append(self.n_objective_evals)
            self.best_history.append(self.best_loss)
            if self.best_loss < previous_best:
                self.best_theta_history.append(self.best_theta.copy())
                self.best_theta_nfev.append(self.n_objective_evals)
                self.best_theta_time.append(elapsed)
        return value

    def _evaluate(self, theta: np.ndarray) -> float:
        """Evaluate with hard-budget checks performed *before* the call."""
        if (
            self.max_nfev is not None
            and self.n_objective_evals >= self.max_nfev
        ):
            raise _BudgetExhausted("max_nfev")
        if (
            self.time_limit is not None
            and time.perf_counter() - self._start_time >= self.time_limit
        ):
            raise _BudgetExhausted("time_limit")
        return self.objective(theta)

    # ------------------------------------------------------------------
    # Main algorithm
    # ------------------------------------------------------------------
    def optimize(self) -> OptimizerResult:
        start = time.perf_counter()
        self._start_time = start

        x0 = self.x0
        d = x0.size
        if d == 0:
            raise ValueError("Nelder-Mead requires at least one parameter.")

        # Adaptive (Gao & Han, 2012) vs. classical coefficients.
        if self.adaptive:
            rho = 1.0
            chi = 1.0 + 2.0 / d
            gamma = 0.75 - 1.0 / (2.0 * d)
            sigma = 1.0 - 1.0 / d
        else:
            rho, chi, gamma, sigma = 1.0, 2.0, 0.5, 0.5

        # ------------------------------------------------------------------
        # Initial simplex: x0 plus one displaced vertex per coordinate.
        # ------------------------------------------------------------------
        simplex = np.tile(x0, (d + 1, 1))
        step = self.initial_simplex_scale * np.maximum(np.abs(x0), 1.0)
        for i in range(d):
            simplex[i + 1, i] += step[i]

        fsim = np.full(d + 1, np.inf, dtype=float)
        reason = "maximum iterations reached"
        n_iterations = 0
        try:
            fsim[0] = self._evaluate(simplex[0])
            for i in range(1, d + 1):
                fsim[i] = self._evaluate(simplex[i])
        except _BudgetExhausted as exc:
            # Budget exhausted while building the simplex: return the best
            # point evaluated so far.
            reason = f"{exc.reason} reached"
            return self._finish(simplex, fsim, n_iterations, reason, start,
                                converged=False)

        # ------------------------------------------------------------------
        # Simplex iterations
        # ------------------------------------------------------------------
        try:
            for iteration in range(self.max_iter):
                n_iterations = iteration + 1

                # Order: fsim[0] <= ... <= fsim[d]; worst is index d.
                order = np.argsort(fsim, kind="stable")
                simplex = simplex[order]
                fsim = fsim[order]

                # Convergence test on BOTH simplex size and objective spread.
                # The x criterion is the standard max_i ||x_i - x_best||_inf
                # (same as SciPy's Nelder-Mead implementation).
                x_spread = float(np.max(np.abs(simplex - simplex[0])))
                f_spread = float(np.max(np.abs(fsim[1:] - fsim[0])))
                if x_spread <= self.xatol and f_spread <= self.fatol:
                    reason = (
                        f"xatol and fatol satisfied (x_spread={x_spread:.3e}, "
                        f"f_spread={f_spread:.3e})"
                    )
                    return self._finish(simplex, fsim, n_iterations, reason,
                                        start, converged=True)

                centroid = simplex[:-1].mean(axis=0)

                # Reflection.
                x_r = centroid + rho * (centroid - simplex[-1])
                f_r = self._evaluate(x_r)

                if f_r < fsim[0]:
                    # Expansion (better than the current best).
                    x_e = centroid + chi * (centroid - simplex[-1])
                    f_e = self._evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1], fsim[-1] = x_e, f_e
                    else:
                        simplex[-1], fsim[-1] = x_r, f_r
                elif f_r < fsim[-2]:
                    # Accept reflection (strictly better than second worst).
                    simplex[-1], fsim[-1] = x_r, f_r
                else:
                    # Contraction.
                    if f_r < fsim[-1]:
                        # Outside contraction.
                        x_c = centroid + gamma * (centroid - simplex[-1])
                        f_c = self._evaluate(x_c)
                        if f_c <= f_r:
                            simplex[-1], fsim[-1] = x_c, f_c
                        else:
                            self._shrink(simplex, fsim, sigma, self._evaluate)
                    else:
                        # Inside contraction.
                        x_c = centroid - gamma * (centroid - simplex[-1])
                        f_c = self._evaluate(x_c)
                        if f_c < fsim[-1]:
                            simplex[-1], fsim[-1] = x_c, f_c
                        else:
                            self._shrink(simplex, fsim, sigma, self._evaluate)
        except _BudgetExhausted as exc:
            reason = f"{exc.reason} reached"

        return self._finish(simplex, fsim, n_iterations, reason, start,
                            converged=False)

    @staticmethod
    def _shrink(simplex: np.ndarray, fsim: np.ndarray, sigma: float, evaluate):
        """Shrink all non-best vertices toward the best vertex."""
        best = simplex[0].copy()
        for i in range(1, len(simplex)):
            simplex[i] = best + sigma * (simplex[i] - best)
            fsim[i] = evaluate(simplex[i])

    def _finish(
        self,
        simplex: np.ndarray,
        fsim: np.ndarray,
        n_iterations: int,
        reason: str,
        start: float,
        converged: bool,
    ) -> OptimizerResult:
        """Pack the best point seen, attach traces, and return."""
        result = self._pack_result(
            self.best_theta,
            self.best_loss,
            success=converged,
            message=reason,
            wall_time=time.perf_counter() - start,
        )
        # Extra diagnostics for the experiment/report driver.  These are
        # attached dynamically so the shared OptimizerResult schema (and the
        # other methods) stay untouched.
        result.n_iterations = int(n_iterations)
        result.termination = reason
        result.nfev_history = np.asarray(self.nfev_history, dtype=int)
        result.time_history = np.asarray(self.time_history, dtype=float)
        result.best_history = np.asarray(self.best_history, dtype=float)
        result.best_theta_history = np.asarray(self.best_theta_history, dtype=float)
        result.best_theta_nfev = np.asarray(self.best_theta_nfev, dtype=int)
        result.best_theta_time = np.asarray(self.best_theta_time, dtype=float)
        result.final_simplex = np.asarray(simplex, dtype=float)
        result.final_fsim = np.asarray(fsim, dtype=float)
        result.initial_simplex_scale = self.initial_simplex_scale
        result.adaptive = self.adaptive
        result.xatol = self.xatol
        result.fatol = self.fatol
        return result
