import time
import warnings
import numpy as np
from scipy.optimize import line_search
from .base import BaseOptimizer


class _Budget(Exception):
    pass


class LBFGSOptimizer(BaseOptimizer):

    def __init__(
        self,
        *args,
        m=10,
        gtol=None,
        ftol=1e-12,
        c1=0.0001,
        c2=0.9,
        maxls=30,
        time_limit=60.0,
        max_evals=100000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.m = m
        self.gtol = self.tol if gtol is None else gtol
        self.ftol = ftol
        self.c1 = c1
        self.c2 = c2
        self.maxls = maxls
        self.time_limit = time_limit
        self.max_evals = max_evals
        if (
            m < 1
            or self.gtol < 0
            or ftol < 0
            or (not 0 < c1 < c2 < 1)
            or (maxls < 1)
            or (time_limit <= 0)
            or (max_evals < 1)
        ):
            raise ValueError("Invalid settings")
        self.history = []
        self.events = []
        self.n_iter = 0

    def _check(self):
        if time.perf_counter() - self.start >= self.time_limit:
            raise _Budget("wall_clock_limit")

    def objective(self, x):
        self._check()
        if self.n_objective_evals >= self.max_evals:
            raise _Budget("objective_evaluation_limit")
        f = super().objective(x)
        self.history.append(
            dict(
                elapsed=time.perf_counter() - self.start,
                nfe=self.n_objective_evals,
                nge=self.n_gradient_evals,
                loss=f,
                best_loss=self.best_loss,
                gradient_norm=float("nan"),
                iteration=self.n_iter,
            )
        )
        return f

    def grad(self, x):
        self._check()
        g = super().grad(x)
        if g.shape != self.x0.shape:
            raise ValueError("Gradient dimension mismatch")
        return g

    @staticmethod
    def direction(g, pairs):
        q = g.copy()
        aa = []
        for s, y, rho in reversed(pairs):
            a = rho * (s @ q)
            aa.append(a)
            q -= a * y
        scale = (
            pairs[-1][0] @ pairs[-1][1] / (pairs[-1][1] @ pairs[-1][1])
            if pairs
            else 1.0
        )
        r = scale * q
        for (s, y, rho), a in zip(pairs, reversed(aa)):
            r += s * (a - rho * (y @ r))
        return -r

    def optimize(self):
        if self.grad_fn is None:
            raise ValueError("Gradient required")
        self.start = time.perf_counter()
        x = self.x0.copy()
        f = float("inf")
        pairs = []
        success = False
        message = "maximum_iterations"
        try:
            f = self.objective(x)
            g = self.grad(x)
            self.history[-1]["gradient_norm"] = float(np.linalg.norm(g, np.inf))
            if not np.isfinite(f) or not np.all(np.isfinite(g)):
                raise _Budget("nonfinite_initial_value")
            for k in range(self.max_iter):
                self._check()
                if np.linalg.norm(g, np.inf) <= self.gtol:
                    success = True
                    message = "gradient_tolerance"
                    break
                p = self.direction(g, pairs)
                if not np.all(np.isfinite(p)) or p @ g >= 0:
                    pairs.clear()
                    p = -g
                    self.events.append("descent_restart")
                slope = float(g @ p)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    a, _, _, fn, _, gn = line_search(
                        self.objective,
                        self.grad,
                        x,
                        p,
                        gfk=g,
                        old_fval=f,
                        c1=self.c1,
                        c2=self.c2,
                        maxiter=self.maxls,
                    )
                valid = a is not None and fn is not None and (gn is not None)
                if valid:
                    valid = (
                        np.isfinite(fn)
                        and np.all(np.isfinite(gn))
                        and (fn <= f + self.c1 * a * slope)
                        and (abs(gn @ p) <= -self.c2 * slope)
                    )
                if not valid:
                    self.events.append("armijo_fallback")
                    a = 1.0
                    for _ in range(self.maxls):
                        fn = self.objective(x + a * p)
                        if np.isfinite(fn) and fn <= f + self.c1 * a * slope:
                            gn = self.grad(x + a * p)
                            if np.all(np.isfinite(gn)):
                                break
                        a *= 0.5
                    else:
                        message = "line_search_failed"
                        break
                    pairs.clear()
                xn = x + a * p
                s = xn - x
                y = gn - g
                sy = float(s @ y)
                if sy > max(
                    np.finfo(float).tiny, 1e-10 * np.linalg.norm(s) * np.linalg.norm(y)
                ):
                    pairs.append((s.copy(), y.copy(), 1 / sy))
                    pairs = pairs[-self.m :]
                else:
                    self.events.append("curvature_pair_skipped")
                oldf = f
                x = xn
                f = float(fn)
                g = np.asarray(gn)
                self.n_iter += 1
                self.history.append(
                    dict(
                        elapsed=time.perf_counter() - self.start,
                        nfe=self.n_objective_evals,
                        nge=self.n_gradient_evals,
                        loss=f,
                        best_loss=self.best_loss,
                        gradient_norm=float(np.linalg.norm(g, np.inf)),
                        iteration=self.n_iter,
                    )
                )
                if np.linalg.norm(g, np.inf) <= self.gtol:
                    success = True
                    message = "gradient_tolerance"
                    break
                if self.ftol and abs(oldf - f) <= self.ftol * max(
                    1.0, abs(oldf), abs(f)
                ):
                    success = True
                    message = "function_tolerance"
                    break
        except _Budget as e:
            message = str(e)
        return self._pack_result(
            x, f, success, message, time.perf_counter() - self.start
        )
