"""CMA-ES optimizer — evolutionary, population-based method (Section 3.4).

Research question: does population-based exploration improve robustness
on a nonconvex objective?

Implementation: covariance matrix adaptation evolution strategy with the
standard rank-one + rank-mu update (Hansen & Ostermeier 2001 / Hansen 2016
tutorial), no negative-weight corrections (lambda is small here).

Note: ``self.grad_fn`` is intentionally unused in this method.
"""

from __future__ import annotations

import time

import numpy as np

from .base import BaseOptimizer, OptimizerResult


class CMAESOptimizer(BaseOptimizer):
    """Adaptive multivariate-normal sampling with covariance updates.

    ``max_iter`` is the number of generations; each generation evaluates
    ``lambda`` candidates (lambda = 4 + floor(3 ln d)), so the objective
    budget is about max_iter * lambda. ``tol`` is a scale threshold:
    the run terminates successfully once sigma * sqrt(largest covariance
    eigenvalue) drops below tol (distribution collapsed around the mean).
    """

    def optimize(self) -> OptimizerResult:
        start = time.perf_counter()
        # Deterministic RNG seeded from the problem instance so runs are
        # reproducible given the same x0 (no explicit seed is threaded
        # through BaseOptimizer).
        rng = np.random.default_rng(
            int.from_bytes(self.x0.tobytes(), "little") % (2 ** 63))
        d = self.x0.size

        # --- Strategy parameters (Hansen tutorial defaults) ---
        lam = 4 + int(np.floor(3.0 * np.log(d)))
        mu = lam // 2
        prime_w = np.log((lam + 1) / 2.0) - np.log(np.arange(1, lam + 1))
        weights = prime_w[:mu].copy()
        weights /= weights.sum()          # positive weights, sum = 1
        mueff = 1.0 / (weights ** 2).sum()

        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        c1 = 2.0 / ((d + 1.3) ** 2 + mueff)
        cmu = min(1 - c1,
                  2.0 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        cs = (mueff + 2) / (d + mueff + 3)
        ds = 1 + cs + 2 * max(0.0, np.sqrt((mueff - 1) / (d + 1)) - 1)

        # --- State ---
        m = self.x0.copy()
        sigma = 0.3
        eigvals = np.ones(d)              # C = B diag(eigvals) B^T, C = I
        B = np.eye(d)
        ps = np.zeros(d)
        pc = np.zeros(d)

        best_theta = m.copy()
        best_loss = self.objective(m)
        success = False
        message = "generation budget exhausted"

        for g in range(self.max_iter):
            # 1. Sample lambda offspring from N(m, sigma^2 C_i): y = B D z.
            z = rng.standard_normal((lam, d))
            y = z @ (B * np.sqrt(np.maximum(eigvals, 1e-20))).T
            x = m + sigma * y

            fit = np.array([self.objective(xi) for xi in x])
            order = np.argsort(fit)
            y_sel = y[order[:mu]]
            y_w = weights @ y_sel         # weighted mean shift in white space

            # 2. Mean update.
            m_old = m
            m = m_old + sigma * y_w

            # 3. Step-size path: conjugate (C^-1/2) evolution path ps.
            invsqrt_c = B @ np.diag(1.0 / np.sqrt(eigvals)) @ B.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrt_c @ (m - m_old) / sigma)
            ps_norm = float(np.linalg.norm(ps))
            hsig = ps_norm / (chi_n * np.sqrt(1 - (1 - cs) ** (2 * (g + 1)))) < 1.4 + 2 / (d + 1)
            sigma *= float(np.exp((cs / ds) * (ps_norm / chi_n - 1)))
            sigma = min(sigma, 1e12)

            # 4. Covariance path (rank-one) and rank-one + rank-mu update.
            delta_m = (m - m_old) / sigma
            pc = (1 - c1) * pc + (np.sqrt(c1 * (2 - c1) * mueff) if hsig else 0.0) * delta_m
            y_w2 = (weights[:, None] * y_sel).T           # columns of weighted y
            C = ((1 - c1 - cmu) * (B * eigvals) @ B.T
                 + c1 * np.outer(pc, pc)
                 + (1 - float(hsig)) * c1 * (2 - c1) * np.diag(eigvals)
                 + cmu * (y_w2 @ y_w2.T))
            C = (C + C.T) / 2.0

            # 5. Eigendecompose updated C for the next generation.
            eigvals, B = np.linalg.eigh(C)
            e_min, e_max = eigvals.min(), eigvals.max()
            if e_min <= 0:                                # repair for sampling
                eigvals = eigvals + (abs(e_min) * 10 + 1e-14)

            # 6. Track the best point ever visited.
            if fit[order[0]] < best_loss:
                best_loss = float(fit[order[0]])
                best_theta = x[order[0]].copy()

            scale = sigma * np.sqrt(max(eigvals.max(), 0.0))
            if scale <= self.tol:
                success = True
                message = (f"distribution collapsed (scale {scale:.3e} <= "
                           f"tol {self.tol:g}) at generation {g}")
                break

        return self._pack_result(best_theta, best_loss, success=success,
                                 message=message,
                                 wall_time=time.perf_counter() - start)
