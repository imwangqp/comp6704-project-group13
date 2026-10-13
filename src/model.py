"""MLP model for binary classification on the breast cancer dataset.

Architecture follows the project spec (Section 4, Experiment B):
    30 -> 2 -> 1, tanh hidden activation, sigmoid output
    => 30*2 + 2 (layer 1) + 2*1 + 1 (layer 2) = 65 trainable parameters.

The parameters are exposed as ONE flat vector ``theta`` so that every
optimizer in ``optimizers/`` can treat the network purely as an
unconstrained real-valued objective:

    f(theta)        -> scalar training loss      (objective)
    grad f(theta)   -> flat gradient             (gradient-based methods)
    hess_vec(theta, v) -> flat Hessian-vector product  (Newton-CG only)

The loss is a numerically stable binary cross-entropy computed from
logits (Section 1 of the spec). Everything is full-batch so the
objective is deterministic — required for derivative-free methods.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit  # numerically stable sigmoid


def bce_with_logits(z: np.ndarray, y: np.ndarray) -> float:
    """Numerically stable mean binary cross-entropy on logits.

    l(z, y) = max(z, 0) - z*y + log(1 + exp(-|z|))
    """
    z = np.asarray(z, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    return float(
        np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z))))
    )


class MLP:
    """30 -> 2 -> 1 MLP with a flat-parameter view for external optimizers."""

    def __init__(self, input_dim: int = 30, hidden_dim: int = 2, seed: int = 0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Deterministic initialization protocol (kept identical across
        # optimizers so the comparison isolates the optimizer, Section 4).
        rng = np.random.default_rng(seed)
        lim1 = np.sqrt(6.0 / (input_dim + hidden_dim))       # Xavier for tanh
        lim2 = np.sqrt(6.0 / (hidden_dim + 1))
        self.W1 = rng.uniform(-lim1, lim1, size=(hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.uniform(-lim2, lim2, size=(1, hidden_dim))
        self.b2 = np.zeros(1)

        self._shapes = [self.W1.shape, self.b1.shape, self.W2.shape, self.b2.shape]
        self.num_params = sum(int(np.prod(s)) for s in self._shapes)

    # ------------------------------------------------------------------
    # Flat-parameter view
    # ------------------------------------------------------------------
    def get_flat_params(self) -> np.ndarray:
        """Pack all parameters into a single 1-D vector theta."""
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_flat_params(self, theta: np.ndarray) -> None:
        """Unpack a 1-D vector theta back into the layer parameters."""
        theta = np.asarray(theta, dtype=float)
        sizes = [int(np.prod(s)) for s in self._shapes]
        chunks = np.split(theta, np.cumsum(sizes)[:-1])
        self.W1 = chunks[0].reshape(self._shapes[0])
        self.b1 = chunks[1].reshape(self._shapes[1])
        self.W2 = chunks[2].reshape(self._shapes[2])
        self.b2 = chunks[3].reshape(self._shapes[3])

    # ------------------------------------------------------------------
    # Objective pieces handed to the optimizers
    # ------------------------------------------------------------------
    def forward_logits(self, X: np.ndarray) -> np.ndarray:
        """Forward pass; returns logits z of shape (N,). Uses tanh hidden."""
        h1 = np.tanh(X @ self.W1.T + self.b1)   # (N, H)
        z = (h1 @ self.W2.T + self.b2).ravel()  # (N,)
        return z

    def make_objective(self, X: np.ndarray, y: np.ndarray):
        """Return f(theta) = full-batch training loss (scalar)."""
        def objective(theta: np.ndarray) -> float:
            self.set_flat_params(theta)
            return bce_with_logits(self.forward_logits(X), y)
        return objective

    def make_grad(self, X: np.ndarray, y: np.ndarray):
        """Return grad f(theta) via backprop (full batch, averaged over N)."""
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)

        def grad(theta: np.ndarray) -> np.ndarray:
            self.set_flat_params(theta)
            a1 = X @ self.W1.T + self.b1                     # (N, H)
            h1 = np.tanh(a1)
            z = (h1 @ self.W2.T + self.b2).ravel()           # (N,)

            # d(BCE)/dz for mean-reduced BCE with logits is (sigmoid(z)-y)/N.
            d_z = (expit(z) - y) / n                         # (N,)

            g_W2 = (d_z[:, None] * h1).sum(axis=0)           # (H,)
            g_b2 = d_z.sum(axis=0)                           # scalar
            d_a1 = (d_z[:, None] * self.W2) * (1.0 - h1 ** 2)  # (N, H)
            g_W1 = d_a1.T @ X                                # (H, D)
            g_b1 = d_a1.sum(axis=0)                          # (H,)
            return np.concatenate([g_W1.ravel(), g_b1, g_W2.ravel(), np.atleast_1d(g_b2)])
        return grad

    def make_hess_vec(self, X: np.ndarray, y: np.ndarray, eps: float = 1e-5):
        """Return hess_vec(theta, v) via central finite differences.

        H(theta) v ~= [grad(theta + eps*v) - grad(theta - eps*v)] / (2*eps)

        This costs 2 gradient evaluations per call, which the training
        loop accounts for separately (see OptimizerResult counters).
        A closed-form HVP could be substituted here later without
        touching any optimizer code.
        """
        grad = self.make_grad(X, y)

        def hess_vec(theta: np.ndarray, v: np.ndarray) -> np.ndarray:
            v = np.asarray(v, dtype=float)
            return (grad(theta + eps * v) - grad(theta - eps * v)) / (2.0 * eps)
        return hess_vec

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicted probability of the positive class (benign, y=1)."""
        return expit(self.forward_logits(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)
