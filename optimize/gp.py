"""Gaussian process regression with Cholesky solves and measured noise."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .kernels import covariance


class GP:
    def __init__(self, kernel="matern52", alpha=1.0, seed=0):
        if kernel not in ("rbf", "matern32", "matern52", "rq"):
            raise ValueError(f"Unsupported kernel: {kernel}")
        self.kernel = kernel
        self.alpha = alpha
        self.seed = seed

    def fit(self, x, y, noise_variance, optimize=True, restarts=3):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        nv = np.broadcast_to(np.asarray(noise_variance, dtype=float), y.shape).copy()
        if (x.ndim != 2 or len(x) != len(y) or len(y) < 2 or
                not np.isfinite(x).all() or not np.isfinite(y).all() or
                not np.isfinite(nv).all() or np.any(nv < 0)):
            raise ValueError("Invalid GP observations or noise variances")
        self.x = x
        self.mean = float(np.mean(y))
        self.scale = float(max(np.std(y), np.sqrt(np.median(nv)), 1e-9))
        self.y = (y - self.mean) / self.scale
        self.noise = nv / (self.scale * self.scale)
        d = x.shape[1]

        def factor(log_params):
            lengths = np.exp(log_params[:d])
            variance = float(np.exp(log_params[d]))
            k = covariance(x, x, self.kernel, lengths, variance, self.alpha)
            k.flat[::len(x)+1] += self.noise + 1e-8
            return cho_factor(k, lower=True, check_finite=True), lengths, variance

        def nll(params):
            try:
                chol, _, _ = factor(params)
                weights = cho_solve(chol, self.y)
                return float(.5 * self.y @ weights +
                             np.log(np.diag(chol[0])).sum() +
                             len(y) * np.log(2*np.pi) / 2)
            except (ValueError, np.linalg.LinAlgError):
                return 1e100

        initial = np.log(np.r_[np.full(d, .25), 1.0])
        bounds = [(np.log(.02), np.log(3.0))] * d + [(np.log(.01), np.log(100.0))]
        best = initial
        if optimize:
            rng = np.random.default_rng(self.seed)
            starts = [initial]
            starts += [rng.uniform([lo for lo, _ in bounds], [hi for _, hi in bounds])
                       for _ in range(max(0, restarts-1))]
            best = min((minimize(nll, p, method="L-BFGS-B", bounds=bounds) for p in starts),
                       key=lambda r: r.fun).x
        self.chol, self.length_scales, self.variance = factor(best)
        self.weights = cho_solve(self.chol, self.y)
        self.log_marginal_likelihood = -nll(best)
        return self

    def predict(self, x, observation_noise=0.0):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        k = covariance(self.x, x, self.kernel, self.length_scales, self.variance, self.alpha)
        mean = self.mean + self.scale * (k.T @ self.weights)
        variance = (self.variance - np.einsum("ij,ij->j", k, cho_solve(self.chol, k)))
        variance = np.maximum(variance, 0) * self.scale ** 2
        return mean, np.sqrt(variance + observation_noise)
