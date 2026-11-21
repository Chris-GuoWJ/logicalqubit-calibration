"""Stationary GP kernels with one length scale per control parameter."""

import numpy as np


def covariance(x, z, kind="matern52", length_scales=1.0, variance=1.0, alpha=1.0):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    z = np.atleast_2d(np.asarray(z, dtype=float))
    lengths = np.asarray(length_scales, dtype=float)
    if x.ndim != 2 or z.ndim != 2 or x.shape[1] != z.shape[1]:
        raise ValueError("Kernel inputs must have the same number of columns")
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("Kernel inputs must be finite")
    if not np.isfinite(lengths).all() or np.any(lengths <= 0) or lengths.size not in (1, x.shape[1]):
        raise ValueError("Length scales must be positive, one per dimension or shared")
    if not np.isfinite(variance) or variance <= 0 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Variance and alpha must be positive")
    r2 = np.sum(((x[:, None, :] - z[None, :, :]) / lengths) ** 2, axis=-1)
    if kind == "rbf":
        return variance * np.exp(-.5 * r2)
    if kind == "matern32":
        r = np.sqrt(3 * r2)
        return variance * (1 + r) * np.exp(-r)
    if kind == "matern52":
        r = np.sqrt(5 * r2)
        return variance * (1 + r + r * r / 3) * np.exp(-r)
    if kind == "rq":
        return variance * (1 + r2 / (2 * alpha)) ** (-alpha)
    raise ValueError(f"Unknown kernel: {kind}")
