"""Acquisition scores for minimizing a measured loss."""

import numpy as np
from scipy.stats import norm


def lcb(mean, std, kappa=2.0):
    if kappa < 0:
        raise ValueError("kappa must be nonnegative")
    return np.asarray(mean) - kappa * np.asarray(std)


def expected_improvement(mean, std, best, xi=0.0):
    if xi < 0:
        raise ValueError("xi must be nonnegative")
    mean, std = np.broadcast_arrays(np.asarray(mean, dtype=float),
                                   np.asarray(std, dtype=float))
    improvement = best - mean - xi
    safe_std = np.where(std > 0, std, 1.0)
    z = improvement / safe_std
    return np.where(std > 0, improvement * norm.cdf(z) + std * norm.pdf(z),
                    np.maximum(improvement, 0.0))
