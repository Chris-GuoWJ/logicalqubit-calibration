"""Ask/tell Bayesian optimization. All hardware I/O belongs to src and notebooks."""

from dataclasses import dataclass
import numpy as np

from .gp import GP
from .acquisition import lcb, expected_improvement


@dataclass(frozen=True)
class Observation:
    x: tuple
    loss: float
    noise_variance: float
    shots: int = 0
    job_id: str = ""


class Search:
    def __init__(self, bounds, kernel="matern52", acquisition="ei", kappa=2.0,
                 xi=0.0, seed=0, initial_points=5):
        self.bounds = np.asarray(bounds, dtype=float)
        if (self.bounds.ndim != 2 or self.bounds.shape[1] != 2 or
                not np.isfinite(self.bounds).all() or
                np.any(self.bounds[:, 1] <= self.bounds[:, 0])):
            raise ValueError("Bounds must be pairs of increasing finite limits")
        if acquisition not in ("ei", "lcb"):
            raise ValueError("acquisition must be ei or lcb")
        self.kernel = kernel
        self.acquisition = acquisition
        self.kappa = kappa
        self.xi = xi
        self.initial_points = max(initial_points, 2)
        self.rng = np.random.default_rng(seed)
        self.observations = []
        self.seed = seed
        self.model = None

    def tell(self, x, loss, noise_variance, shots=0, job_id=""):
        x = np.asarray(x, dtype=float)
        if (x.shape != (len(self.bounds),) or not np.isfinite(x).all() or
                np.any(x < self.bounds[:, 0]) or np.any(x > self.bounds[:, 1]) or
                not np.isfinite(loss) or not np.isfinite(noise_variance) or
                noise_variance < 0 or shots < 0):
            raise ValueError("Observation outside bounds or has invalid loss/noise")
        self.observations.append(Observation(tuple(x), float(loss),
                                             float(noise_variance), int(shots), str(job_id)))

    def ask(self, candidates=2048, refit=True):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        if len(self.observations) < self.initial_points:
            return self.rng.uniform(lo, hi)
        x = np.asarray([o.x for o in self.observations])
        y = np.asarray([o.loss for o in self.observations])
        noise = np.asarray([o.noise_variance for o in self.observations])
        normalized = (x - lo) / (hi - lo)
        self.model = GP(kernel=self.kernel, seed=self.seed).fit(normalized, y, noise,
                                                                 optimize=refit)
        grid = self.rng.uniform(0, 1, (candidates, len(lo)))
        # Evaluate near the current best as well as over the full parameter box.
        incumbent = normalized[np.argmin(self.model.predict(normalized)[0])]
        local = np.clip(incumbent + self.rng.normal(0, .12, (max(64, candidates//4), len(lo))), 0, 1)
        grid = np.vstack([grid, local])
        mean, std = self.model.predict(grid)
        if self.acquisition == "lcb":
            order = np.argsort(lcb(mean, std, self.kappa))
        else:
            best = float(np.min(self.model.predict(normalized)[0]))
            order = np.argsort(-expected_improvement(mean, std, best, self.xi))
        for index in order:
            point = lo + grid[index] * (hi - lo)
            if all(np.linalg.norm((point - np.asarray(o.x)) / (hi - lo)) > 1e-6
                   for o in self.observations):
                return point
        raise RuntimeError("No new point found in candidate grid")
