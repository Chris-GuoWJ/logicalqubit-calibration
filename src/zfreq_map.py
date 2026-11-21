"""Active sampling of Z bias for a drive-IF versus Z frequency map."""

from dataclasses import asdict

import numpy as np

from .adaptive import RawSession
from .readout import center_distance
from .runs import data_dir, load_manifest, run_lock, save_csv
from .zfreq import ZSettings, build_plan


def fit_spectral_peak(rows, seed=0):
    """Estimate a single interior resonance and its conditional frequency noise."""
    frequency = np.asarray([row["drive_if_ghz"] for row in rows], dtype=float)
    signal = np.asarray([row["S"] for row in rows], dtype=float)
    sigma = np.asarray([row["mean_displacement_noise"] for row in rows], dtype=float)
    if (len(rows) < 5 or not np.isfinite([*frequency, *signal, *sigma]).all()
            or np.any(np.diff(frequency) <= 0) or np.any(sigma < 0)):
        raise ValueError("Need at least five ordered, finite IF observations")
    index = int(np.argmax(signal))
    if index in (0, len(rows) - 1):
        raise ValueError("Spectroscopy peak is at an IF boundary; widen frequency_bounds")
    neighbors = slice(index - 1, index + 2)
    x = frequency[neighbors]
    y = signal[neighbors]
    coefficients = np.polyfit(x, y, 2)
    if coefficients[0] >= 0:
        raise ValueError("No concave local spectroscopy peak; increase shots or IF resolution")
    peak = float(-coefficients[1] / (2 * coefficients[0]))
    if not x[0] < peak < x[-1]:
        raise ValueError("Spectroscopy peak interpolation left its IF bracket")
    # Shot-noise bootstrap for the three-point vertex. Baseline drift and
    # branch ambiguity are not represented by this conditional variance.
    rng = np.random.default_rng(seed)
    draws = rng.normal(y, sigma[neighbors], size=(512, 3))
    coeffs = np.linalg.solve(np.vander(x, 3), draws.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        vertices = -coeffs[:, 1] / (2 * coeffs[:, 0])
    valid = (coeffs[:, 0] < 0) & (vertices > x[0]) & (vertices < x[-1])
    variance = (float(np.var(vertices[valid], ddof=1))
                if valid.sum() >= 32 else float(((x[-1] - x[0]) / 4) ** 2))
    return {"peak_drive_if_ghz": peak,
            "peak_noise_variance": max(variance, 1e-12),
            "peak_iq_center_distance": float(signal[index]),
            "peak_if_grid_ghz": float(frequency[index]),
            "peak_method": "local_quadratic",
            "valid_bootstrap_fraction": float(valid.mean()),
            "at_search_boundary": False}


class ZMapSession:
    """One complete frequency sweep per adaptively proposed Z bias."""

    def __init__(self, backend, qid, qconf, pi_cal, z_bounds,
                 frequency_bounds, frequency_points, output_root, plan,
                 settings=None, shots=500, run_mode="new", resume_run_tag="",
                 job_loader=None, timeout=900):
        self.qid, self.qconf, self.pi_cal = int(qid), dict(qconf), dict(pi_cal)
        self.settings = settings or ZSettings()
        self.z_bounds = np.asarray(z_bounds, dtype=float)
        self.frequency_bounds = np.asarray(frequency_bounds, dtype=float)
        if (self.z_bounds.shape != (2,) or self.frequency_bounds.shape != (2,)
                or not np.isfinite([*self.z_bounds, *self.frequency_bounds]).all()
                or self.z_bounds[0] >= self.z_bounds[1]
                or self.frequency_bounds[0] >= self.frequency_bounds[1]
                or isinstance(frequency_points, bool)
                or not isinstance(frequency_points, int) or frequency_points < 5):
            raise ValueError("Use increasing Z/IF bounds and at least five IF points")
        self.frequencies = np.linspace(*self.frequency_bounds, frequency_points)
        for z in self.z_bounds:
            for frequency in self.frequency_bounds:
                build_plan(self.qid, self.qconf, self.pi_cal, float(z),
                           float(frequency), self.settings)
        expected = {"schema_version": 1, "kind": "zfreq", "strategy": "bo_map",
                    "backend": str(backend.name), "qubit": self.qid,
                    "z_bounds": self.z_bounds.tolist(),
                    "frequency_bounds_ghz": self.frequency_bounds.tolist(),
                    "frequency_points": frequency_points, "shots": shots,
                    "settings": asdict(self.settings),
                    "pi_cal": self.pi_cal,
                    "qconf": {key: float(qconf[key]) for key in
                              ("read_amp", "df_read", "read_length")},
                    "plan": plan, "readout_correction": False}
        self.raw = RawSession(backend, output_root, "zfreq", expected, shots,
                              timeout, run_mode, resume_run_tag, job_loader)
        self.run = self.raw.run

    @property
    def manifest(self):
        return self.raw.manifest

    def _saved_sweep(self, index):
        prefix = f"map_{index:04d}"
        base = prefix + "_base"
        measurement_labels = [f"{prefix}_f{fi:04d}"
                              for fi in range(len(self.frequencies))]
        if not self.raw.path(base).is_file():
            if any(self.raw.path(label).is_file() for label in measurement_labels):
                raise ValueError("Saved spectroscopy measurements have no original baseline")
            return None
        entry = next((job for job in reversed(self.manifest["jobs"])
                      if job["label"] == base), None)
        if entry is None:
            raise ValueError("Missing Z map baseline job metadata")
        z = float(entry["parameters"]["z_amplitude"])
        if not self.z_bounds[0] <= z <= self.z_bounds[1]:
            raise ValueError("Saved Z proposal is outside its bounds")
        reference, base_job_id = self.raw.saved(base, [self.qid])
        spectrum = []
        missing = False
        for fi, (label, frequency) in enumerate(zip(measurement_labels, self.frequencies)):
            if not self.raw.path(label).is_file():
                missing = True
                continue
            if missing:
                raise ValueError("Nonconsecutive frequency checkpoints in Z sweep")
            meta = next((job for job in reversed(self.manifest["jobs"])
                         if job["label"] == label), None)
            if (meta is None or meta["parameters"]["z_amplitude"] != z
                    or meta["parameters"]["drive_if_ghz"] != float(frequency)
                    or meta["parameters"]["baseline_job_id"] != base_job_id):
                raise ValueError("Frequency checkpoint differs from Z or baseline")
            iq, job_id = self.raw.saved(label, [self.qid])
            stat = center_distance(iq[self.qid], reference[self.qid])
            spectrum.append({"qubit": self.qid, "z_index": index,
                             "z_amplitude": z, "drive_if_ghz": float(frequency),
                             "shots": self.raw.shots, "job_id": job_id,
                             "baseline_job_id": base_job_id, **stat})
        if len(spectrum) != len(self.frequencies):
            return None
        peak = fit_spectral_peak(spectrum, seed=1777 + index)
        candidate = {"qubit": self.qid, "z_index": index, "z_amplitude": z,
                     **peak, "baseline_job_id": base_job_id}
        return spectrum, candidate

    def observations(self):
        result = []
        for index in range(int(self.manifest["plan"]["steps"])):
            sweep = self._saved_sweep(index)
            if sweep is None:
                # Reject a later complete sweep after a gap in the sequence.
                if any(self.raw.path(f"map_{later:04d}_base").is_file()
                       for later in range(index + 1, int(self.manifest["plan"]["steps"]))):
                    raise ValueError("Nonconsecutive Z map sweeps")
                break
            result.append(sweep[1])
        return result

    def measure(self, index, z):
        z = float(z)
        if not np.isfinite(z) or not self.z_bounds[0] <= z <= self.z_bounds[1]:
            raise ValueError("Proposed Z is outside its calibrated bounds")
        with run_lock(self.run):
            self.raw.manifest = load_manifest(self.run)
            if index != len(self.observations()):
                raise ValueError("Z map proposal index differs from completed sweeps")
            prefix = f"map_{index:04d}"
            base = prefix + "_base"
            if not self.raw.path(base).is_file() and any(
                    self.raw.path(f"{prefix}_f{fi:04d}").is_file()
                    for fi in range(len(self.frequencies))):
                raise ValueError("Cannot replace a baseline used by saved IF measurements")
            _, base_id = self.raw.acquire(
                base, [self.qid], {"z_index": index, "z_amplitude": z, "baseline": True},
                lambda: build_plan(self.qid, self.qconf, self.pi_cal, z,
                                   float(self.frequencies[0]), self.settings,
                                   baseline=True))
            for fi, frequency in enumerate(self.frequencies):
                f = float(frequency)
                self.raw.acquire(
                    f"{prefix}_f{fi:04d}", [self.qid],
                    {"z_index": index, "z_amplitude": z, "drive_if_ghz": f,
                     "baseline_job_id": base_id},
                    lambda f=f: build_plan(self.qid, self.qconf, self.pi_cal,
                                            z, f, self.settings))
            sweep = self._saved_sweep(index)
            spectra = []
            for past in range(index):
                spectra.extend(self._saved_sweep(past)[0])
            spectra.extend(sweep[0])
            self.save_observations(spectra, self.observations())
            return sweep[1]

    def save_observations(self, spectra, candidates):
        saved_spectra = save_csv(data_dir(self.run) / "observations.csv",
                                 spectra, allow_pending=True)
        saved_peaks = save_csv(data_dir(self.run) / "frequency_candidates.csv",
                               candidates, allow_pending=True)
        self.raw.manifest["observation_csv"] = saved_spectra.name
        self.raw.manifest["candidate_csv"] = saved_peaks.name
        self.raw.save()

    def save_search_state(self, suggestion, status="running"):
        self.raw.state(suggestion, None, status)

