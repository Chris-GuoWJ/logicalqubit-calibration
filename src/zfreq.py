"""Z bias versus drive-IF spectroscopy with resumable raw IQ acquisition."""

from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import tempfile
from time import monotonic

import numpy as np

from .readout import extract_iq, center_distance
from .waveforms import square, xy_wave, readout
from .runs import new_run, resume_run, data_dir, fig_dir, save_json, load_manifest, save_csv, run_lock
from .hardware import restore_job


@dataclass(frozen=True)
class ZSettings:
    z_start_ns: int = 20
    z_duration_ns: int = 2200
    xy_start_ns: int = 260
    read_start_ns: int = 2400
    total_ns: int = 6000

    def validate(self, read_length, pi_duration):
        values = [*asdict(self).values(), read_length, pi_duration]
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values):
            raise ValueError("Waveform timings must be nonnegative integer ns")
        if (pi_duration < 4 or self.xy_start_ns < self.z_start_ns
                or self.xy_start_ns + pi_duration > self.z_start_ns + self.z_duration_ns
                or self.read_start_ns < self.z_start_ns + self.z_duration_ns
                or self.read_start_ns + read_length > self.total_ns
                or self.total_ns % 8):
            raise ValueError("Invalid spectroscopy pulse/readout timing")


def build_plan(qid, qconf, pi_cal, z_amplitude, drive_if_ghz, settings=None, baseline=False):
    settings = settings or ZSettings()
    duration = int(pi_cal["duration_ns"])
    read_length = int(qconf["read_length"])
    settings.validate(read_length, duration)
    if pi_cal["envelope"] != "raised_cosine":
        raise ValueError("Z spectroscopy requires the calibrated raised-cosine pi pulse")
    if not np.isfinite([z_amplitude, drive_if_ghz, pi_cal["pi_amp"]]).all():
        raise ValueError("Nonfinite spectroscopy parameter")
    if abs(drive_if_ghz) >= .5 or abs(float(qconf["df_read"])) >= .5:
        raise ValueError("At 1 ns sampling the digital IF must be within +/-0.5 GHz")
    wave = (np.zeros(settings.total_ns, dtype=np.complex64) if baseline else
            xy_wave(settings.total_ns, settings.xy_start_ns, duration,
                    float(pi_cal["pi_amp"]), drive_if_ghz))
    return {f"q{qid}.xy": [wave],
            f"q{qid}.z": [square(settings.total_ns, settings.z_start_ns,
                                  settings.z_duration_ns, z_amplitude)],
            **readout(qid, qconf, settings.total_ns, settings.read_start_ns)}


def _raw_path(run, qid, zi, fi=None):
    suffix = "base" if fi is None else f"f{fi:04d}"
    return data_dir(run) / "raw" / f"q{qid}_z{zi:04d}_{suffix}.npz"


def _load_raw(path, shots):
    with np.load(path, allow_pickle=False) as saved:
        iq = np.asarray(saved["iq"], dtype=float)
        job_id = str(saved["job_id"].item())
    if iq.shape != (shots, 2) or not np.isfinite(iq).all() or not job_id:
        raise ValueError(f"Corrupt raw IQ checkpoint: {path}")
    return iq, job_id


def _save_raw(path, iq, job_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as out:
        tmp = Path(out.name)
        np.savez_compressed(out, iq=iq, job_id=str(job_id))
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


def run_calibration(backend, qconf, pi_cal, z_values, frequencies_by_qubit,
                    output_root, settings=None, shots=500, timeout=900,
                    run_mode="new", resume_run_tag="", job_loader=None):
    """Run a fixed spectroscopy grid; each new job ID is checkpointed."""
    settings = settings or ZSettings()
    if isinstance(shots, bool) or not isinstance(shots, int) or not 2 <= shots <= 50000:
        raise ValueError("shots must be in [2, 50000]")
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive")
    z = np.asarray(z_values, dtype=float)
    if z.ndim != 1 or len(z) < 2 or not np.isfinite(z).all() or np.any(np.diff(z) <= 0):
        raise ValueError("Use at least two finite increasing Z values")
    if not frequencies_by_qubit:
        raise ValueError("Select at least one qubit")
    freqs = {}
    for qid, values in frequencies_by_qubit.items():
        if qid not in qconf or qid not in pi_cal:
            raise ValueError(f"Missing hardware or pi calibration for Q{qid}")
        x = np.asarray(values, dtype=float)
        if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
            raise ValueError(f"Q{qid}: use at least three increasing drive IF values")
        for f in x:
            build_plan(qid, qconf[qid], pi_cal[qid], float(z[-1]), float(f), settings)
        freqs[int(qid)] = x
    expected = {"schema_version": 1, "kind": "zfreq", "backend": str(backend.name),
                "settings": asdict(settings), "shots": shots, "z_values": z.tolist(),
                "frequencies": {str(q): f.tolist() for q, f in freqs.items()},
                "pi_cal": {str(q): dict(pi_cal[q]) for q in freqs},
                "readout_correction": False, "jobs": [], "status": "running"}
    root = Path(output_root).resolve()
    if run_mode == "new":
        if resume_run_tag:
            raise ValueError("Clear resume_run_tag when creating a new run")
        run = new_run(root, "zfreq")
    elif run_mode == "resume":
        run = resume_run(root, resume_run_tag, "zfreq")
    else:
        raise ValueError("run_mode must be new or resume")
    manifest_file = data_dir(run) / "manifest.json"
    with run_lock(run):
        if run_mode == "resume":
            manifest = load_manifest(run)
            for key in expected:
                if key not in ("jobs", "status") and manifest.get(key) != expected[key]:
                    raise ValueError(f"Resume refused: {key} differs")
            manifest["status"] = "running"
        else:
            manifest = expected
        save_json(manifest_file, manifest)

        def acquire(qid, zi, fi, z_amp, frequency):
            started = monotonic()
            path = _raw_path(run, qid, zi, fi)
            if path.is_file():
                return _load_raw(path, shots)
            label = path.stem
            previous = next((entry for entry in reversed(manifest["jobs"])
                             if entry["label"] == label), None)
            if previous is not None:
                if not previous.get("job_id"):
                    raise RuntimeError(f"Submission status unknown for {label}; inspect server before resuming")
                loader = job_loader or restore_job
                job = loader(backend, previous["job_id"])
                status_obj = job.status()
                status = str(getattr(status_obj, "name", status_obj)).upper()
                if status in ("FAILED", "CANCELLED"):
                    previous["status"] = status.lower()
                    save_json(manifest_file, manifest)
                    job = None
                elif status not in ("COMPLETED", "DONE", "SUCCESS", "RUNNING", "QUEUED", "PENDING"):
                    raise RuntimeError(f"Unknown server status: {status}")
            else:
                job = None
            if job is None:
                item = {"label": label, "job_id": None,
                        "submitted_utc": datetime.now(timezone.utc).isoformat()}
                manifest["jobs"].append(item)
                save_json(manifest_file, manifest)
                plan = build_plan(qid, qconf[qid], pi_cal[qid], z_amp, frequency,
                                  settings, baseline=fi is None)
                job = backend.run_wave(plan, shots=shots, result_format="raw",
                                       readout_correction=False)
                item["job_id"] = str(job.job_id)
                save_json(manifest_file, manifest)
            else:
                item = previous
            iq = extract_iq(job.result(timeout=timeout, verbose=False), qid, shots)
            _save_raw(path, iq, str(job.job_id))
            item["status"] = "saved"
            item["elapsed_seconds"] = round(monotonic() - started, 3)
            save_json(manifest_file, manifest)
            return iq, str(job.job_id)

        try:
            rows = []
            for qid, grid in freqs.items():
                for zi, z_amp in enumerate(z):
                    baseline, baseline_id = acquire(qid, zi, None, float(z_amp), float(grid[0]))
                    for fi, frequency in enumerate(grid):
                        iq, job_id = acquire(qid, zi, fi, float(z_amp), float(frequency))
                        rows.append({"qubit": qid, "z_amplitude": float(z_amp),
                                     "drive_if_ghz": float(frequency), "shots": shots,
                                     "job_id": job_id, "baseline_job_id": baseline_id,
                                     **center_distance(iq, baseline)})
                        saved = save_csv(data_dir(run) / "observations.csv", rows, allow_pending=True)
                        manifest["observation_csv"] = saved.name
                        save_json(manifest_file, manifest)
            manifest["status"] = "completed"
            manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
            save_json(manifest_file, manifest)
        except BaseException as exc:
            manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            manifest["error_type"] = type(exc).__name__
            save_json(manifest_file, manifest)
            raise
    return run


def analyze(run):
    """Return measured IF peaks and save diagnostics and a plot."""
    import pandas as pd
    import matplotlib.pyplot as plt
    manifest = load_manifest(run)
    if manifest.get("strategy") in ("bo", "bo_map"):
        raise ValueError("BO frequency candidates need independent spectral-peak validation")
    if manifest.get("status") != "completed":
        raise ValueError("Only a complete spectroscopy run can be analyzed")
    shots = manifest["shots"]
    rows = []
    for qkey, grid in manifest["frequencies"].items():
        qid = int(qkey)
        for zi, z_amp in enumerate(manifest["z_values"]):
            baseline, _ = _load_raw(_raw_path(run, qid, zi), shots)
            for fi, frequency in enumerate(grid):
                iq, _ = _load_raw(_raw_path(run, qid, zi, fi), shots)
                rows.append({"qubit": qid, "z_amplitude": float(z_amp),
                             "drive_if_ghz": float(frequency),
                             "center_distance": center_distance(iq, baseline)["S"]})
    table = pd.DataFrame(rows)
    features = []
    qubits = sorted(table.qubit.unique())
    fig, axes = plt.subplots(len(qubits), 1, figsize=(8, 3.5 * len(qubits)), squeeze=False)
    for ax, qid in zip(axes[:, 0], qubits):
        group = table[table.qubit == qid]
        for z_amp, trace in group.groupby("z_amplitude"):
            trace = trace.sort_values("drive_if_ghz")
            x, y = trace.drive_if_ghz.to_numpy(float), trace.center_distance.to_numpy(float)
            baseline = float(np.median(y))
            index = int(np.argmax(y))
            contrast = float(y[index] - baseline)
            noise = 1.4826 * float(np.median(np.abs(y - baseline)))
            score = contrast / max(noise, 1e-12)
            features.append({"qubit": int(qid), "z_amplitude": float(z_amp),
                             "qubit_frequency_ghz": float(x[index]),
                             "feature_kind": "peak", "distance_feature": float(y[index]),
                             "distance_baseline": baseline, "signed_contrast": contrast,
                             "significance": score, "at_scan_boundary": index in (0, len(x)-1),
                             "valid": bool(score >= 3 and index not in (0, len(x)-1))})
            ax.plot(x, y, "o-", label=f"Z={z_amp:.3g}")
        ax.set(title=f"Q{qid}", xlabel="Drive IF / GHz", ylabel="IQ center distance")
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(fig_dir(run) / "spectroscopy.png", dpi=180)
    plt.close(fig)
    for qid in qubits:
        ordered = sorted((r for r in features if r["qubit"] == qid),
                         key=lambda r: r["z_amplitude"])
        for previous, current in zip(ordered, ordered[1:]):
            current["frequency_jump_ghz"] = abs(current["qubit_frequency_ghz"]
                                                - previous["qubit_frequency_ghz"])
        ordered[0]["frequency_jump_ghz"] = None
    save_csv(data_dir(run) / "frequency_map.csv", features)
    return features


class ZSession:
    """Independent BO of drive IF at each specified Z bias."""

    def __init__(self, backend, qid, qconf, pi_cal, z_values, bounds, output_root,
                 plan, settings=None, shots=500, run_mode="new",
                 resume_run_tag="", job_loader=None, timeout=900):
        from .adaptive import RawSession
        self.qid, self.qconf, self.pi_cal = int(qid), dict(qconf), dict(pi_cal)
        self.settings = settings or ZSettings()
        self.z_values = np.asarray(z_values, dtype=float)
        self.bounds = np.asarray(bounds, dtype=float)
        if (self.z_values.ndim != 1 or len(self.z_values) < 2 or
                not np.isfinite(self.z_values).all() or
                np.any(np.diff(self.z_values) <= 0)):
            raise ValueError("At least two increasing Z values are required")
        if (self.bounds.shape != (2,) or not np.isfinite(self.bounds).all() or
                self.bounds[0] >= self.bounds[1]):
            raise ValueError("Drive IF bounds must be two increasing GHz values")
        for z in self.z_values:
            for f in self.bounds:
                build_plan(self.qid, self.qconf, self.pi_cal, float(z), float(f), self.settings)
        expected = {"schema_version": 1, "kind": "zfreq", "strategy": "bo",
                    "backend": str(backend.name), "qubit": self.qid,
                    "z_values": self.z_values.tolist(), "bounds_ghz": self.bounds.tolist(),
                    "shots": shots, "settings": asdict(self.settings),
                    "pi_cal": self.pi_cal,
                    "qconf": {k: float(qconf[k]) for k in
                              ("read_amp", "df_read", "read_length")},
                    "plan": plan, "readout_correction": False}
        self.raw = RawSession(backend, output_root, "zfreq", expected, shots, timeout,
                              run_mode, resume_run_tag, job_loader)
        self.run = self.raw.run

    @property
    def manifest(self):
        return self.raw.manifest

    def _baseline(self, zi):
        label = f"z{zi:04d}_base"
        if (not self.raw.path(label).is_file() and any(
                self.raw.path(e["label"]).is_file() for e in self.manifest["jobs"]
                if e["label"].startswith(f"z{zi:04d}_bo_"))):
            raise RuntimeError("Baseline missing after BO points; cannot replace it")
        return self.raw.acquire(label, [self.qid], {"z_index": zi, "baseline": True},
                lambda: build_plan(self.qid, self.qconf, self.pi_cal,
                                   float(self.z_values[zi]), float(self.bounds[0]),
                                   self.settings, baseline=True))

    def observations(self, zi):
        label = f"z{zi:04d}_base"
        if not self.raw.path(label).is_file():
            if any(e["label"].startswith(f"z{zi:04d}_bo_") for e in self.manifest["jobs"]):
                raise ValueError("BO points exist but their baseline is missing")
            return []
        reference, baseline_id = self.raw.saved(label, [self.qid])
        reference = reference[self.qid]
        rows = []
        labels = sorted({e["label"] for e in self.manifest["jobs"]
                         if e["label"].startswith(f"z{zi:04d}_bo_")})
        for index, point in enumerate(labels):
            if point != f"z{zi:04d}_bo_{index:04d}":
                raise ValueError("Nonconsecutive Z BO checkpoint labels")
            if not self.raw.path(point).is_file():
                break
            entry = next(e for e in reversed(self.manifest["jobs"]) if e["label"] == point)
            if entry["parameters"]["baseline_job_id"] != baseline_id:
                raise ValueError("Baseline changed during a Z BO run")
            iq, job_id = self.raw.saved(point, [self.qid])
            stat = center_distance(iq[self.qid], reference)
            rows.append({"qubit": self.qid, "z_amplitude": float(self.z_values[zi]),
                         "drive_if_ghz": float(entry["parameters"]["frequency_ghz"]),
                         "loss": -stat["S"],
                         "noise_variance": max(stat["mean_displacement_noise"] ** 2, 1e-12),
                         "shots": self.raw.shots, "job_id": job_id,
                         "baseline_job_id": baseline_id, **stat})
        return rows

    def measure(self, zi, frequency):
        frequency = float(frequency)
        build_plan(self.qid, self.qconf, self.pi_cal,
                   float(self.z_values[zi]), frequency, self.settings)
        with run_lock(self.run):
            self.raw.manifest = load_manifest(self.run)
            _, baseline_id = self._baseline(zi)
            rows = self.observations(zi)
            label = f"z{zi:04d}_bo_{len(rows):04d}"
            self.raw.acquire(label, [self.qid],
                {"z_index": zi, "frequency_ghz": frequency,
                 "baseline_job_id": baseline_id},
                lambda: build_plan(self.qid, self.qconf, self.pi_cal,
                    float(self.z_values[zi]), frequency, self.settings))
            rows = self.observations(zi)
            all_rows = [row for i in range(len(self.z_values))
                        for row in self.observations(i)]
            saved = save_csv(data_dir(self.run) / "observations.csv",
                             all_rows, allow_pending=True)
            self.raw.manifest["observation_csv"] = saved.name
            self.raw.save()
            return rows[-1]

    def save_search_state(self, suggestion, rng_state, status="running"):
        self.raw.state(suggestion, rng_state, status)
