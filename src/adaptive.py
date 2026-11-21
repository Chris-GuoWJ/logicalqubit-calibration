"""Shared durable acquisition for adaptive calibration sessions."""

from datetime import datetime, timezone
from pathlib import Path
import os
import tempfile
from time import monotonic

import numpy as np

from .hardware import restore_job
from .runs import data_dir, load_manifest, new_run, resume_run, run_lock, save_json


class RawSession:
    """A single BO run with atomic raw IQ and recoverable server jobs."""

    def __init__(self, backend, output_root, kind, expected, shots, timeout=900,
                 run_mode="new", resume_run_tag="", job_loader=None):
        if isinstance(shots, bool) or not isinstance(shots, int) or not 2 <= shots <= 50000:
            raise ValueError("shots must be an integer in [2, 50000]")
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self.backend, self.shots, self.timeout = backend, shots, timeout
        self.job_loader = job_loader
        if run_mode == "new":
            if resume_run_tag:
                raise ValueError("Clear resume_run_tag for a new run")
            self.run = new_run(output_root, kind)
            self.manifest = dict(expected, jobs=[], status="running")
            self.save()
        elif run_mode == "resume":
            self.run = resume_run(output_root, resume_run_tag, kind)
            self.manifest = load_manifest(self.run)
            for key, value in expected.items():
                if self.manifest.get(key) != value:
                    raise ValueError(f"Resume refused because {key} differs")
        else:
            raise ValueError("run_mode must be new or resume")
        (data_dir(self.run) / "raw").mkdir(exist_ok=True)

    def save(self):
        save_json(data_dir(self.run) / "manifest.json", self.manifest)

    def path(self, label):
        return data_dir(self.run) / "raw" / f"{label}.npz"

    def saved(self, label, qids):
        path = self.path(label)
        with np.load(path, allow_pickle=False) as raw:
            iq = {q: np.asarray(raw[f"q{q}"], dtype=float) for q in qids}
            job_id = str(raw["job_id"].item())
        if (not job_id or any(a.shape != (self.shots, 2) or not np.isfinite(a).all()
                              for a in iq.values())):
            raise ValueError(f"Invalid raw IQ checkpoint: {path}")
        entry = next((e for e in reversed(self.manifest["jobs"]) if e["label"] == label), None)
        if entry is None or entry.get("job_id") != job_id:
            raise ValueError(f"Raw IQ and job manifest disagree: {path}")
        return iq, job_id

    def acquire(self, label, qids, parameters, plan):
        """Call under the run lock, with a deterministic label and parameters."""
        entry = next((e for e in reversed(self.manifest["jobs"]) if e["label"] == label), None)
        if entry is not None and entry["parameters"] != parameters:
            raise ValueError(f"Pending parameters changed for {label}")
        if self.path(label).is_file():
            return self.saved(label, qids)
        job = None
        if entry is not None:
            if not entry.get("job_id"):
                raise RuntimeError(f"Submission unconfirmed for {label}; reconcile with server")
            job = (self.job_loader(self.backend, entry["job_id"]) if self.job_loader
                   else restore_job(self.backend, entry["job_id"]))
            server_status = job.status()
            status = str(getattr(server_status, "name", server_status)).upper()
            if status in ("FAILED", "CANCELLED"):
                entry["status"] = status.lower()
                self.save()
                job = None
            elif status not in ("COMPLETED", "DONE", "SUCCESS", "RUNNING",
                                "QUEUED", "PENDING", "INITIALIZING"):
                raise RuntimeError(f"Unknown server status for {label}: {status}")
        if job is None:
            entry = {"label": label, "parameters": parameters, "job_id": None,
                     "submitted_utc": datetime.now(timezone.utc).isoformat()}
            self.manifest["jobs"].append(entry)
            self.save()
            try:
                job = self.backend.run_wave(plan(), shots=self.shots, result_format="raw",
                                            readout_correction=False)
                entry["job_id"] = str(job.job_id)
                self.save()
            except BaseException as exc:
                self.manifest["status"] = ("interrupted" if isinstance(exc, KeyboardInterrupt)
                                           else "failed")
                self.manifest["error_type"] = type(exc).__name__
                self.save()
                raise
        started = monotonic()
        try:
            result = job.result(timeout=self.timeout, verbose=False)
            from .readout import extract_iq
            iq = {q: extract_iq(result, q, self.shots) for q in qids}
        except BaseException as exc:
            self.manifest["status"] = ("interrupted" if isinstance(exc, KeyboardInterrupt)
                                       else "failed")
            self.manifest["error_type"] = type(exc).__name__
            self.save()
            raise
        path = self.path(label)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as out:
            tmp = Path(out.name)
            np.savez_compressed(out, **{f"q{q}": a for q, a in iq.items()},
                                job_id=str(job.job_id))
            out.flush()
            os.fsync(out.fileno())
        tmp.replace(path)
        entry["status"] = "saved"
        entry["elapsed_seconds"] = round(monotonic() - started, 3)
        self.save()
        return iq, str(job.job_id)

    def state(self, suggestion, rng_state, status="running"):
        with run_lock(self.run):
            self.manifest = load_manifest(self.run)
            self.manifest.update(suggestion=suggestion, rng_state=rng_state, status=status)
            self.save()
