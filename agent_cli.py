"""Deterministic command entry point for an agent working with Platform_LQ.

Examples: python agent_cli.py test
          python agent_cli.py plan config/plans/pi_bo_example.json
          python agent_cli.py run config/plans/pi_bo_example.json --execute
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.runs import load_manifest

KINDS = {"pi_grid", "pi_bo", "zfreq", "zfreq_bo", "coupler", "coupler_bo"}


def _ids(values):
    if not isinstance(values, list) or not values or any(
            isinstance(q, bool) or not isinstance(q, int) or q < 0 for q in values):
        raise ValueError("qubits must be a nonempty list of nonnegative IDs")
    if len(values) != len(set(values)):
        raise ValueError("qubits must be distinct")
    return values


def _axis(values, name, count=2):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < count or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError(f"{name} must contain at least {count} finite increasing values")
    return x


def read_spec(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("Experiment specification must be a JSON object")
    return spec


def preflight(spec):
    """Count the exact planned jobs/shots without importing LQCloud."""
    kind = spec.get("kind")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    backend = spec.get("backend")
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend must be a nonempty name")
    shots = spec.get("shots")
    if isinstance(shots, bool) or not isinstance(shots, int) or not 2 <= shots <= 50000:
        raise ValueError("shots must be an integer in [2, 50000]")
    if kind in ("zfreq_bo", "coupler_bo"):
        default_initial = 3 if kind == "zfreq_bo" and "z_bounds" in spec else 5
        steps, initial = spec.get("steps"), spec.get("initial_points", default_initial)
        if (isinstance(steps, bool) or not isinstance(steps, int) or
                isinstance(initial, bool) or not isinstance(initial, int) or
                initial < 2 or steps <= initial):
            raise ValueError("BO needs >=2 initial points and >=1 GP-chosen point")
        if spec.get("kernel", "matern52") not in ("rbf", "matern32", "matern52", "rq"):
            raise ValueError("Unsupported kernel")
        if spec.get("acquisition", "ei") not in ("ei", "lcb"):
            raise ValueError("Unsupported acquisition")
        if not np.isfinite(float(spec.get("kappa", 2.0))) or float(spec.get("kappa", 2.0)) < 0:
            raise ValueError("kappa must be nonnegative")
        if isinstance(spec.get("seed", 0), bool) or not isinstance(spec.get("seed", 0), int):
            raise ValueError("seed must be an integer")
    if kind == "pi_bo":
        _ids([spec.get("qubit")])
        bounds = _axis(spec.get("bounds"), "bounds", 2)
        if len(bounds) != 2 or bounds[0] < 0:
            raise ValueError("pi amplitude bounds must be two nonnegative numbers")
        steps = spec.get("steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be positive")
        if spec.get("kernel", "matern52") not in ("rbf", "matern32", "matern52", "rq"):
            raise ValueError("Unsupported kernel")
        if spec.get("acquisition", "ei") not in ("ei", "lcb"):
            raise ValueError("Unsupported acquisition")
        jobs = steps + 1
    elif kind == "pi_grid":
        qubits = _ids(spec.get("qubits"))
        axes = spec.get("amplitudes")
        if not isinstance(axes, dict) or set(axes) != {str(q) for q in qubits}:
            raise ValueError("Specify one amplitude grid per qubit")
        for q in qubits:
            axis = _axis(axes[str(q)], f"Q{q} amplitudes", 3)
            if axis[0] < 0:
                raise ValueError("Negative drive amplitude")
        jobs = sum(1 + len(axes[str(q)]) for q in qubits)
    elif kind == "zfreq_bo":
        _ids([spec.get("qubit")])
        if "z_bounds" in spec:
            if any(field in spec for field in ("z_values", "bounds", "acquisition", "kappa")):
                raise ValueError("Adaptive Z-map plans use z_bounds and frequency_bounds; acquisition is GP uncertainty")
            z_bounds = _axis(spec["z_bounds"], "Z sampling bounds")
            bounds = _axis(spec.get("frequency_bounds"), "drive IF bounds")
            points = spec.get("frequency_points")
            if (len(z_bounds) != 2 or len(bounds) != 2
                    or np.any(np.abs(bounds) >= .5)
                    or isinstance(points, bool) or not isinstance(points, int)
                    or points < 5):
                raise ValueError("Z/IF bounds need two values, IF within +/-0.5 GHz, and >=5 IF points")
            jobs = spec["steps"] * (points + 1)
        else:
            # An immutable historical plan may still resume its fixed-Z BO run.
            z_values = _axis(spec.get("z_values"), "Z amplitudes")
            bounds = _axis(spec.get("bounds"), "drive IF bounds")
            if len(bounds) != 2 or np.any(np.abs(bounds) >= .5):
                raise ValueError("Drive IF bounds must be within +/-0.5 GHz")
            jobs = len(z_values) * (spec["steps"] + 1)
    elif kind == "coupler_bo":
        qubits = _ids(spec.get("qubits"))
        if len(qubits) != 2:
            raise ValueError("Coupler experiment requires two qubits")
        bounds = _axis(spec.get("bounds"), "Coupler Z bounds")
        if len(bounds) != 2:
            raise ValueError("Coupler Z bounds must contain two numbers")
        times = _axis(spec.get("times_ns"), "Interaction times", 8)
        if not np.all(times == times.astype(int)):
            raise ValueError("Interaction times must be integer ns")
        jobs = 4 + spec["steps"] * len(times)
    elif kind == "zfreq":
        qubits = _ids(spec.get("qubits"))
        z_values = _axis(spec.get("z_values"), "Z amplitudes")
        grids = spec.get("frequencies")
        if not isinstance(grids, dict) or set(grids) != {str(q) for q in qubits}:
            raise ValueError("Specify drive IF grid in GHz for every qubit")
        for q in qubits:
            _axis(grids[str(q)], f"Q{q} drive IF", 3)
        jobs = len(z_values) * sum(len(grids[str(q)]) + 1 for q in qubits)
    else:
        qubits = _ids(spec.get("qubits"))
        if len(qubits) != 2:
            raise ValueError("Coupler experiment requires two qubits")
        _axis(spec.get("coupler_z_values"), "Coupler Z amplitudes")
        times = _axis(spec.get("times_ns"), "Interaction times", 8)
        if not np.all(times == times.astype(int)):
            raise ValueError("Interaction times must be integer ns")
        jobs = 4 + len(spec["coupler_z_values"]) * len(times)

    max_jobs, max_shots = spec.get("max_jobs"), spec.get("max_shots")
    if (isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs < jobs
            or isinstance(max_shots, bool) or not isinstance(max_shots, int)
            or max_shots < jobs * shots):
        raise ValueError(f"Budget too small or absent: needs {jobs} jobs and {jobs * shots} shots")
    mode = spec.get("run_mode", "new")
    tag = spec.get("resume_run_tag", "")
    if mode not in ("new", "resume") or (mode == "new" and tag) or (mode == "resume" and not tag):
        raise ValueError("Use run_mode=new with no tag, or resume with a run directory tag")
    preview = {"kind": kind, "backend": backend, "jobs": jobs, "shots_per_job": shots,
               "total_shots": jobs * shots, "run_mode": mode, "max_jobs": max_jobs,
               "max_shots": max_shots, "publishes_config": False}
    if kind == "zfreq_bo":
        preview["z_sampling"] = ("adaptive_gp_uncertainty" if "z_bounds" in spec
                                 else "legacy_preset_biases")
    return preview



class BudgetBackend:
    """Enforce actual submissions, including retries, against the reviewed budget."""

    def __init__(self, backend, max_jobs, max_shots, prior_jobs=0, prior_shots=0):
        self.backend = backend
        self.max_jobs = max_jobs
        self.max_shots = max_shots
        self.jobs = prior_jobs
        self.shots = prior_shots
        if self.jobs > max_jobs or self.shots > max_shots:
            raise ValueError("Previous attempts already exceed this run's budget")

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def run_wave(self, *args, **kwargs):
        shots = kwargs.get("shots")
        if not isinstance(shots, int) or self.jobs + 1 > self.max_jobs or self.shots + shots > self.max_shots:
            raise RuntimeError("QPU job/shot budget exceeded; review a new experiment plan")
        # Reserve budget before calling the SDK; ambiguous submissions also count.
        self.jobs += 1
        self.shots += shots
        return self.backend.run_wave(*args, **kwargs)

def run(spec):
    """Submit the reviewed finite plan; all calibration files remain unchanged."""
    preview = preflight(spec)
    from lqcloud import LQCloudProvider
    from src.config import save_hardware, load_pi
    from src.pi import PulseSettings
    from src.zfreq import ZSettings
    from src.coupler import (CouplerPulseSettings, probe_coupled_pair,
                              load_pi_calibration, load_z_frequency_map,
                              frequency_at_z, z_at_frequency)
    from optimize.runner import run_pi, run_zfreq, run_zfreq_map, run_coupler

    physical_backend = LQCloudProvider().get_backend(spec["backend"])
    kind, shots = spec["kind"], spec["shots"]
    category = ("zfreq" if kind.startswith("zfreq") else
                "coupler" if kind.startswith("coupler") else "pi")
    prior_jobs = 0
    if spec.get("run_mode", "new") == "resume":
        from src.runs import resume_run
        previous = resume_run(ROOT / "result" / category, spec["resume_run_tag"], category)
        prior_jobs = len(load_manifest(previous)["jobs"])
    backend = BudgetBackend(physical_backend, spec["max_jobs"], spec["max_shots"],
                            prior_jobs=prior_jobs, prior_shots=prior_jobs * shots)
    qconf = {int(item["id"]): item for item in backend.config["properties"]["qubit_metrics"]}
    kwargs = {"run_mode": spec.get("run_mode", "new"),
              "resume_run_tag": spec.get("resume_run_tag", "")}
    if kind == "pi_bo":
        from src.pi import build_plan
        qid = spec["qubit"]
        settings = PulseSettings(**spec.get("settings", {}))
        frequency = float(spec.get("drive_if_ghz", qconf[qid]["df_10"]))
        build_plan(qid, qconf[qid], float(spec["bounds"][1]), frequency, settings)
        save_hardware(ROOT / "config", backend)
        run_dir, rows = run_pi(
            backend, qid, qconf[qid], spec["bounds"], ROOT / "result" / "pi",
            spec["steps"], settings=settings, shots=shots, drive_if_ghz=frequency,
            kernel=spec.get("kernel", "matern52"),
            acquisition=spec.get("acquisition", "ei"),
            kappa=float(spec.get("kappa", 2.0)),
            seed=int(spec.get("seed", 0)), initial_points=int(spec.get("initial_points", 5)),
            **kwargs)
        summary = {"candidate_amplitude": max(rows, key=lambda r: r["S"])["amplitude"]}
    elif kind == "pi_grid":
        from src.pi import run_calibration
        qubits = spec["qubits"]
        scans = {q: _axis(spec["amplitudes"][str(q)], f"Q{q} amplitudes", 3)
                 for q in qubits}
        settings = PulseSettings(**spec.get("settings", {}))
        drives = {int(q): float(f) for q, f in spec.get("drive_ifs", {}).items()}
        for q in qubits:
            from src.pi import build_plan
            build_plan(q, qconf[q], float(scans[q][-1]), drives.get(q, qconf[q]["df_10"]), settings)
        save_hardware(ROOT / "config", backend)
        run_dir, candidates = run_calibration(
            backend, qconf, scans, ROOT / "result" / "pi",
            settings=settings, drive_ifs=drives, shots=shots, config_dir=None, **kwargs)
        summary = {"candidates": candidates}
    elif kind == "zfreq_bo":
        from src.zfreq import build_plan
        qid = spec["qubit"]
        cal = load_pi(ROOT / "config", spec["backend"], [qid])[qid]
        settings = ZSettings(**spec.get("settings", {}))
        if "z_bounds" in spec:
            for z in spec["z_bounds"]:
                for frequency in spec["frequency_bounds"]:
                    build_plan(qid, qconf[qid], cal, z, frequency, settings)
            save_hardware(ROOT / "config", backend)
            run_dir, candidates = run_zfreq_map(
                backend, qid, qconf[qid], cal, spec["z_bounds"],
                spec["frequency_bounds"], spec["frequency_points"],
                ROOT / "result" / "zfreq", spec["steps"], settings=settings, shots=shots,
                kernel=spec.get("kernel", "matern52"), seed=spec.get("seed", 0),
                initial_points=spec.get("initial_points", 3), **kwargs)
            summary = {"frequency_candidates": candidates,
                       "frequency_map_csv": str(run_dir / "data" /
                                                load_manifest(run_dir)["frequency_map_csv"])}
        else:
            for z in spec["z_values"]:
                for frequency in spec["bounds"]:
                    build_plan(qid, qconf[qid], cal, z, frequency, settings)
            save_hardware(ROOT / "config", backend)
            run_dir, candidates = run_zfreq(
                backend, qid, qconf[qid], cal, spec["z_values"], spec["bounds"],
                ROOT / "result" / "zfreq", spec["steps"], settings=settings, shots=shots,
                kernel=spec.get("kernel", "matern52"),
                acquisition=spec.get("acquisition", "ei"),
                kappa=float(spec.get("kappa", 2.0)),
                seed=spec.get("seed", 0), initial_points=spec.get("initial_points", 5),
                **kwargs)
            summary = {"frequency_candidates": candidates}
    elif kind == "zfreq":
        from src.zfreq import run_calibration
        qubits = spec["qubits"]
        cal = load_pi(ROOT / "config", spec["backend"], qubits)
        settings = ZSettings(**spec.get("settings", {}))
        freqs = {q: _axis(spec["frequencies"][str(q)], f"Q{q} drive IF", 3)
                 for q in qubits}
        from src.zfreq import build_plan
        for q in qubits:
            for frequency in freqs[q]:
                build_plan(q, qconf[q], cal[q], spec["z_values"][-1], frequency, settings)
        save_hardware(ROOT / "config", backend)
        run_dir = run_calibration(
            backend, qconf, cal, spec["z_values"], freqs,
            ROOT / "result" / "zfreq", settings=settings, shots=shots, **kwargs)
        summary = {}
    else:
        from src.coupler import run_calibration, build_swap_plan, fixed_read_start
        q_a, q_b = spec["qubits"]
        pair = probe_coupled_pair(backend.config, (q_a, q_b))
        pi_file = ROOT / "config" / "calibrations" / "pi_pulse_amplitudes.json"
        z_file = ROOT / "config" / "calibrations" / "qubit_frequency_vs_z.csv"
        pi_cal = load_pi_calibration(pi_file, spec["backend"], (q_a, q_b))
        zmap = load_z_frequency_map(z_file, (q_a, q_b))
        reference_z = float(spec.get("reference_z", 0.0))
        target = float(frequency_at_z(zmap[q_b], [reference_z])[0])
        mapping = z_at_frequency(zmap[q_a], target,
                                 preferred_z=spec.get("tuning_z_hint"),
                                 z_range=spec.get("tuning_z_range"))
        settings = CouplerPulseSettings(**spec.get("settings", {}))
        read_start = fixed_read_start(pi_cal, max(spec["times_ns"]), settings)
        amplitude = (spec["bounds"][-1] if kind == "coupler_bo"
                     else spec["coupler_z_values"][-1])
        build_swap_plan(q_a, q_b, pair["coupler_channel"], qconf, pi_cal, q_a,
                        mapping["z_amplitude"], amplitude,
                        spec["times_ns"][-1], settings, read_start_ns=read_start)
        save_hardware(ROOT / "config", backend)
        metadata = {**mapping, "reference_qubit": q_b, "reference_z": reference_z}
        if kind == "coupler_bo":
            run_dir, rows = run_coupler(
                backend, qconf, q_a, q_b, q_a, mapping["z_amplitude"],
                spec["times_ns"], spec["bounds"], pi_file,
                ROOT / "result" / "coupler", spec["steps"], settings=settings,
                shots=shots, resonance_metadata=metadata,
                kernel=spec.get("kernel", "matern52"),
                acquisition=spec.get("acquisition", "ei"),
                kappa=float(spec.get("kappa", 2.0)),
                seed=spec.get("seed", 0),
                initial_points=spec.get("initial_points", 5), **kwargs)
            best = min(rows, key=lambda row: row["loss"])
            summary = {"candidate_coupler_z": best["coupler_z"],
                       "candidate_score": best["score"],
                       "resonance_mapping": mapping}
        else:
            run_dir = run_calibration(
                backend, qconf, q_a, q_b, pair["coupler_channel"], q_a,
                spec["coupler_z_values"], mapping["z_amplitude"], spec["times_ns"],
                pi_file, ROOT / "result" / "coupler", settings=settings, shots=shots,
                resonance_metadata=metadata, **kwargs)
            summary = {"resonance_mapping": mapping}
    return {"plan": preview, "run_dir": str(run_dir),
            "run_status": load_manifest(run_dir)["status"], **summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("test")
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("spec", type=Path)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("spec", type=Path)
    run_parser.add_argument("--execute", action="store_true",
                            help="Actually connect to LQCloud and submit the plan")
    args = parser.parse_args(argv)
    if args.command == "test":
        return subprocess.call([sys.executable, "-m", "unittest", "discover",
                                "-s", "tests", "-v"], cwd=ROOT)
    spec = read_spec(args.spec)
    preview = preflight(spec)
    if args.command == "plan":
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return 0
    if not args.execute:
        parser.error("Use plan to review budgets; run needs --execute for QPU submission")
    print(json.dumps(run(spec), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
