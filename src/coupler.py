"""Nearest-neighbour SWAP calibration with LQCloud raw-wave experiments.

The experiment scans coupler Z amplitude, tuning-qubit Z amplitude and
interaction time.  Both qubits are read in the same shot.  Their raw IQ values
are classified with the 0/1 centres exported by pi_amplitude_calibration.py,
then accumulated as P00/P01/P10/P11.  No cloud connection occurs on import.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import os
from time import monotonic, sleep

import numpy as np
from .runs import (new_run, resume_run, data_dir, fig_dir,
                   save_json as _json, save_csv as _csv, run_lock as _run_lock,
                   load_manifest)
from .readout import extract_pair as extract_joint_iq
from .hardware import restore_job as _restore_job
from .waveforms import (raised_cosine as _raised_cosine, align8 as _align8,
                        flat_top as _flat_top, readout as _readout_channels)


STATES = ("00", "01", "10", "11")


def available_coupled_pairs(backend_config, eligible_qubits=None):
    """Return the backend's valid nearest-neighbour edges.

    ``eligible_qubits`` may restrict the result to qubits that already have
    the prerequisite π/readout and Z-frequency calibrations.
    """
    topology = backend_config.get("topology", {}) if isinstance(backend_config, dict) else {}
    raw_edges = topology.get("coupling_map", [])
    allowed = None if eligible_qubits is None else {int(q) for q in eligible_qubits}
    edges = []
    for edge in raw_edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        try:
            pair = tuple(sorted((int(edge[0]), int(edge[1]))))
        except (TypeError, ValueError):
            continue
        if pair[0] == pair[1] or (allowed is not None and not set(pair) <= allowed):
            continue
        if pair not in edges:
            edges.append(pair)
    return sorted(edges)


def probe_coupled_pair(backend_config, scan_qubits, coupler_channel="auto"):
    """Validate a user-selected scan edge before any QPU job is submitted.

    The SDK maps numeric waveform channel names to physical device labels.
    Therefore the canonical coupler key is derived from the same two logical
    indices, for example scan_qubits=(4, 10) -> ``c4_10.z``.
    """
    if not isinstance(scan_qubits, (list, tuple)) or len(scan_qubits) != 2:
        raise ValueError("SCAN_QUBITS must contain exactly two qubit IDs")
    q_a, q_b = map(int, scan_qubits)
    if q_a == q_b:
        raise ValueError("SCAN_QUBITS must contain two distinct qubits")
    edges = available_coupled_pairs(backend_config)
    if not edges:
        raise ValueError("Backend topology.coupling_map is missing or empty; QPU submission refused")
    selected = tuple(sorted((q_a, q_b)))
    if selected not in edges:
        neighbours_a = sorted(b if a == q_a else a for a, b in edges if q_a in (a, b))
        neighbours_b = sorted(b if a == q_b else a for a, b in edges if q_b in (a, b))
        raise ValueError(
            f"Q{q_a}-Q{q_b} is not a nearest-neighbour edge on this backend. "
            f"Q{q_a} neighbours={neighbours_a}; Q{q_b} neighbours={neighbours_b}; "
            f"valid edges={edges}"
        )
    # Keep the user's Q_A/Q_B order in the waveform key.  Adjacency itself is
    # undirected, but this makes SCAN_QUBITS=(9, 4) visibly map to c9_4.z.
    canonical = f"c{q_a}_{q_b}.z"
    requested = str(coupler_channel).strip()
    if requested.lower() in ("", "auto", "none"):
        requested = canonical
    if requested != canonical:
        raise ValueError(
            f"Coupler channel {requested!r} does not match SCAN_QUBITS={scan_qubits}; "
            f"use 'auto' or {canonical!r}"
        )
    coordinates = backend_config.get("topology", {}).get("qubit_coordinates", {})
    return {"q_a": q_a, "q_b": q_b, "edge": selected,
            "coupler_channel": canonical,
            "coordinates": {q: coordinates.get(str(q)) for q in (q_a, q_b)}}


@dataclass(frozen=True)
class CouplerPulseSettings:
    relaxation_ns: int = 80_000
    pi_to_interaction_ns: int = 20
    z_edge_ns: int = 8
    interaction_to_read_ns: int = 80
    tail_ns: int = 1_000
    ram_limit_ns: int = 120_000
    ram_reserve_ns: int = 30_000

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer in ns")
        if self.relaxation_ns == 0 or self.z_edge_ns < 1:
            raise ValueError("relaxation_ns > 0 and z_edge_ns >= 1 required")
        if self.ram_limit_ns <= self.ram_reserve_ns:
            raise ValueError("ram_limit_ns must exceed ram_reserve_ns")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_pi_calibration(path, backend_name, qubits):
    """Load π pulses and single-qubit 0/1 IQ centres exported by this project."""
    data = _read_json(path)
    if data.get("schema_version") != 1 or data.get("backend") != backend_name:
        raise ValueError("π calibration schema/backend does not match")
    result = {}
    for qid in qubits:
        item = data.get("qubits", {}).get(str(qid))
        if not item:
            raise ValueError(f"π calibration missing Q{qid}")
        if item.get("at_scan_boundary") or item.get("low_signal"):
            raise ValueError(f"Q{qid} π calibration has a quality warning")
        c0 = item["iq_centers"]["state_0"]
        c1 = item["iq_centers"]["state_1"]
        result[qid] = {
            "pi_amp": float(item["pi_amp"]),
            "drive_if_ghz": float(item["drive_if_ghz"]),
            "duration_ns": int(item["pulse_settings"]["duration_ns"]),
            "envelope": item["pulse_settings"]["envelope"],
            "center_0": np.array([c0["I"], c0["Q"]], dtype=float),
            "center_1": np.array([c1["I"], c1["Q"]], dtype=float),
            "source_run_tag": item.get("source_run_tag"),
        }
        if not np.isfinite(np.r_[result[qid]["center_0"], result[qid]["center_1"]]).all():
            raise ValueError(f"Q{qid} IQ centres are not finite")
        if np.linalg.norm(result[qid]["center_1"] - result[qid]["center_0"]) == 0:
            raise ValueError(f"Q{qid} IQ centres coincide")
    return result


def load_z_frequency_map(path, qubits):
    """Load valid Z-amplitude/frequency rows without choosing a flux branch."""
    import pandas as pd
    frame = pd.read_csv(path)
    frequency_column = ("drive_if_ghz" if "drive_if_ghz" in frame.columns
                        else "qubit_frequency_ghz")
    required = {"qubit", "z_amplitude", frequency_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"Z-frequency CSV missing columns: {sorted(required - set(frame.columns))}")
    if "valid" in frame.columns:
        frame = frame[frame.valid.astype(str).str.lower().isin(("true", "1"))]
    out = {}
    for qid in qubits:
        rows = frame[frame.qubit.astype(int) == qid].copy().sort_values("z_amplitude")
        if len(rows) < 2:
            raise ValueError(f"Z-frequency map has fewer than two valid rows for Q{qid}")
        rows["frequency_ghz"] = rows[frequency_column].astype(float)
        out[qid] = rows
    return out


def frequency_at_z(rows, z_amplitudes):
    """Interpolate frequency only inside one monotonic, explicitly selected Z branch."""
    z = rows.z_amplitude.to_numpy(float)
    column = "frequency_ghz" if "frequency_ghz" in rows.columns else "qubit_frequency_ghz"
    f = rows[column].to_numpy(float)
    query = np.asarray(z_amplitudes, dtype=float)
    if np.any(np.diff(z) <= 0) or not np.isfinite(np.r_[z, f, query]).all():
        raise ValueError("Selected Z branch must contain finite, strictly increasing Z values")
    if query.min() < z.min() or query.max() > z.max():
        raise ValueError("Tuning-qubit Z scan exceeds the selected frequency-map branch")
    return np.interp(query, z, f)


def z_at_frequency(rows, target_frequency_ghz, preferred_z=None, z_range=None):
    """Invert a measured Z-frequency mapping on an explicitly selected branch.

    Every adjacent pair that brackets the target produces one interpolation
    candidate.  ``preferred_z`` resolves multiple flux branches; ambiguity is
    rejected when no preference is supplied.
    """
    branch = rows.copy()
    if z_range is not None:
        lo, hi = sorted(map(float, z_range))
        branch = branch[(branch.z_amplitude.astype(float) >= lo) &
                        (branch.z_amplitude.astype(float) <= hi)]
    column = "frequency_ghz" if "frequency_ghz" in branch.columns else "qubit_frequency_ghz"
    branch = branch.assign(
        z_amplitude=branch.z_amplitude.astype(float),
        _frequency=branch[column].astype(float)).sort_values("z_amplitude")
    z = branch.z_amplitude.to_numpy(float); f = branch._frequency.to_numpy(float)
    target = float(target_frequency_ghz)
    if len(z) < 2 or not np.isfinite(np.r_[z, f, target]).all():
        raise ValueError("Selected Z-frequency branch has insufficient finite data")
    candidates = []
    exact = np.flatnonzero(np.isclose(f, target, rtol=0, atol=1e-12))
    candidates.extend(float(z[i]) for i in exact)
    for i in range(len(z) - 1):
        f0, f1 = f[i], f[i + 1]
        if f0 == f1 or not (min(f0, f1) <= target <= max(f0, f1)):
            continue
        value = float(z[i] + (target - f0) * (z[i + 1] - z[i]) / (f1 - f0))
        if not any(np.isclose(value, old, atol=1e-12) for old in candidates):
            candidates.append(value)
    if not candidates:
        raise ValueError(
            f"Target frequency {target:.9g} GHz is not bracketed by the selected Z branch")
    if len(candidates) > 1 and preferred_z is None:
        raise ValueError(
            f"Target frequency has multiple Z solutions {candidates}; set preferred_z or z_range")
    chosen = (candidates[0] if preferred_z is None else
              min(candidates, key=lambda value: abs(value - float(preferred_z))))
    return {"z_amplitude": float(chosen), "target_frequency_ghz": target,
            "candidate_z_amplitudes": candidates,
            "branch_z_range": [float(z.min()), float(z.max())]}








def fixed_read_start(pi_cal, maximum_duration_ns, settings):
    """One absolute readout time shared by references and every scan point."""
    pi_len = max(pi_cal[q]["duration_ns"] for q in pi_cal)
    interaction_start = settings.relaxation_ns + pi_len + settings.pi_to_interaction_ns
    return _align8(interaction_start + int(maximum_duration_ns) + settings.interaction_to_read_ns)


def waveform_budget(qconf_a, qconf_b, pi_cal, duration_ns, settings, read_start_ns=None):
    settings.validate()
    pi_len = max(pi_cal[q]["duration_ns"] for q in pi_cal)
    read_len = max(int(qconf_a["read_length"]), int(qconf_b["read_length"]))
    interaction_start = settings.relaxation_ns + pi_len + settings.pi_to_interaction_ns
    earliest_read = interaction_start + int(duration_ns) + settings.interaction_to_read_ns
    read_start = _align8(earliest_read) if read_start_ns is None else int(read_start_ns)
    if read_start < earliest_read or read_start % 8:
        raise ValueError("read_start_ns must follow the interaction and lie on the 8 ns grid")
    total = _align8(read_start + read_len + settings.tail_ns)
    if total + settings.ram_reserve_ns > settings.ram_limit_ns:
        raise ValueError("Waveform plus reserve exceeds the configured QPU RAM limit")
    return total




def build_swap_plan(q_a, q_b, coupler_channel, qconf, pi_cal, tuning_qubit,
                    tuning_z, coupler_z, duration_ns, settings,
                    prepared_state="10", enable_interaction=True,
                    read_start_ns=None):
    """Build one simultaneous two-qubit raw-IQ waveform experiment.

    Bit order is always Q_a,Q_b.  π pulses occur at idle; Z pulses are then
    applied for the interaction and return to zero before idle-point readout.
    """
    settings.validate()
    if tuning_qubit not in (q_a, q_b) or prepared_state not in STATES:
        raise ValueError("Invalid tuning qubit or prepared state")
    if not isinstance(coupler_channel, str) or not coupler_channel.endswith(".z"):
        raise ValueError("coupler_channel must be an exact SDK key such as 'c1_2.z'")
    duration_ns = int(duration_ns)
    total = waveform_budget(qconf[q_a], qconf[q_b], pi_cal, duration_ns, settings,
                            read_start_ns=read_start_ns)
    pi_start = settings.relaxation_ns
    pi_max = max(pi_cal[q_a]["duration_ns"], pi_cal[q_b]["duration_ns"])
    interaction_start = pi_start + pi_max + settings.pi_to_interaction_ns
    # ADC trigger delays on this backend must lie on an 8 ns grid.  Interaction
    # duration remains the requested value; only the following idle is padded.
    earliest_read = interaction_start + duration_ns + settings.interaction_to_read_ns
    read_start = _align8(earliest_read) if read_start_ns is None else int(read_start_ns)
    if read_start % 8:
        raise AssertionError("Internal error: read_start is not 8 ns aligned")
    waves = {}
    for bit, qid in zip(prepared_state, (q_a, q_b)):
        xy = np.zeros(total, dtype=np.complex64)
        if bit == "1":
            cal = pi_cal[qid]
            n = cal["duration_ns"]
            env = _raised_cosine(n) if cal["envelope"] == "raised_cosine" else np.ones(n)
            t = pi_start + np.arange(n)
            xy[pi_start:pi_start + n] = cal["pi_amp"] * env * np.exp(
                -2j * np.pi * cal["drive_if_ghz"] * t)
        z_amp = float(tuning_z) if enable_interaction and qid == tuning_qubit else 0.
        waves[f"q{qid}.xy"] = [xy]
        waves[f"q{qid}.z"] = [_flat_top(total, interaction_start, duration_ns,
                                          z_amp, settings.z_edge_ns)]
        waves.update(_readout_channels(qid, qconf[qid], total, read_start))
    waves[coupler_channel] = [_flat_top(total, interaction_start, duration_ns,
                                         float(coupler_z) if enable_interaction else 0.,
                                         settings.z_edge_ns)]
    return waves




def classify_iq(iq, center_0, center_1):
    """Nearest-centre classification for every shot; returns uint8 0/1."""
    iq = np.asarray(iq, dtype=float)
    return (np.sum((iq - center_1) ** 2, axis=1) <
            np.sum((iq - center_0) ** 2, axis=1)).astype(np.uint8)


def joint_probabilities(iq_a, iq_b, cal_a, cal_b):
    a = classify_iq(iq_a, cal_a["center_0"], cal_a["center_1"])
    b = classify_iq(iq_b, cal_b["center_0"], cal_b["center_1"])
    if len(a) != len(b):
        raise ValueError("The two IQ arrays do not contain paired shots")
    code = 2 * a + b
    counts = np.bincount(code, minlength=4)
    return {state: float(counts[i] / len(code)) for i, state in enumerate(STATES)}


def assignment_matrix(reference_probabilities):
    """Columns are prepared 00/01/10/11; rows are measured states."""
    matrix = np.array([[reference_probabilities[p][m] for p in STATES] for m in STATES])
    if not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 1e6:
        raise ValueError("Joint readout assignment matrix is singular or ill-conditioned")
    return matrix


def correct_probabilities(raw, matrix):
    vector = np.array([raw[s] for s in STATES], dtype=float)
    corrected = np.linalg.pinv(matrix) @ vector
    corrected = np.clip(corrected, 0, None)
    corrected /= corrected.sum()
    return {s: float(corrected[i]) for i, s in enumerate(STATES)}












def _point_name(cz_i, t_i):
    return f"cz_{cz_i:04d}_t_{t_i:04d}"


def run_calibration(backend, qconf, q_a, q_b, coupler_channel, tuning_qubit,
                    coupler_z_values, tuning_z_value, interaction_times_ns,
                    pi_calibration_path, output_root, settings=None, shots=1000,
                    timeout=900, run_mode="new", resume_run_tag="",
                    resonance_metadata=None):
    """Acquire four joint references and the Coupler-Z/time scan at mapped resonance."""
    # This is deliberately the first hardware preflight: an invalid pair or a
    # mismatched coupler key is rejected before a run directory or QPU job exists.
    pair_info = probe_coupled_pair(backend.config, (q_a, q_b), coupler_channel)
    coupler_channel = pair_info["coupler_channel"]
    settings = settings or CouplerPulseSettings()
    settings.validate()
    if q_a == q_b or q_a not in qconf or q_b not in qconf:
        raise ValueError("Choose two distinct configured qubits")
    if not 2 <= int(shots) <= 50_000:
        raise ValueError("shots must be between 2 and 50000")
    czs = np.asarray(coupler_z_values, dtype=float)
    times_float = np.asarray(interaction_times_ns, dtype=float)
    tuning_z_value = float(tuning_z_value)
    if (czs.ndim != 1 or len(czs) < 2 or times_float.ndim != 1 or
            len(times_float) < 8 or not np.isfinite(np.r_[czs, times_float, tuning_z_value]).all()):
        raise ValueError("Use >=2 Coupler Z points, >=8 time points, and one finite tuning Z")
    if np.any(np.diff(czs) <= 0) or np.any(np.diff(times_float) <= 0):
        raise ValueError("Scan axes must be strictly increasing")
    if np.any(times_float != times_float.astype(int)):
        raise ValueError("Interaction times must be integer ns")
    times = times_float.astype(int)
    pi_cal = load_pi_calibration(pi_calibration_path, str(backend.name), (q_a, q_b))
    read_start_ns = fixed_read_start(pi_cal, int(times[-1]), settings)
    for t in times:
        build_swap_plan(q_a, q_b, coupler_channel, qconf, pi_cal, tuning_qubit,
                        tuning_z_value, czs[-1], int(t), settings,
                        read_start_ns=read_start_ns)
    root = Path(output_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    mode = str(run_mode).lower().strip()
    if mode == "new":
        if resume_run_tag:
            raise ValueError("Clear resume_run_tag in new mode")
        run_dir = new_run(root, 'coupler')
    elif mode == "resume":
        if not resume_run_tag or Path(resume_run_tag).name != resume_run_tag:
            raise ValueError("resume_run_tag must be one directory name")
        run_dir = resume_run(root, resume_run_tag, 'coupler')
    else:
        raise ValueError("run_mode must be new or resume")
    expected = {
        "schema_version": 3, "waveform_version": "mapped_resonance_fixed_readout_v3",
        "backend": str(backend.name), "q_a": q_a, "q_b": q_b,
        "bit_order": [q_a, q_b], "coupler_channel": coupler_channel,
        "tuning_qubit": tuning_qubit, "tuning_z_value": tuning_z_value,
        "resonance_mapping": resonance_metadata or {},
        "coupler_z_values": czs.tolist(), "interaction_times_ns": times.tolist(),
        "settings": asdict(settings), "shots": int(shots),
        "fixed_read_start_ns": read_start_ns,
        "pi_calibration_path": str(Path(pi_calibration_path).resolve()),
        "pi_sources": {str(q): pi_cal[q]["source_run_tag"] for q in (q_a, q_b)},
        "readout_correction": False,
        "probability_definition": "paired-shot nearest-centre classification; bit order q_a,q_b",
        "status": "running", "jobs": []}
    manifest_path = data_dir(run_dir) / "manifest.json"
    with _run_lock(run_dir):
        if mode == "resume":
            manifest = load_manifest(run_dir)
            for key in expected:
                if key not in ("status", "jobs") and manifest.get(key) != expected[key]:
                    raise RuntimeError(f"Resume refused because {key} changed")
            manifest["status"] = "running"
            manifest.pop("error_type", None)
            manifest.pop("error", None)
            _json(manifest_path, manifest)
        else:
            manifest = expected; _json(manifest_path, manifest)
        raw_dir = data_dir(run_dir) / "raw"; raw_dir.mkdir(exist_ok=True)

        def acquire(label, plan):
            path = raw_dir / f"{label}.npz"
            if path.exists():
                saved = np.load(path, allow_pickle=False)
                a, b = saved["iq_a"], saved["iq_b"]
                if a.shape != (shots, 2) or b.shape != (shots, 2):
                    raise ValueError(f"Invalid saved checkpoint: {path}")
                return a, b, str(saved["job_id"].item())
            entry = next((x for x in reversed(manifest["jobs"]) if x["label"] == label), None)
            job = None
            if entry:
                if not entry.get("job_id"):
                    # The client stopped after writing the manifest placeholder
                    # but before a task id was persisted.  With no raw checkpoint
                    # and no id that can be restored, this point is safe to retry.
                    entry.update(status="submission_unconfirmed",
                                 abandoned_utc=datetime.now(timezone.utc).isoformat())
                    _json(manifest_path, manifest)
                    entry = None
                else:
                    job = _restore_job(backend, entry["job_id"])
                    server_status = job.status()
                    status = str(getattr(server_status, "name", server_status)).upper()
                    if status in ("FAILED", "CANCELLED"):
                        entry.update(status=status.lower(),
                                     abandoned_utc=datetime.now(timezone.utc).isoformat())
                        _json(manifest_path, manifest)
                        job = None
                    elif status not in ("COMPLETED", "DONE", "SUCCESS", "RUNNING", "QUEUED", "PENDING"):
                        raise RuntimeError(f"Unknown server status {status} for {label}")
            if job is None:
                entry = {"label": label, "job_id": None, "submitted_utc": datetime.now(timezone.utc).isoformat()}
                manifest["jobs"].append(entry); _json(manifest_path, manifest)
                job = backend.run_wave(plan, shots=int(shots), result_format="raw", readout_correction=False)
                entry["job_id"] = str(job.job_id); _json(manifest_path, manifest)
            result = job.result(timeout=timeout, verbose=False)
            a, b = extract_joint_iq(result, q_a, q_b, int(shots))
            np.savez_compressed(path, iq_a=a, iq_b=b, job_id=str(job.job_id))
            entry["status"] = "saved"; _json(manifest_path, manifest)
            return a, b, str(job.job_id)

        try:
            reference_iq = {}
            reference_time = 2 * settings.z_edge_ns
            for state in STATES:
                plan = build_swap_plan(q_a, q_b, coupler_channel, qconf, pi_cal,
                                       tuning_qubit, 0., 0., reference_time, settings,
                                       prepared_state=state, enable_interaction=False,
                                       read_start_ns=read_start_ns)
                a, b, _ = acquire(f"reference_{state}", plan)
                reference_iq[state] = (a, b)
            # Re-centre the classifiers with references acquired at this run's
            # exact simultaneous-readout time and hardware state.
            run_cal = {q: dict(pi_cal[q]) for q in (q_a, q_b)}
            run_cal[q_a]["center_0"] = np.vstack([reference_iq["00"][0], reference_iq["01"][0]]).mean(axis=0)
            run_cal[q_a]["center_1"] = np.vstack([reference_iq["10"][0], reference_iq["11"][0]]).mean(axis=0)
            run_cal[q_b]["center_0"] = np.vstack([reference_iq["00"][1], reference_iq["10"][1]]).mean(axis=0)
            run_cal[q_b]["center_1"] = np.vstack([reference_iq["01"][1], reference_iq["11"][1]]).mean(axis=0)
            reference_probs = {
                state: joint_probabilities(*reference_iq[state], run_cal[q_a], run_cal[q_b])
                for state in STATES
            }
            matrix = assignment_matrix(reference_probs)
            _json(data_dir(run_dir) / "readout_calibration.json", {
                "states": list(STATES), "columns_prepared_rows_measured": matrix.tolist(),
                "reference_probabilities": reference_probs,
                "fixed_read_start_ns": read_start_ns,
                "run_iq_centers": {str(q): {
                    "state_0": run_cal[q]["center_0"].tolist(),
                    "state_1": run_cal[q]["center_1"].tolist()} for q in (q_a, q_b)}})
            rows = []
            csv_path = data_dir(run_dir) / "joint_probabilities.csv"
            pending_path = data_dir(run_dir) / "joint_probabilities.pending.csv"
            candidates = [p for p in (csv_path, pending_path) if p.exists()]
            if candidates:
                import pandas as pd
                tables = [pd.read_csv(p) for p in candidates]
                rows = max(tables, key=len).to_dict("records")
            done = {(float(r["coupler_z"]), int(r["duration_ns"])) for r in rows}
            started = monotonic(); total_points = len(czs) * len(times)
            for ci, cz in enumerate(czs):
                for ti, duration in enumerate(times):
                    key = (float(cz), int(duration))
                    if key in done:
                        continue
                    plan = build_swap_plan(q_a, q_b, coupler_channel, qconf, pi_cal,
                                           tuning_qubit, tuning_z_value, cz, duration, settings,
                                           read_start_ns=read_start_ns)
                    a, b, job_id = acquire(_point_name(ci, ti), plan)
                    raw = joint_probabilities(a, b, run_cal[q_a], run_cal[q_b])
                    corrected = correct_probabilities(raw, matrix)
                    rows.append({"coupler_z": float(cz), "tuning_z": tuning_z_value,
                        "duration_ns": int(duration), "job_id": job_id,
                        **{f"p{s}_raw": raw[s] for s in STATES},
                        **{f"p{s}": corrected[s] for s in STATES}})
                    saved_csv = _csv(csv_path, rows, allow_pending=True)
                    manifest["probability_csv"] = saved_csv.name
                    _json(manifest_path, manifest)
                    print(f"[{len(rows)}/{total_points}] CZ={cz:.6g}, fixed QZ={tuning_z_value:.6g}, t={duration} ns", flush=True)
            manifest.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                            elapsed_seconds=monotonic() - started)
            _json(manifest_path, manifest)
            return run_dir
        except BaseException as exc:
            manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                            error_type=type(exc).__name__, error=str(exc))
            _json(manifest_path, manifest)
            raise


def _safe_corr(x, y):
    return np.nan if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0 else float(np.corrcoef(x, y)[0, 1])


def _joint_model(x, oa, aa, ob, ab, g_mhz, decay_ns, phase):
    channel, t = x
    osc = np.exp(-t / decay_ns) * np.cos(4 * np.pi * g_mhz * 1e-3 * t + phase)
    return np.where(channel < .5, oa + aa * osc, ob - ab * osc)


def fit_swap_trace(trace, p10_col="p10", p01_col="p01"):
    from scipy.optimize import curve_fit
    trace = trace.sort_values("duration_ns")
    t = trace.duration_ns.to_numpy(float)
    p10 = trace[p10_col].to_numpy(float); p01 = trace[p01_col].to_numpy(float)
    if len(t) < 8 or np.any(np.diff(t) <= 0):
        raise ValueError("At least eight increasing time points are required")
    dt = np.median(np.diff(t)); signal = p10 - p01 - np.mean(p10 - p01)
    freq = np.fft.rfftfreq(len(signal), d=dt)
    population_frequency = freq[1 + np.argmax(np.abs(np.fft.rfft(signal))[1:])]
    g0 = max(float(population_frequency * 1e3 / 2), .01)
    channel = np.r_[np.zeros_like(t), np.ones_like(t)]; tt = np.r_[t, t]
    measured = np.r_[p10, p01]
    p0 = [p10.mean(), max(np.ptp(p10) / 2, .01), p01.mean(), max(np.ptp(p01) / 2, .01),
          g0, max(t.max(), 100.), 0.]
    popt, pcov = curve_fit(_joint_model, (channel, tt), measured, p0=p0,
        bounds=([-0.2, 0, -0.2, 0, 0, 10, -2*np.pi],
                [1.2, 1.2, 1.2, 1.2, 125, 1e6, 2*np.pi]), maxfev=100000)
    fitted = _joint_model((channel, tt), *popt)
    ss_res = np.sum((measured - fitted) ** 2); ss_tot = np.sum((measured - measured.mean()) ** 2)
    return {"g_mhz": abs(float(popt[4])), "g_std_mhz": float(np.sqrt(np.diag(pcov))[4]),
            "decay_ns": float(popt[5]), "r_squared": float(1 - ss_res / ss_tot) if ss_tot else np.nan,
            "fit_p10": _joint_model((np.zeros_like(t), t), *popt),
            "fit_p01": _joint_model((np.ones_like(t), t), *popt)}


def analyze_run(run_dir, use_corrected=True):
    """Fit |g|/2π at the one resonance Z supplied by the frequency mapping."""
    import pandas as pd
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir)
    csv_name = manifest.get("probability_csv", "joint_probabilities.csv")
    data = pd.read_csv(data_dir(run_dir) / csv_name)
    suffix = "" if use_corrected else "_raw"
    p10_col, p01_col = "p10" + suffix, "p01" + suffix
    diagnostics = []
    for cz, group in data.groupby("coupler_z", sort=True):
        diagnostics.append({"coupler_z": cz, "tuning_z": float(group.tuning_z.iloc[0]),
            "p01_contrast": float(np.ptp(group[p01_col])),
            "p10_contrast": float(np.ptp(group[p10_col])),
            "anti_corr": -_safe_corr(group[p10_col], group[p01_col]),
            "single_excitation_range": float(np.ptp(group[p10_col] + group[p01_col]))})
    diagnostic = pd.DataFrame(diagnostics)
    rows = []; fits = {}
    for cz in sorted(diagnostic.coupler_z.unique()):
        diag = diagnostic[np.isclose(diagnostic.coupler_z, cz)].iloc[0]
        trace = data[np.isclose(data.coupler_z, cz)]
        error = ""
        try:
            fit = fit_swap_trace(trace, p10_col, p01_col); fits[float(cz)] = fit
            relative_std = fit["g_std_mhz"] / fit["g_mhz"] if fit["g_mhz"] else np.inf
            reliable = bool(diag.p01_contrast >= .10 and diag.anti_corr >= .30 and
                            fit["r_squared"] >= .50 and relative_std <= .50)
        except Exception as exc:
            fit = {"g_mhz": np.nan, "g_std_mhz": np.nan, "decay_ns": np.nan, "r_squared": np.nan}
            reliable = False; error = repr(exc)
        rows.append({"coupler_z": float(cz), "mapped_tuning_z": float(trace.tuning_z.iloc[0]),
            "p01_contrast": float(diag.p01_contrast), "anti_corr": float(diag.anti_corr),
            "g_mhz": fit["g_mhz"], "g_std_mhz": fit["g_std_mhz"], "decay_ns": fit["decay_ns"],
            "fit_r_squared": fit["r_squared"], "fit_reliable": reliable, "fit_error": error})
    comparison = pd.DataFrame(rows)
    diagnostic.to_csv(data_dir(run_dir) / "resonance_diagnostics.csv", index=False)
    comparison.to_csv(data_dir(run_dir) / "coupling_strength_vs_coupler_z.csv", index=False)
    plot_analysis(run_dir, data, diagnostic, comparison, fits, p10_col, p01_col)
    return comparison


def plot_analysis(run_dir, data, diagnostic, comparison, fits, p10_col="p10", p01_col="p01"):
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    selectable = comparison[comparison.fit_reliable & np.isfinite(comparison.g_mhz)]
    best_is_reliable = not selectable.empty
    if not best_is_reliable:
        selectable = comparison[np.isfinite(comparison.g_mhz)]
    best = None if selectable.empty else selectable.loc[selectable.g_mhz.idxmax()]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    if best is not None:
        trace = data[np.isclose(data.coupler_z, best.coupler_z)].sort_values("duration_ns")
        t = trace.duration_ns.to_numpy(float); fit = fits[float(best.coupler_z)]
        axes[0, 0].plot(t, trace[p10_col], "o", ms=4, label="P10 data")
        axes[0, 0].plot(t, trace[p01_col], "s", ms=4, label="P01 data")
        axes[0, 0].plot(t, fit["fit_p10"], "-", lw=1.8, label="P10 joint fit")
        axes[0, 0].plot(t, fit["fit_p01"], "--", lw=1.8, label="P01 joint fit")
        label = "Maximum reliable coupling" if best_is_reliable else "Largest finite fit (unreliable)"
        axes[0, 0].set_title(f"{label}: |g|/2π={best.g_mhz:.3f} MHz")
    axes[0, 0].set(xlabel="Interaction time (ns)", ylabel="Probability")
    axes[0, 0].grid(alpha=.3); axes[0, 0].legend()

    heat = data.pivot(index="duration_ns", columns="coupler_z", values=p01_col).sort_index().sort_index(axis=1)
    x = heat.columns.to_numpy(float); y = heat.index.to_numpy(float)
    im = axes[0, 1].imshow(heat.values, origin="lower", aspect="auto",
        extent=[x.min(), x.max(), y.min(), y.max()], cmap="turbo")
    fig.colorbar(im, ax=axes[0, 1], label="P01")
    axes[0, 1].set(xlabel="Coupler Z amplitude", ylabel="Interaction time (ns)",
                   title="SWAP probability map at mapped resonance")

    axes[1, 0].plot(diagnostic.coupler_z, diagnostic.p01_contrast, "o-", label="P01 contrast")
    axes[1, 0].plot(diagnostic.coupler_z, diagnostic.p10_contrast, "s--", label="P10 contrast")
    axes[1, 0].plot(diagnostic.coupler_z, diagnostic.anti_corr, "^:", label="anti-correlation")
    axes[1, 0].axhline(.10, color="gray", ls=":", lw=1, label="contrast threshold")
    axes[1, 0].set(xlabel="Coupler Z amplitude", ylabel="Diagnostic value",
                   title="Exchange visibility and anti-correlation")
    axes[1, 0].grid(alpha=.3); axes[1, 0].legend()

    axes[1, 1].errorbar(comparison.coupler_z, comparison.g_mhz,
        yerr=comparison.g_std_mhz, fmt="o-", capsize=4, label="SWAP calibration |g|/2π")
    if best is not None:
        axes[1, 1].axvline(best.coupler_z, color="r", ls=":", lw=1.2,
                           label="maximum reliable g" if best_is_reliable else "largest finite fit")
    axes[1, 1].set(xlabel="Coupler Z amplitude", ylabel="Coupling strength (MHz)",
                   title="Coupling strength versus Coupler Z")
    axes[1, 1].grid(alpha=.3); axes[1, 1].legend()
    fig.suptitle("Coupler Z - exchange coupling calibration")
    fig.tight_layout(); fig.savefig(fig_dir(run_dir) / "coupler_calibration.png", dpi=180, bbox_inches="tight")
    return fig


class CouplerSession:
    """One BO candidate is a full coupler-Z interaction-time trace."""

    def __init__(self, backend, qconf, q_a, q_b, tuning_qubit,
                 tuning_z_value, times_ns, bounds, pi_calibration_path,
                 output_root, plan, settings=None, shots=1000,
                 resonance_metadata=None, run_mode="new",
                 resume_run_tag="", job_loader=None, timeout=900):
        from .adaptive import RawSession
        import hashlib
        self.q_a, self.q_b = int(q_a), int(q_b)
        self.qconf = qconf
        self.tuning_qubit = int(tuning_qubit)
        if self.tuning_qubit not in (self.q_a, self.q_b):
            raise ValueError("Tuning qubit must be in the selected pair")
        self.channel = probe_coupled_pair(backend.config, (self.q_a, self.q_b))["coupler_channel"]
        self.pi_file = Path(pi_calibration_path).resolve()
        self.pi_cal = load_pi_calibration(self.pi_file, str(backend.name),
                                           (self.q_a, self.q_b))
        self.settings = settings or CouplerPulseSettings()
        self.settings.validate()
        self.times = np.asarray(times_ns, dtype=float)
        self.bounds = np.asarray(bounds, dtype=float)
        self.tuning_z = float(tuning_z_value)
        if (self.bounds.shape != (2,) or not np.isfinite(self.bounds).all()
                or self.bounds[0] >= self.bounds[1]
                or not np.isfinite(self.tuning_z)):
            raise ValueError("Coupler-Z bounds must be increasing and finite")
        if (self.times.ndim != 1 or len(self.times) < 8
                or not np.isfinite(self.times).all()
                or np.any(np.diff(self.times) <= 0)
                or np.any(self.times != self.times.astype(int))):
            raise ValueError("Use at least eight increasing integer time points")
        self.times = self.times.astype(int)
        self.read_start = fixed_read_start(self.pi_cal, int(self.times[-1]),
                                           self.settings)
        for amplitude in self.bounds:
            for duration in self.times:
                self._plan(float(amplitude), int(duration))
        expected = {"schema_version": 3, "kind": "coupler", "strategy": "bo",
                    "backend": str(backend.name), "q_a": self.q_a, "q_b": self.q_b,
                    "bit_order": [self.q_a, self.q_b],
                    "coupler_channel": self.channel,
                    "tuning_qubit": self.tuning_qubit,
                    "tuning_z_value": self.tuning_z,
                    "resonance_mapping": resonance_metadata or {},
                    "bounds": self.bounds.tolist(),
                    "interaction_times_ns": self.times.tolist(),
                    "settings": asdict(self.settings), "shots": shots,
                    "fixed_read_start_ns": self.read_start,
                    "pi_calibration_path": str(self.pi_file),
                    "pi_calibration_sha256": hashlib.sha256(
                        self.pi_file.read_bytes()).hexdigest(),
                    "pi_sources": {str(q): self.pi_cal[q]["source_run_tag"]
                                   for q in (self.q_a, self.q_b)},
                    "qconf": {str(q): {key: float(qconf[q][key])
                                      for key in ("read_amp", "df_read", "read_length")}
                              for q in (self.q_a, self.q_b)},
                    "readout_correction": False, "plan": plan}
        self.raw = RawSession(backend, output_root, "coupler", expected, shots,
                              timeout, run_mode, resume_run_tag, job_loader)
        self.run = self.raw.run

    @property
    def manifest(self):
        return self.raw.manifest

    def _plan(self, amplitude, duration, state="10", interaction=True):
        return build_swap_plan(
            self.q_a, self.q_b, self.channel, self.qconf, self.pi_cal,
            self.tuning_qubit, self.tuning_z if interaction else 0.,
            amplitude if interaction else 0., duration, self.settings,
            prepared_state=state, enable_interaction=interaction,
            read_start_ns=self.read_start)

    def _references(self, acquire):
        labels = [f"ref_{state}" for state in STATES]
        if (any(not self.raw.path(label).is_file() for label in labels)
                and any(self.raw.path(e["label"]).is_file()
                        for e in self.manifest["jobs"] if e["label"].startswith("bo_"))):
            raise RuntimeError("Cannot replace references used by saved BO traces")
        pair = (self.q_a, self.q_b)
        data, ids = {}, []
        for state in STATES:
            label = f"ref_{state}"
            if acquire:
                iq, job_id = self.raw.acquire(
                    label, pair, {"prepared_state": state, "interaction": False},
                    lambda state=state: self._plan(
                        0., 2 * self.settings.z_edge_ns, state, False))
            else:
                if not self.raw.path(label).is_file():
                    return None
                iq, job_id = self.raw.saved(label, pair)
            data[state] = iq
            ids.append(job_id)
        run_cal = {q: dict(self.pi_cal[q]) for q in pair}
        run_cal[self.q_a]["center_0"] = np.vstack(
            [data["00"][self.q_a], data["01"][self.q_a]]).mean(axis=0)
        run_cal[self.q_a]["center_1"] = np.vstack(
            [data["10"][self.q_a], data["11"][self.q_a]]).mean(axis=0)
        run_cal[self.q_b]["center_0"] = np.vstack(
            [data["00"][self.q_b], data["10"][self.q_b]]).mean(axis=0)
        run_cal[self.q_b]["center_1"] = np.vstack(
            [data["01"][self.q_b], data["11"][self.q_b]]).mean(axis=0)
        reference_probs = {
            state: joint_probabilities(data[state][self.q_a],
                                       data[state][self.q_b],
                                       run_cal[self.q_a], run_cal[self.q_b])
            for state in STATES}
        matrix = assignment_matrix(reference_probs)
        _json(data_dir(self.run) / "readout_calibration.json", {
            "states": list(STATES), "columns_prepared_rows_measured": matrix.tolist(),
            "reference_job_ids": ids, "reference_probabilities": reference_probs,
            "run_iq_centers": {str(q): {
                "state_0": run_cal[q]["center_0"].tolist(),
                "state_1": run_cal[q]["center_1"].tolist()} for q in pair}})
        return run_cal, matrix, ids

    @staticmethod
    def _score(rows):
        p01 = np.asarray([r["p01"] for r in rows])
        p10 = np.asarray([r["p10"] for r in rows])
        corr = _safe_corr(p01, p10)
        anti_corr = max(0., -float(corr)) if np.isfinite(corr) else 0.
        contrast = float(np.ptp(p01))
        return contrast * anti_corr, contrast, anti_corr

    def _trace(self, index, cal, matrix, ids):
        rows, raw_probs = [], []
        for ti, duration in enumerate(self.times):
            label = f"bo_{index:04d}_t_{ti:04d}"
            if not self.raw.path(label).is_file():
                return None
            entry = next((e for e in reversed(self.manifest["jobs"])
                          if e["label"] == label), None)
            if entry is None or entry["parameters"]["reference_job_ids"] != ids:
                raise ValueError("Reference job IDs changed during coupler BO")
            iq, job_id = self.raw.saved(label, (self.q_a, self.q_b))
            raw = joint_probabilities(iq[self.q_a], iq[self.q_b],
                                      cal[self.q_a], cal[self.q_b])
            corrected = correct_probabilities(raw, matrix)
            raw_probs.append(np.asarray([raw[s] for s in STATES]))
            rows.append({"coupler_z": entry["parameters"]["coupler_z"],
                         "tuning_z": self.tuning_z,
                         "duration_ns": int(duration), "job_id": job_id,
                         **{f"p{s}_raw": raw[s] for s in STATES},
                         **{f"p{s}": corrected[s] for s in STATES}})
        score, contrast, anti_corr = self._score(rows)
        # Shot-noise bootstrap conditional on this run's reference matrix.
        rng = np.random.default_rng(1777 + index)
        samples = []
        for _ in range(48):
            simulated = []
            for row, probs in zip(rows, raw_probs):
                counts = rng.multinomial(self.raw.shots, probs / probs.sum())
                corrected = correct_probabilities(
                    dict(zip(STATES, counts / self.raw.shots)), matrix)
                simulated.append(corrected)
            samples.append(self._score([{"p01": r["01"], "p10": r["10"]}
                                        for r in simulated])[0])
        return rows, {"coupler_z": rows[0]["coupler_z"], "loss": -score,
                       "score": score, "p01_contrast": contrast,
                       "anti_corr": anti_corr,
                       "noise_variance": max(float(np.var(samples, ddof=1)), 1e-8),
                       "shots": self.raw.shots * len(self.times),
                       "job_ids": [r["job_id"] for r in rows],
                       "reference_job_ids": ids}

    def observations(self):
        ref = self._references(False)
        if ref is None:
            if any(e["label"].startswith("bo_") for e in self.manifest["jobs"]):
                raise ValueError("BO traces exist but joint references are missing")
            return []
        cal, matrix, ids = ref
        labels = {e["label"] for e in self.manifest["jobs"] if e["label"].startswith("bo_")}
        indices = sorted({int(label.split("_")[1]) for label in labels})
        records = []
        for ci in indices:
            if ci != len(records):
                raise ValueError("Nonconsecutive coupler BO checkpoint labels")
            trace = self._trace(ci, cal, matrix, ids)
            if trace is None:
                break
            records.append(trace[1])
        return records

    def measure(self, amplitude):
        amplitude = float(amplitude)
        for duration in self.times:
            self._plan(amplitude, int(duration))
        with _run_lock(self.run):
            self.raw.manifest = load_manifest(self.run)
            cal, matrix, ids = self._references(True)
            ci = len(self.observations())
            for ti, duration in enumerate(self.times):
                self.raw.acquire(
                    f"bo_{ci:04d}_t_{ti:04d}", (self.q_a, self.q_b),
                    {"coupler_z": amplitude, "duration_ns": int(duration),
                     "reference_job_ids": ids},
                    lambda duration=duration: self._plan(amplitude, int(duration)))
            completed = self._trace(ci, cal, matrix, ids)
            all_rows, scores = [], []
            for index in range(ci + 1):
                trace = completed if index == ci else self._trace(index, cal, matrix, ids)
                all_rows.extend(trace[0])
                scores.append(trace[1])
            saved = _csv(data_dir(self.run) / "joint_probabilities.csv",
                         all_rows, allow_pending=True)
            observations = _csv(data_dir(self.run) / "observations.csv",
                                scores, allow_pending=True)
            self.raw.manifest["probability_csv"] = saved.name
            self.raw.manifest["observation_csv"] = observations.name
            self.raw.save()
            return completed[1]

    def save_search_state(self, suggestion, rng_state, status="running"):
        self.raw.state(suggestion, rng_state, status)
