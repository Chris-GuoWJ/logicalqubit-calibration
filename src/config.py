"""Validated calibration configuration and provenance snapshots."""

from datetime import datetime, timezone
from pathlib import Path
import json
import shutil

from .runs import save_json, save_csv, run_lock


def paths(root):
    root = Path(root)
    return {"hardware": root / "hardware", "calibrations": root / "calibrations",
            "history": root / "history"}


def _backup(path, history):
    if path.is_file():
        history.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        shutil.copy2(path, history / f"{path.stem}_{stamp}{path.suffix}")


def save_hardware(root, backend):
    """Record the hardware properties actually used, excluding credentials."""
    dest = paths(root)["hardware"]
    dest.mkdir(parents=True, exist_ok=True)
    snapshot = {"schema_version": 1, "backend": str(backend.name),
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "properties": backend.config.get("properties", {}),
                "topology": backend.config.get("topology", {})}
    out = dest / (snapshot["captured_at_utc"].replace(":", "").replace(".", "_") + ".json")
    # Normalize SDK-backed scalars while retaining readable machine data.
    def normalize(value):
        if hasattr(value, "tolist"):
            return normalize(value.tolist())
        if hasattr(value, "item"):
            return normalize(value.item())
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [normalize(v) for v in value]
        return value
    save_json(out, normalize(snapshot))
    return out


def load_pi(root, backend_name, qubits):
    path = paths(root)["calibrations"] / "pi_pulse_amplitudes.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("backend") != backend_name:
        raise ValueError(f"Incompatible pi calibration: {path}")
    result = {}
    for qid in qubits:
        row = data["qubits"][str(qid)]
        settings = row["pulse_settings"]
        result[int(qid)] = {"pi_amp": float(row["pi_amp"]),
                            "duration_ns": int(settings["duration_ns"]),
                            "envelope": settings["envelope"],
                            "source_run_tag": row.get("source_run_tag")}
    return result


def publish_pi(run, root):
    """Explicitly publish a completed pi run; the old config is retained."""
    from .pi import export_pi_amplitudes
    folders = paths(root)
    target = folders["calibrations"] / "pi_pulse_amplitudes.json"
    with run_lock(root):
        _backup(target, folders["history"])
        export_pi_amplitudes(run, folders["calibrations"])
    return target


def publish_z(run, root, backend_name):
    """Merge only measured Z points; retain unmeasured qubits and point history."""
    from .zfreq import analyze
    from .runs import load_manifest
    if load_manifest(run).get("backend") != backend_name:
        raise ValueError("Run belongs to another backend")
    rows = analyze(run)
    folders = paths(root)
    target = folders["calibrations"] / "qubit_frequency_vs_z.json"
    csv_path = folders["calibrations"] / "qubit_frequency_vs_z.csv"
    folders["calibrations"].mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        data = (json.loads(target.read_text(encoding="utf-8")) if target.exists() else
                {"schema_version": 1, "backend": backend_name, "qubits": {},
                 "frequency_unit": "GHz IF, NOT absolute RF",
                 "z_amplitude_unit": "platform waveform units"})
        if data.get("schema_version") != 1 or data.get("backend") != backend_name:
            raise ValueError("Existing Z frequency map belongs to another backend/schema")
        _backup(target, folders["history"])
        _backup(csv_path, folders["history"])
        timestamp = datetime.now(timezone.utc).isoformat()
        for qid in sorted({int(r["qubit"]) for r in rows}):
            old = dict(data["qubits"].get(str(qid), {}))
            points = {format(float(p["z_amplitude"]), ".12g"): dict(p)
                      for p in old.get("z_frequency_points", [])}
            for row in (r for r in rows if int(r["qubit"]) == qid):
                key = format(float(row["z_amplitude"]), ".12g")
                current = {"z_amplitude": float(row["z_amplitude"]),
                           "drive_if_ghz": float(row["qubit_frequency_ghz"]),
                           "center_distance": float(row["distance_feature"]),
                           "distance_baseline": float(row["distance_baseline"]),
                           "significance": float(row["significance"]),
                           "at_scan_boundary": bool(row["at_scan_boundary"]),
                           "valid": bool(row["valid"]),
                           "frequency_jump_ghz": (float(row["frequency_jump_ghz"])
                                                  if row["frequency_jump_ghz"] is not None else None),
                           "source_run_tag": Path(run).name,
                           "source_run_dir": str(Path(run).resolve()),
                           "measured_at": timestamp}
                prior = points.get(key)
                if prior is not None:
                    history = list(prior.get("history", []))
                    previous = {k: v for k, v in prior.items() if k != "history"}
                    if previous != current:
                        history.append(previous)
                    if history:
                        current["history"] = history
                points[key] = current
            old["z_frequency_points"] = sorted(points.values(), key=lambda p: p["z_amplitude"])
            old["last_updated_run_tag"] = Path(run).name
            old["updated_at"] = timestamp
            data["qubits"][str(qid)] = old
        data["updated_at"] = timestamp
        save_json(target, data)
        flattened = [{"qubit": int(q), **{k: v for k, v in p.items() if k != "history"},
                      "history_count": len(p.get("history", []))}
                     for q, item in data["qubits"].items()
                     for p in item.get("z_frequency_points", [])]
        if flattened:
            save_csv(csv_path, flattened)
    return target
