"""Print a read-only BO run summary; never import SDK or modify run files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


def number(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {name}")
    return result


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (position - low) * (values[high] - values[low])


def summarize(run_dir, root):
    root = root.resolve(strict=True)
    results = (root / "result").resolve(strict=True)
    run = run_dir.resolve(strict=True)
    if not run.is_relative_to(results) or run.parent.parent != results:
        raise ValueError("Select exactly result/<kind>/<run_tag> inside this project")
    manifest_path = (run / "data" / "manifest.json").resolve(strict=True)
    if not manifest_path.is_relative_to(run):
        raise ValueError("Manifest resolves outside the selected run")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8-sig"))
    kind = manifest.get("kind")
    if kind not in ("pi", "zfreq", "coupler") or manifest.get("strategy") != "bo":
        raise ValueError("This summary supports BO runs only; inspect grid data separately")
    if run.parent.name != kind:
        raise ValueError("Run category and manifest kind differ")
    warnings = []
    if manifest.get("status") != "completed":
        warnings.append("Run is incomplete: inspect saved jobs and pending submission before planning a new run")
    jobs = manifest.get("jobs", [])
    if not isinstance(jobs, list) or any(not isinstance(j, dict) for j in jobs):
        raise ValueError("Invalid jobs manifest")
    labels = [j.get("label") for j in jobs]
    if len(labels) != len(set(labels)):
        warnings.append("Repeated job labels: entries are not necessarily distinct completed measurements")
    pending = [{"label": j.get("label"), "job_id": j.get("job_id"), "status": j.get("status")}
               for j in jobs if j.get("status") != "saved"]
    if pending:
        warnings.append("Non-saved jobs exist: reconcile status; do not blindly resubmit")
    field = manifest.get("observation_csv", "observations.csv")
    if not isinstance(field, str) or Path(field).name != field:
        raise ValueError("Observation CSV must be one filename inside data")
    csv_path = (run / "data" / field).resolve(strict=True)
    if not csv_path.is_relative_to(run):
        raise ValueError("Observation CSV resolves outside the selected run")
    csv_bytes = csv_path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"), newline="")))
    parameter, score = ("coupler_z", "score") if kind == "coupler" else (
        "drive_if_ghz", "S") if kind == "zfreq" else ("amplitude", "S")
    plan = manifest.get("plan", {})
    bounds = plan.get("bounds", manifest.get("bounds"))
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("Missing search bounds")
    lo, hi = [number(x, "bounds") for x in bounds]
    if hi <= lo:
        raise ValueError("Search bounds must increase")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        x, y = number(row[parameter], parameter), number(row[score], score)
        loss = number(row["loss"], "loss")
        variance = number(row["noise_variance"], "noise_variance")
        if variance < 0 or not lo <= x <= hi:
            raise ValueError(f"Invalid observation row {index}")
        if not math.isclose(loss, -y, rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError(f"Loss does not equal negative score at row {index}")
        group = "all"
        if kind == "zfreq":
            group = "Z=" + str(number(row["z_amplitude"], "z_amplitude"))
        groups[group].append((x, y, variance))
    summaries = []
    for group, points in sorted(groups.items()):
        x, y, variance = max(points, key=lambda item: item[1])
        scores = [p[1] for p in points]
        errors = [math.sqrt(p[2]) for p in points]
        near_boundary = min(x - lo, hi - x) < 0.05 * (hi - lo)
        if near_boundary:
            warnings.append(f"{group}: best measured point is within 5% of a search boundary")
        summaries.append({
            "group": group, "observations": len(points),
            "parameter": parameter, "score_field": score,
            "best_measured_parameter": x, "best_measured_score": y,
            "best_point_standard_error_model": math.sqrt(variance),
            "median_point_standard_error_model": median(errors),
            "score_range": [min(scores), max(scores)],
            "score_p90_minus_p10": quantile(scores, .9) - quantile(scores, .1),
            "near_search_boundary_5pct": near_boundary,
            "low_signal_heuristic": y <= 3 * math.sqrt(variance) if kind != "coupler" else None,
        })
    if not rows:
        warnings.append("No saved BO observations; do not infer an optimum")
    if kind == "zfreq" and len(groups) != len(manifest.get("z_values", [])):
        warnings.append("Observed Z-group count differs from manifest; inspect missing/partial traces")
    if manifest_path.read_bytes() != raw:
        warnings.append("Manifest changed while reading; take a new consistent snapshot before deciding")
    shots = manifest.get("shots")
    return {
        "run_dir": str(run), "kind": kind, "strategy": "bo",
        "status": manifest.get("status"), "backend": manifest.get("backend"),
        "qubits": [manifest.get("q_a"), manifest.get("q_b")] if kind == "coupler" else [manifest.get("qubit")],
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "observation_csv": str(csv_path),
        "observations_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "recorded_job_entries": len(jobs), "shots_per_job": shots,
        "listed_shots_estimate": len(jobs) * shots if isinstance(shots, int) else None,
        "cost_caveat": "Manifest entries do not prove total billed attempts or the cross-round remaining budget",
        "pending_jobs": pending, "pending_suggestion": manifest.get("suggestion"),
        "bounds": [lo, hi], "settings": manifest.get("settings"),
        "groups": summaries, "warnings": warnings,
        "interpretation": "Measured candidates only; noise is conditional and may omit shared-reference uncertainty and drift; no gate fidelity inferred",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    try:
        report = summarize(args.run_dir, args.root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Cannot summarize run: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
