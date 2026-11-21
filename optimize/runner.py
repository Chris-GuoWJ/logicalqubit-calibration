"""Resumable Bayesian loops for pi amplitude, Z spectroscopy and coupler Z."""

import numpy as np
from .search import Search
from src.pi import PiSession


def run_pi(backend, qid, qconf, bounds, output_root, steps,
           settings=None, shots=1000, drive_if_ghz=None,
           kernel="matern52", acquisition="ei", kappa=2.0,
           seed=0, initial_points=5, run_mode="new",
           resume_run_tag="", job_loader=None):
    """Return the run directory and IQ candidates; never publish a gate calibration."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    limits = np.asarray(bounds, dtype=float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] < 0 or limits[1] <= limits[0]:
        raise ValueError("pi amplitude bounds must be increasing and nonnegative")
    plan = {"bounds": limits.tolist(), "steps": steps, "kernel": kernel,
            "acquisition": acquisition, "kappa": float(kappa),
            "seed": int(seed), "initial_points": int(initial_points)}
    session = PiSession(backend, qid, qconf, output_root, plan, settings=settings,
                        shots=shots, drive_if_ghz=drive_if_ghz, run_mode=run_mode,
                        resume_run_tag=resume_run_tag, job_loader=job_loader)
    search = Search([limits.tolist()], kernel=kernel, acquisition=acquisition,
                    kappa=kappa, seed=seed, initial_points=initial_points)
    records = session.observations()
    if len(records) > steps:
        raise ValueError("Existing run exceeds the requested budget")
    for row in records:
        search.tell([row["amplitude"]], row["loss"], row["noise_variance"],
                    shots=row["shots"], job_id=row["job_id"])
    if session.manifest.get("rng_state") is not None:
        search.rng.bit_generator.state = session.manifest["rng_state"]
    while len(records) < steps:
        pending = session.manifest.get("suggestion")
        if pending is not None and pending["index"] < len(records):
            session.save_search_state(None, search.rng.bit_generator.state)
            pending = None
        if pending is None:
            x = float(search.ask()[0])
            pending = {"index": len(records), "amplitude": x}
            session.save_search_state(pending, search.rng.bit_generator.state)
        if pending["index"] != len(records):
            raise RuntimeError("Saved suggestion index differs from recorded raw IQ")
        row = session.measure(pending["amplitude"])
        search.tell([row["amplitude"]], row["loss"], row["noise_variance"],
                    shots=row["shots"], job_id=row["job_id"])
        records = session.observations()
        session.save_search_state(None, search.rng.bit_generator.state)
    session.save_search_state(None, search.rng.bit_generator.state, status="completed")
    return session.run, records


def _bo_plan(bounds, steps, kernel, acquisition, kappa, seed, initial_points):
    limits = np.asarray(bounds, dtype=float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError("BO bounds must contain two increasing finite values")
    if (isinstance(steps, bool) or not isinstance(steps, int)
            or isinstance(initial_points, bool) or not isinstance(initial_points, int)
            or initial_points < 2 or steps <= initial_points):
        raise ValueError("BO needs at least two initial points and one GP-chosen point")
    if kernel not in ("rbf", "matern32", "matern52", "rq"):
        raise ValueError("Unsupported BO kernel")
    if acquisition not in ("ei", "lcb") or not np.isfinite(kappa) or kappa < 0:
        raise ValueError("Invalid BO acquisition or kappa")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("BO seed must be an integer")
    return limits, {"bounds": limits.tolist(), "steps": steps, "kernel": kernel,
                    "acquisition": acquisition, "kappa": float(kappa),
                    "seed": seed, "initial_points": initial_points}



def _z_map_posterior(rows, bounds, kernel, seed, points=401):
    """Predict resonance frequency, not spectroscopy signal, across Z."""
    from .gp import GP

    lo, hi = bounds
    x = np.asarray([row["z_amplitude"] for row in rows], dtype=float)
    frequency = np.asarray([row["peak_drive_if_ghz"] for row in rows], dtype=float)
    noise = np.asarray([row["peak_noise_variance"] for row in rows], dtype=float)
    model = GP(kernel=kernel, seed=seed).fit(
        ((x - lo) / (hi - lo))[:, None], frequency, noise)
    grid = np.linspace(lo, hi, points)
    mean, std = model.predict(((grid - lo) / (hi - lo))[:, None])
    return grid, mean, std


def suggest_z_bias(rows, bounds, kernel="matern52", seed=0):
    """Sample the least certain unsampled part of the frequency map."""
    lo, hi = map(float, bounds)
    grid, _, std = _z_map_posterior(rows, (lo, hi), kernel, seed, points=1025)
    measured = np.asarray([row["z_amplitude"] for row in rows], dtype=float)
    # Avoid proposing a nearly identical Z when measurement noise dominates.
    allowed = np.min(np.abs(grid[:, None] - measured[None, :]), axis=1) >= .005 * (hi - lo)
    if not allowed.any():
        raise ValueError("No distinct Z bias remains within the sampling interval")
    return float(grid[np.argmax(np.where(allowed, std, -np.inf))])


def run_zfreq_map(backend, qid, qconf, pi_cal, z_bounds, frequency_bounds,
                  frequency_points, output_root, steps, settings=None, shots=500,
                  kernel="matern52", seed=0, initial_points=3, run_mode="new",
                  resume_run_tag="", job_loader=None):
    """Choose Z biases actively; each Z gets one full IF sweep and one peak."""
    from src.zfreq_map import ZMapSession
    from src.runs import data_dir, save_csv

    limits, _ = _bo_plan(z_bounds, steps, kernel, "lcb", 2., seed, initial_points)
    if (isinstance(frequency_points, bool) or not isinstance(frequency_points, int)
            or frequency_points < 5):
        raise ValueError("Frequency sweep needs at least five points")
    frequencies = np.asarray(frequency_bounds, dtype=float)
    if (frequencies.shape != (2,) or not np.isfinite(frequencies).all()
            or frequencies[0] >= frequencies[1] or np.any(np.abs(frequencies) >= .5)):
        raise ValueError("Invalid digital IF bounds in GHz")
    plan = {"bounds": limits.tolist(), "steps": steps, "kernel": kernel,
            "seed": seed, "initial_points": initial_points,
            "frequency_bounds": frequencies.tolist(),
            "frequency_points": frequency_points,
            "sampling": "max_gp_posterior_std"}
    session = ZMapSession(
        backend, qid, qconf, pi_cal, limits, frequencies, frequency_points,
        output_root, plan, settings, shots, run_mode, resume_run_tag, job_loader)
    rows = session.observations()
    if len(rows) > steps:
        raise ValueError("Saved map exceeds the requested Z-point budget")
    initial = np.linspace(*limits, initial_points)
    while len(rows) < steps:
        pending = session.manifest.get("suggestion")
        if pending is not None and pending["index"] > len(rows):
            raise ValueError("Pending Z proposal exceeds saved sweeps")
        if pending is not None and pending["index"] < len(rows):
            session.save_search_state(None)
            pending = None
        if pending is None:
            z = (float(initial[len(rows)]) if len(rows) < initial_points
                 else suggest_z_bias(rows, limits, kernel=kernel, seed=seed))
            pending = {"index": len(rows), "z_amplitude": z}
            session.save_search_state(pending)
        row = session.measure(pending["index"], pending["z_amplitude"])
        rows = session.observations()
        if row["z_amplitude"] != float(pending["z_amplitude"]):
            raise ValueError("Saved peak differs from proposed Z bias")
        session.save_search_state(None)
    spectra = [point for i in range(len(rows)) for point in session._saved_sweep(i)[0]]
    session.save_observations(spectra, rows)
    grid, mean, std = _z_map_posterior(rows, limits, kernel, seed)
    map_rows = [{"z_amplitude": float(z), "drive_if_ghz": float(f),
                 "frequency_std_ghz": float(s)}
                for z, f, s in zip(grid, mean, std)]
    saved = save_csv(data_dir(session.run) / "frequency_map_posterior.csv",
                     map_rows, allow_pending=True)
    session.raw.manifest["frequency_map_csv"] = saved.name
    session.raw.save()
    session.save_search_state(None, status="completed")
    return session.run, rows

def run_zfreq(backend, qid, qconf, pi_cal, z_values, bounds, output_root, steps,
              settings=None, shots=500, kernel="matern52", acquisition="ei",
              kappa=2.0, seed=0, initial_points=5, run_mode="new",
              resume_run_tag="", job_loader=None):
    """Optimize drive IF separately for each Z; return candidate frequency rows."""
    from src.zfreq import ZSession
    limits, plan = _bo_plan(bounds, steps, kernel, acquisition, kappa, seed,
                            initial_points)
    session = ZSession(backend, qid, qconf, pi_cal, z_values, limits, output_root,
                       plan, settings, shots, run_mode, resume_run_tag, job_loader)
    candidates = []
    for zi, z in enumerate(session.z_values):
        search = Search([limits.tolist()], kernel=kernel, acquisition=acquisition,
                        kappa=kappa, seed=seed + zi, initial_points=initial_points)
        rows = session.observations(zi)
        if len(rows) > steps:
            raise ValueError("Existing Z BO run exceeds the planned steps")
        for row in rows:
            search.tell([row["drive_if_ghz"]], row["loss"],
                        row["noise_variance"], shots=row["shots"], job_id=row["job_id"])
        states = session.manifest.get("rng_state") or {}
        if str(zi) in states:
            search.rng.bit_generator.state = states[str(zi)]
        while len(rows) < steps:
            pending = session.manifest.get("suggestion")
            if pending is not None and (pending["z_index"] != zi
                                        or pending["index"] > len(rows)):
                raise ValueError("Pending Z BO suggestion differs from checkpoint")
            if pending is not None and pending["index"] < len(rows):
                pending = None
                session.save_search_state(None, states)
            if pending is None:
                pending = {"z_index": zi, "index": len(rows),
                           "drive_if_ghz": float(search.ask()[0])}
                states[str(zi)] = search.rng.bit_generator.state
                session.save_search_state(pending, states)
            row = session.measure(zi, pending["drive_if_ghz"])
            search.tell([row["drive_if_ghz"]], row["loss"], row["noise_variance"],
                        shots=row["shots"], job_id=row["job_id"])
            rows = session.observations(zi)
            session.save_search_state(None, states)
        best = min(rows, key=lambda row: row["loss"])
        candidates.append({"qubit": int(qid), "z_amplitude": float(z),
                           "drive_if_ghz": best["drive_if_ghz"],
                           "iq_center_distance": best["S"],
                           "at_search_boundary": bool(
                               min(best["drive_if_ghz"] - limits[0],
                                   limits[1] - best["drive_if_ghz"])
                               < .05 * (limits[1] - limits[0]))})
    from src.runs import data_dir, save_csv
    save_csv(data_dir(session.run) / "frequency_candidates.csv", candidates)
    session.save_search_state(None, session.manifest.get("rng_state") or {},
                              status="completed")
    return session.run, candidates


def run_coupler(backend, qconf, q_a, q_b, tuning_qubit, tuning_z_value,
                times_ns, bounds, pi_calibration_path, output_root, steps,
                settings=None, shots=1000, resonance_metadata=None,
                kernel="matern52", acquisition="ei", kappa=2.0, seed=0,
                initial_points=5, run_mode="new", resume_run_tag="",
                job_loader=None):
    """Optimize coupler Z using a full SWAP trace for every BO observation."""
    from src.coupler import CouplerSession
    limits, plan = _bo_plan(bounds, steps, kernel, acquisition, kappa, seed,
                            initial_points)
    session = CouplerSession(
        backend, qconf, q_a, q_b, tuning_qubit, tuning_z_value,
        times_ns, limits, pi_calibration_path, output_root, plan,
        settings, shots, resonance_metadata, run_mode, resume_run_tag, job_loader)
    search = Search([limits.tolist()], kernel=kernel, acquisition=acquisition,
                    kappa=kappa, seed=seed, initial_points=initial_points)
    rows = session.observations()
    if len(rows) > steps:
        raise ValueError("Existing coupler BO run exceeds the planned steps")
    for row in rows:
        search.tell([row["coupler_z"]], row["loss"], row["noise_variance"],
                    shots=row["shots"], job_id=",".join(row["job_ids"]))
    if session.manifest.get("rng_state") is not None:
        search.rng.bit_generator.state = session.manifest["rng_state"]
    while len(rows) < steps:
        pending = session.manifest.get("suggestion")
        if pending is not None and pending["index"] > len(rows):
            raise ValueError("Pending coupler suggestion exceeds saved traces")
        if pending is not None and pending["index"] < len(rows):
            pending = None
            session.save_search_state(None, search.rng.bit_generator.state)
        if pending is None:
            pending = {"index": len(rows), "coupler_z": float(search.ask()[0])}
            session.save_search_state(pending, search.rng.bit_generator.state)
        row = session.measure(pending["coupler_z"])
        search.tell([row["coupler_z"]], row["loss"], row["noise_variance"],
                    shots=row["shots"], job_id=",".join(row["job_ids"]))
        rows = session.observations()
        session.save_search_state(None, search.rng.bit_generator.state)
    # Also cover a resumed run whose raw traces were complete before the
    # observation CSV was written, including runs made before this plotter.
    from src.runs import data_dir, save_csv
    saved = save_csv(data_dir(session.run) / "observations.csv",
                     rows, allow_pending=True)
    session.raw.manifest["observation_csv"] = saved.name
    session.raw.save()
    session.save_search_state(None, search.rng.bit_generator.state,
                              status="completed")
    return session.run, rows
