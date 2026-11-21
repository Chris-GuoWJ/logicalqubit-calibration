"""Offline BO plots for pi amplitude, Z spectroscopy and coupler calibration."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_solve

from src.runs import load_manifest
from .gp import GP


def _posterior(frame, bounds, kernel, seed, parameter, points=350):
    lo, hi = map(float, bounds)
    x = frame[parameter].to_numpy(float)
    loss = frame["loss"].to_numpy(float)
    noise = frame["noise_variance"].to_numpy(float)
    model = GP(kernel=kernel, seed=seed).fit(
        ((x - lo) / (hi - lo))[:, None], loss, noise)
    grid = np.linspace(lo, hi, points)
    mean_loss, std = model.predict(((grid - lo) / (hi - lo))[:, None])
    return grid, -mean_loss, std, model  # BO minimizes loss = -score.


def show_bo_evolution(frame, run_dir, bounds, kernel, seed, initial_points,
                      *, parameter, score, x_label, y_label, title, prefix):
    """Show saved measurement prefixes and their refitted GP posteriors."""
    total = len(frame)
    lo, hi = map(float, bounds)
    first_gp = min(max(2, initial_points), total)
    extras = (np.linspace(first_gp + 1, total, 4, dtype=int).tolist()
              if total > first_gp else [])
    stages = sorted(set([1, first_gp, total] + extras))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    try:
        for ax, count in zip(axes.flat, stages):
            subset = frame.iloc[:count]
            x = subset[parameter].to_numpy(float)
            y = subset[score].to_numpy(float)
            if count >= max(2, initial_points):
                grid, mean, std, _ = _posterior(
                    subset, bounds, kernel, seed, parameter)
                ax.plot(grid, mean, color="tab:blue", lw=1.8, label="GP mean")
                ax.fill_between(grid, mean - 1.96 * std, mean + 1.96 * std,
                                color="tab:blue", alpha=.2,
                                label="95% model interval")
            ax.scatter(x, y, c=np.arange(count), cmap="viridis",
                       vmin=0, vmax=max(total - 1, 1), s=30, zorder=3)
            ax.scatter(x[-1], y[-1], marker="*", color="tab:red", s=130,
                       zorder=4, label="Latest observation")
            ax.axvline(x[int(np.argmax(y))], color="tab:green", ls="--", lw=1)
            ax.set_title(f"After {count}/{total} measurements")
            ax.grid(alpha=.2)
            ax.set_xlim(lo, hi)
        for ax in list(axes.flat)[len(stages):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel(x_label)
        for ax in axes[:, 0]:
            ax.set_ylabel(y_label)
        handles, labels = axes.flat[min(len(stages) - 1, 5)].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.suptitle(f"{title}: GP fit to loss = -score", fontsize=14)
        fig.tight_layout(rect=(0, .04, 1, .96))
        destination = Path(run_dir) / "fig" / f"{prefix}_bo_evolution.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180)
        if plt.get_backend().lower() != "agg":
            plt.show()
        return destination
    finally:
        plt.close(fig)


def show_bo_fit(frame, run_dir, bounds, kernel, seed, *, parameter, score,
                x_label, y_label, title, prefix):
    """Show final posterior, best observed score and conditional LOO checks."""
    if len(frame) < 2:
        print(f"{title}: at least two observations are required for GP diagnostics.")
        return None
    grid, mean, std, model = _posterior(frame, bounds, kernel, seed, parameter)
    x = frame[parameter].to_numpy(float)
    y = frame[score].to_numpy(float)
    err = np.sqrt(frame["noise_variance"].to_numpy(float))
    n = len(frame)
    best_index = int(np.argmax(y))
    best_so_far = np.maximum.accumulate(y)
    # Hold full-data fitted kernel hyperparameters fixed; this is not
    # independent validation. The LOO predictive std includes observed noise.
    precision_diagonal = np.diag(cho_solve(model.chol, np.eye(n)))
    loo_loss = model.y - model.weights / precision_diagonal
    loo_prediction = -(model.mean + model.scale * loo_loss)
    loo_std = model.scale / np.sqrt(precision_diagonal)
    rmse = float(np.sqrt(np.mean((loo_prediction - y) ** 2)))
    coverage = float(np.mean(np.abs(loo_prediction - y) <= 1.96 * loo_std))

    fig, (ax, convergence, cv) = plt.subplots(1, 3, figsize=(16, 4.6))
    try:
        ax.plot(grid, mean, color="tab:blue", label="GP posterior mean")
        ax.fill_between(grid, mean - 1.96 * std, mean + 1.96 * std,
                        color="tab:blue", alpha=.2, label="95% model interval")
        ax.errorbar(x, y, yerr=err, fmt="o", ms=4, capsize=2,
                    color="tab:orange", label="Measured score")
        gp_peak = float(grid[int(np.argmax(mean))])
        ax.axvline(x[best_index], ls="--", color="tab:green",
                   label=f"Best measured {x[best_index]:.4g}")
        ax.axvline(gp_peak, ls=":", color="tab:purple",
                   label=f"GP peak {gp_peak:.4g}")
        ax.set(xlabel=x_label, ylabel=y_label, title="Final GP fit")
        ax.legend(fontsize=8)
        convergence.plot(np.arange(1, n + 1), best_so_far, "o-", ms=3)
        convergence.set(xlabel="Measured parameter points",
                        ylabel="Best observed score",
                        title="Improvement with experiment count")
        convergence.set_xlim(1, n)
        cv.errorbar(y, loo_prediction, yerr=1.96 * loo_std, fmt="o",
                    ms=4, capsize=2, color="tab:purple")
        low = min(float(y.min()), float(loo_prediction.min()))
        high = max(float(y.max()), float(loo_prediction.max()))
        cv.plot([low, high], [low, high], "k--", lw=1)
        cv.set(xlabel="Measured score", ylabel="Leave-one-out predicted score",
               title=f"LOO* RMSE={rmse:.3g}, 95% coverage={coverage:.0%}")
        for subplot in (ax, convergence, cv):
            subplot.grid(alpha=.2)
        fig.suptitle(title)
        fig.text(.5, .025,
                 "*LOO keeps final fitted kernel hyperparameters; diagnostic only",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .07, 1, .94))
        destination = Path(run_dir) / "fig" / f"{prefix}_bo_fit.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=180)
        if plt.get_backend().lower() != "agg":
            plt.show()
        print(f"{title}: Best measured {parameter}={x[best_index]:.6g}, "
              f"{score}={y[best_index]:.6g}; GP peak {parameter}={gp_peak:.6g}")
        print(f"Conditional LOO: RMSE={rmse:.4g}, "
              f"95% interval coverage={coverage:.1%}")
        return destination
    finally:
        plt.close(fig)


def plot_bo_run(run_dir):
    """Read a saved BO run, display plots and return paths grouped by trace."""
    run_dir = Path(run_dir).resolve()
    manifest = load_manifest(run_dir)
    kind = manifest.get("kind")
    if kind == "zfreq" and manifest.get("strategy") == "bo_map":
        return plot_zfreq_map(run_dir, manifest)
    if kind not in ("pi", "zfreq", "coupler") or manifest.get("strategy") != "bo":
        raise ValueError("Selected run is not a supported BO calibration run")
    csv_name = manifest.get("observation_csv", "observations.csv")
    if Path(csv_name).name != csv_name:
        raise ValueError("Invalid BO observation filename in manifest")
    csv_path = run_dir / "data" / csv_name
    if not csv_path.is_file():
        if kind == "coupler" and (run_dir / "data" / "joint_probabilities.csv").is_file():
            raise FileNotFoundError(
                "This coupler run predates persisted BO scores; its joint "
                "probabilities alone do not contain the GP noise variances.")
        print(f"No BO observations yet: {csv_path}")
        return {}
    frame = pd.read_csv(csv_path)
    plan = manifest["plan"]
    bounds = plan["bounds"]
    kernel = plan["kernel"]
    initial_points = int(plan["initial_points"])
    seed = int(plan["seed"])
    if (len(bounds) != 2 or not np.isfinite(bounds).all()
            or float(bounds[0]) >= float(bounds[1])):
        raise ValueError("Invalid BO bounds in manifest")

    if kind == "pi":
        traces = [("pi", frame, "amplitude", "S", "Pi amplitude",
                   "IQ centre distance S (larger is better)", "Pi amplitude BO", seed)]
    elif kind == "zfreq":
        if "z_amplitude" not in frame:
            raise ValueError("Z-frequency BO observations need z_amplitude")
        z = pd.to_numeric(frame["z_amplitude"], errors="raise").to_numpy(float)
        z_values = manifest["z_values"]
        masks = [np.isclose(z, value, rtol=0, atol=1e-12) for value in z_values]
        if not all(np.sum(np.column_stack(masks), axis=1) == 1):
            raise ValueError("Z-frequency observations do not match unique manifest biases")
        traces = [(f"z{i:03d}", frame.loc[mask].reset_index(drop=True),
                   "drive_if_ghz", "S", "Drive IF (GHz)",
                   "IQ centre distance S",
                   f"Z-frequency BO: Z={float(value):.6g}", seed + i)
                  for i, (value, mask) in enumerate(zip(z_values, masks)) if mask.any()]
    else:
        traces = [("coupler", frame, "coupler_z", "score", "Coupler Z amplitude",
                   "SWAP contrast x anticorrelation score",
                   "Coupler Z BO", seed)]

    results = {}
    for prefix, group, parameter, score, x_label, y_label, title, trace_seed in traces:
        required = [parameter, score, "loss", "noise_variance"]
        if not set(required).issubset(group.columns) or group.empty:
            raise ValueError(f"{title}: missing or empty BO observations")
        numeric = group[required].apply(pd.to_numeric, errors="raise")
        if (not np.isfinite(numeric.to_numpy(float)).all()
                or (numeric["noise_variance"] < 0).any()
                or not numeric[parameter].between(*bounds).all()):
            raise ValueError(f"{title}: invalid BO observations")
        group = group.copy()
        group[required] = numeric
        options = dict(parameter=parameter, score=score, x_label=x_label,
                       y_label=y_label, title=title, prefix=prefix)
        evolution = show_bo_evolution(group, run_dir, bounds, kernel, trace_seed,
                                      initial_points, **options)
        fit = show_bo_fit(group, run_dir, bounds, kernel, trace_seed, **options)
        results[prefix] = {"evolution": evolution, "fit": fit}
        print(f"Figures ({prefix}): {evolution}, {fit}")
    return results




def plot_zfreq_map(run_dir, manifest):
    """Display adaptive Z sampling and the GP model of resonance IF versus Z."""
    from .runner import _z_map_posterior

    run_dir = Path(run_dir)
    peak_path = run_dir / "data" / manifest.get("candidate_csv", "frequency_candidates.csv")
    if not peak_path.is_file():
        print(f"No completed Z-frequency sweep yet: {peak_path}")
        return {}
    peaks = pd.read_csv(peak_path)
    needed = ["z_amplitude", "peak_drive_if_ghz", "peak_noise_variance"]
    if (peaks.empty or not set(needed).issubset(peaks)
            or not np.isfinite(peaks[needed].to_numpy(float)).all()
            or (peaks["peak_noise_variance"] < 0).any()):
        raise ValueError("Invalid Z-frequency map peak observations")
    plan = manifest["plan"]
    bounds, kernel, seed = plan["bounds"], plan["kernel"], int(plan["seed"])
    initial_points = int(plan["initial_points"])
    x = peaks["z_amplitude"].to_numpy(float)
    f = peaks["peak_drive_if_ghz"].to_numpy(float)
    noise = peaks["peak_noise_variance"].to_numpy(float)
    if not np.all((x >= bounds[0]) & (x <= bounds[1])):
        raise ValueError("Saved Z proposal lies outside map bounds")
    total = len(peaks)
    first_gp = min(initial_points, total)
    stages = sorted(set([1, first_gp, total] + (
        np.linspace(first_gp + 1, total, 4, dtype=int).tolist()
        if total > first_gp else [])))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    try:
        for ax, count in zip(axes.flat, stages):
            if count >= 2:
                grid, mean, std = _z_map_posterior(
                    peaks.iloc[:count].to_dict("records"), bounds, kernel, seed)
                ax.plot(grid, mean, color="tab:blue", label="GP mean IF")
                ax.fill_between(grid, mean - 1.96 * std, mean + 1.96 * std,
                                color="tab:blue", alpha=.2, label="95% model interval")
            ax.errorbar(x[:count], f[:count], yerr=np.sqrt(noise[:count]),
                        fmt="o", capsize=3, color="tab:orange", label="Measured peak IF")
            ax.scatter(x[count-1], f[count-1], color="tab:red", marker="*",
                       s=140, zorder=4, label="Latest Z")
            ax.set_title(f"After {count}/{total} Z biases")
            ax.set_xlim(*bounds)
            ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(stages):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("Z bias amplitude")
        for ax in axes[:, 0]:
            ax.set_ylabel("Resonance drive IF (GHz)")
        handles, labels = axes.flat[min(len(stages)-1, 5)].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.suptitle("Adaptive Z sampling: GP fit to resonance frequency f(Z)")
        fig.tight_layout(rect=(0, .04, 1, .96))
        evolution = run_dir / "fig" / "zfreq_map_bo_evolution.png"
        evolution.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(evolution, dpi=180)
        if plt.get_backend().lower() != "agg":
            plt.show()
    finally:
        plt.close(fig)
    if total < 2:
        print("At least two complete Z sweeps are needed for GP diagnostics.")
        return {"zfreq_map": {"evolution": evolution, "fit": None}}
    from .gp import GP

    lo, hi = map(float, bounds)
    model = GP(kernel=kernel, seed=seed).fit(
        ((x - lo) / (hi - lo))[:, None], f, noise)
    grid, mean, std = _z_map_posterior(peaks.to_dict("records"), bounds, kernel, seed)
    diagonal = np.diag(cho_solve(model.chol, np.eye(total)))
    loo_frequency = model.mean + model.scale * (model.y - model.weights / diagonal)
    loo_std = model.scale / np.sqrt(diagonal)
    rmse = float(np.sqrt(np.mean((loo_frequency - f) ** 2)))
    coverage = float(np.mean(np.abs(loo_frequency - f) <= 1.96 * loo_std))
    fig, (ax, spectra_ax, cv) = plt.subplots(1, 3, figsize=(17, 5))
    try:
        ax.plot(grid, mean, color="tab:blue", label="GP predicted IF")
        ax.fill_between(grid, mean - 1.96 * std, mean + 1.96 * std,
                        color="tab:blue", alpha=.2, label="95% model interval")
        ax.errorbar(x, f, yerr=np.sqrt(noise), fmt="o", capsize=3,
                    color="tab:orange", label="Measured peaks")
        for index, (z, frequency) in enumerate(zip(x, f), 1):
            ax.annotate(str(index), (z, frequency), xytext=(5, 4),
                        textcoords="offset points", fontsize=8)
        ax.set(xlabel="Z bias amplitude", ylabel="Resonance drive IF (GHz)",
               title="Final frequency map f(Z)")
        ax.legend(fontsize=8)
        spectrum_path = run_dir / "data" / manifest.get("observation_csv", "observations.csv")
        if spectrum_path.is_file():
            spectra = pd.read_csv(spectrum_path)
            for index, sweep in spectra.groupby("z_index", sort=True):
                sweep = sweep.sort_values("drive_if_ghz")
                spectra_ax.plot(sweep["drive_if_ghz"], sweep["S"], "o-",
                                ms=3, label=f"Z={x[int(index)]:.4g}")
            spectra_ax.legend(fontsize=7)
        spectra_ax.set(xlabel="Drive IF (GHz)", ylabel="IQ centre distance S",
                       title="Measured spectra at sampled Z")
        cv.errorbar(f, loo_frequency, yerr=1.96 * loo_std, fmt="o",
                    capsize=2, color="tab:purple")
        low = min(float(f.min()), float(loo_frequency.min()))
        high = max(float(f.max()), float(loo_frequency.max()))
        cv.plot([low, high], [low, high], "k--", lw=1)
        cv.set(xlabel="Measured peak IF (GHz)",
               ylabel="LOO predicted IF (GHz)",
               title=f"LOO* RMSE={rmse:.3g} GHz, coverage={coverage:.0%}")
        for subplot in (ax, spectra_ax, cv):
            subplot.grid(alpha=.2)
        fig.text(.5, .02, "*LOO keeps final fitted GP hyperparameters; diagnostic only",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .05, 1, 1))
        fit = run_dir / "fig" / "zfreq_map_bo_fit.png"
        fig.savefig(fit, dpi=180)
        if plt.get_backend().lower() != "agg":
            plt.show()
    finally:
        plt.close(fig)
    print(f"Z-frequency map: {total} sampled Z values; "
          f"conditional LOO RMSE={rmse:.4g} GHz, 95% coverage={coverage:.1%}")
    print(f"Figures (zfreq_map): {evolution}, {fit}")
    return {"zfreq_map": {"evolution": evolution, "fit": fit}}

