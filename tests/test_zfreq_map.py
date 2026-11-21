"""Offline checks of adaptive Z sampling and resumable frequency mapping."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_cli import preflight
from optimize.runner import run_zfreq_map
from optimize.show_fig import plot_bo_run
from src.runs import load_manifest
from src.zfreq import ZSettings, analyze
from src.zfreq_map import fit_spectral_peak
from test_bo import SpectroscopyBackend, QCONF


PI_CAL = {"pi_amp": .8, "duration_ns": 40, "envelope": "raised_cosine",
          "source_run_tag": "test"}


class ZMapTests(unittest.TestCase):
    def test_peak_and_exact_budget(self):
        frequencies = np.linspace(-.13, .17, 11)
        rows = [{"drive_if_ghz": float(f),
                 "S": float(np.exp(-((f - .03) / .04) ** 2)),
                 "mean_displacement_noise": .01} for f in frequencies]
        fitted = fit_spectral_peak(rows, seed=2)
        self.assertAlmostEqual(fitted["peak_drive_if_ghz"], .03, delta=.005)
        self.assertGreater(fitted["peak_noise_variance"], 0)
        with self.assertRaisesRegex(ValueError, "IF boundary"):
            fit_spectral_peak([{**row, "S": f} for f, row in zip(frequencies, rows)])
        spec = json.loads((ROOT / "config/plans/zfreq_map_bo_example.json").read_text())
        self.assertEqual(preflight(spec)["jobs"], 72)

    def test_complete_map_after_interrupted_frequency_sweep(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temp:
            backend = SpectroscopyBackend(interrupt_at=6)
            args = (backend, 9, QCONF[9], PI_CAL,
                    [-.2, .2], [-.13, .17], 11, Path(temp) / "zfreq", 4)
            options = dict(settings=ZSettings(), shots=150, seed=7,
                           initial_points=3,
                           job_loader=lambda b, job_id: b.jobs[job_id])
            with self.assertRaises(TimeoutError):
                run_zfreq_map(*args, **options)
            run = next((Path(temp) / "zfreq").iterdir())
            completed, peaks = run_zfreq_map(
                *args, run_mode="resume", resume_run_tag=run.name, **options)
            self.assertEqual(completed, run)
            self.assertEqual(len(peaks), 4)
            self.assertEqual(backend.calls, 4 * 12)
            self.assertEqual(load_manifest(run)["status"], "completed")
            z = [row["z_amplitude"] for row in peaks]
            np.testing.assert_allclose(z[:3], [-.2, 0., .2], atol=1e-9)
            self.assertTrue(-.2 < z[3] < .2 and abs(z[3]) > .002)
            self.assertTrue((run / "data" / "frequency_map_posterior.csv").is_file())
            saved = pd.read_csv(run / "data" / "frequency_candidates.csv")
            self.assertEqual(len(saved), 4)
            self.assertEqual(len(pd.read_csv(run / "data" / "observations.csv")), 44)
            result = plot_bo_run(run)["zfreq_map"]
            self.assertTrue(result["evolution"].is_file())
            self.assertTrue(result["fit"].is_file())
            with self.assertRaisesRegex(ValueError, "independent spectral-peak"):
                analyze(run)
            rerun, again = run_zfreq_map(
                *args, run_mode="resume", resume_run_tag=run.name, **options)
            self.assertEqual(rerun, run)
            self.assertEqual(backend.calls, 48)
            self.assertEqual(len(again), 4)


if __name__ == "__main__":
    unittest.main()

