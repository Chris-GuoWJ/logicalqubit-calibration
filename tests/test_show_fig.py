"""Offline checks for the shared BO figures; no QPU jobs are submitted."""

import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimize.show_fig import plot_bo_run
from src.runs import new_run, save_csv, save_json


class BOFigureTests(unittest.TestCase):
    def _run(self, root, kind, rows, *, z_values=None):
        run = new_run(root / kind, kind)
        manifest = {
            "jobs": [], "kind": kind, "strategy": "bo",
            "plan": {"bounds": ([0., 1.] if kind == "pi" else [-.2, .2]),
                     "kernel": "matern52", "seed": 7, "initial_points": 2},
        }
        if z_values is not None:
            manifest["z_values"] = z_values
        save_json(run / "data" / "manifest.json", manifest)
        save_csv(run / "data" / "observations.csv", rows)
        return run

    def test_pi_replay_and_partial_coupler_run(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temp:
            root = Path(temp)
            scores = [.3, .5, .4, .6]
            pi = self._run(root, "pi", [
                {"amplitude": a, "S": y, "loss": -y, "noise_variance": .0004}
                for a, y in zip([.1, .4, .7, .9], scores)])
            with redirect_stdout(io.StringIO()) as output:
                paths = plot_bo_run(pi)["pi"]
            self.assertTrue(paths["evolution"].is_file())
            self.assertTrue(paths["fit"].is_file())
            self.assertIn("Conditional LOO: RMSE=", output.getvalue())

            coupler = self._run(root, "coupler", [
                {"coupler_z": x, "score": s, "loss": -s,
                 "noise_variance": .0001}
                for x, s in [(-.1, .2), (0., .5)]])
            paths = plot_bo_run(coupler)["coupler"]
            self.assertTrue(paths["evolution"].is_file())
            self.assertTrue(paths["fit"].is_file())

    def test_z_biases_have_separate_figures_and_missing_run_is_safe(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temp:
            root = Path(temp)
            rows = [
                {"z_amplitude": z, "drive_if_ghz": x, "S": s,
                 "loss": -s, "noise_variance": .0001}
                for z, x, s in [(-.1, -.15, .2), (.1, -.15, .5),
                                (-.1, 0., .8), (.1, 0., .3),
                                (-.1, .1, .3), (.1, .1, .2)]]
            run = self._run(root, "zfreq", rows, z_values=[-.1, .1])
            paths = plot_bo_run(run)
            self.assertEqual(set(paths), {"z000", "z001"})
            for prefix in paths:
                self.assertTrue(paths[prefix]["evolution"].is_file())
                self.assertTrue(paths[prefix]["fit"].is_file())
                self.assertIn(prefix, paths[prefix]["fit"].name)
            empty = new_run(root / "zfreq", "zfreq")
            save_json(empty / "data" / "manifest.json", {
                "jobs": [], "kind": "zfreq", "strategy": "bo"})
            self.assertEqual(plot_bo_run(empty), {})


if __name__ == "__main__":
    unittest.main()

