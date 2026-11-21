"""Offline BO integration: synthetic raw IQ and job recovery, never LQCloud."""

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_cli import BudgetBackend, preflight, read_spec
from optimize.runner import run_zfreq, run_coupler
from src.pi import PulseSettings
from src.zfreq import ZSettings
from src.coupler import CouplerPulseSettings
from src.runs import load_manifest


QCONF = {q: {"read_length": 64, "read_amp": .15,
             "df_read": .03, "df_10": .07} for q in (9, 4)}


class SpectroscopyBackend:
    name = "offline"

    def __init__(self, interrupt_at=None):
        self.config = {}
        self.jobs = {}
        self.calls = 0
        self.interrupt_at = interrupt_at

    def run_wave(self, plan, shots, result_format, readout_correction):
        self.calls += 1
        xy = plan["q9.xy"][0]
        z = plan["q9.z"][0]
        z_amp = float(z[np.argmax(np.abs(z))])
        idx = np.flatnonzero(np.abs(xy) > .05)
        if len(idx) < 2:
            excited = 0.
        else:
            f = -np.angle(xy[idx[1]] / xy[idx[0]]) / (2 * np.pi)
            excited = .75 * np.exp(-((f - (.02 + .25 * z_amp)) / .03) ** 2)
        rng = np.random.default_rng(100 + self.calls)
        samples = (rng.random(shots) < excited).astype(float)
        iq = np.column_stack([samples + rng.normal(0, .01, shots),
                              rng.normal(0, .01, shots)])[:, None, :]
        value = SimpleNamespace(get_iq_data=lambda: {"q9": iq})
        job = SimpleNamespace(job_id=f"z{self.calls}", status=lambda: "COMPLETED",
                              result=lambda **kw: value)
        if self.calls == self.interrupt_at:
            first = [True]
            def transient(**kwargs):
                if first[0]:
                    first[0] = False
                    raise TimeoutError("synthetic result timeout")
                return value
            job.result = transient
        self.jobs[job.job_id] = job
        return job


class SwapBackend:
    name = "offline"

    def __init__(self, times, interrupt_at=None):
        self.times = times
        self.config = {"topology": {"coupling_map": [[9, 4]]}}
        self.jobs = {}
        self.calls = 0
        self.interrupt_at = interrupt_at

    def run_wave(self, plan, shots, result_format, readout_correction):
        self.calls += 1
        rng = np.random.default_rng(600 + self.calls)
        if self.calls <= 4:
            bit_a = np.max(np.abs(plan["q9.xy"][0])) > .1
            bit_b = np.max(np.abs(plan["q4.xy"][0])) > .1
            a = np.full(shots, int(bit_a), dtype=float)
            b = np.full(shots, int(bit_b), dtype=float)
        else:
            ci = self.calls - 5
            t = self.times[ci % len(self.times)]
            wave = plan["c9_4.z"][0]
            amp = float(wave[np.argmax(np.abs(wave))])
            strength = .12 + .74 * np.exp(-((amp - .055) / .08) ** 2)
            p01 = strength * (1 - np.cos(4 * np.pi * .006 * t)) / 2
            transferred = rng.random(shots) < p01
            a = (~transferred).astype(float)
            b = transferred.astype(float)
        a = np.column_stack([a + rng.normal(0, .02, shots),
                             rng.normal(0, .02, shots)])[:, None, :]
        b = np.column_stack([b + rng.normal(0, .02, shots),
                             rng.normal(0, .02, shots)])[:, None, :]
        value = SimpleNamespace(get_iq_data=lambda: {"q9": a, "q4": b})
        job = SimpleNamespace(job_id=f"c{self.calls}", status=lambda: "COMPLETED",
                              result=lambda **kw: value)
        if self.calls == self.interrupt_at:
            first = [True]
            def transient(**kwargs):
                if first[0]:
                    first[0] = False
                    raise TimeoutError("synthetic result timeout")
                return value
            job.result = transient
        self.jobs[job.job_id] = job
        return job


class AdaptiveTests(unittest.TestCase):
    def test_specs_and_hard_budget(self):
        z = preflight(read_spec(ROOT / "config/plans/zfreq_bo_example.json"))
        c = preflight(read_spec(ROOT / "config/plans/coupler_bo_example.json"))
        self.assertEqual((z["jobs"], c["jobs"]), (35, 100))
        class Backend:
            def run_wave(self, *args, **kwargs):
                return "submitted"
        budget = BudgetBackend(Backend(), 1, 12)
        self.assertEqual(budget.run_wave({}, shots=12), "submitted")
        with self.assertRaises(RuntimeError):
            budget.run_wave({}, shots=12)

    def test_z_bo_and_resume_after_result_timeout(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
            backend = SpectroscopyBackend(interrupt_at=3)
            kwargs = dict(settings=ZSettings(), shots=40, seed=7,
                          initial_points=2,
                          job_loader=lambda b, job_id: b.jobs[job_id])
            args = (backend, 9, QCONF[9],
                    {"pi_amp": .8, "duration_ns": 40,
                     "envelope": "raised_cosine", "source_run_tag": "test"},
                    [-.1, .1], [-.08, .10], Path(tmp) / "zfreq", 3)
            with self.assertRaises(TimeoutError):
                run_zfreq(*args, **kwargs)
            run = next((Path(tmp) / "zfreq").iterdir())
            assert backend.calls == 3
            completed, candidates = run_zfreq(*args, run_mode="resume",
                                                resume_run_tag=run.name, **kwargs)
            self.assertEqual(completed, run)
            self.assertEqual(backend.calls, 8)
            self.assertEqual(len(candidates), 2)
            self.assertEqual(load_manifest(run)["status"], "completed")
            self.assertTrue((run / "data" / "frequency_candidates.csv").exists())
            from src.zfreq import analyze
            with self.assertRaisesRegex(ValueError, "independent spectral-peak"):
                analyze(run)
            again, _ = run_zfreq(*args, run_mode="resume",
                                  resume_run_tag=run.name, **kwargs)
            self.assertEqual(again, run)
            self.assertEqual(backend.calls, 8)

    def test_coupler_bo_and_resume_after_result_timeout(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
            root = Path(tmp)
            pi_file = root / "pi.json"
            pi_file.write_text(json.dumps({
                "schema_version": 1, "backend": "offline",
                "qubits": {str(q): {
                    "pi_amp": 1., "drive_if_ghz": .07,
                    "pulse_settings": {"duration_ns": 40,
                                       "envelope": "raised_cosine"},
                    "iq_centers": {"state_0": {"I": 0., "Q": 0.},
                                   "state_1": {"I": 1., "Q": 0.}},
                    "source_run_tag": "test"} for q in (9, 4)}}),
                encoding="utf-8")
            times = list(range(20, 84, 8))
            backend = SwapBackend(times, interrupt_at=7)
            args = (backend, QCONF, 9, 4, 9, .15, times,
                    [-.15, .15], pi_file, root / "coupler", 3)
            kwargs = dict(settings=CouplerPulseSettings(relaxation_ns=1000),
                          shots=40, initial_points=2, seed=8,
                          job_loader=lambda b, job_id: b.jobs[job_id])
            with self.assertRaises(TimeoutError):
                run_coupler(*args, **kwargs)
            run = next((root / "coupler").iterdir())
            self.assertEqual(backend.calls, 7)
            completed, rows = run_coupler(*args, run_mode="resume",
                                           resume_run_tag=run.name, **kwargs)
            self.assertEqual(completed, run)
            self.assertEqual(backend.calls, 28)
            self.assertEqual(len(rows), 3)
            self.assertEqual(load_manifest(run)["status"], "completed")
            self.assertTrue((run / "data" / "joint_probabilities.csv").is_file())
            import pandas as pd
            saved = pd.read_csv(run / "data" / "observations.csv")
            self.assertEqual(len(saved), len(rows))
            np.testing.assert_allclose(saved["score"], [row["score"] for row in rows])
            np.testing.assert_allclose(saved["noise_variance"],
                                       [row["noise_variance"] for row in rows])
            self.assertEqual(load_manifest(run)["observation_csv"], "observations.csv")
            again, _ = run_coupler(*args, run_mode="resume",
                                    resume_run_tag=run.name, **kwargs)
            self.assertEqual(again, run)
            self.assertEqual(backend.calls, 28)
            pi_file.write_text(pi_file.read_text(encoding="utf-8") + " ",
                               encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "pi_calibration_sha256"):
                run_coupler(*args, run_mode="resume",
                            resume_run_tag=run.name, **kwargs)


    def test_agent_cli_dispatches_both_bo_kinds_without_real_sdk(self):
        from unittest.mock import patch
        import agent_cli

        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
            root = Path(tmp)
            calibrations = root / "config" / "calibrations"
            calibrations.mkdir(parents=True)
            pi_data = {"schema_version": 1, "backend": "offline",
                       "qubits": {str(q): {
                           "pi_amp": .8, "drive_if_ghz": .07,
                           "pulse_settings": {"duration_ns": 40,
                                              "envelope": "raised_cosine"},
                           "iq_centers": {"state_0": {"I": 0., "Q": 0.},
                                          "state_1": {"I": 1., "Q": 0.}},
                           "source_run_tag": "test"} for q in (9, 4)}}
            (calibrations / "pi_pulse_amplitudes.json").write_text(
                json.dumps(pi_data), encoding="utf-8")
            (calibrations / "qubit_frequency_vs_z.csv").write_text(
                "qubit,z_amplitude,drive_if_ghz,valid\n"
                "9,0.0,0.08,True\n9,0.2,0.12,True\n"
                "4,0.0,0.11,True\n4,0.2,0.11,True\n",
                encoding="utf-8")
            times = list(range(20, 84, 8))
            backends = {"z": SpectroscopyBackend(),
                        "c": SwapBackend(times)}
            for name, backend in backends.items():
                backend.name = name
                backend.config["properties"] = {
                    "qubit_metrics": [{"id": q, **QCONF[q]} for q in (9, 4)]}

            class Provider:
                def get_backend(self, name):
                    return backends[name]

            import sys
            sdk = SimpleNamespace(LQCloudProvider=Provider)
            common = {"steps": 3, "initial_points": 2, "shots": 40,
                      "seed": 3, "run_mode": "new", "resume_run_tag": ""}
            z_spec = dict(common, kind="zfreq_bo", backend="z",
                          qubit=9, z_values=[-.1, .1], bounds=[-.08, .10],
                          max_jobs=8, max_shots=320)
            c_spec = dict(common, kind="coupler_bo", backend="c",
                          qubits=[9, 4], bounds=[-.15, .15], times_ns=times,
                          reference_z=0., tuning_z_range=[0., .2],
                          tuning_z_hint=.15,
                          settings={"relaxation_ns": 1000},
                          max_jobs=28, max_shots=1120)
            # Calibration sources must match each backend in the test.
            with patch.dict(sys.modules, {"lqcloud": sdk}), patch.object(agent_cli, "ROOT", root):
                for spec in (z_spec, c_spec):
                    pi_data["backend"] = spec["backend"]
                    (calibrations / "pi_pulse_amplitudes.json").write_text(
                        json.dumps(pi_data), encoding="utf-8")
                    outcome = agent_cli.run(spec)
                    self.assertEqual(outcome["run_status"], "completed")
                    self.assertEqual(len(load_manifest(outcome["run_dir"])["jobs"]),
                                     spec["max_jobs"])


if __name__ == "__main__":
    unittest.main()
