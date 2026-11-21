"""Offline regressions: no lqcloud connection or real jobs."""

from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import pi, coupler, zfreq
from src.runs import new_run, resume_run, data_dir, fig_dir, load_manifest
from optimize.kernels import covariance
from optimize.gp import GP
from optimize.search import Search


class FakeBackend:
    name = "offline"
    use_async = True

    def __init__(self):
        self.jobs = {}
        self.calls = 0
        self.config = {"topology": {"coupling_map": [[9, 4]]}}

    def run_wave(self, plan, shots, result_format, readout_correction):
        assert result_format == "raw" and readout_correction is False
        self.calls += 1
        job_id = f"fake_{self.calls}"
        raw = {}
        for key, value in plan.items():
            if not key.endswith(".xy"):
                continue
            qid = key.split(".")[0]
            amplitude = float(np.max(np.abs(value[0])))
            rng = np.random.default_rng(self.calls + int(qid[1:]))
            centers = np.column_stack([np.full(shots, amplitude), np.zeros(shots)])
            raw[qid] = (centers + rng.normal(0, .02, (shots, 2)))[:, None, :]
        job = SimpleNamespace(job_id=job_id,
                              result=lambda **kw: SimpleNamespace(get_iq_data=lambda: raw),
                              status=lambda: "COMPLETED")
        self.jobs[job_id] = job
        return job


QCONF = {9: {"read_length": 64, "read_amp": .15, "df_read": .03, "df_10": .07},
         4: {"read_length": 64, "read_amp": .15, "df_read": .03, "df_10": .07}}


class ProjectTests(unittest.TestCase):
    def test_run_names_and_layout(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
            run = new_run(Path(tmp) / "pi", "pi")
            self.assertRegex(run.name, r"^\d{8}T\d{6}_\d{6}Z_PI$")
            self.assertTrue(data_dir(run).is_dir())
            self.assertTrue(fig_dir(run).is_dir())
            self.assertEqual(resume_run(run.parent, run.name, "pi"), run)
            with self.assertRaises(ValueError):
                resume_run(run.parent, "../outside", "pi")

    def test_pi_waveform_matches_original(self):
        origin = ROOT.parent / "LogicalQ" / "pi_amplitude_calibration.py"
        spec = importlib.util.spec_from_file_location("old_pi", origin)
        old = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = old
        spec.loader.exec_module(old)
        for amplitude in (0.0, .32, 1.1):
            settings = pi.PulseSettings(relaxation_ns=1200)
            before = old.build_plan(9, QCONF[9], amplitude, .07,
                                    old.PulseSettings(relaxation_ns=1200))
            after = pi.build_plan(9, QCONF[9], amplitude, .07, settings)
            for channel in before:
                for expected, actual in zip(before[channel], after[channel]):
                    np.testing.assert_allclose(expected, actual, rtol=0, atol=1e-6)

    def test_pi_scan_and_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
            backend = FakeBackend()
            run, candidates = pi.run_calibration(
                backend, {9: QCONF[9]}, {9: [0., .5, 1.]},
                Path(tmp) / "pi", shots=10,
                settings=pi.PulseSettings(relaxation_ns=1200),
                job_loader=lambda job_id: backend.jobs[job_id])
            count = backend.calls
            self.assertEqual(count, 4)
            self.assertEqual(load_manifest(run)["status"], "completed")
            self.assertTrue((run / "data" / "q9" / "baseline.npz").is_file())
            pi.plot_qubit(run, 9)
            self.assertTrue((run / "fig" / "q9" / "calibration.png").is_file())
            self.assertEqual(len(candidates), 1)
            again, _ = pi.run_calibration(
                backend, {9: QCONF[9]}, {9: [0., .5, 1.]},
                Path(tmp) / "pi", shots=10,
                settings=pi.PulseSettings(relaxation_ns=1200),
                run_mode="resume", resume_run_tag=run.name,
                job_loader=lambda job_id: backend.jobs[job_id])
            self.assertEqual(again, run)
            self.assertEqual(backend.calls, count)

    def test_z_waveform_matches_notebook(self):
        import ast
        notebook = json.loads((ROOT.parent / "LogicalQ" / "z_frequency_calibration.ipynb")
                              .read_text(encoding="utf-8"))
        cell = ast.parse("".join(notebook["cells"][3]["source"]))
        names = {"square_wave", "spectroscopy_wave", "readout_wave",
                 "build_spectroscopy_plan"}
        funcs = ast.Module(body=[n for n in cell.body
                                 if isinstance(n, ast.FunctionDef) and n.name in names],
                           type_ignores=[])
        settings = zfreq.ZSettings()
        old = {"np": np, "QCONF": QCONF, "TOTAL_LENGTH_NS": settings.total_ns,
               "Z_START_NS": settings.z_start_ns, "Z_DURATION_NS": settings.z_duration_ns,
               "XY_START_NS": settings.xy_start_ns, "READ_START_NS": settings.read_start_ns,
               "PI_CALIBRATIONS": {9: {"duration_ns": 40, "pi_amp": .8}}}
        exec(compile(funcs, "<old Z notebook functions>", "exec"), old)
        before = old["build_spectroscopy_plan"](9, .12, .07)
        after = zfreq.build_plan(9, QCONF[9],
                                 {"duration_ns": 40, "pi_amp": .8,
                                  "envelope": "raised_cosine"}, .12, .07)
        for channel in before:
            for expected, actual in zip(before[channel], after[channel]):
                np.testing.assert_allclose(expected, actual, rtol=0, atol=1e-6)

    def test_z_scan_resume_and_analysis(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
            backend = FakeBackend()
            cal = {9: {"pi_amp": 1., "duration_ns": 40,
                       "envelope": "raised_cosine", "source_run_tag": "test"}}
            args = (backend, {9: QCONF[9]}, cal, [-.1, .1],
                    {9: [.04, .07, .10]}, Path(tmp) / "zfreq")
            run = zfreq.run_calibration(*args, shots=10)
            self.assertEqual(backend.calls, 8)
            self.assertTrue((run / "data" / "observations.csv").exists())
            self.assertEqual(len(zfreq.analyze(run)), 2)
            zfreq.run_calibration(*args, shots=10, run_mode="resume",
                                  resume_run_tag=run.name,
                                  job_loader=lambda b, id: backend.jobs[id])
            self.assertEqual(backend.calls, 8)

    def test_z_config_export_from_complete_run(self):
        from src.config import publish_z
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
            backend = FakeBackend()
            cal = {9: {"pi_amp": 1., "duration_ns": 40,
                       "envelope": "raised_cosine", "source_run_tag": "test"}}
            run = zfreq.run_calibration(
                backend, {9: QCONF[9]}, cal, [-.1, .1],
                {9: [.04, .07, .10]}, Path(tmp) / "zfreq", shots=10)
            config = Path(tmp) / "config"
            target = publish_z(run, config, "offline")
            data = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(len(data["qubits"]["9"]["z_frequency_points"]), 2)
            self.assertTrue((config / "calibrations" / "qubit_frequency_vs_z.csv").is_file())

    def test_z_resumes_server_job_after_result_error(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
            backend = FakeBackend()
            submit = backend.run_wave
            def fail_once(*args, **kwargs):
                job = submit(*args, **kwargs)
                if backend.calls == 1:
                    good_result = job.result
                    calls = [0]
                    def first_result(**kw):
                        calls[0] += 1
                        if calls[0] == 1:
                            raise TimeoutError("transient")
                        return good_result(**kw)
                    job.result = first_result
                return job
            backend.run_wave = fail_once
            cal = {9: {"pi_amp": 1., "duration_ns": 40,
                       "envelope": "raised_cosine", "source_run_tag": "test"}}
            args = (backend, {9: QCONF[9]}, cal, [-.1, .1],
                    {9: [.04, .07, .10]}, Path(tmp) / "zfreq")
            with self.assertRaises(TimeoutError):
                zfreq.run_calibration(*args, shots=10)
            self.assertEqual(backend.calls, 1)
            run = next((Path(tmp) / "zfreq").iterdir())
            zfreq.run_calibration(*args, shots=10, run_mode="resume",
                                  resume_run_tag=run.name,
                                  job_loader=lambda b, id: backend.jobs[id])
            self.assertEqual(backend.calls, 8)
            self.assertEqual(load_manifest(run)["status"], "completed")

    def test_coupler_build_matches_original(self):
        origin = ROOT.parent / "LogicalQ" / "Coupler_calibration.py"
        spec = importlib.util.spec_from_file_location("old_coupler", origin)
        old = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = old
        spec.loader.exec_module(old)
        cal = {q: {"pi_amp": 1., "drive_if_ghz": .07, "duration_ns": 40,
                   "envelope": "raised_cosine"} for q in (9, 4)}
        args = (9, 4, "c9_4.z", QCONF, cal, 9, .12, .05, 24)
        a = old.build_swap_plan(*args, settings=old.CouplerPulseSettings(relaxation_ns=1000))
        b = coupler.build_swap_plan(*args, settings=coupler.CouplerPulseSettings(relaxation_ns=1000))
        for channel in a:
            for expected, actual in zip(a[channel], b[channel]):
                np.testing.assert_allclose(expected, actual, rtol=0, atol=1e-6)

    def test_pi_bayesian_loop_resume(self):
        from optimize.runner import run_pi
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
            backend = FakeBackend()
            settings = pi.PulseSettings(relaxation_ns=1200)
            kwargs = dict(steps=4, shots=10, settings=settings, seed=11,
                          initial_points=3, job_loader=lambda job_id: backend.jobs[job_id])
            run, rows = run_pi(backend, 9, QCONF[9], (.2, 1.), Path(tmp) / "pi", **kwargs)
            self.assertEqual(len(rows), 4)
            self.assertEqual(backend.calls, 5)  # one XY-off reference
            self.assertEqual(load_manifest(run)["strategy"], "bo")
            self.assertTrue((run / "data" / "observations.csv").exists())
            resumed, rows_again = run_pi(
                backend, 9, QCONF[9], (.2, 1.), Path(tmp) / "pi",
                run_mode="resume", resume_run_tag=run.name, **kwargs)
            self.assertEqual(run, resumed)
            self.assertEqual(len(rows_again), 4)
            self.assertEqual(backend.calls, 5)

    def test_coupler_scan_and_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
            directory = Path(tmp)
            pi_file = directory / "pi.json"
            pi_file.write_text(json.dumps({
                "schema_version": 1, "backend": "offline",
                "qubits": {str(q): {
                    "pi_amp": 1., "drive_if_ghz": .07, "pulse_settings": {
                        "duration_ns": 40, "envelope": "raised_cosine"},
                    "iq_centers": {"state_0": {"I": 0., "Q": 0.},
                                   "state_1": {"I": 1., "Q": 0.}}}
                    for q in (9, 4)}}), encoding="utf-8")
            backend = FakeBackend()
            args = (backend, QCONF, 9, 4, "c9_4.z", 9,
                    [-.1, .1], .12, list(range(20, 52, 4)),
                    pi_file, directory / "coupler")
            run = coupler.run_calibration(
                *args, settings=coupler.CouplerPulseSettings(relaxation_ns=1000), shots=10)
            self.assertEqual(backend.calls, 20)  # 4 joint references + 16 points
            self.assertTrue((run / "data" / "joint_probabilities.csv").is_file())
            self.assertEqual(load_manifest(run)["status"], "completed")
            coupler.analyze_run(run)
            self.assertTrue((run / "fig" / "coupler_calibration.png").is_file())
            coupler.run_calibration(
                *args, settings=coupler.CouplerPulseSettings(relaxation_ns=1000),
                shots=10, run_mode="resume", resume_run_tag=run.name)
            self.assertEqual(backend.calls, 20)

    def test_pi_publish_is_explicit(self):
        from src.config import publish_pi
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:
            backend = FakeBackend()
            run, _ = pi.run_calibration(
                backend, {9: QCONF[9]}, {9: [0., .5, 1.]},
                Path(tmp) / "pi", shots=10,
                settings=pi.PulseSettings(relaxation_ns=1200))
            target = Path(tmp) / "config" / "calibrations" / "pi_pulse_amplitudes.json"
            self.assertFalse(target.exists())
            self.assertEqual(publish_pi(run, Path(tmp) / "config"), target)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["backend"], "offline")

    def test_gp_kernels_and_search(self):
        x = np.linspace(0, 1, 12)[:, None]
        y = (x[:, 0] - .35) ** 2
        for kernel in ("rbf", "matern32", "matern52", "rq"):
            k = covariance(x, x, kind=kernel, length_scales=.3)
            self.assertTrue(np.all(np.linalg.eigvalsh(k) > -1e-9))
            gp = GP(kernel=kernel).fit(x, y, 1e-5)
            mean, std = gp.predict(x)
            self.assertEqual(mean.shape, (12,))
            self.assertTrue(np.all(np.isfinite(std)))
        search = Search([[0, 1]], seed=7, initial_points=3)
        for v in (.1, .5, .9):
            search.tell([v], (v-.35)**2, 1e-5)
        value = search.ask()
        self.assertTrue(0 <= value[0] <= 1)


if __name__ == "__main__":
    unittest.main()
