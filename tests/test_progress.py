"""Progress monitor checks use checkpoint files, never QPU jobs."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.progress import checkpoint_progress, watch_run
from src.runs import load_manifest, new_run, save_json


class ProgressTests(unittest.TestCase):
    def test_retry_and_resume_keep_the_right_completed_count(self):
        jobs = [
            {"label": "base", "status": "failed", "job_id": "old"},
            {"label": "base", "status": "saved", "job_id": "recovered"},
            {"label": "point_0", "status": "submitted", "job_id": "waiting"},
        ]
        self.assertEqual(checkpoint_progress({"jobs": jobs}, 3)[0], 1)
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temp:
            root = Path(temp) / "pi"
            with redirect_stdout(io.StringIO()):
                with watch_run(root, 3, interval=.02) as monitor:
                    run = new_run(root, "pi")
                    self.assertEqual(monitor.run_dir, run)
                    save_json(run / "data" / "manifest.json", {
                        "jobs": jobs, "status": "running"})
                    done, waiting, phase, status = monitor._state()
                    self.assertEqual((done, waiting["job_id"], status),
                                     (1, "waiting", "running"))
                    self.assertIn("point_0", phase)
                with watch_run(root, 3, run_mode="resume",
                               resume_run_tag=run.name, interval=.02) as resumed:
                    self.assertEqual(resumed._baseline, 1)
                    self.assertEqual(resumed.run_dir, run)
                    jobs[2]["status"] = "saved"
                    jobs.append({"label": "point_1", "status": "completed"})
                    save_json(run / "data" / "manifest.json", {
                        "jobs": jobs, "status": "completed"})
                    self.assertEqual(resumed._state()[0], 3)
            self.assertEqual(load_manifest(run)["status"], "completed")

    def test_monitor_never_swallows_experiment_error(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temp:
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "offline failure"):
                    with watch_run(Path(temp) / "zfreq", 3, interval=.02):
                        raise RuntimeError("offline failure")


if __name__ == "__main__":
    unittest.main()

