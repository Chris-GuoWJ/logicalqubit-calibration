"""Notebook progress display based on durable QPU job checkpoints.

The monitor reads manifests only. It never submits jobs or alters run state.
"""

from contextvars import ContextVar
from html import escape
from pathlib import Path
from threading import Event, Thread
from time import monotonic

from .runs import load_manifest, resume_run


_ACTIVE_MONITOR = ContextVar("lq_run_progress", default=None)
_FINISHED = {"completed", "saved"}


def attach_run(run_dir):
    """Bind a newly created run to the monitor active in this execution."""
    monitor = _ACTIVE_MONITOR.get()
    if monitor is not None:
        monitor.run_dir = Path(run_dir)


def checkpoint_progress(manifest, total_jobs):
    """Count distinct saved job labels; retries never inflate completion."""
    latest = {}
    for entry in manifest.get("jobs", []):
        label = entry.get("label")
        if label:
            latest[(entry.get("qubit"), label)] = entry
    completed = sum(entry.get("status") in _FINISHED for entry in latest.values())
    waiting = next((entry for entry in reversed(list(latest.values()))
                    if entry.get("status") not in _FINISHED
                    and entry.get("status") not in ("failed", "cancelled")), None)
    return min(completed, total_jobs), waiting


def _clock(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


class RunProgress:
    """Live notebook display for one new or resumed calibration run."""

    def __init__(self, root, total_jobs, *, run_mode="new", resume_run_tag="",
                 title="真机标定", interval=1.0):
        if isinstance(total_jobs, bool) or not isinstance(total_jobs, int) or total_jobs < 1:
            raise ValueError("total_jobs must be a positive integer")
        if interval <= 0:
            raise ValueError("interval must be positive")
        self.root = Path(root).resolve()
        self.total_jobs = total_jobs
        self.title = title
        self.interval = interval
        self.run_mode = run_mode
        self.run_dir = None
        if run_mode == "resume":
            self.run_dir = resume_run(self.root, resume_run_tag, self.root.name)
        elif run_mode != "new" or resume_run_tag:
            raise ValueError("Use run_mode=new without a tag or resume with a run tag")
        self._stop = Event()
        self._handle = None
        self._started = None
        self._baseline = 0
        self._last_print = None
        self._last_print_time = 0

    def _state(self):
        if self.run_dir is None:
            return 0, None, "准备连接或校验参数", None
        try:
            manifest = load_manifest(self.run_dir)
        except (FileNotFoundError, OSError, ValueError):
            return 0, None, "等待首次检查点", None
        done, waiting = checkpoint_progress(manifest, self.total_jobs)
        status = manifest.get("status", "running")
        if status == "completed":
            phase = "实验完成"
        elif status in ("failed", "interrupted"):
            phase = "实验失败" if status == "failed" else "实验中断"
        elif waiting:
            label = str(waiting.get("label", "任务"))
            job_id = waiting.get("job_id")
            phase = (f"等待任务 {label} ({job_id})" if job_id
                     else f"提交任务 {label}")
        elif done >= self.total_jobs:
            phase = "采集完成，整理结果中"
        else:
            phase = "准备下一项测量"
        return done, waiting, phase, status

    def _render(self, ending=None):
        done, waiting, phase, status = self._state()
        elapsed = monotonic() - self._started
        if ending == "failed":
            phase = "运行中断或失败；请检查异常和原运行记录"
        elif ending == "finished" and status == "completed":
            phase = "实验完成"
        elif ending == "finished" and self.run_dir is None:
            phase = "没有创建实验目录"
        new_done = done - self._baseline
        if done >= self.total_jobs:
            eta = "00:00:00" if status == "completed" else "整理中"
        elif new_done > 0:
            eta = _clock(elapsed * (self.total_jobs - done) / new_done)
        else:
            eta = "计算中"
        percent = 100 * done / self.total_jobs
        bar = "█" * int(20 * done / self.total_jobs) + "░" * (20 - int(20 * done / self.total_jobs))
        tag = self.run_dir.name if self.run_dir else "尚未创建"
        message = (f"{self.title} [{bar}] {done}/{self.total_jobs} ({percent:.1f}%) | "
                   f"已用 {_clock(elapsed)} | 预计剩余 {eta} | {phase} | RUN_TAG={tag}")
        if self._handle is not None:
            from IPython.display import HTML
            self._handle.update(HTML(
                f"<div style='font-family:system-ui;max-width:850px'>"
                f"<strong>{escape(self.title)}</strong>　"
                f"{done}/{self.total_jobs}（{percent:.1f}%）<br>"
                f"<progress value='{done}' max='{self.total_jobs}' "
                f"style='width:100%;height:18px'></progress><br>"
                f"已用 {_clock(elapsed)}　预计剩余 {eta}<br>"
                f"{escape(phase)}<br>"
                f"<small>RUN_TAG: {escape(tag)}</small></div>"))
        else:
            key = (done, phase, tag, ending)
            now = monotonic()
            if key != self._last_print or now - self._last_print_time >= 5:
                print(message, flush=True)
                self._last_print, self._last_print_time = key, now

    def _poll(self):
        while not self._stop.wait(self.interval):
            try:
                self._render()
            except Exception:
                # Checkpoint writes can temporarily race a read. The experiment
                # must not fail because its optional display could not update.
                continue

    def __enter__(self):
        if self._started is not None:
            raise RuntimeError("A progress monitor can only be started once")
        self._started = monotonic()
        if self.run_dir is not None:
            self._baseline = self._state()[0]
        self._token = _ACTIVE_MONITOR.set(self)
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display
            ip = get_ipython()
            if ip is not None and getattr(ip, "kernel", None) is not None:
                self._handle = display(HTML("准备开始真机标定…"), display_id=True)
        except ImportError:
            pass
        self._render()
        self._thread = Thread(target=self._poll, daemon=True, name="lq-run-progress")
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join(timeout=self.interval + 1)
        try:
            self._render(ending="failed" if exc_type else "finished")
        except Exception:
            pass  # Never mask the hardware exception with a display error.
        finally:
            _ACTIVE_MONITOR.reset(self._token)
        return False


def watch_run(root, total_jobs, *, run_mode="new", resume_run_tag="",
              title="真机标定", interval=1.0):
    """Return a context manager for a hardware call in execute/*.ipynb."""
    return RunProgress(root, total_jobs, run_mode=run_mode,
                       resume_run_tag=resume_run_tag, title=title,
                       interval=interval)

