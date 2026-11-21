"""Shared run names, paths and durable checkpoints for all calibrations."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import sleep, time_ns
import csv
import json
import os
import tempfile

KINDS = {"pi": "PI", "zfreq": "ZF", "coupler": "CP"}


def new_run(root, kind):
    if kind not in KINDS:
        raise ValueError(f"Unknown experiment kind: {kind}")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + KINDS[kind]
        run = root / tag
        try:
            run.mkdir()
        except FileExistsError:
            sleep(.001)
            continue
        (run / "data").mkdir()
        (run / "fig").mkdir()
        # A notebook monitor is optional; bind this exact new run when present.
        from .progress import attach_run
        attach_run(run)
        return run
    raise FileExistsError("Could not create a unique run directory")


def resume_run(root, tag, kind):
    if kind not in KINDS:
        raise ValueError(f"Unknown experiment kind: {kind}")
    if (not isinstance(tag, str) or not tag.endswith("_" + KINDS[kind])
            or not tag or Path(tag).name != tag or tag in (".", "..")
            or "/" in tag or "\\" in tag or ":" in tag):
        raise ValueError("resume tag must be one run directory name for this experiment")
    root = Path(root).resolve()
    run = (root / tag).resolve()
    if (run.parent != root or not run.is_dir()
            or not (run / "data").is_dir() or not (run / "fig").is_dir()):
        raise FileNotFoundError(f"Run not found or incomplete: {run}")
    return run


def data_dir(run):
    path = Path(run) / "data"
    path.mkdir(exist_ok=True)
    return path


def fig_dir(run):
    path = Path(run) / "fig"
    path.mkdir(exist_ok=True)
    return path


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name == "manifest.json":
        value["checkpoint_revision"] = max(time_ns(), value.get("checkpoint_revision", 0) + 1)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as out:
        tmp = Path(out.name)
        out.write(payload)
        out.flush()
        os.fsync(out.fileno())
    for attempt in range(4):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt < 3:
                sleep(.1 * 2 ** attempt)
            elif path.name == "manifest.json":
                print(f"Manifest locked; latest checkpoint saved at {tmp}", flush=True)
                return
            else:
                raise


def load_manifest(run):
    directory = data_dir(run)
    paths = [directory / "manifest.json", directory / "manifest.json.tmp"]
    paths.extend(directory.glob("manifest.json.*.tmp"))
    snapshots = []
    for path in paths:
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict) or not isinstance(value.get("jobs"), list):
                    raise ValueError("invalid manifest structure")
                revision = int(value.get("checkpoint_revision", path.stat().st_mtime_ns))
                snapshots.append((revision, value))
            except (ValueError, OSError) as exc:
                raise ValueError(f"Unreadable checkpoint: {path}: {exc}") from exc
    if not snapshots:
        raise FileNotFoundError(f"No manifest checkpoint in {directory}")
    return max(snapshots, key=lambda item: item[0])[1]


def save_csv(path, rows, allow_pending=False):
    path = Path(path)
    if not rows:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                     dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as out:
        tmp = Path(out.name)
        writer = csv.DictWriter(out, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
        out.flush()
        os.fsync(out.fileno())
    for attempt in range(6):
        try:
            tmp.replace(path)
            return path
        except PermissionError:
            if attempt < 5:
                sleep(.2 * (attempt + 1))
    if not allow_pending:
        raise PermissionError(f"CSV cannot be replaced: {path}")
    pending = path.with_name(path.stem + ".pending" + path.suffix)
    tmp.replace(pending)
    return pending


@contextmanager
def run_lock(directory):
    """The OS releases this lock on process exit."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Run is already in use: {directory}") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
