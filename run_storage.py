"""Archive each run's inputs, artifacts, and trace onto the persistent volume.

The local disk (``/tmp`` and ``/root`` share one filesystem on the Studio
image) is already holding the GPU runtimes that keep workers off the
intermittent volume. Run directories from ``mkdtemp`` used to land there and
were never deleted. ``/mnt/workspace`` is the persistent volume.

That volume has a history of intermittent ``OSError: [Errno 5]`` on small
files, so every write here must be allowed to fail: archival failure is
logged and the local path is kept; the run itself never fails because an
archive write did.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


_DEFAULT_ARCHIVE_ROOT = "/mnt/workspace/runs"
_ALLOWED_SWEEP_PREFIXES = ("rsmem_interactive_", "rsmem_cpu_")
_FORBIDDEN_PATH_PARTS = frozenset({"rsmem-local"})
_DOWNLOADS_DIR_NAME = "rsmem_downloads"
DEFAULT_LOCAL_MAX_AGE_S = 8 * 3600


def archive_root() -> Path | None:
    """Return the persistent archive root, or None when archiving is disabled.

    ``RSMEM_RUN_ARCHIVE_ROOT`` defaults to ``/mnt/workspace/runs``. An empty
    value disables archiving entirely, which is what tests and a laptop
    checkout without the volume should do.
    """
    raw = os.getenv("RSMEM_RUN_ARCHIVE_ROOT", _DEFAULT_ARCHIVE_ROOT).strip()
    if not raw:
        return None
    return Path(raw)


def _as_dir(archive_dir: Path | str | None) -> Path | None:
    if archive_dir is None:
        return None
    text = str(archive_dir).strip()
    if not text:
        return None
    return Path(text)


def new_run_dir(run_id: str) -> Path | None:
    """Create ``<root>/<YYYYMMDD>/<run_id>/{inputs,outputs}`` or return None."""
    root = archive_root()
    if root is None:
        return None
    try:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = root / day / run_id
        (path / "inputs").mkdir(parents=True, exist_ok=True)
        (path / "outputs").mkdir(parents=True, exist_ok=True)
        return path
    except OSError as exc:
        print(
            f"[run-storage] creating archive directory under {root} failed ({exc}); "
            "keeping files on the local disk",
            flush=True,
        )
        return None


def _is_under(path: Path, root: Path) -> bool:
    try:
        return path.is_relative_to(root)
    except (OSError, ValueError):
        return False


def _unique_destination(directory: Path, name: str) -> Path:
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stem = Path(name).stem
    suffix = Path(name).suffix
    index = 2
    while True:
        candidate = directory / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _copy_into(directory: Path, paths: list[str]) -> list[str]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        dest_root = directory.resolve()
    except OSError as exc:
        print(
            f"[run-storage] preparing {directory} failed ({exc}); keeping the local copies",
            flush=True,
        )
        return list(paths)
    copied: list[str] = []
    for item in paths:
        source = Path(item)
        try:
            resolved = source.resolve()
        except OSError as exc:
            print(
                f"[run-storage] resolving {source} failed ({exc}); keeping the local copy",
                flush=True,
            )
            copied.append(item)
            continue
        if resolved.is_file() and _is_under(resolved, dest_root):
            copied.append(str(resolved))
            continue
        if not resolved.is_file():
            copied.append(item)
            continue
        try:
            dest = dest_root / resolved.name
            if dest.exists():
                try:
                    if dest.stat().st_size == resolved.stat().st_size:
                        copied.append(str(dest.resolve()))
                        continue
                except OSError:
                    pass
            dest = _unique_destination(dest_root, resolved.name)
            shutil.copy2(resolved, dest)
            copied.append(str(dest.resolve()))
        except OSError as exc:
            print(
                f"[run-storage] copying {resolved} to {dest_root} failed ({exc}); "
                "keeping the local copy",
                flush=True,
            )
            copied.append(item)
    return copied


def archive_inputs(archive_dir: Path | str | None, paths: list[str]) -> list[str]:
    """Copy input rasters into ``inputs/``. Failures keep the original path."""
    directory = _as_dir(archive_dir)
    if directory is None:
        return list(paths)
    return _copy_into(directory / "inputs", paths)


def archive_outputs(archive_dir: Path | str | None, paths: list[str]) -> list[str]:
    """Copy generated artifacts into ``outputs/``. Same-name files are not overwritten."""
    directory = _as_dir(archive_dir)
    if directory is None:
        return list(paths)
    return _copy_into(directory / "outputs", paths)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_trace(archive_dir: Path | str | None, payload: dict) -> str | None:
    """Write ``trace.json``. Returns the path, or None when archiving is off or fails."""
    directory = _as_dir(archive_dir)
    if directory is None:
        return None
    path = directory / "trace.json"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(path, payload)
        return str(path)
    except OSError as exc:
        print(
            f"[run-storage] writing {path} failed ({exc}); keeping the local copy",
            flush=True,
        )
        return None


def downloads_root() -> Path:
    return Path(tempfile.gettempdir()) / _DOWNLOADS_DIR_NAME


def _rewrite_hidden_paths(text: str, roots: list[str]) -> str:
    rewritten = text
    for root in sorted(set(roots), key=len, reverse=True):
        root = root.rstrip("/")
        if not root:
            continue
        pattern = re.compile(re.escape(root) + r"(?:/[^\s\"',;\]\}]+)*")

        def replace(match: re.Match[str], *, _root: str = root) -> str:
            matched = match.group(0).rstrip("/")
            if matched == _root:
                return Path(_root).name
            return Path(matched).name

        rewritten = pattern.sub(replace, rewritten)
    return rewritten


def sanitize_trace_for_download(
    payload: dict,
    *,
    archive_dir: Path | str | None = None,
    run_dir: Path | str | None = None,
) -> dict:
    """Strip directory keys and rewrite known roots to filenames for visitors."""
    data = copy.deepcopy(payload)
    for key in ("archive_dir", "run_dir", "checkpoint_path"):
        data.pop(key, None)
    roots: list[str] = []
    for candidate in (archive_dir, run_dir, archive_root(), downloads_root()):
        if candidate is None:
            continue
        try:
            roots.append(str(Path(candidate).resolve()))
        except OSError:
            roots.append(str(candidate))

    def walk(value):
        if isinstance(value, str):
            return _rewrite_hidden_paths(value, roots)
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, tuple):
            return [walk(item) for item in value]
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        return value

    return walk(data)


def publish_trace(run_id: str | None, payload: dict) -> str | None:
    """Write the visitor-facing trace JSON under the downloads directory."""
    name = (run_id or "trace").replace("/", "_")
    dest = downloads_root() / f"{name}.json"
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        _write_json(dest, payload)
        return str(dest.resolve())
    except OSError as exc:
        print(f"[run-storage] writing visitor trace {dest} failed ({exc})", flush=True)
        return None


def publish_downloads(run_id: str | None, paths: list[str]) -> list[str]:
    """Hard-link (or copy) artifacts into the visitor downloads directory."""
    name = (run_id or "run").replace("/", "_")
    dest_dir = downloads_root() / name
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_root = dest_dir.resolve()
    except OSError as exc:
        print(f"[run-storage] preparing downloads dir failed ({exc})", flush=True)
        return list(paths)
    published: list[str] = []
    for item in paths:
        source = Path(item)
        try:
            resolved = source.resolve()
        except OSError:
            published.append(item)
            continue
        if not resolved.is_file():
            published.append(item)
            continue
        dest = dest_root / resolved.name
        try:
            if dest.exists():
                published.append(str(dest.resolve()))
                continue
            try:
                os.link(resolved, dest)
            except OSError:
                shutil.copy2(resolved, dest)
            published.append(str(dest.resolve()))
        except OSError as exc:
            print(
                f"[run-storage] publishing {resolved} failed ({exc}); keeping the local copy",
                flush=True,
            )
            published.append(item)
    return published


def _local_run_max_age_s() -> float:
    raw = os.getenv("RSMEM_LOCAL_RUN_MAX_AGE_S", "").strip()
    if not raw:
        return float(DEFAULT_LOCAL_MAX_AGE_S)
    try:
        return max(0.0, float(raw))
    except ValueError:
        return float(DEFAULT_LOCAL_MAX_AGE_S)


def sweep_local_runs(
    prefixes: tuple[str, ...] = _ALLOWED_SWEEP_PREFIXES,
    max_age_s: float | None = None,
    keep: set[str] | None = None,
    parent: Path | str | None = None,
) -> int:
    """Delete aged ``rsmem_interactive_*`` / ``rsmem_cpu_*`` directories.

    Only directories whose name starts with one of ``prefixes``, whose
    resolved parent is exactly ``parent`` (default: the process temp dir,
    ``/tmp`` on the Studio image), and whose mtime is older than
    ``max_age_s`` (default 8 hours) are removed. Symlinks, anything under
    ``rsmem-local``, and paths in ``keep`` are skipped. This function must
    never be pointed at ``/root/rsmem-local``.
    """
    parent_path = Path(parent) if parent is not None else Path(tempfile.gettempdir())
    try:
        parent_path = parent_path.resolve()
    except OSError as exc:
        print(f"[run-storage] cannot resolve sweep parent {parent_path} ({exc})", flush=True)
        return 0
    if "rsmem-local" in parent_path.parts:
        print(
            f"[run-storage] refusing to sweep {parent_path}: it is under rsmem-local",
            flush=True,
        )
        return 0
    age = _local_run_max_age_s() if max_age_s is None else max_age_s
    kept = set()
    for item in keep or set():
        try:
            kept.add(str(Path(item).resolve()))
        except OSError:
            kept.add(str(item))
    try:
        entries = list(parent_path.iterdir())
    except OSError as exc:
        print(f"[run-storage] listing {parent_path} failed ({exc})", flush=True)
        return 0
    now = time.time()
    deleted = 0
    for path in entries:
        if path.is_symlink() or not path.is_dir():
            continue
        if not path.name.startswith(prefixes):
            continue
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved.parent != parent_path:
            continue
        if any(part in _FORBIDDEN_PATH_PARTS for part in resolved.parts):
            continue
        if not _is_under(resolved, parent_path):
            continue
        if str(resolved) in kept:
            continue
        try:
            if now - path.stat().st_mtime < age:
                continue
        except OSError:
            continue
        # Last-line defence: never delete outside the sweep parent, and never
        # delete a directory whose name we did not match above.
        if resolved.parent != parent_path:
            continue
        try:
            shutil.rmtree(resolved)
            deleted += 1
        except OSError as exc:
            print(f"[run-storage] deleting {resolved} failed ({exc})", flush=True)
    return deleted


def sweep_downloads(max_age_s: float | None = None) -> int:
    """Delete aged visitor-facing files under ``<tempdir>/rsmem_downloads``.

    The directory itself is kept. Symlinks and anything under ``rsmem-local``
    are skipped, matching :func:`sweep_local_runs`.
    """
    parent_path = downloads_root()
    try:
        parent_path = parent_path.resolve()
    except OSError as exc:
        print(f"[run-storage] cannot resolve downloads dir {parent_path} ({exc})", flush=True)
        return 0
    if "rsmem-local" in parent_path.parts:
        print(
            f"[run-storage] refusing to sweep {parent_path}: it is under rsmem-local",
            flush=True,
        )
        return 0
    try:
        parent_path.mkdir(parents=True, exist_ok=True)
        entries = list(parent_path.iterdir())
    except OSError as exc:
        print(f"[run-storage] listing {parent_path} failed ({exc})", flush=True)
        return 0
    age = _local_run_max_age_s() if max_age_s is None else max_age_s
    now = time.time()
    deleted = 0
    for path in entries:
        try:
            if path.is_symlink():
                continue
            resolved = path.resolve()
        except OSError:
            continue
        if resolved.parent != parent_path:
            continue
        if any(part in _FORBIDDEN_PATH_PARTS for part in resolved.parts):
            continue
        try:
            if now - path.stat().st_mtime < age:
                continue
        except OSError:
            continue
        try:
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()
            deleted += 1
        except OSError as exc:
            print(f"[run-storage] deleting {resolved} failed ({exc})", flush=True)
    return deleted
