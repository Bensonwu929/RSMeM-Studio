"""Install persistent xGPU runtimes exclusively from offline conda-pack archives."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ENV_SCHEMA = 2
OFFLINE_ENV_ARCHIVES = {
    "sam2": "offline/sam2-py310-cu121-v1.tar.gz",
    "remoteclip": "offline/remoteclip-py310-cu121-v1.tar.gz",
    "changeos": "offline/changeos-py310-cu121-v1.tar.gz",
    "bright": "offline/bright-py310-cu121-v1.tar.gz",
    "oea-e1": "offline/oea-e1-py310-cu124-v1.tar.gz",
    "oea-e3": "offline/oea-e3-py310-cu124-v1.tar.gz",
}

# Expanded sizes measured on the QY_11 deployment on 2026-09-06 (see
# docs/NEXT_AGENT_HANDOFF.md), used only to pick which runtimes fit the local
# disk before extraction starts. A stale estimate never corrupts anything --
# _runtime_is_ready and the archive checksum are what actually gate
# correctness -- it only risks placing the wrong runtime on the wrong disk.
RUNTIME_EXPANDED_GIB = {
    "oea-e1": 10.50,
    "oea-e3": 7.11,
    "sam2": 9.09,
    "changeos": 8.99,
    "remoteclip": 8.99,
    "bright": 5.67,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained(root: str, candidate: str) -> bool:
    """Report whether a path stays inside root, without touching the disk.

    Deliberately textual: resolving would follow symlinks already written by a
    concurrent writer, which is exactly the race an archive could exploit.
    """
    normalized = os.path.normpath(candidate)
    return normalized == root or normalized.startswith(root + os.sep)


def _validated_target(destination: Path, root: str, name: str) -> Path:
    target = destination / name
    if not _contained(root, str(target)):
        raise RuntimeError(f"Unsafe path in offline archive: {name}")
    return target


class _SymlinkGuard:
    """Stop later members from being written through a symlink out of the tree.

    Conda environments legitimately contain links to absolute build-machine
    paths (GDB auto-load scripts, for one), so rejecting those breaks real
    archives. What actually has to be prevented is a subsequent member whose
    path traverses such a link, which no amount of textual checking on the
    member itself would catch.
    """

    def __init__(self) -> None:
        self._links: set[str] = set()
        self._lock = threading.Lock()

    def record(self, target: Path) -> None:
        with self._lock:
            self._links.add(str(target))

    def blocks(self, target: Path) -> bool:
        parts = str(target).split(os.sep)
        with self._lock:
            if not self._links:
                return False
            return any(
                os.sep.join(parts[: index + 1]) in self._links
                for index in range(len(parts) - 1)
            )


class _DirCache:
    """Remember which parents exist so each file does not re-probe the volume."""

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def ensure(self, parent: Path) -> None:
        key = str(parent)
        with self._lock:
            if key in self._seen:
                return
        # mkdir outside the lock: it is a network round trip, and concurrent
        # creates of the same directory are harmless with exist_ok.
        parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._seen.add(key)


class _WriteBatch:
    """Track the writes one archive has in flight and surface their failures."""

    def __init__(self) -> None:
        self._pending = 0
        self._condition = threading.Condition()
        self._errors: list[BaseException] = []

    def submitted(self) -> None:
        with self._condition:
            self._pending += 1

    def finished(self, error: BaseException | None) -> None:
        with self._condition:
            self._pending -= 1
            if error is not None:
                self._errors.append(error)
            if self._pending == 0:
                self._condition.notify_all()

    def wait(self) -> None:
        """Block until nothing is in flight, without reporting failures.

        Used when an error is already propagating: writers must not outlive the
        extraction, but the original exception is the one worth surfacing.
        """
        with self._condition:
            while self._pending:
                self._condition.wait()

    def drain(self) -> None:
        self.wait()
        with self._condition:
            if self._errors:
                raise self._errors[0]


_WRITE_POOL: ThreadPoolExecutor | None = None
_WRITE_SLOTS: threading.Semaphore | None = None
_WRITE_POOL_LOCK = threading.Lock()


def _write_pool() -> tuple[ThreadPoolExecutor, threading.Semaphore]:
    """One pool shared by every runtime, so concurrency is budgeted globally.

    Giving each of the six runtimes its own pool would multiply out to far more
    parallel writes than the volume is likely to serve well.
    """
    global _WRITE_POOL, _WRITE_SLOTS
    with _WRITE_POOL_LOCK:
        if _WRITE_POOL is None:
            threads = max(1, int(os.getenv("RSMEM_EXTRACT_WRITE_THREADS", "24")))
            _WRITE_POOL = ThreadPoolExecutor(
                max_workers=threads, thread_name_prefix="rsmem-extract"
            )
            # Bounded queue depth keeps the reader from racing ahead of the
            # writers and buffering the whole archive in memory.
            _WRITE_SLOTS = threading.Semaphore(threads * 8)
        return _WRITE_POOL, _WRITE_SLOTS


def _write_regular(
    target: Path, data: bytes, mode: int, dirs: _DirCache, resume: bool = False
) -> None:
    dirs.ensure(target.parent)
    if resume:
        # One stat is cheaper than the open/write/close it can replace, and an
        # interrupted install leaves most of the tree already correct. Contents
        # come from a sha256-verified archive, so matching size means matching
        # file. Only worth paying for when there is something to resume.
        try:
            if target.stat().st_size == len(data):
                return
        except OSError:
            pass
    with open(target, "wb") as handle:
        handle.write(data)
    # Owner and mtime are never read back, and every one of them is a separate
    # round trip on this volume. Only the executable bit has to survive.
    if mode & 0o111:
        os.chmod(target, mode)


def _safe_extract(
    archive: Path, destination: Path, label: str = "", resume: bool = False
) -> int:
    """Stream one pass over the archive, writing members concurrently.

    A gzip stream cannot be rewound, so the reader stays single threaded; the
    per-file cost is dominated by network latency rather than bandwidth, so the
    writes are handed to a shared pool to overlap those round trips.
    """
    destination.mkdir(parents=True, exist_ok=True)
    root = os.path.normpath(str(destination))
    pool, slots = _write_pool()
    dirs = _DirCache()
    dirs.ensure(destination)
    guard = _SymlinkGuard()
    batch = _WriteBatch()
    started = time.time()
    last_report = started
    extracted = 0

    def submit(target: Path, data: bytes, mode: int) -> None:
        slots.acquire()
        batch.submitted()

        def run() -> None:
            error: BaseException | None = None
            try:
                _write_regular(target, data, mode, dirs, resume)
            except BaseException as exc:  # noqa: BLE001 - reported via drain()
                error = exc
            finally:
                slots.release()
                batch.finished(error)

        try:
            pool.submit(run)
        except BaseException:
            # The slot and the pending count were already taken; releasing them
            # here keeps drain() from waiting on work that will never run.
            slots.release()
            batch.finished(None)
            raise

    try:
        with tarfile.open(archive, "r|gz") as bundle:
            for member in bundle:
                target = _validated_target(destination, root, member.name)
                if guard.blocks(target):
                    raise RuntimeError(
                        f"Member would be written through a symlink: {member.name}"
                    )
                if member.isdir():
                    dirs.ensure(target)
                elif member.issym() or member.islnk():
                    dirs.ensure(target.parent)
                    # Regular members are simply overwritten, but linking onto
                    # an existing name fails, and a prefix left over from an
                    # interrupted install is the normal case here.
                    if os.path.lexists(target):
                        os.unlink(target)
                    if member.issym():
                        os.symlink(member.linkname, target)
                        guard.record(target)
                    else:
                        # Hard links need their target on disk first; drain so
                        # the pending writes are guaranteed to have landed.
                        batch.drain()
                        os.link(destination / member.linkname, target)
                elif member.isreg():
                    source = bundle.extractfile(member)
                    if source is None:
                        continue
                    if member.size > LARGE_FILE_BYTES:
                        # Streaming a big member through the queue would pin its
                        # whole size in memory, so write it here instead.
                        batch.drain()
                        dirs.ensure(target.parent)
                        with open(target, "wb") as handle:
                            shutil.copyfileobj(source, handle)
                        if member.mode & 0o111:
                            os.chmod(target, member.mode)
                    else:
                        submit(target, source.read(), member.mode)
                else:
                    continue
                extracted += 1
                now = time.time()
                if label and now - last_report >= EXTRACT_PROGRESS_INTERVAL_SECONDS:
                    elapsed = max(now - started, 1e-6)
                    print(
                        f"[xgpu] {label}: extracted {extracted} files "
                        f"in {elapsed:.0f}s ({extracted / elapsed:.1f} files/s)",
                        flush=True,
                    )
                    last_report = now
    except BaseException:
        # Let the writers finish before unwinding, but keep the original error:
        # draining here would replace it with whatever a writer happened to hit.
        batch.wait()
        raise
    batch.drain()
    return extracted


def _install_archive(
    prefix: Path, archive: Path, expected_sha256: str, label: str = ""
) -> None:
    if not archive.is_file():
        raise RuntimeError(f"Offline runtime archive is missing: {archive}")
    name = label or prefix.name
    # Expand straight into the final path. Moving a finished tree into place
    # looks atomic but costs ~586 ms per file here, and clearing the prefix up
    # front would destroy an extraction another instance still has in flight.
    # A half-written tree is instead made obvious by dropping the ready marker.
    marker = prefix / ".rsmem-runtime.json"
    prefix.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    # An earlier attempt that got most of the way through is common enough here
    # that rewriting its output would dominate the retry.
    resume = any(prefix.iterdir())
    if resume:
        print(f"[xgpu] {name}: resuming over an existing tree", flush=True)

    started = time.time()
    extracted = _safe_extract(archive, prefix, label, resume=resume)
    print(
        f"[xgpu] {name}: extracted {extracted} files "
        f"in {time.time() - started:.0f}s",
        flush=True,
    )
    unpack = prefix / "bin" / "conda-unpack"
    if unpack.is_file():
        started = time.time()
        subprocess.run(
            [str(unpack)], check=True, cwd=str(prefix),
            timeout=CONDA_UNPACK_TIMEOUT_SECONDS,
        )
        print(
            f"[xgpu] {name}: conda-unpack finished "
            f"in {time.time() - started:.0f}s",
            flush=True,
        )
    python = prefix / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(f"Offline runtime has no Python interpreter: {prefix}")
    metadata = {
        "schema": ENV_SCHEMA,
        "archive": archive.name,
        "sha256": expected_sha256,
    }
    marker.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _checksums(models: Path) -> dict[str, str]:
    manifest = models / "offline" / "SHA256SUMS"
    if not manifest.is_file():
        raise RuntimeError(f"Offline checksum manifest is missing: {manifest}")
    values: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            checksum, filename = line.split(maxsplit=1)
        except ValueError as exc:
            raise RuntimeError(f"Malformed checksum manifest line: {line!r}") from exc
        values[filename.lstrip("*")] = checksum
    return values


DOWNLOAD_ATTEMPTS = int(os.getenv("RSMEM_DOWNLOAD_ATTEMPTS", "4"))
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("RSMEM_DOWNLOAD_TIMEOUT", "2700"))
# oea-e1 extrapolates to roughly 3940 s from remoteclip's measured 31.8 ms per
# entry, so the previous 3600 s budget would have failed the whole bootstrap
# after the extraction had already succeeded.
CONDA_UNPACK_TIMEOUT_SECONDS = int(os.getenv("RSMEM_CONDA_UNPACK_TIMEOUT", "7200"))
EXTRACT_PROGRESS_INTERVAL_SECONDS = float(
    os.getenv("RSMEM_EXTRACT_PROGRESS_INTERVAL", "60")
)
LARGE_FILE_BYTES = int(os.getenv("RSMEM_EXTRACT_LARGE_FILE_BYTES", str(8 * 1024 * 1024)))

# Run the download in a child process so a stalled connection can be killed.
# A stalled in-process download once held the bootstrap for four hours before
# the peer reset it, leaving the GPU workers permanently unavailable.
_DOWNLOAD_SCRIPT = """
import os, sys
from modelscope import snapshot_download

repo_id, relative, revision, local_dir = sys.argv[1:5]
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    revision=revision,
    local_dir=local_dir,
    token=os.getenv("MODELSCOPE_API_TOKEN") or None,
    allow_file_pattern=[relative],
)
"""


def _downloaded_archive(download_root: Path, relative: str) -> Path | None:
    for candidate in (download_root / relative, download_root / Path(relative).name):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _download_archive(models: Path, relative: str, workspace: Path) -> Path:
    """Fetch one offline archive into a staging directory private to this call.

    ModelScope's "double instance" deploy quirk can start two containers (or
    two bootstrap attempts) back to back, and both can decide the same
    archive is missing at the same moment. A staging directory shared
    between them let one instance's cleanup delete the file the other was
    still verifying or installing, raising FileNotFoundError with no data
    problem at all -- just two writers racing over one mutable path. This
    call's own ``<pid>-<random>`` directory is visible to nobody else, so
    the caller can verify and publish what lands in it without racing
    anyone else doing the same for the same archive. The returned path's
    parent is always exactly that private directory (never a subdirectory
    of it), so the caller can remove the whole thing in one place when done.

    The directory is still created under ``workspace`` (not e.g. ``/tmp``):
    publishing the verified file relies on ``os.replace``, which requires
    the source and destination to share a filesystem.
    """
    download_root = workspace / ".rsmem-runtime-downloads" / f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    download_root.mkdir(parents=True, exist_ok=True)
    repo_id = os.getenv("RSMEM_MODELSCOPE_MODEL_REPO", "wbx929/RSMeM-Perception-Models")
    revision = os.getenv("RSMEM_MODEL_REVISION", "master")
    reason = "no attempt was made"

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        print(
            f"[xgpu] downloading {relative} (attempt {attempt}/{DOWNLOAD_ATTEMPTS})",
            flush=True,
        )
        try:
            subprocess.run(
                [sys.executable, "-c", _DOWNLOAD_SCRIPT,
                 repo_id, relative, revision, str(download_root)],
                check=True,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            reason = f"stalled for more than {DOWNLOAD_TIMEOUT_SECONDS}s"
        except subprocess.CalledProcessError as exc:
            reason = f"downloader exited with code {exc.returncode}"
        else:
            downloaded = _downloaded_archive(download_root, relative)
            if downloaded is not None:
                # Flatten: the downloader mirrors `relative`'s own
                # subdirectories (e.g. "offline/x.tar.gz") inside
                # download_root, which would otherwise leave the returned
                # path's parent one level short of the directory this call
                # actually owns and the caller is meant to remove.
                flat = download_root / Path(relative).name
                if downloaded != flat:
                    downloaded.rename(flat)
                return flat
            reason = "downloader succeeded but produced no file"

        print(f"[xgpu] download failed for {relative}: {reason}", flush=True)
        if attempt < DOWNLOAD_ATTEMPTS:
            time.sleep(min(60, 10 * attempt))

    shutil.rmtree(download_root, ignore_errors=True)
    raise RuntimeError(
        f"Offline runtime download failed after {DOWNLOAD_ATTEMPTS} attempts "
        f"for {relative}: {reason}"
    )


def _small_file_write_rate(directory: Path, samples: int = 200) -> str:
    """Time a burst of small writes, the operation the bootstrap is bound by."""
    probe = directory / f".rsmem-probe-{os.getpid()}"
    payload = b"x" * 4096
    try:
        probe.mkdir(parents=True, exist_ok=True)
        started = time.time()
        for index in range(samples):
            (probe / str(index)).write_bytes(payload)
        elapsed = max(time.time() - started, 1e-6)
        return f"{samples / elapsed:.0f} files/s"
    except OSError as exc:
        return f"unavailable ({type(exc).__name__})"
    finally:
        shutil.rmtree(probe, ignore_errors=True)


def report_storage_options(workspace: Path) -> None:
    """Record what the container-local disks offer, for the next design round.

    Extraction currently targets the persistent volume, whose per-file latency
    dominates the whole bootstrap. Deciding whether a local disk is a viable
    target needs capacity and speed numbers, and there is no shell on this
    instance to collect them by hand. Purely diagnostic: nothing here changes
    where the runtimes are installed.
    """
    for candidate in (workspace, Path("/tmp"), Path("/root")):
        try:
            usage = shutil.disk_usage(candidate)
        except OSError as exc:
            print(f"[xgpu] storage {candidate}: unavailable ({exc})", flush=True)
            continue
        print(
            f"[xgpu] storage {candidate}: {usage.free / 1024 ** 3:.1f} GiB free "
            f"of {usage.total / 1024 ** 3:.1f} GiB · "
            f"small writes {_small_file_write_rate(candidate)}",
            flush=True,
        )


def _runtime_is_ready(prefix: Path, filename: str, expected_sha256: str) -> bool:
    try:
        current = json.loads((prefix / ".rsmem-runtime.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (prefix / "bin" / "python").is_file() and current == {
        "schema": ENV_SCHEMA,
        "archive": filename,
        "sha256": expected_sha256,
    }


def _select_local_targets(
    pending: list[tuple[str, str, str]], local_root: Path | None, min_free_gib: float
) -> dict[str, Path]:
    """Decide which pending runtimes fit the local disk, once, before any
    extraction starts -- not per worker thread, which would let several
    threads each see the same free space and jointly overcommit it. The
    total (50.35 GB) is close to what the local disk offers (55 GiB free at
    last measurement, itself shared with /tmp), so this has to bin-pack
    rather than assume everything fits.

    A runtime that is already validly installed under ``local_root`` does
    not consume any of that budget: the space it occupies is already gone
    from ``free``, so charging for it again would evict a healthy local
    copy the moment the disk got tight and send the worker back to the
    volume -- the EIO failure this placement exists to avoid.
    """
    if local_root is None or not pending:
        return {}
    try:
        local_root.mkdir(parents=True, exist_ok=True)
        budget = shutil.disk_usage(local_root).free / 1024 ** 3 - min_free_gib
    except OSError as exc:
        print(f"[xgpu] local disk {local_root} unavailable ({exc}); using the persistent volume", flush=True)
        return {}

    targets: dict[str, Path] = {}
    not_ready: list[tuple[str, str, str]] = []
    for name, relative, sha in pending:
        prefix = local_root / "envs" / name
        if _runtime_is_ready(prefix, Path(relative).name, sha):
            targets[name] = prefix
            continue
        not_ready.append((name, relative, sha))

    # Largest first: a big runtime landing on the fast disk removes more of
    # the slow-volume failure risk per GB spent than several small ones would.
    for name, _relative, _sha in sorted(
        not_ready, key=lambda item: RUNTIME_EXPANDED_GIB.get(item[0], 0.0), reverse=True
    ):
        size = RUNTIME_EXPANDED_GIB.get(name, 0.0)
        if size <= budget:
            targets[name] = local_root / "envs" / name
            budget -= size
        else:
            print(
                f"[xgpu] {name}: staying on the persistent volume "
                f"({size:.1f} GiB needed, {budget:.1f} GiB local budget left)",
                flush=True,
            )
    return targets


def _prepare_runtime(
    name: str, relative: str, expected_sha256: str, prefix: Path, workspace: Path, models: Path
) -> None:
    """Verify (or fetch) one runtime archive, publish it, then install it.

    The archive is never deleted afterwards. Leaving a verified copy at
    ``models / relative`` lets the next call's ``archive.is_file()`` check
    below short-circuit straight past downloading -- and that is what
    removes the cross-process download race for good in steady state: no
    download means no shared mutable staging file for two instances to
    fight over. A canonical copy that fails its checksum (an earlier
    transfer interrupted mid-write, for instance) is treated the same as a
    missing one and re-fetched within this same call, rather than failing
    the runtime until the next deploy happens to retry it.
    """
    filename = Path(relative).name
    started = time.time()
    print(f"[xgpu] preparing offline runtime {name} -> {prefix}", flush=True)
    archive = models / relative
    downloaded_for = 0.0
    verify_started = time.time()
    valid = archive.is_file() and _sha256(archive) == expected_sha256
    verified_for = time.time() - verify_started
    if not valid:
        download_started = time.time()
        downloaded = _download_archive(models, relative, workspace)
        downloaded_for = time.time() - download_started
        staging_dir = downloaded.parent
        try:
            verify_started = time.time()
            if _sha256(downloaded) != expected_sha256:
                raise RuntimeError(f"Offline runtime checksum mismatch: {filename}")
            verified_for += time.time() - verify_started
            archive.parent.mkdir(parents=True, exist_ok=True)
            # Same-filesystem atomic rename: a concurrent instance publishing
            # the same, equally-verified archive at the same moment just
            # overwrites with byte-identical content, and a reader that
            # already opened the previous path keeps its own file
            # descriptor regardless of the rename underneath it.
            os.replace(downloaded, archive)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
    install_started = time.time()
    _install_archive(prefix, archive, expected_sha256, label=name)
    installed_for = time.time() - install_started
    free_gb = shutil.disk_usage(prefix).free / 1024 ** 3
    # Reported per phase because a single total once hid a 6.6 h directory
    # rename behind an otherwise unremarkable "ready in 37301s".
    print(
        f"[xgpu] runtime {name} ready in {time.time() - started:.0f}s "
        f"(download {downloaded_for:.0f}s · verify {verified_for:.0f}s "
        f"· install {installed_for:.0f}s) · {free_gb:.1f} GiB free at {prefix.parent}",
        flush=True,
    )


@contextlib.contextmanager
def _local_lock(local_root: Path | None):
    """A second, host-local lock guarding local-disk runtime placement.

    ``ensure_xgpu_runtime_envs`` already takes a lock on the persistent
    volume, but that volume's flock reliability is exactly as suspect as
    every other read on it (see docs/NEXT_AGENT_HANDOFF.md): a lock that
    silently fails to serialize two same-host instances would let both
    extract into the same ``local_root / "envs" / <name>`` directory at
    once. A lock on genuinely local disk carries no such doubt -- flock is a
    same-host kernel construct, and this path never leaves the host it runs
    on. A no-op when ``local_root`` is None, matching every existing caller.
    """
    if local_root is None:
        yield
        return
    local_root.mkdir(parents=True, exist_ok=True)
    lock_path = local_root / ".rsmem-env-bootstrap.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def ensure_xgpu_runtime_envs(
    workspace: Path,
    models: Path,
    local_root: Path | None = None,
    min_free_gib: float = 4.0,
) -> tuple[dict[str, Path], dict[str, str]]:
    """Extract versioned offline runtimes once; never invoke conda or pip.

    Each archive expands into tens of thousands of small files. On the
    persistent volume the per-file write latency dominates *and* that volume
    fails reads intermittently even once extraction succeeds -- workers have
    died importing their own already-installed site-packages with
    ``OSError: [Errno 5]``. When ``local_root`` is given, local placement is
    decided first, against every runtime, and wins over a stale volume
    cache: a runtime already cached on the volume from a deploy that
    predates this local-disk support would otherwise keep reusing it, and
    keep failing, forever. Only a runtime that does not fit the local budget
    falls back to checking whether the volume already has a valid copy
    before re-extracting it there. A local target that is already validly
    installed -- for instance by a sibling instance that won the race to
    extract it first -- is reused rather than re-extracted; see
    ``_local_lock``. Passing ``local_root=None`` (the default) reproduces
    the pre-existing volume-only behaviour precisely, including for every
    existing caller.

    The runtimes are independent, so they are prepared concurrently.
    Returns the interpreters that are ready and the reasons the rest are
    not. Reporting per runtime rather than raising lets the workers backed
    by a healthy environment start while a slower or broken one is still
    missing.
    """
    lock_path = workspace / ".rsmem-env-bootstrap.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    failures: dict[str, str] = {}
    ready: dict[str, Path] = {}
    with lock_path.open("a+") as lock, _local_lock(local_root):
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        checksums = _checksums(models)
        candidates: list[tuple[str, str, str]] = []
        for name, relative in OFFLINE_ENV_ARCHIVES.items():
            filename = Path(relative).name
            expected_sha256 = checksums.get(filename)
            if not expected_sha256:
                failures[name] = f"missing checksum for {filename}"
                continue
            candidates.append((name, relative, expected_sha256))

        # Local placement is decided against every candidate up front, before
        # any volume-cache check, so it outranks whatever happened to get
        # cached on the volume by an earlier deploy.
        local_targets = _select_local_targets(candidates, local_root, min_free_gib)

        pending = []
        for name, relative, expected_sha256 in candidates:
            filename = Path(relative).name
            if name in local_targets:
                local_prefix = local_targets[name]
                if _runtime_is_ready(local_prefix, filename, expected_sha256):
                    ready[name] = local_prefix
                else:
                    pending.append((name, relative, expected_sha256, local_prefix))
                continue
            volume_prefix = workspace / "envs" / name
            if _runtime_is_ready(volume_prefix, filename, expected_sha256):
                ready[name] = volume_prefix
            else:
                pending.append((name, relative, expected_sha256, volume_prefix))

        if pending:
            workers = max(1, min(int(os.getenv("RSMEM_ENV_EXTRACT_WORKERS", "6")), len(pending)))
            free_gb = shutil.disk_usage(workspace).free / 1024 ** 3
            print(
                f"[xgpu] preparing {len(pending)} runtime(s) with {workers} worker(s) "
                f"· {len(local_targets)} on local disk · {free_gb:.1f} GiB free on the volume",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_prepare_runtime, name, relative, sha, prefix, workspace, models): (name, prefix)
                    for name, relative, sha, prefix in pending
                }
                for future in as_completed(futures):
                    name, prefix = futures[future]
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001 - report every failure
                        failures[name] = f"{type(exc).__name__}: {exc}"
                        print(f"[xgpu] runtime {name} failed: {exc}", flush=True)
                    else:
                        ready[name] = prefix

    return (
        {name: prefix / "bin" / "python" for name, prefix in sorted(ready.items())},
        failures,
    )


RUNTIME_PYTHON_VARIABLES = {
    "sam2": "RSMEM_EARTHBENCH_PYTHON",
    "remoteclip": "RSMEM_REMOTECLIP_PYTHON",
    "changeos": "RSMEM_CHANGEOS_PYTHON",
    "bright": "RSMEM_BRIGHT_PYTHON",
    "oea-e1": "RSMEM_OEA_E1_PYTHON",
    "oea-e3": "RSMEM_OEA_E3_PYTHON",
}


def configure_runtime_environment(interpreters: dict[str, Path]) -> None:
    for name, variable in RUNTIME_PYTHON_VARIABLES.items():
        path = interpreters.get(name)
        if path is not None:
            os.environ.setdefault(variable, str(path))
