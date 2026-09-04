"""Install persistent xGPU runtimes exclusively from offline conda-pack archives."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe path in offline archive: {member.name}")
        bundle.extractall(destination, filter="fully_trusted")


def _install_archive(prefix: Path, archive: Path, expected_sha256: str) -> None:
    if not archive.is_file():
        raise RuntimeError(f"Offline runtime archive is missing: {archive}")
    if prefix.exists():
        shutil.rmtree(prefix)
    _safe_extract(archive, prefix)
    unpack = prefix / "bin" / "conda-unpack"
    if unpack.is_file():
        subprocess.run([str(unpack)], check=True, cwd=str(prefix))
    python = prefix / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(f"Offline runtime has no Python interpreter: {prefix}")
    metadata = {
        "schema": ENV_SCHEMA,
        "archive": archive.name,
        "sha256": expected_sha256,
    }
    (prefix / ".rsmem-runtime.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


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
    """Fetch one offline archive, retrying on stalls and transport failures."""
    download_root = workspace / ".rsmem-runtime-downloads"
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
                return downloaded
            reason = "downloader succeeded but produced no file"

        print(f"[xgpu] download failed for {relative}: {reason}", flush=True)
        if attempt < DOWNLOAD_ATTEMPTS:
            time.sleep(min(60, 10 * attempt))

    raise RuntimeError(
        f"Offline runtime download failed after {DOWNLOAD_ATTEMPTS} attempts "
        f"for {relative}: {reason}"
    )


def ensure_xgpu_runtime_envs(workspace: Path, models: Path) -> dict[str, Path]:
    """Extract versioned offline runtimes once; never invoke conda or pip."""
    env_root = workspace / "envs"
    lock_path = workspace / ".rsmem-env-bootstrap.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        checksums = _checksums(models)
        interpreters: dict[str, Path] = {}
        for name, relative in OFFLINE_ENV_ARCHIVES.items():
            filename = Path(relative).name
            expected_sha256 = checksums.get(filename)
            if not expected_sha256:
                raise RuntimeError(f"Missing checksum for offline runtime: {filename}")
            archive = models / relative
            prefix = env_root / name
            marker = prefix / ".rsmem-runtime.json"
            current = {}
            try:
                current = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            ready = (prefix / "bin" / "python").is_file() and current == {
                "schema": ENV_SCHEMA,
                "archive": filename,
                "sha256": expected_sha256,
            }
            if not ready:
                print(f"[xgpu] extracting offline runtime {name}", flush=True)
                if not archive.is_file():
                    archive = _download_archive(models, relative, workspace)
                if _sha256(archive) != expected_sha256:
                    archive.unlink(missing_ok=True)
                    raise RuntimeError(f"Offline runtime checksum mismatch: {filename}")
                _install_archive(prefix, archive, expected_sha256)
                archive.unlink(missing_ok=True)
            interpreters[name] = prefix / "bin" / "python"
    return interpreters


def configure_runtime_environment(interpreters: dict[str, Path]) -> None:
    os.environ.setdefault("RSMEM_EARTHBENCH_PYTHON", str(interpreters["sam2"]))
    os.environ.setdefault("RSMEM_REMOTECLIP_PYTHON", str(interpreters["remoteclip"]))
    os.environ.setdefault("RSMEM_CHANGEOS_PYTHON", str(interpreters["changeos"]))
    os.environ.setdefault("RSMEM_BRIGHT_PYTHON", str(interpreters["bright"]))
    os.environ.setdefault("RSMEM_OEA_E1_PYTHON", str(interpreters["oea-e1"]))
    os.environ.setdefault("RSMEM_OEA_E3_PYTHON", str(interpreters["oea-e3"]))
