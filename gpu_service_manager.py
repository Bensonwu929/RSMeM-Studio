"""Start the configured xGPU perception workers without embedding models in Gradio.

All large sources and checkpoints live under ``/mnt/workspace`` by default.  The
Git repository contains only launch metadata; this module never downloads or
serializes credentials.
"""

from __future__ import annotations

import atexit
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests

from model_assets import prepare_xgpu_assets
from xgpu_runtime import (
    configure_runtime_environment,
    ensure_xgpu_runtime_envs,
    report_storage_options,
)


WORKSPACE = Path(os.getenv("RSMEM_XGPU_WORKSPACE", "/mnt/workspace"))
RS_ROOT = Path(os.getenv("RSMEM_EARTHBENCH_ROOT", str(WORKSPACE / "RSMeM-public")))
OEA_ROOT = Path(os.getenv("RSMEM_OEA_ROOT", str(WORKSPACE / "OpenEarthAgent")))
MODELS = Path(os.getenv("RSMEM_MODELS_DIR", str(WORKSPACE / "models")))

# The persistent volume fails small-file reads intermittently (measured 14
# files/s there vs ~2000x that locally, and idle, not just under load -- see
# docs/NEXT_AGENT_HANDOFF.md). Source trees, the two OEA detector packages,
# runtime environments and worker logs are all small-file-heavy and land here
# when there is room; large sequential reads (model weights) stay on the
# volume, which has never failed at those. Set RSMEM_LOCAL_ROOT="" to disable
# local placement entirely and reproduce the pre-fix, volume-only behaviour.
_LOCAL_ROOT_RAW = os.getenv("RSMEM_LOCAL_ROOT", "/root/rsmem-local").strip()
LOCAL_ROOT: Path | None = Path(_LOCAL_ROOT_RAW) if _LOCAL_ROOT_RAW else None
LOCAL_MIN_FREE_GIB = float(os.getenv("RSMEM_LOCAL_MIN_FREE_GIB", "4"))

LOG_DIR = Path(os.getenv("RSMEM_GPU_LOG_DIR", str((LOCAL_ROOT or WORKSPACE) / "gpu-logs")))

# GroundingDINO and LAE-DINO's *source* packages (imported via PYTHONPATH, not
# their checkpoints) travel with the source bundle -- see _apply_source_root.
# This starts equal to MODELS so behaviour is unchanged until that runs.
SOURCE_MODELS_ROOT = MODELS


def _apply_source_root(source_root: Path) -> None:
    """Point RS_ROOT/OEA_ROOT/SOURCE_MODELS_ROOT at wherever the source bundle
    actually landed (see model_assets.ensure_source_trees), unless the
    operator pinned one of the first two explicitly via its own env var.
    """
    global RS_ROOT, OEA_ROOT, SOURCE_MODELS_ROOT
    if not os.getenv("RSMEM_EARTHBENCH_ROOT"):
        RS_ROOT = source_root / "RSMeM-public"
    if not os.getenv("RSMEM_OEA_ROOT"):
        OEA_ROOT = source_root / "OpenEarthAgent"
    SOURCE_MODELS_ROOT = source_root / "models"

CONTROLLER_NAME = "oea-controller"
QWEN_NAME = "oea-qwen3-vl"
OEA_DEPENDENT = frozenset({
    "oea-grounding-dino",
    "oea-lae-dino",
    "oea-ocr",
    "oea-qwen3-vl",
})


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    command: tuple[str, ...]
    cwd: Path
    health_url: str
    expected_model: str = ""
    required_paths: tuple[Path, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()
    # Which offline runtime supplies this worker's interpreter. Without it a
    # missing environment would silently fall back to the container's Python.
    runtime: str = ""
    # Overrides the shared budget for workers that legitimately take longer to
    # become reachable, such as one loading a 22.5 GB checkpoint.
    startup_timeout_sec: int = 0


def _python(variable: str) -> str:
    return os.getenv(variable, sys.executable)


def service_specs() -> list[ServiceSpec]:
    deploy = RS_ROOT / "tools" / "deploy_tools"
    workers = OEA_ROOT / "tool_server" / "tool_workers" / "online_workers"
    controller = "http://127.0.0.1:21001"
    return [
        ServiceSpec("earthbench-sam2", (_python("RSMEM_EARTHBENCH_PYTHON"), str(deploy / "sam2_api.py")), deploy, "http://127.0.0.1:5554/health", required_paths=(MODELS / "sam2.1_hiera_small.pt",), runtime="sam2"),
        ServiceSpec("earthbench-remoteclip", (_python("RSMEM_REMOTECLIP_PYTHON"), str(deploy / "remoteclip_api.py")), deploy, "http://127.0.0.1:5555/health", required_paths=(MODELS / "RemoteCLIP-ViT-L-14.pt",), environment=(("REMOTECLIP_PORT", "5555"),), runtime="remoteclip"),
        ServiceSpec("earthbench-changeos", (_python("RSMEM_CHANGEOS_PYTHON"), str(deploy / "changeos_api.py")), deploy, "http://127.0.0.1:8019/health", required_paths=(MODELS / "changeos_r101.pt",), runtime="changeos"),
        ServiceSpec("earthbench-bright", (_python("RSMEM_BRIGHT_PYTHON"), str(deploy / "siamcrnn_api_biany.py")), deploy, "http://127.0.0.1:8120/health", required_paths=(MODELS / "SiamCRNN_20250211_033615_1" / "best_model.pth",), runtime="bright"),
        ServiceSpec("oea-controller", (_python("RSMEM_OEA_E1_PYTHON"), str(workers / "controller.py"), "--host", "0.0.0.0", "--port", "21001"), workers, f"{controller}/list_models", runtime="oea-e1"),
        ServiceSpec("oea-grounding-dino", (_python("RSMEM_OEA_E1_PYTHON"), str(workers / "TextToBbox_worker.py"), "--controller-address", controller, "--host", "0.0.0.0", "--port", "21009", "--model-path", str(MODELS / "groundingdino_swint_ogc.pth"), "--model-config", str(SOURCE_MODELS_ROOT / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"), "--limit-model-concurrency", "1"), workers, f"{controller}/list_models", "TextToBbox", (MODELS / "groundingdino_swint_ogc.pth", SOURCE_MODELS_ROOT / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"), runtime="oea-e1"),
        ServiceSpec("oea-lae-dino", (_python("RSMEM_OEA_E3_PYTHON"), str(workers / "ObjectDetection_worker.py"), "--controller-address", controller, "--host", "0.0.0.0", "--port", "21016", "--model-path", str(MODELS / "lae_dino_swint_lae1m-28ca3a15.pth"), "--model-config", str(SOURCE_MODELS_ROOT / "LAE-DINO" / "mmdetection_lae" / "configs" / "lae_dino" / "lae_dino_swin-t_pretrain_LAE-1M.py"), "--limit-model-concurrency", "1"), workers, f"{controller}/list_models", "ObjectDetection", (MODELS / "lae_dino_swint_lae1m-28ca3a15.pth", SOURCE_MODELS_ROOT / "LAE-DINO" / "mmdetection_lae" / "configs" / "lae_dino" / "lae_dino_swin-t_pretrain_LAE-1M.py"), runtime="oea-e3"),
        ServiceSpec("oea-ocr", (_python("RSMEM_OEA_E1_PYTHON"), str(workers / "OCR_worker.py"), "--controller-address", controller, "--host", "0.0.0.0", "--port", "21003", "--limit-model-concurrency", "1"), workers, f"{controller}/list_models", "OCR", runtime="oea-e1"),
        # Load the 22.5 GB VLM last to keep initialization peaks predictable.
        # It needs far longer than the light workers: the shared budget once
        # declared it failed while its process was still loading normally.
        ServiceSpec("oea-qwen3-vl", (_python("RSMEM_OEA_E1_PYTHON"), str(workers / "BaseModel_worker.py"), "--controller-address", controller, "--host", "0.0.0.0", "--port", "21010", "--model_path", str(MODELS / "Qwen3-VL-8B-Instruct"), "--limit-model-concurrency", "1"), workers, f"{controller}/list_models", "BaseModel", (MODELS / "Qwen3-VL-8B-Instruct" / "config.json",), runtime="oea-e1", startup_timeout_sec=_vlm_startup_timeout_sec()),
    ]


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# One retry for a crash or an unhandled exception, none for the rest: those two
# reasons are how an intermittent volume read failure shows up, and a retry has
# recovered workers in practice (SAM2 came back healthy once concurrency gave it
# a clean pass). A timeout already spent its full budget -- doubling it has no
# evidence of helping when the process is still alive and simply never
# registered -- and a missing asset or source will look identical on a second
# attempt, so retrying either only doubles the time spent reporting the same
# failure.
DEFAULT_START_RETRIES = 1
# The controller gates every OEA-dependent worker (see OEA_DEPENDENT below), so
# losing it is the single highest-leverage failure in the whole startup: one
# extra retry attempt here is worth four workers, not one.
CONTROLLER_START_RETRIES = 2


def _is_retryable_failure(reason: str) -> bool:
    return "exited during startup" in reason or reason.startswith("unexpected error:")


def _reachable(spec: ServiceSpec) -> bool:
    try:
        method = requests.post if spec.health_url.endswith("/list_models") else requests.get
        response = method(spec.health_url, timeout=1.5)
        if not response.ok:
            return False
        if spec.expected_model:
            return spec.expected_model in response.json().get("models", [])
        return True
    except (requests.RequestException, ValueError):
        return False


def _startup_concurrency() -> int:
    return max(1, int(os.getenv("RSMEM_GPU_STARTUP_CONCURRENCY", "4")))


def _startup_timeout_sec() -> int:
    return int(os.getenv("RSMEM_GPU_STARTUP_TIMEOUT_SEC", "300"))


def _vlm_startup_timeout_sec() -> int:
    return int(os.getenv("RSMEM_GPU_VLM_STARTUP_TIMEOUT_SEC", "1800"))


# A random id generated once per process, distinct from the PID (which the
# platform can reuse across restarts). Comparing hostname/boot_id/pid/this
# id across repeated /rsmem/gpu-status polls is how "double instance" gets
# told apart in production, where there is no shell: identical hostname and
# boot_id but different pid means one host running the bootstrap twice
# (host-local lock territory); a different hostname entirely means two
# separate machines (only per-archive atomic publishing helps there). It is
# also the evidence this is reported to ModelScope as a platform bug.
_INSTANCE_ID = uuid.uuid4().hex[:12]


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _instance_identity() -> dict:
    return {
        "hostname": socket.gethostname(),
        "boot_id": _boot_id(),
        "pid": os.getpid(),
        "instance_id": _INSTANCE_ID,
    }


def _storage_snapshot() -> dict:
    """Read-only free space for the three disks a concurrency probe cares about.

    ``/tmp`` and ``/root`` share one filesystem on the current Studio image;
    run directories from ``mkdtemp`` land on that disk and are never deleted.
    ``/mnt/workspace`` is the persistent volume. ``status()`` exposes this so
    ``/rsmem/gpu-status`` can be polled during a load test without a shell.
    """
    report = {}
    for name, path in (
        ("tmp", Path("/tmp")),
        ("root", Path("/root")),
        ("workspace", WORKSPACE),
    ):
        try:
            usage = shutil.disk_usage(path)
        except OSError as exc:
            report[name] = {"path": str(path), "error": str(exc)}
            continue
        report[name] = {
            "path": str(path),
            "free_gib": round(usage.free / 1024 ** 3, 2),
            "used_gib": round(usage.used / 1024 ** 3, 2),
            "total_gib": round(usage.total / 1024 ** 3, 2),
        }
    return report


def read_worker_log(name: str, tail_bytes: int = 16384) -> str:
    """Return the tail of one worker's startup log.

    Workers that exit during startup leave their traceback here and nowhere
    else; the platform log stream only carries the manager's own output.
    """
    if name not in {spec.name for spec in service_specs()}:
        raise KeyError(name)
    path = LOG_DIR / f"{name}.log"
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - tail_bytes))
        return handle.read().decode("utf-8", errors="replace")


class GPUServiceManager:
    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self.skipped: dict[str, str] = {}
        self.startup_timings: dict[str, float] = {}
        self._lock = threading.Lock()
        self._startup_thread: threading.Thread | None = None

    def start(self) -> "GPUServiceManager":
        if not _truthy("RSMEM_GPU_AUTOSTART"):
            self.skipped["all"] = "RSMEM_GPU_AUTOSTART is disabled"
            print("[xgpu] autostart disabled", flush=True)
            return self
        print("[xgpu] starting background asset/runtime bootstrap", flush=True)
        self._startup_thread = threading.Thread(
            target=self._start_selected_services,
            name="rsmem-gpu-startup",
            daemon=True,
        )
        self._startup_thread.start()
        return self

    def _mark_skipped(self, name: str, reason: str) -> None:
        with self._lock:
            self.skipped[name] = reason

    def _record_process(self, name: str, process: subprocess.Popen) -> None:
        with self._lock:
            self.processes[name] = process

    def _record_timing(self, name: str, seconds: float) -> None:
        with self._lock:
            self.startup_timings[name] = seconds

    def _start_one(self, spec: ServiceSpec) -> bool:
        """Launch one worker and wait until it is healthy. Return True on success.

        Every step here can touch the persistent volume (``script.is_file()``,
        ``path.exists()``, opening the log, ``Popen`` reading the interpreter
        and cwd), and that volume fails reads with ``OSError: [Errno 5]``
        while otherwise idle -- not only under concurrent load. Path.is_file/
        exists do not swallow EIO the way they swallow ENOENT, so an
        uncaught one here used to propagate out of _start_one entirely. Phase
        A and Phase C call this with no try/except of their own, so that
        escape killed the whole rsmem-gpu-startup thread and left later
        phases (including the one loading Qwen3-VL) never attempted, with no
        record in either started or skipped. The try/except below is the
        fix: every exit from this method goes through _mark_skipped, so a
        volume hiccup costs one worker instead of every worker after it.
        """
        try:
            return self._start_one_unguarded(spec)
        except Exception as exc:  # noqa: BLE001 - a bad read must not end the thread
            self._mark_skipped(spec.name, f"unexpected error: {type(exc).__name__}: {exc}")
            print(f"[xgpu] {spec.name}: unexpected error: {type(exc).__name__}: {exc}", flush=True)
            return False

    def _start_one_unguarded(self, spec: ServiceSpec) -> bool:
        script = Path(spec.command[1])
        if not script.is_file():
            self._mark_skipped(spec.name, f"missing source: {script}")
            return False
        missing = [path for path in spec.required_paths if not path.exists()]
        if missing:
            self._mark_skipped(
                spec.name,
                "missing assets: " + ", ".join(str(path) for path in missing),
            )
            return False
        if _reachable(spec):
            self._mark_skipped(spec.name, "already running")
            print(f"[xgpu] {spec.name}: already healthy", flush=True)
            return True

        log_path = LOG_DIR / f"{spec.name}.log"
        print(f"[xgpu] {spec.name}: starting; log={log_path}", flush=True)
        started = time.time()
        try:
            handle = log_path.open("ab", buffering=0)
        except OSError as exc:
            # A log we cannot open is not a reason to skip a worker that
            # would otherwise run; only the log itself is lost.
            print(
                f"[xgpu] {spec.name}: could not open {log_path} ({exc}); "
                "logging to /dev/null instead",
                flush=True,
            )
            handle = open(os.devnull, "ab", buffering=0)
        environment = os.environ.copy()
        environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
        environment.setdefault("RSMEM_MODELS_DIR", str(MODELS))
        environment.setdefault("EASYOCR_MODULE_PATH", str(WORKSPACE / "easyocr"))
        python_paths = [
            str(RS_ROOT),
            str(RS_ROOT / "tools" / "deploy_tools"),
            str(OEA_ROOT),
            str(SOURCE_MODELS_ROOT / "GroundingDINO"),
            str(SOURCE_MODELS_ROOT / "LAE-DINO" / "mmdetection_lae"),
        ]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        environment.update(dict(spec.environment))
        process = subprocess.Popen(
            spec.command,
            cwd=str(spec.cwd),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._record_process(spec.name, process)

        # A wall-clock deadline, not an iteration count: each pass also spends
        # up to 1.5 s on the health probe, so counting loops overshot the
        # configured budget by more than a factor of two.
        budget = spec.startup_timeout_sec or _startup_timeout_sec()
        deadline = time.monotonic() + budget
        while time.monotonic() < deadline:
            if _reachable(spec):
                elapsed = time.time() - started
                self._record_timing(spec.name, elapsed)
                print(f"[xgpu] {spec.name}: healthy in {elapsed:.0f}s", flush=True)
                return True
            if process.poll() is not None:
                reason = f"exited during startup; see {log_path}"
                self._mark_skipped(spec.name, reason)
                print(f"[xgpu] {spec.name}: {reason}", flush=True)
                return False
            time.sleep(1)

        reason = f"startup timed out after {budget}s; see {log_path}"
        self._mark_skipped(spec.name, reason)
        print(f"[xgpu] {spec.name}: {reason}", flush=True)
        return False

    def _start_with_retries(self, spec: ServiceSpec, retries: int) -> bool:
        for attempt in range(retries + 1):
            if self._start_one(spec):
                return True
            if attempt >= retries:
                return False
            reason = self.skipped.get(spec.name, "")
            if not _is_retryable_failure(reason):
                return False
            print(f"[xgpu] {spec.name}: retrying (attempt {attempt + 2}/{retries + 1}) after: {reason}", flush=True)
        return False

    def _eligible(self, specs: list[ServiceSpec], selected: set[str]) -> list[ServiceSpec]:
        eligible = []
        for spec in specs:
            if selected and spec.name not in selected:
                self._mark_skipped(spec.name, "not selected")
                continue
            eligible.append(spec)
        return eligible

    def _start_phase(self, specs: list[ServiceSpec], concurrency: int = 1) -> None:
        if not specs:
            return
        workers = max(1, min(concurrency, len(specs)))
        if workers == 1:
            for spec in specs:
                self._start_with_retries(spec, DEFAULT_START_RETRIES)
            return
        print(
            f"[xgpu] starting {len(specs)} worker(s) with concurrency {workers}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._start_with_retries, spec, DEFAULT_START_RETRIES): spec.name
                for spec in specs
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001 - isolate per-service failures
                    self._mark_skipped(name, f"{type(exc).__name__}: {exc}")
                    print(f"[xgpu] {name}: failed: {exc}", flush=True)

    def _start_selected_services(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["RSMEM_ENABLE_GPU_PERCEPTION"] = "1"
        runtime_failures: dict[str, str] = {}
        try:
            print(f"[xgpu] validating model assets in {MODELS}", flush=True)
            _, source_root = prepare_xgpu_assets(MODELS, WORKSPACE, LOCAL_ROOT, LOCAL_MIN_FREE_GIB)
            _apply_source_root(source_root)
            print(f"[xgpu] model assets and source trees are ready (sources under {source_root})", flush=True)
            if _truthy("RSMEM_GPU_BOOTSTRAP_ENVS", "1"):
                report_storage_options(WORKSPACE)
                print("[xgpu] validating persistent Python 3.10 worker environments", flush=True)
                interpreters, runtime_failures = ensure_xgpu_runtime_envs(
                    WORKSPACE, MODELS, LOCAL_ROOT, LOCAL_MIN_FREE_GIB
                )
                configure_runtime_environment(interpreters)
                print(
                    f"[xgpu] worker environments ready: {sorted(interpreters)}"
                    + (f"; unavailable: {sorted(runtime_failures)}" if runtime_failures else ""),
                    flush=True,
                )
        except Exception as exc:
            self._mark_skipped("asset-bootstrap", f"{type(exc).__name__}: {exc}")
            print(f"[xgpu] bootstrap failed: {type(exc).__name__}: {exc}", flush=True)
            return

        selected = {
            item.strip()
            for item in os.getenv("RSMEM_GPU_SERVICES", "").split(",")
            if item.strip()
        }
        by_name = {spec.name: spec for spec in self._eligible(service_specs(), selected)}

        # A runtime that is still extracting must not hold back the workers
        # backed by the environments that are already in place.
        for name, spec in list(by_name.items()):
            reason = runtime_failures.get(spec.runtime)
            if reason:
                self._mark_skipped(name, f"runtime {spec.runtime} unavailable: {reason}")
                by_name.pop(name)
                print(
                    f"[xgpu] {name}: skipped, runtime {spec.runtime} unavailable",
                    flush=True,
                )

        # Phase A: controller must be healthy before any OEA worker can report in.
        controller = by_name.pop(CONTROLLER_NAME, None)
        controller_ok = True
        if controller is not None:
            print("[xgpu] phase A: starting controller", flush=True)
            controller_ok = self._start_with_retries(controller, CONTROLLER_START_RETRIES)
            if not controller_ok:
                for name in list(by_name):
                    if name in OEA_DEPENDENT:
                        self._mark_skipped(name, "controller unavailable")
                        by_name.pop(name, None)
                        print(
                            f"[xgpu] {name}: skipped because controller is unavailable",
                            flush=True,
                        )

        # Phase B: remaining light workers in parallel (excludes the large VLM).
        qwen = by_name.pop(QWEN_NAME, None)
        phase_b = list(by_name.values())
        if phase_b:
            print(f"[xgpu] phase B: starting {len(phase_b)} worker(s)", flush=True)
            self._start_phase(phase_b, concurrency=_startup_concurrency())

        # Phase C: load Qwen3-VL last to keep GPU memory peaks predictable.
        if qwen is not None:
            print("[xgpu] phase C: starting Qwen3-VL", flush=True)
            self._start_with_retries(qwen, DEFAULT_START_RETRIES)

        timings = ", ".join(
            f"{name}={seconds:.0f}s"
            for name, seconds in sorted(self.startup_timings.items())
        )
        if timings:
            print(f"[xgpu] worker startup timings: {timings}", flush=True)

    def status(self) -> dict:
        with self._lock:
            started = {
                name: {"pid": process.pid, "running": process.poll() is None}
                for name, process in self.processes.items()
            }
            skipped = dict(self.skipped)
            timings = dict(self.startup_timings)
        return {
            "workspace": str(WORKSPACE),
            "started": started,
            "skipped": skipped,
            "startup_timings": timings,
            "log_dir": str(LOG_DIR),
            "startup_running": bool(self._startup_thread and self._startup_thread.is_alive()),
            "identity": _instance_identity(),
            "storage": _storage_snapshot(),
        }

    def stop(self) -> None:
        with self._lock:
            for process in self.processes.values():
                if process.poll() is None:
                    process.terminate()


_MANAGER: GPUServiceManager | None = None


def start_configured_gpu_services() -> GPUServiceManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = GPUServiceManager().start()
        atexit.register(_MANAGER.stop)
    return _MANAGER


def get_gpu_service_manager() -> GPUServiceManager | None:
    return _MANAGER
