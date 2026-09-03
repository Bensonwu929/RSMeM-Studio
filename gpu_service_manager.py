"""Start the configured xGPU perception workers without embedding models in Gradio.

All large sources and checkpoints live under ``/mnt/workspace`` by default.  The
Git repository contains only launch metadata; this module never downloads or
serializes credentials.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from model_assets import prepare_xgpu_assets
from xgpu_runtime import configure_runtime_environment, ensure_xgpu_runtime_envs


WORKSPACE = Path(os.getenv("RSMEM_XGPU_WORKSPACE", "/mnt/workspace"))
RS_ROOT = Path(os.getenv("RSMEM_EARTHBENCH_ROOT", str(WORKSPACE / "RSMeM-public")))
OEA_ROOT = Path(os.getenv("RSMEM_OEA_ROOT", str(WORKSPACE / "OpenEarthAgent")))
MODELS = Path(os.getenv("RSMEM_MODELS_DIR", str(WORKSPACE / "models")))
LOG_DIR = Path(os.getenv("RSMEM_GPU_LOG_DIR", str(WORKSPACE / "rsmem-gpu-logs")))


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    command: tuple[str, ...]
    cwd: Path
    health_url: str
    expected_model: str = ""
    required_paths: tuple[Path, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()


def _python(variable: str) -> str:
    return os.getenv(variable, sys.executable)


def service_specs() -> list[ServiceSpec]:
    deploy = RS_ROOT / "tools" / "deploy_tools"
    workers = OEA_ROOT / "tool_server" / "tool_workers" / "online_workers"
    controller = "http://127.0.0.1:21001"
    return [
        ServiceSpec("earthbench-sam2", (_python("RSMEM_EARTHBENCH_PYTHON"), str(deploy / "sam2_api.py")), deploy, "http://127.0.0.1:5554/health", required_paths=(MODELS / "sam2.1_hiera_small.pt",)),
        ServiceSpec("earthbench-remoteclip", (_python("RSMEM_REMOTECLIP_PYTHON"), str(deploy / "remoteclip_api.py")), deploy, "http://127.0.0.1:5555/health", required_paths=(MODELS / "RemoteCLIP-ViT-L-14.pt",), environment=(("REMOTECLIP_PORT", "5555"),)),
        ServiceSpec("earthbench-changeos", (_python("RSMEM_CHANGEOS_PYTHON"), str(deploy / "changeos_api.py")), deploy, "http://127.0.0.1:8019/health", required_paths=(MODELS / "changeos_r101.pt",)),
        ServiceSpec("earthbench-bright", (_python("RSMEM_BRIGHT_PYTHON"), str(deploy / "siamcrnn_api_biany.py")), deploy, "http://127.0.0.1:8120/health", required_paths=(MODELS / "SiamCRNN_20250211_033615_1" / "best_model.pth",)),
        ServiceSpec("oea-controller", (_python("RSMEM_OEA_E1_PYTHON"), str(workers / "controller.py"), "--host", "0.0.0.0", "--port", "21001"), workers, f"{controller}/list_models"),
        ServiceSpec("oea-grounding-dino", (_python("RSMEM_OEA_E1_PYTHON"), str(workers / "TextToBbox_worker.py"), "--controller-address", controller, "--host", "0.0.0.0", "--port", "21009", "--model-path", str(MODELS / "groundingdino_swint_ogc.pth"), "--model-config", str(MODELS / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"), "--limit-model-concurrency", "1"), workers, f"{controller}/list_models", "TextToBbox", (MODELS / "groundingdino_swint_ogc.pth", MODELS / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")),
        ServiceSpec("oea-lae-dino", (_python("RSMEM_OEA_E3_PYTHON"), str(workers / "ObjectDetection_worker.py"), "--controller-address", controller, "--host", "0.0.0.0", "--port", "21016", "--model-path", str(MODELS / "lae_dino_swint_lae1m-28ca3a15.pth"), "--model-config", str(MODELS / "LAE-DINO" / "mmdetection_lae" / "configs" / "lae_dino" / "lae_dino_swin-t_pretrain_LAE-1M.py"), "--limit-model-concurrency", "1"), workers, f"{controller}/list_models", "ObjectDetection", (MODELS / "lae_dino_swint_lae1m-28ca3a15.pth", MODELS / "LAE-DINO" / "mmdetection_lae" / "configs" / "lae_dino" / "lae_dino_swin-t_pretrain_LAE-1M.py")),
        ServiceSpec("oea-ocr", (_python("RSMEM_OEA_E1_PYTHON"), str(workers / "OCR_worker.py"), "--controller-address", controller, "--host", "0.0.0.0", "--port", "21003", "--limit-model-concurrency", "1"), workers, f"{controller}/list_models", "OCR"),
        # Load the 22.5 GB VLM last to keep initialization peaks predictable.
        ServiceSpec("oea-qwen3-vl", (_python("RSMEM_OEA_E1_PYTHON"), str(workers / "BaseModel_worker.py"), "--controller-address", controller, "--host", "0.0.0.0", "--port", "21010", "--model_path", str(MODELS / "Qwen3-VL-8B-Instruct"), "--limit-model-concurrency", "1"), workers, f"{controller}/list_models", "BaseModel", (MODELS / "Qwen3-VL-8B-Instruct" / "config.json",)),
    ]


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


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


class GPUServiceManager:
    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self.skipped: dict[str, str] = {}
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

    def _start_selected_services(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["RSMEM_ENABLE_GPU_PERCEPTION"] = "1"
        try:
            print(f"[xgpu] validating model assets in {MODELS}", flush=True)
            prepare_xgpu_assets(MODELS, WORKSPACE)
            print("[xgpu] model assets and source trees are ready", flush=True)
            if _truthy("RSMEM_GPU_BOOTSTRAP_ENVS", "1"):
                print("[xgpu] validating persistent Python 3.10 worker environments", flush=True)
                interpreters = ensure_xgpu_runtime_envs(WORKSPACE, MODELS)
                configure_runtime_environment(interpreters)
                print("[xgpu] persistent worker environments are ready", flush=True)
        except Exception as exc:
            self.skipped["asset-bootstrap"] = f"{type(exc).__name__}: {exc}"
            print(f"[xgpu] bootstrap failed: {type(exc).__name__}: {exc}", flush=True)
            return
        selected = {
            item.strip() for item in os.getenv("RSMEM_GPU_SERVICES", "").split(",") if item.strip()
        }
        for spec in service_specs():
            if selected and spec.name not in selected:
                self.skipped[spec.name] = "not selected"
                continue
            script = Path(spec.command[1])
            if not script.is_file():
                self.skipped[spec.name] = f"missing source: {script}"
                continue
            missing = [path for path in spec.required_paths if not path.exists()]
            if missing:
                self.skipped[spec.name] = "missing assets: " + ", ".join(str(path) for path in missing)
                continue
            if _reachable(spec):
                self.skipped[spec.name] = "already running"
                print(f"[xgpu] {spec.name}: already healthy", flush=True)
                continue
            log_path = LOG_DIR / f"{spec.name}.log"
            print(f"[xgpu] {spec.name}: starting; log={log_path}", flush=True)
            handle = log_path.open("ab", buffering=0)
            environment = os.environ.copy()
            environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
            environment.setdefault("RSMEM_MODELS_DIR", str(MODELS))
            environment.setdefault("EASYOCR_MODULE_PATH", str(WORKSPACE / "easyocr"))
            python_paths = [
                str(RS_ROOT),
                str(RS_ROOT / "tools" / "deploy_tools"),
                str(OEA_ROOT),
                str(MODELS / "GroundingDINO"),
                str(MODELS / "LAE-DINO" / "mmdetection_lae"),
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
            self.processes[spec.name] = process
            wait_seconds = int(os.getenv("RSMEM_GPU_STARTUP_TIMEOUT_SEC", "300"))
            for _ in range(wait_seconds):
                if _reachable(spec):
                    print(f"[xgpu] {spec.name}: healthy", flush=True)
                    break
                if process.poll() is not None:
                    self.skipped[spec.name] = f"exited during startup; see {log_path}"
                    print(f"[xgpu] {spec.name}: exited during startup; see {log_path}", flush=True)
                    break
                time.sleep(1)
            else:
                self.skipped[spec.name] = f"startup timed out; see {log_path}"
                print(f"[xgpu] {spec.name}: startup timed out; see {log_path}", flush=True)

    def status(self) -> dict:
        return {
            "workspace": str(WORKSPACE),
            "started": {
                name: {"pid": process.pid, "running": process.poll() is None}
                for name, process in self.processes.items()
            },
            "skipped": dict(self.skipped),
            "log_dir": str(LOG_DIR),
            "startup_running": bool(self._startup_thread and self._startup_thread.is_alive()),
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
