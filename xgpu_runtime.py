"""Create persistent, isolated Python runtimes for xGPU perception workers.

The Gradio image remains the application environment.  GPU workers use three
Python 3.10 prefixes under ``/mnt/workspace/envs`` so incompatible OpenMMLab
and EarthBench dependencies never leak into the web application.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


ENV_SCHEMA = 1
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
MMCV_INDEX = "https://miropsota.github.io/torch_packages_builder"


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    torch: tuple[str, ...]
    packages: tuple[str, ...]
    editable: tuple[Path, ...] = ()
    extra: tuple[tuple[str, ...], ...] = ()
    probes: tuple[str, ...] = ()


def _manager() -> str:
    configured = os.getenv("RSMEM_CONDA_EXE")
    if configured:
        return configured
    for candidate in ("micromamba", "mamba", "conda"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise RuntimeError(
        "A conda-compatible executable is required to create persistent "
        "Python 3.10 GPU runtimes. Set RSMEM_CONDA_EXE to conda, mamba, or micromamba."
    )


def _run(command: list[str], **kwargs) -> None:
    subprocess.run(command, check=True, **kwargs)


def _create_prefix(manager: str, prefix: Path) -> None:
    if (prefix / "bin" / "python").is_file():
        return
    prefix.parent.mkdir(parents=True, exist_ok=True)
    base = Path(manager).name
    command = [manager, "create", "-y", "-p", str(prefix), "python=3.10", "pip"]
    if base == "conda":
        command.insert(2, "--quiet")
    _run(command)


def _pip(python: Path, *arguments: str) -> None:
    _run([str(python), "-m", "pip", *arguments])


def _marker_matches(prefix: Path, spec: RuntimeSpec) -> bool:
    marker = prefix / ".rsmem-runtime.json"
    python = prefix / "bin" / "python"
    if not marker.is_file() or not python.is_file():
        return False
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return metadata == {"schema": ENV_SCHEMA, "name": spec.name}


def _install(prefix: Path, spec: RuntimeSpec) -> None:
    python = prefix / "bin" / "python"
    _pip(python, "install", "--upgrade", "pip", "wheel", "setuptools<81")
    _pip(python, "install", *spec.torch, "--index-url", TORCH_INDEX)
    if spec.packages:
        _pip(python, "install", *spec.packages)
    for command in spec.extra:
        _pip(python, *command)
    for source in spec.editable:
        if not source.is_dir():
            raise RuntimeError(f"Editable dependency is missing: {source}")
        _pip(python, "install", "-e", str(source), "--no-build-isolation")
    for probe in spec.probes:
        _run([str(python), "-c", probe])
    (prefix / ".rsmem-runtime.json").write_text(
        json.dumps({"schema": ENV_SCHEMA, "name": spec.name}, indent=2) + "\n",
        encoding="utf-8",
    )


def runtime_specs(workspace: Path, models: Path) -> tuple[RuntimeSpec, ...]:
    return (
        RuntimeSpec(
            name="earthbench",
            torch=("torch==2.4.0", "torchvision==0.19.0"),
            packages=(
                "numpy<2", "fastapi", "uvicorn[standard]", "opencv-python-headless==4.9.0.80",
                "pydantic", "pillow", "python-multipart", "timm", "einops", "hydra-core",
                "omegaconf", "iopath", "open-clip-torch", "openai", "requests", "tabulate",
                "ftfy", "regex", "changeos==0.2", "scipy", "scikit-image",
            ),
            probes=(
                "import torch, open_clip, cv2, changeos; assert torch.cuda.is_available()",
                "import fastapi, uvicorn, skimage, hydra, iopath",
            ),
        ),
        RuntimeSpec(
            name="oea-e1",
            torch=("torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"),
            packages=(
                "numpy<2", "python-box", "loguru", "openai", "tiktoken", "uvicorn",
                "fastapi", "func_timeout", "easyocr", "rasterio", "accelerate",
                "transformers==4.57.3", "qwen-vl-utils", "python-multipart", "addict", "yapf",
                "timm", "opencv-python-headless", "supervision>=0.22.0", "pycocotools",
                "matplotlib", "pyyaml", "tqdm",
            ),
            editable=(models / "GroundingDINO",),
            probes=(
                "import torch, easyocr, transformers; assert torch.cuda.is_available()",
                "import groundingdino, rasterio",
            ),
        ),
        RuntimeSpec(
            name="oea-e3",
            torch=("torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"),
            packages=(
                "numpy<2", "gdown", "fastapi==0.115.12", "uvicorn==0.34.2",
                "transformers==4.42.4", "open_clip_torch", "fairscale", "python-box", "loguru",
                "requests", "pillow", "pyyaml", "tqdm",
            ),
            extra=(("install", "mmcv==2.2.0+pt2.5.1cu121", "--extra-index-url", MMCV_INDEX),),
            editable=(models / "LAE-DINO" / "mmdetection_lae",),
            probes=(
                "import torch, mmcv, mmdet; assert torch.cuda.is_available()",
                "import transformers; assert transformers.__version__ == '4.42.4'",
            ),
        ),
    )


def ensure_xgpu_runtime_envs(workspace: Path, models: Path) -> dict[str, Path]:
    """Build each environment once and return its interpreter path."""
    env_root = workspace / "envs"
    lock_path = workspace / ".rsmem-env-bootstrap.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manager = _manager()
        interpreters: dict[str, Path] = {}
        for spec in runtime_specs(workspace, models):
            prefix = env_root / spec.name
            _create_prefix(manager, prefix)
            if not _marker_matches(prefix, spec):
                _install(prefix, spec)
            interpreters[spec.name] = prefix / "bin" / "python"
    return interpreters


def configure_runtime_environment(interpreters: dict[str, Path]) -> None:
    earth = str(interpreters["earthbench"])
    os.environ.setdefault("RSMEM_EARTHBENCH_PYTHON", earth)
    os.environ.setdefault("RSMEM_REMOTECLIP_PYTHON", earth)
    os.environ.setdefault("RSMEM_CHANGEOS_PYTHON", earth)
    os.environ.setdefault("RSMEM_BRIGHT_PYTHON", earth)
    os.environ.setdefault("RSMEM_OEA_E1_PYTHON", str(interpreters["oea-e1"]))
    os.environ.setdefault("RSMEM_OEA_E3_PYTHON", str(interpreters["oea-e3"]))
