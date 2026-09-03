"""Persistent model/source bootstrap for the ModelScope xGPU Studio."""

from __future__ import annotations

import fcntl
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_REPO_ID = "wbx929/RSMeM-Perception-Models"

# Exact byte sizes from the tested QY_11 deployment.  Validate every large
# Qwen shard as well as its index so a later partial deletion cannot be mistaken
# for a ready persistent cache.  snapshot_download additionally validates the
# repository objects while downloading them.
EXPECTED_FILES = {
    "sam2.1_hiera_small.pt": 184_416_285,
    "RemoteCLIP-ViT-L-14.pt": 1_710_613_765,
    "changeos_r101.pt": 233_553_904,
    "SiamCRNN_20250211_033615_1/best_model.pth": 132_432_363,
    "groundingdino_swint_ogc.pth": 693_997_677,
    "lae_dino_swint_lae1m-28ca3a15.pth": 724_694_893,
    "Qwen3-VL-8B-Instruct/config.json": 1_474,
    "Qwen3-VL-8B-Instruct/model-00001-of-00004.safetensors": 4_902_275_944,
    "Qwen3-VL-8B-Instruct/model-00002-of-00004.safetensors": 4_915_962_496,
    "Qwen3-VL-8B-Instruct/model-00003-of-00004.safetensors": 4_999_831_048,
    "Qwen3-VL-8B-Instruct/model-00004-of-00004.safetensors": 2_716_270_024,
    "Qwen3-VL-8B-Instruct/model.safetensors.index.json": 67_759,
    "bert-base-uncased/config.json": 570,
    "bert-base-uncased/model.safetensors": 440_449_768,
    "bert-base-uncased/tokenizer.json": 466_062,
    "bert-base-uncased/tokenizer_config.json": 48,
    "bert-base-uncased/vocab.txt": 231_508,
}


def assets_ready(models_dir: Path) -> bool:
    return all(
        (models_dir / relative).is_file()
        and (models_dir / relative).stat().st_size == expected_size
        for relative, expected_size in EXPECTED_FILES.items()
    )


def ensure_model_assets(models_dir: Path, workspace: Path) -> Path:
    """Download the private bundle once, under an inter-process lock."""
    models_dir.mkdir(parents=True, exist_ok=True)
    if assets_ready(models_dir):
        return models_dir

    lock_path = workspace / ".rsmem-model-download.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if assets_ready(models_dir):
            return models_dir

        from modelscope import snapshot_download

        repo_id = os.getenv("RSMEM_MODELSCOPE_MODEL_REPO", DEFAULT_REPO_ID)
        token = os.getenv("MODELSCOPE_API_TOKEN") or None
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=os.getenv("RSMEM_MODEL_REVISION", "master"),
            local_dir=str(models_dir),
            token=token,
            max_workers=int(os.getenv("RSMEM_MODEL_DOWNLOAD_WORKERS", "4")),
            enable_file_lock=True,
        )
        if not assets_ready(models_dir):
            missing = [
                relative
                for relative, size in EXPECTED_FILES.items()
                if not (models_dir / relative).is_file()
                or (models_dir / relative).stat().st_size != size
            ]
            raise RuntimeError(
                "Model repository download completed but validation failed: "
                + ", ".join(missing)
            )
    return models_dir


def _clone_once(url: str, destination: Path) -> None:
    if destination.is_dir():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(destination)],
        check=True,
    )


def ensure_source_trees(workspace: Path, models_dir: Path) -> None:
    """Clone public worker/config sources only on the first xGPU startup."""
    _clone_once("https://github.com/AI9Stars/RSMeM.git", workspace / "RSMeM-public")
    _clone_once("https://github.com/mbzuai-oryx/OpenEarthAgent.git", workspace / "OpenEarthAgent")
    _clone_once("https://github.com/IDEA-Research/GroundingDINO.git", models_dir / "GroundingDINO")
    _clone_once("https://github.com/jaychempan/LAE-DINO.git", models_dir / "LAE-DINO")

    # Keep the deployed workers byte-for-byte aligned with the versions tested
    # on QY_11.  The public repositories remain the source trees; these three
    # small overrides contain deployment fixes (ChangeOS preprocessing, LAE
    # BERT attention, and register-only-after-model-init ordering).
    override_root = Path(__file__).resolve().parent / "xgpu_workers"
    overrides = {
        override_root / "changeos_api.py": (
            workspace / "RSMeM-public" / "tools" / "deploy_tools" / "changeos_api.py"
        ),
        override_root / "ObjectDetection_worker.py": (
            workspace / "OpenEarthAgent" / "tool_server" / "tool_workers"
            / "online_workers" / "ObjectDetection_worker.py"
        ),
        override_root / "base_tool_worker.py": (
            workspace / "OpenEarthAgent" / "tool_server" / "tool_workers"
            / "online_workers" / "base_tool_worker.py"
        ),
    }
    for source, destination in overrides.items():
        if not source.is_file():
            raise RuntimeError(f"Required worker override is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    # The tested LAE-DINO config contains a QY_11-specific absolute BERT path.
    # Replace only that assignment in the persistent clone; the checkpoint and
    # tokenizer themselves remain immutable files in the model bundle.
    lae_config = (
        models_dir
        / "LAE-DINO"
        / "mmdetection_lae"
        / "configs"
        / "lae_dino"
        / "lae_dino_swin-t_pretrain_LAE-1M.py"
    )
    if lae_config.is_file():
        source = lae_config.read_text(encoding="utf-8")
        lines = source.splitlines()
        replacement = f"lang_model_name = {str(models_dir / 'bert-base-uncased')!r}"
        patched = [replacement if line.startswith("lang_model_name =") else line for line in lines]
        if patched != lines:
            lae_config.write_text("\n".join(patched) + "\n", encoding="utf-8")


def prepare_xgpu_assets(models_dir: Path, workspace: Path) -> Path:
    ensure_model_assets(models_dir, workspace)
    ensure_source_trees(workspace, models_dir)
    easyocr_dir = workspace / "easyocr"
    easyocr_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("EASYOCR_MODULE_PATH", str(easyocr_dir))
    return models_dir
