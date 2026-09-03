"""Prepare persistent xGPU source/model directories and report missing assets.

This command does not accept or print tokens.  ModelScope authentication, when
needed, must be supplied through the platform's own environment/credential UI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_assets import DEFAULT_REPO_ID, prepare_xgpu_assets
from xgpu_runtime import configure_runtime_environment, ensure_xgpu_runtime_envs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="/mnt/workspace")
    parser.add_argument(
        "--bootstrap-envs",
        action="store_true",
        help="extract the packaged EarthBench and OEA Python 3.10 environments",
    )
    parser.add_argument(
        "--download-deployment-bundle",
        action="store_true",
        help=f"download and verify the complete offline bundle from {DEFAULT_REPO_ID}",
    )
    args = parser.parse_args()
    root = Path(args.workspace).expanduser().resolve()
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)

    if args.download_deployment_bundle:
        prepare_xgpu_assets(models, root)

    if args.bootstrap_envs:
        interpreters = ensure_xgpu_runtime_envs(root, models)
        configure_runtime_environment(interpreters)
        for name, python in interpreters.items():
            print(f"runtime  {name}: {python}")

    required = [
        root / "RSMeM-public" / "tools" / "deploy_tools" / "remoteclip_api.py",
        root / "OpenEarthAgent" / "tool_server" / "tool_workers" / "online_workers" / "controller.py",
        models / "sam2.1_hiera_small.pt",
        models / "RemoteCLIP-ViT-L-14.pt",
        models / "changeos_r101.pt",
        models / "SiamCRNN_20250211_033615_1" / "best_model.pth",
        models / "groundingdino_swint_ogc.pth",
        models / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py",
        models / "lae_dino_swint_lae1m-28ca3a15.pth",
        models / "bert-base-uncased" / "model.safetensors",
        models / "bert-base-uncased" / "tokenizer.json",
        models / "LAE-DINO" / "mmdetection_lae" / "configs" / "lae_dino" / "lae_dino_swin-t_pretrain_LAE-1M.py",
        models / "Qwen3-VL-8B-Instruct" / "config.json",
        models / "Qwen3-VL-8B-Instruct" / "model-00001-of-00004.safetensors",
        models / "Qwen3-VL-8B-Instruct" / "model-00002-of-00004.safetensors",
        models / "Qwen3-VL-8B-Instruct" / "model-00003-of-00004.safetensors",
        models / "Qwen3-VL-8B-Instruct" / "model-00004-of-00004.safetensors",
    ]
    missing = [path for path in required if not path.exists()]
    print(f"workspace: {root}")
    if missing:
        print("missing assets:")
        for path in missing:
            print(f"  - {path}")
        return 2
    print("all required source trees and checkpoints are present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
