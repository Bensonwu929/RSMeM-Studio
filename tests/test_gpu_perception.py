from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import gpu_service_manager as manager
import model_assets
import xgpu_runtime


def _write_offline_bundles(models: Path) -> None:
    checksums = []
    for relative in model_assets.OFFLINE_BUNDLE_FILES:
        path = models / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
        checksums.append(f"{model_assets._sha256(path)}  {path.name}")
    manifest = models / model_assets.OFFLINE_CHECKSUMS
    manifest.write_text("\n".join(checksums) + "\n", encoding="utf-8")
    model_assets._verify_offline_bundles(models)


def _mark_source_ready(workspace: Path, models: Path) -> None:
    archive = models / "offline" / "rsmem-perception-sources-v1.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz"):
        pass
    required = (
        workspace / "RSMeM-public" / "tools" / "deploy_tools" / "sam2_api.py",
        workspace / "OpenEarthAgent" / "tool_server" / "tool_workers" / "online_workers" / "controller.py",
        workspace / "OpenEarthAgent" / "tool_server" / "utils" / "server_utils.py",
        models / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py",
        models / "LAE-DINO" / "mmdetection_lae" / "__init__.py",
        workspace / "easyocr" / "model" / "craft_mlt_25k.pth",
    )
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    marker = workspace / ".rsmem-source-bundle.json"
    marker.write_text(json.dumps({"sha256": model_assets._sha256(archive)}), encoding="utf-8")


def _load_perception(tmp_path: Path, monkeypatch):
    module_path = Path(__file__).parents[1] / "tools" / "Perception.py"
    monkeypatch.setattr(sys, "argv", [str(module_path), "--temp_dir", str(tmp_path)])
    spec = importlib.util.spec_from_file_location("test_perception_facade", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sam2_artifact_is_copied_into_request_directory(tmp_path, monkeypatch):
    perception = _load_perception(tmp_path, monkeypatch)

    class Response:
        content = b"png-bytes"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(perception._session, "get", lambda *args, **kwargs: Response())
    destination = tmp_path / "mask.png"
    perception._download_worker_artifact("/results/request_mask.png", destination)
    assert destination.read_bytes() == b"png-bytes"


def test_sam2_artifact_rejects_external_url(tmp_path, monkeypatch):
    perception = _load_perception(tmp_path, monkeypatch)
    destination = tmp_path / "mask.png"
    try:
        perception._download_worker_artifact("https://example.com/mask.png", destination)
    except RuntimeError as exc:
        assert "invalid result URL" in str(exc)
    else:
        raise AssertionError("external artifact URL was accepted")


def test_gpu_manager_skips_service_when_checkpoint_is_missing(tmp_path, monkeypatch):
    source = tmp_path / "worker.py"
    source.write_text("pass\n", encoding="utf-8")
    missing = tmp_path / "model.pt"
    spec = manager.ServiceSpec(
        "test-worker",
        (sys.executable, str(source)),
        tmp_path,
        "http://127.0.0.1:9/health",
        required_paths=(missing,),
    )
    monkeypatch.setattr(manager, "service_specs", lambda: [spec])
    monkeypatch.setattr(manager, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(manager, "prepare_xgpu_assets", lambda *_args: None)
    # Track the manager's process-wide opt-in mutation so pytest restores it.
    monkeypatch.setenv("RSMEM_ENABLE_GPU_PERCEPTION", "0")
    monkeypatch.setenv("RSMEM_GPU_BOOTSTRAP_ENVS", "0")
    instance = manager.GPUServiceManager()
    instance._start_selected_services()
    assert instance.processes == {}
    assert str(missing) in instance.skipped["test-worker"]


def test_asset_validation_uses_exact_qy11_sizes(tmp_path):
    for relative, size in model_assets.EXPECTED_FILES.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            handle.truncate(size)
    _write_offline_bundles(tmp_path)
    assert model_assets.assets_ready(tmp_path)
    (tmp_path / "Qwen3-VL-8B-Instruct" / "config.json").write_bytes(b"short")
    assert not model_assets.assets_ready(tmp_path)


def test_runtime_environment_preserves_qy11_isolation_boundaries(tmp_path, monkeypatch):
    interpreters = {
        "sam2": tmp_path / "sam2" / "python",
        "remoteclip": tmp_path / "remoteclip" / "python",
        "changeos": tmp_path / "changeos" / "python",
        "bright": tmp_path / "bright" / "python",
        "oea-e1": tmp_path / "e1" / "python",
        "oea-e3": tmp_path / "e3" / "python",
    }
    for variable in (
        "RSMEM_EARTHBENCH_PYTHON", "RSMEM_REMOTECLIP_PYTHON",
        "RSMEM_CHANGEOS_PYTHON", "RSMEM_BRIGHT_PYTHON",
        "RSMEM_OEA_E1_PYTHON", "RSMEM_OEA_E3_PYTHON",
    ):
        monkeypatch.delenv(variable, raising=False)
    xgpu_runtime.configure_runtime_environment(interpreters)
    assert manager.service_specs()[0].command[0] == str(interpreters["sam2"])
    assert manager.service_specs()[1].command[0] == str(interpreters["remoteclip"])
    assert manager.service_specs()[2].command[0] == str(interpreters["changeos"])
    assert manager.service_specs()[3].command[0] == str(interpreters["bright"])
    assert manager.service_specs()[4].command[0] == str(interpreters["oea-e1"])
    assert manager.service_specs()[6].command[0] == str(interpreters["oea-e3"])


def test_remoteclip_worker_is_forced_to_facade_port(monkeypatch):
    spec = next(item for item in manager.service_specs() if item.name == "earthbench-remoteclip")
    assert spec.health_url == "http://127.0.0.1:5555/health"
    assert dict(spec.environment)["REMOTECLIP_PORT"] == "5555"


def test_lae_config_rewrites_qy11_bert_path(tmp_path, monkeypatch):
    models = tmp_path / "models"
    config = (
        models / "LAE-DINO" / "mmdetection_lae" / "configs" / "lae_dino"
        / "lae_dino_swin-t_pretrain_LAE-1M.py"
    )
    config.parent.mkdir(parents=True)
    config.write_text("lang_model_name = '/server-only/bert'\nother = 1\n", encoding="utf-8")
    _mark_source_ready(tmp_path, models)
    model_assets.ensure_source_trees(tmp_path, models)
    rewritten = config.read_text(encoding="utf-8")
    assert str(models / "bert-base-uncased") in rewritten
    assert "/server-only/bert" not in rewritten


def test_tested_worker_overrides_replace_cloned_sources(tmp_path, monkeypatch):
    models = tmp_path / "models"
    targets = [
        tmp_path / "RSMeM-public" / "tools" / "deploy_tools" / "changeos_api.py",
        tmp_path / "OpenEarthAgent" / "tool_server" / "tool_workers" / "online_workers"
        / "ObjectDetection_worker.py",
        tmp_path / "OpenEarthAgent" / "tool_server" / "tool_workers" / "online_workers"
        / "base_tool_worker.py",
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("stale\n", encoding="utf-8")
    lae = (
        models / "LAE-DINO" / "mmdetection_lae" / "configs" / "lae_dino"
        / "lae_dino_swin-t_pretrain_LAE-1M.py"
    )
    lae.parent.mkdir(parents=True)
    lae.write_text("lang_model_name = '/old'\n", encoding="utf-8")
    _mark_source_ready(tmp_path, models)
    model_assets.ensure_source_trees(tmp_path, models)
    assert "_preprocess_pair" in targets[0].read_text(encoding="utf-8")
    assert "_patch_bert_sdpa_mask_utils" in targets[1].read_text(encoding="utf-8")
    assert targets[2].read_text(encoding="utf-8").index("self.init_model()") < targets[2].read_text(encoding="utf-8").index("self.register_to_controller()")


def test_offline_runtime_extracts_without_conda_or_pip(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    checksums = []
    for relative in xgpu_runtime.OFFLINE_ENV_ARCHIVES.values():
        archive = models / relative
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as bundle:
            payload = b"#!/bin/sh\n"
            info = tarfile.TarInfo("bin/python")
            info.mode = 0o755
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
        checksums.append(f"{xgpu_runtime._sha256(archive)}  {archive.name}")
    (models / "offline" / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(xgpu_runtime.subprocess, "run", lambda *args, **kwargs: None)
    interpreters = xgpu_runtime.ensure_xgpu_runtime_envs(workspace, models)
    assert set(interpreters) == {
        "sam2", "remoteclip", "changeos", "bright", "oea-e1", "oea-e3"
    }
    assert all(path.is_file() for path in interpreters.values())
    assert not any((models / relative).exists() for relative in xgpu_runtime.OFFLINE_ENV_ARCHIVES.values())


def test_runtime_downloads_only_missing_archive_and_deletes_it(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    download_root = tmp_path / "remote"
    checksums = []
    for relative in xgpu_runtime.OFFLINE_ENV_ARCHIVES.values():
        archive = download_root / relative
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "w:gz") as bundle:
            payload = b"#!/bin/sh\n"
            info = tarfile.TarInfo("bin/python")
            info.mode = 0o755
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
        checksums.append(f"{xgpu_runtime._sha256(archive)}  {archive.name}")
    (models / "offline").mkdir(parents=True)
    (models / "offline" / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    downloaded = []

    def fake_download(_models, relative, _workspace):
        source = download_root / relative
        target = workspace / ".rsmem-runtime-downloads" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        downloaded.append(Path(relative).name)
        return target

    monkeypatch.setattr(xgpu_runtime, "_download_archive", fake_download)
    monkeypatch.setattr(xgpu_runtime.subprocess, "run", lambda *args, **kwargs: None)
    interpreters = xgpu_runtime.ensure_xgpu_runtime_envs(workspace, models)
    assert downloaded == [Path(item).name for item in xgpu_runtime.OFFLINE_ENV_ARCHIVES.values()]
    assert all(path.is_file() for path in interpreters.values())
    assert not (workspace / ".rsmem-runtime-downloads" / "offline").exists() or not any(
        (workspace / ".rsmem-runtime-downloads" / "offline").iterdir()
    )
