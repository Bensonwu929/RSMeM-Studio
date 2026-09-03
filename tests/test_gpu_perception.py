from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import gpu_service_manager as manager
import model_assets
import xgpu_runtime


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
    assert model_assets.assets_ready(tmp_path)
    (tmp_path / "Qwen3-VL-8B-Instruct" / "config.json").write_bytes(b"short")
    assert not model_assets.assets_ready(tmp_path)


def test_runtime_environment_maps_shared_earthbench_and_isolated_oea(tmp_path, monkeypatch):
    interpreters = {
        "earthbench": tmp_path / "earth" / "python",
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
    assert manager.service_specs()[0].command[0] == str(interpreters["earthbench"])
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
    monkeypatch.setattr(model_assets, "_clone_once", lambda *_args: None)
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
    monkeypatch.setattr(model_assets, "_clone_once", lambda *_args: None)
    model_assets.ensure_source_trees(tmp_path, models)
    assert "_preprocess_pair" in targets[0].read_text(encoding="utf-8")
    assert "_patch_bert_sdpa_mask_utils" in targets[1].read_text(encoding="utf-8")
    assert targets[2].read_text(encoding="utf-8").index("self.init_model()") < targets[2].read_text(encoding="utf-8").index("self.register_to_controller()")
