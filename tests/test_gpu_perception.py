from __future__ import annotations

import builtins
import importlib.util
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import pytest

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


class _FakeHttpResponse:
    def __init__(self, payload=None, content: bytes = b""):
        self._payload = payload or {}
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _sam2_ready_image(perception, tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(perception, "_ready", lambda _name: True)
    image = tmp_path / "scene.png"
    image.write_bytes(b"fake-image-bytes")
    return image


def test_sam2_passes_an_xyxy_bbox_through_to_the_worker_unmodified(tmp_path, monkeypatch):
    """RSMeM-public/tools/deploy_tools/sam2_api.py documents bbox_str as
    (x, y, w, h) but hands it straight to SAM2's predict(box=...), which
    requires XYXY. Passing [x, y, width, height] here once produced a
    degenerate box (x2 < x1) and a mask nowhere near the requested object;
    the fix is to declare and pass through xyxy so no conversion ever
    happens on either side."""
    perception = _load_perception(tmp_path, monkeypatch)
    image = _sam2_ready_image(perception, tmp_path, monkeypatch)
    captured = {}

    def fake_post(_url, files, data, timeout):
        captured["bbox_str"] = data["bbox_str"]
        return _FakeHttpResponse({
            "mask_url": "/results/mask.png",
            "bounding_boxes": [[717, 209, 1232, 420]],
        })

    monkeypatch.setattr(perception._session, "post", fake_post)
    monkeypatch.setattr(
        perception._session, "get", lambda *a, **k: _FakeHttpResponse(content=b"mask-bytes")
    )

    result = perception.SAM2(str(image), [717, 209, 1232, 420])

    assert captured["bbox_str"] == "717,209,1232,420"
    assert Path(result["output_path"]).name == "sam2_mask.png"
    assert Path(result["output_path"]).read_bytes() == b"mask-bytes"
    assert "warning" not in result


def test_sam2_rejects_a_degenerate_box(tmp_path, monkeypatch):
    perception = _load_perception(tmp_path, monkeypatch)
    image = _sam2_ready_image(perception, tmp_path, monkeypatch)

    # This is exactly [x, y, width, height] for the box (717,209,1232,420)
    # misread as xyxy -- x2 (515) < x1 (717), the real incident this guards.
    with pytest.raises(ValueError, match="x2 > x1"):
        perception.SAM2(str(image), [717, 209, 515, 211])


def test_sam2_downloads_the_boxed_visualization_as_a_local_artifact(tmp_path, monkeypatch):
    """The worker also renders a boxed/masked visualization -- what a user
    asking for "the image with the box drawn on it" actually wants -- but
    only the raw mask ever reached the request-scoped output directory
    before this fix, so that image could never be delivered."""
    perception = _load_perception(tmp_path, monkeypatch)
    image = _sam2_ready_image(perception, tmp_path, monkeypatch)

    monkeypatch.setattr(
        perception._session, "post",
        lambda *a, **k: _FakeHttpResponse({
            "mask_url": "/results/mask.png",
            "visualization_url": "/results/vis.jpg",
            "bounding_boxes": [[717, 209, 1232, 420]],
        }),
    )
    monkeypatch.setattr(
        perception._session, "get", lambda *a, **k: _FakeHttpResponse(content=b"vis-bytes")
    )

    result = perception.SAM2(str(image), [717, 209, 1232, 420])

    assert Path(result["visualization_path"]).name == "sam2_mask_vis.jpg"
    assert Path(result["visualization_path"]).read_bytes() == b"vis-bytes"
    # Worker-internal /results paths are unreachable by anyone else; they
    # must not survive into a payload the model might cite in its answer.
    assert "mask_url" not in result
    assert "visualization_url" not in result


def test_sam2_warns_when_the_mask_has_no_contour(tmp_path, monkeypatch):
    """An empty bounding_boxes list is the worker's own signal that the mask
    found nothing -- exactly what the degenerate-box incident produced (a
    couple hundred stray pixels with no extractable contour). This must
    surface as an observable warning rather than a silent "success"."""
    perception = _load_perception(tmp_path, monkeypatch)
    image = _sam2_ready_image(perception, tmp_path, monkeypatch)

    monkeypatch.setattr(
        perception._session, "post",
        lambda *a, **k: _FakeHttpResponse({"mask_url": "/results/mask.png", "bounding_boxes": []}),
    )
    monkeypatch.setattr(
        perception._session, "get", lambda *a, **k: _FakeHttpResponse(content=b"mask-bytes")
    )

    result = perception.SAM2(str(image), [717, 209, 1232, 420])

    assert "no contour" in result["warning"]


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
    monkeypatch.setattr(manager, "prepare_xgpu_assets", lambda *args, **kwargs: (args[0], args[1]))
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


def test_apply_source_root_updates_paths_when_not_pinned(monkeypatch, tmp_path):
    monkeypatch.setattr(manager, "RS_ROOT", manager.RS_ROOT)
    monkeypatch.setattr(manager, "OEA_ROOT", manager.OEA_ROOT)
    monkeypatch.setattr(manager, "SOURCE_MODELS_ROOT", manager.SOURCE_MODELS_ROOT)
    monkeypatch.delenv("RSMEM_EARTHBENCH_ROOT", raising=False)
    monkeypatch.delenv("RSMEM_OEA_ROOT", raising=False)

    source_root = tmp_path / "local"
    manager._apply_source_root(source_root)

    assert manager.RS_ROOT == source_root / "RSMeM-public"
    assert manager.OEA_ROOT == source_root / "OpenEarthAgent"
    assert manager.SOURCE_MODELS_ROOT == source_root / "models"


def test_apply_source_root_respects_a_pinned_env_override(monkeypatch, tmp_path):
    """RS_ROOT/OEA_ROOT are computed from their env vars once, at import time;
    _apply_source_root's contract is to leave a pinned one exactly as it
    already is, not to re-derive it from the env var a second time."""
    pinned = tmp_path / "pinned-rsmem-public"
    monkeypatch.setattr(manager, "RS_ROOT", pinned)  # as if RSMEM_EARTHBENCH_ROOT set this at import
    monkeypatch.setattr(manager, "OEA_ROOT", manager.OEA_ROOT)
    monkeypatch.setattr(manager, "SOURCE_MODELS_ROOT", manager.SOURCE_MODELS_ROOT)
    monkeypatch.setenv("RSMEM_EARTHBENCH_ROOT", str(pinned))
    monkeypatch.delenv("RSMEM_OEA_ROOT", raising=False)

    manager._apply_source_root(tmp_path / "local")

    assert manager.RS_ROOT == pinned  # untouched
    assert manager.OEA_ROOT == (tmp_path / "local") / "OpenEarthAgent"  # not pinned, so it follows


def test_service_specs_use_the_relocated_source_root_for_groundingdino_and_lae_dino(monkeypatch, tmp_path):
    """GroundingDINO_SwinT_OGC.py and the LAE-DINO config are the source files
    that actually crashed in production; the checkpoints stay on the volume."""
    monkeypatch.setattr(manager, "RS_ROOT", manager.RS_ROOT)
    monkeypatch.setattr(manager, "OEA_ROOT", manager.OEA_ROOT)
    monkeypatch.setattr(manager, "SOURCE_MODELS_ROOT", manager.SOURCE_MODELS_ROOT)
    monkeypatch.delenv("RSMEM_EARTHBENCH_ROOT", raising=False)
    monkeypatch.delenv("RSMEM_OEA_ROOT", raising=False)
    local_root = tmp_path / "local"
    manager._apply_source_root(local_root)
    local_models = local_root / "models"

    specs = {spec.name: spec for spec in manager.service_specs()}

    grounding = specs["oea-grounding-dino"]
    grounding_config = local_models / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    assert str(grounding_config) in grounding.command
    assert grounding_config in grounding.required_paths
    assert manager.MODELS / "groundingdino_swint_ogc.pth" in grounding.required_paths

    lae = specs["oea-lae-dino"]
    lae_config = (
        local_models / "LAE-DINO" / "mmdetection_lae" / "configs" / "lae_dino"
        / "lae_dino_swin-t_pretrain_LAE-1M.py"
    )
    assert str(lae_config) in lae.command
    assert lae_config in lae.required_paths
    assert manager.MODELS / "lae_dino_swint_lae1m-28ca3a15.pth" in lae.required_paths


def test_start_one_puts_the_relocated_source_root_on_pythonpath(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "RS_ROOT", manager.RS_ROOT)
    monkeypatch.setattr(manager, "OEA_ROOT", manager.OEA_ROOT)
    monkeypatch.setattr(manager, "SOURCE_MODELS_ROOT", manager.SOURCE_MODELS_ROOT)
    monkeypatch.delenv("RSMEM_EARTHBENCH_ROOT", raising=False)
    monkeypatch.delenv("RSMEM_OEA_ROOT", raising=False)
    local_root = tmp_path / "local"
    manager._apply_source_root(local_root)

    source = tmp_path / "worker.py"
    source.write_text("pass\n", encoding="utf-8")
    spec = manager.ServiceSpec(
        "oea-grounding-dino", (sys.executable, str(source)), tmp_path,
        "http://127.0.0.1:9/health",
    )
    monkeypatch.setattr(manager, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(manager, "_reachable", lambda _spec: False)
    captured = {}

    def fake_popen(*args, **kwargs):
        captured["env"] = kwargs["env"]

        class ExitsImmediately:
            pid = 1

            def poll(self):
                return 0

        return ExitsImmediately()

    monkeypatch.setattr(manager.subprocess, "Popen", fake_popen)
    instance = manager.GPUServiceManager()
    instance._start_one(spec)

    pythonpath = captured["env"]["PYTHONPATH"]
    assert str(local_root / "models" / "GroundingDINO") in pythonpath
    assert str(local_root / "models" / "LAE-DINO" / "mmdetection_lae") in pythonpath


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


def test_grounding_dino_config_rewrites_qy11_bert_path(tmp_path, monkeypatch):
    """GroundingDINO's own config hardcodes the same kind of QY_11-specific
    absolute BERT path (its text encoder), separate from LAE-DINO's config.
    This surfaced only once the volume-EIO fix let oea-grounding-dino import
    far enough to actually load it."""
    models = tmp_path / "models"
    config = (
        models / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    )
    config.parent.mkdir(parents=True)
    config.write_text(
        "modelname = 'groundingdino'\n"
        "text_encoder_type = '/ultrafast_share/gzh_data/baijin/OpenEarthAgent/models/bert-base-uncased'\n"
        "use_text_enhancer = True\n",
        encoding="utf-8",
    )
    _mark_source_ready(tmp_path, models)
    model_assets.ensure_source_trees(tmp_path, models)
    rewritten = config.read_text(encoding="utf-8")
    assert str(models / "bert-base-uncased") in rewritten
    assert "/ultrafast_share" not in rewritten
    assert "use_text_enhancer = True" in rewritten  # unrelated lines survive untouched


def test_tested_worker_overrides_replace_cloned_sources(tmp_path, monkeypatch):
    models = tmp_path / "models"
    targets = [
        tmp_path / "RSMeM-public" / "tools" / "deploy_tools" / "changeos_api.py",
        tmp_path / "RSMeM-public" / "tools" / "deploy_tools" / "sam2_api.py",
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
    assert "multimask_output = bbox is None" in targets[1].read_text(encoding="utf-8")
    assert "_patch_bert_sdpa_mask_utils" in targets[2].read_text(encoding="utf-8")
    assert targets[3].read_text(encoding="utf-8").index("self.init_model()") < targets[3].read_text(encoding="utf-8").index("self.register_to_controller()")


def _mark_source_ready_at(extract_root: Path, models_dir: Path) -> None:
    """Like _mark_source_ready, but for the case where the source bundle
    landed on a local disk root distinct from the persistent-volume
    models_dir that still holds the checkpoints."""
    archive = models_dir / "offline" / "rsmem-perception-sources-v1.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz"):
        pass
    bundled_models = extract_root / "models"
    required = (
        extract_root / "RSMeM-public" / "tools" / "deploy_tools" / "sam2_api.py",
        extract_root / "OpenEarthAgent" / "tool_server" / "tool_workers" / "online_workers" / "controller.py",
        extract_root / "OpenEarthAgent" / "tool_server" / "utils" / "server_utils.py",
        bundled_models / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py",
        bundled_models / "LAE-DINO" / "mmdetection_lae" / "__init__.py",
        extract_root / "easyocr" / "model" / "craft_mlt_25k.pth",
    )
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    marker = extract_root / ".rsmem-source-bundle.json"
    marker.write_text(json.dumps({"sha256": model_assets._sha256(archive)}), encoding="utf-8")


def _with_free_gib(monkeypatch, gib: float) -> None:
    monkeypatch.setattr(
        model_assets.shutil, "disk_usage",
        lambda _path: type("Usage", (), {"free": gib * 1024 ** 3, "total": 0})(),
    )


def test_ensure_source_trees_returns_workspace_without_a_local_root(tmp_path):
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    _mark_source_ready(workspace, models)
    assert model_assets.ensure_source_trees(workspace, models) == workspace


def test_ensure_source_trees_extracts_locally_when_room_allows(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    local_root = tmp_path / "local"
    _mark_source_ready_at(local_root, models)
    _with_free_gib(monkeypatch, 100.0)

    extract_root = model_assets.ensure_source_trees(workspace, models, local_root=local_root, min_free_gib=4.0)

    assert extract_root == local_root
    # Already valid at the local root: nothing should have touched the volume.
    assert not (workspace / "RSMeM-public").exists()


def test_ensure_source_trees_falls_back_to_the_volume_without_room(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    local_root = tmp_path / "local"
    _mark_source_ready(workspace, models)
    _with_free_gib(monkeypatch, 1.0)  # below _SOURCE_TREE_ESTIMATED_GIB + min_free_gib

    extract_root = model_assets.ensure_source_trees(workspace, models, local_root=local_root, min_free_gib=4.0)

    assert extract_root == workspace


def test_lae_config_patch_still_points_at_the_volume_weights_when_sources_are_local(tmp_path, monkeypatch):
    """GroundingDINO/LAE-DINO checkpoints are large, sequentially read, and have
    never failed on the volume; only their small source/config files move."""
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    local_root = tmp_path / "local"
    config = (
        local_root / "models" / "LAE-DINO" / "mmdetection_lae" / "configs" / "lae_dino"
        / "lae_dino_swin-t_pretrain_LAE-1M.py"
    )
    config.parent.mkdir(parents=True)
    config.write_text("lang_model_name = '/server-only/bert'\nother = 1\n", encoding="utf-8")
    _mark_source_ready_at(local_root, models)
    _with_free_gib(monkeypatch, 100.0)

    extract_root = model_assets.ensure_source_trees(workspace, models, local_root=local_root, min_free_gib=4.0)

    assert extract_root == local_root
    rewritten = config.read_text(encoding="utf-8")
    # The path written into the (now-local) config still names the volume,
    # where the actual BERT checkpoint remains.
    assert str(models / "bert-base-uncased") in rewritten
    assert "/server-only/bert" not in rewritten


def test_prepare_xgpu_assets_returns_models_and_source_root(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    _mark_source_ready(workspace, models)
    monkeypatch.setattr(model_assets, "ensure_model_assets", lambda _models, _workspace: _models)
    monkeypatch.delenv("EASYOCR_MODULE_PATH", raising=False)

    result_models, source_root = model_assets.prepare_xgpu_assets(models, workspace)

    assert result_models == models
    assert source_root == workspace
    assert os.environ["EASYOCR_MODULE_PATH"] == str(workspace / "easyocr")


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
    interpreters, failures = xgpu_runtime.ensure_xgpu_runtime_envs(workspace, models)
    assert not failures
    assert set(interpreters) == {
        "sam2", "remoteclip", "changeos", "bright", "oea-e1", "oea-e3"
    }
    assert all(path.is_file() for path in interpreters.values())
    # The archive is never deleted: the next deploy's archive.is_file() check
    # in _prepare_runtime short-circuits straight past downloading, which is
    # what removes the cross-instance download race for good in steady state.
    assert all((models / relative).is_file() for relative in xgpu_runtime.OFFLINE_ENV_ARCHIVES.values())


def test_runtime_publishes_downloaded_archive_for_reuse_by_the_next_deploy(tmp_path, monkeypatch):
    """A downloaded archive must land at the canonical models/offline/...
    path (not stay in a scratch directory) and must survive being installed,
    so a later call's archive.is_file() short-circuits straight past
    downloading -- no download means no shared mutable staging file for a
    concurrent "double instance" to race over."""
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    download_root = tmp_path / "remote"
    checksums = []
    for relative in xgpu_runtime.OFFLINE_ENV_ARCHIVES.values():
        checksums.append(f"{_write_runtime_archive(download_root, relative)}  {Path(relative).name}")
    (models / "offline").mkdir(parents=True)
    (models / "offline" / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    downloaded = []

    def fake_run(command, **kwargs):
        # Mirrors the real downloader: writes under the private staging
        # directory passed as the last argument, nested under `relative`.
        relative = command[-3]
        staging = Path(command[-1])
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((download_root / relative).read_bytes())
        downloaded.append(relative)

    monkeypatch.setattr(xgpu_runtime.subprocess, "run", fake_run)
    interpreters, failures = xgpu_runtime.ensure_xgpu_runtime_envs(workspace, models)
    assert not failures
    # Runtimes are prepared concurrently, so completion order is not fixed.
    assert sorted(downloaded) == sorted(xgpu_runtime.OFFLINE_ENV_ARCHIVES.values())
    assert all(path.is_file() for path in interpreters.values())
    # Published to the canonical path and left there.
    assert all((models / relative).is_file() for relative in xgpu_runtime.OFFLINE_ENV_ARCHIVES.values())
    # No leftover staging directories anywhere under the download root.
    assert not any((workspace / ".rsmem-runtime-downloads").rglob("*.tar.gz"))

    # Second call: every archive is already valid at the canonical path, so
    # nothing may be downloaded at all.
    def forbidden_run(*_args, **_kwargs):
        raise AssertionError("must not re-download an archive already on disk")

    monkeypatch.setattr(xgpu_runtime.subprocess, "run", forbidden_run)
    interpreters_again, failures_again = xgpu_runtime.ensure_xgpu_runtime_envs(workspace, models)
    assert not failures_again
    assert interpreters_again == interpreters


def test_select_local_targets_returns_empty_without_a_local_root():
    pending = [("oea-e1", "offline/oea-e1-py310-cu124-v1.tar.gz", "sha")]
    assert xgpu_runtime._select_local_targets(pending, None, 4.0) == {}


def test_select_local_targets_fits_largest_first_within_budget(tmp_path, monkeypatch):
    """50.35 GB of runtimes against ~55 GiB free locally is tight enough that
    this has to bin-pack, not just check that at least one thing fits."""
    local_root = tmp_path / "local"
    monkeypatch.setattr(
        xgpu_runtime.shutil, "disk_usage",
        lambda _path: type("Usage", (), {"free": 20 * 1024 ** 3, "total": 0})(),
    )
    pending = [
        ("bright", "offline/bright-py310-cu121-v1.tar.gz", "sha"),      # 5.67 GiB
        ("oea-e3", "offline/oea-e3-py310-cu124-v1.tar.gz", "sha"),      # 7.11 GiB
        ("oea-e1", "offline/oea-e1-py310-cu124-v1.tar.gz", "sha"),      # 10.50 GiB
    ]
    targets = xgpu_runtime._select_local_targets(pending, local_root, min_free_gib=2.0)
    # Budget is 18 GiB. Largest first: oea-e1 (10.50) fits, leaving 7.5; oea-e3
    # (7.11) fits, leaving 0.39; bright (5.67) does not.
    assert set(targets) == {"oea-e1", "oea-e3"}
    assert targets["oea-e1"] == local_root / "envs" / "oea-e1"
    assert local_root.is_dir()


def test_select_local_targets_falls_back_when_local_disk_is_unavailable(tmp_path, monkeypatch):
    unreachable = tmp_path / "not-a-real-mount" / "local"
    monkeypatch.setattr(
        xgpu_runtime.Path, "mkdir",
        lambda self, *a, **k: (_ for _ in ()).throw(OSError(5, "Input/output error")),
    )
    pending = [("bright", "offline/bright-py310-cu121-v1.tar.gz", "sha")]
    assert xgpu_runtime._select_local_targets(pending, unreachable, 4.0) == {}


def _mark_runtime_ready(prefix: Path, filename: str, sha: str) -> None:
    (prefix / "bin").mkdir(parents=True, exist_ok=True)
    (prefix / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (prefix / ".rsmem-runtime.json").write_text(
        json.dumps({
            "schema": xgpu_runtime.ENV_SCHEMA,
            "archive": filename,
            "sha256": sha,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


def test_select_local_targets_keeps_ready_runtimes_without_charging_the_budget(tmp_path, monkeypatch):
    """A runtime already installed locally must stay local even when free
    space is below min_free_gib -- charging it against the remaining budget
    would evict it on every tight redeploy."""
    local_root = tmp_path / "local"
    pending = [
        ("bright", "offline/bright-py310-cu121-v1.tar.gz", "sha-bright"),
        ("oea-e1", "offline/oea-e1-py310-cu124-v1.tar.gz", "sha-e1"),
    ]
    _mark_runtime_ready(
        local_root / "envs" / "bright", "bright-py310-cu121-v1.tar.gz", "sha-bright"
    )
    monkeypatch.setattr(
        xgpu_runtime.shutil, "disk_usage",
        lambda _path: type("Usage", (), {"free": 0, "total": 0})(),
    )
    targets = xgpu_runtime._select_local_targets(pending, local_root, min_free_gib=4.0)
    assert set(targets) == {"bright"}
    assert targets["bright"] == local_root / "envs" / "bright"


def _with_runtime_free_gib(monkeypatch, gib: float) -> None:
    monkeypatch.setattr(
        xgpu_runtime.shutil, "disk_usage",
        lambda _path: type("Usage", (), {"free": gib * 1024 ** 3, "total": 0})(),
    )


def _write_runtime_archive(models: Path, relative: str) -> str:
    archive = models / relative
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as bundle:
        payload = b"#!/bin/sh\n"
        info = tarfile.TarInfo("bin/python")
        info.mode = 0o755
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    return xgpu_runtime._sha256(archive)


def test_runtime_that_does_not_fit_locally_reuses_an_existing_volume_cache(tmp_path, monkeypatch):
    """A runtime a previous deploy already extracted to the volume, and that
    still does not fit the local budget, must be reused rather than
    re-extracted -- steady-state redeploys stay fast for whatever the local
    disk cannot hold."""
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    checksums = []
    for relative in xgpu_runtime.OFFLINE_ENV_ARCHIVES.values():
        checksums.append(f"{_write_runtime_archive(models, relative)}  {Path(relative).name}")
    (models / "offline" / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    monkeypatch.setattr(xgpu_runtime.subprocess, "run", lambda *a, **k: None)

    # First pass: everything lands on the volume (no local_root offered).
    interpreters, failures = xgpu_runtime.ensure_xgpu_runtime_envs(workspace, models)
    assert not failures
    assert all(str(path).startswith(str(workspace / "envs")) for path in interpreters.values())

    # Second pass: a local disk is offered but has no room for anything, so
    # every runtime must fall back to (and find) its existing volume copy.
    local_root = tmp_path / "local"
    _with_runtime_free_gib(monkeypatch, 0.0)
    interpreters_again, failures_again = xgpu_runtime.ensure_xgpu_runtime_envs(
        workspace, models, local_root=local_root, min_free_gib=1.0
    )
    assert not failures_again
    assert interpreters_again == interpreters


def test_local_placement_wins_over_a_stale_volume_cache(tmp_path, monkeypatch):
    """Regression for the incident where oea-e1/oea-e3 kept reading from the
    volume -- and kept dying with EIO -- because earlier deploys (predating
    local-disk support) had already cached them there, and the volume-cache
    check ran before the local-placement decision. Local placement must win
    whenever it fits, even though a valid volume copy already exists."""
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    checksums = []
    for relative in xgpu_runtime.OFFLINE_ENV_ARCHIVES.values():
        checksums.append(f"{_write_runtime_archive(models, relative)}  {Path(relative).name}")
    (models / "offline" / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    monkeypatch.setattr(xgpu_runtime.subprocess, "run", lambda *a, **k: None)

    # Every runtime is already validly cached on the volume, exactly like the
    # live deployment after several redeploys.
    interpreters, failures = xgpu_runtime.ensure_xgpu_runtime_envs(workspace, models)
    assert not failures
    # The archive is never deleted after installing, so it is still exactly
    # what the second pass below needs to install onto the local disk --
    # nothing has to be recreated.
    assert all((models / relative).is_file() for relative in xgpu_runtime.OFFLINE_ENV_ARCHIVES.values())

    local_root = tmp_path / "local"
    _with_runtime_free_gib(monkeypatch, 200.0)  # comfortably covers all six (50.35 GB)

    interpreters_again, failures_again = xgpu_runtime.ensure_xgpu_runtime_envs(
        workspace, models, local_root=local_root, min_free_gib=4.0
    )

    assert not failures_again
    # Every runtime moved to the local disk despite the volume already having
    # a valid, ready-to-use copy of each one.
    assert all(str(path).startswith(str(local_root / "envs")) for path in interpreters_again.values())


def test_corrupt_canonical_archive_is_redownloaded_within_the_same_call(tmp_path, monkeypatch):
    """A canonical archive that fails its checksum -- an earlier transfer
    interrupted mid-write, for instance -- must not fail the runtime until
    the next deploy happens to retry it; it is refetched right here."""
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    download_root = tmp_path / "remote"
    checksums = {}
    for relative in xgpu_runtime.OFFLINE_ENV_ARCHIVES.values():
        checksums[relative] = _write_runtime_archive(download_root, relative)
        # The canonical path holds a corrupt copy, as if an earlier transfer
        # had been interrupted partway through.
        corrupt = models / relative
        corrupt.parent.mkdir(parents=True, exist_ok=True)
        corrupt.write_bytes(b"not the real archive")
    (models / "offline" / "SHA256SUMS").write_text(
        "\n".join(f"{sha}  {Path(rel).name}" for rel, sha in checksums.items()) + "\n",
        encoding="utf-8",
    )

    def fake_run(command, **kwargs):
        relative = command[-3]
        staging = Path(command[-1])
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((download_root / relative).read_bytes())

    monkeypatch.setattr(xgpu_runtime.subprocess, "run", fake_run)
    interpreters, failures = xgpu_runtime.ensure_xgpu_runtime_envs(workspace, models)

    assert not failures
    assert all(path.is_file() for path in interpreters.values())
    # The corrupt copy was replaced by the freshly downloaded, verified one,
    # not left in place for the next deploy to trip over again.
    for relative, expected_sha in checksums.items():
        assert xgpu_runtime._sha256(models / relative) == expected_sha


def test_local_target_ready_check_skips_a_redundant_second_extraction(tmp_path, monkeypatch):
    """Regression for the local-target branch always re-extracting: once a
    runtime is validly installed at its local_root target -- for instance by
    a sibling instance that already won the race under _local_lock -- a
    second call must reuse it, not redo 55-110s of extraction per runtime."""
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    local_root = tmp_path / "local"
    checksums = []
    for relative in xgpu_runtime.OFFLINE_ENV_ARCHIVES.values():
        checksums.append(f"{_write_runtime_archive(models, relative)}  {Path(relative).name}")
    (models / "offline" / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    monkeypatch.setattr(xgpu_runtime.subprocess, "run", lambda *a, **k: None)
    _with_runtime_free_gib(monkeypatch, 200.0)  # comfortably covers all six (50.35 GB)

    install_calls = []
    real_install = xgpu_runtime._install_archive

    def counting_install(prefix, archive, expected_sha256, label=""):
        install_calls.append(label)
        return real_install(prefix, archive, expected_sha256, label)

    monkeypatch.setattr(xgpu_runtime, "_install_archive", counting_install)

    interpreters, failures = xgpu_runtime.ensure_xgpu_runtime_envs(
        workspace, models, local_root=local_root, min_free_gib=4.0
    )
    assert not failures
    assert sorted(install_calls) == sorted(xgpu_runtime.OFFLINE_ENV_ARCHIVES)
    assert all(str(path).startswith(str(local_root / "envs")) for path in interpreters.values())

    install_calls.clear()
    interpreters_again, failures_again = xgpu_runtime.ensure_xgpu_runtime_envs(
        workspace, models, local_root=local_root, min_free_gib=4.0
    )
    assert not failures_again
    assert install_calls == [], "an already-ready local target must not be re-extracted"
    assert interpreters_again == interpreters


def test_already_ready_local_runtimes_are_kept_when_local_disk_is_full(tmp_path, monkeypatch):
    """Regression for the next-deploy eviction: 6.99 GiB free minus a 4 GiB
    reserve cannot fit even the smallest runtime, but the copies already on
    the local disk must still be selected rather than abandoned for the
    volume."""
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    local_root = tmp_path / "local"
    checksums = []
    for name, relative in xgpu_runtime.OFFLINE_ENV_ARCHIVES.items():
        sha = _write_runtime_archive(models, relative)
        checksums.append(f"{sha}  {Path(relative).name}")
        _mark_runtime_ready(local_root / "envs" / name, Path(relative).name, sha)
    (models / "offline" / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    monkeypatch.setattr(xgpu_runtime.subprocess, "run", lambda *a, **k: None)
    _with_runtime_free_gib(monkeypatch, 0.0)

    def refuse_install(*args, **kwargs):
        raise AssertionError("must not re-extract a ready local runtime")

    monkeypatch.setattr(xgpu_runtime, "_install_archive", refuse_install)
    interpreters, failures = xgpu_runtime.ensure_xgpu_runtime_envs(
        workspace, models, local_root=local_root, min_free_gib=4.0
    )
    assert not failures
    assert set(interpreters) == set(xgpu_runtime.OFFLINE_ENV_ARCHIVES)
    assert all(str(path).startswith(str(local_root / "envs")) for path in interpreters.values())


def test_ensure_xgpu_runtime_envs_places_pending_runtimes_locally_when_room_allows(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    checksums = []
    for relative in xgpu_runtime.OFFLINE_ENV_ARCHIVES.values():
        checksums.append(f"{_write_runtime_archive(models, relative)}  {Path(relative).name}")
    (models / "offline" / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    monkeypatch.setattr(xgpu_runtime.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(
        xgpu_runtime.shutil, "disk_usage",
        lambda _path: type("Usage", (), {"free": 200 * 1024 ** 3, "total": 0})(),
    )

    local_root = tmp_path / "local"
    interpreters, failures = xgpu_runtime.ensure_xgpu_runtime_envs(
        workspace, models, local_root=local_root, min_free_gib=4.0
    )
    assert not failures
    assert set(interpreters) == set(xgpu_runtime.OFFLINE_ENV_ARCHIVES)
    # 200 GiB free comfortably covers all six (50.35 GB total): every one
    # should have landed on the local disk, none on the volume.
    assert all(str(path).startswith(str(local_root / "envs")) for path in interpreters.values())
    assert not (workspace / "envs").exists()


def test_runtimes_are_prepared_concurrently(tmp_path, monkeypatch):
    import threading

    workspace = tmp_path / "workspace"
    models = workspace / "models"
    (models / "offline").mkdir(parents=True)
    checksums = []
    for relative in xgpu_runtime.OFFLINE_ENV_ARCHIVES.values():
        archive = models / relative
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

    barrier = threading.Barrier(len(xgpu_runtime.OFFLINE_ENV_ARCHIVES), timeout=10)
    real_install = xgpu_runtime._install_archive

    def blocking_install(prefix, archive, expected_sha256, label=""):
        # Deadlocks unless every runtime is installed at the same time.
        barrier.wait()
        return real_install(prefix, archive, expected_sha256, label)

    monkeypatch.setattr(xgpu_runtime, "_install_archive", blocking_install)
    monkeypatch.setattr(xgpu_runtime.subprocess, "run", lambda *args, **kwargs: None)
    interpreters, failures = xgpu_runtime.ensure_xgpu_runtime_envs(workspace, models)
    assert not failures
    assert set(interpreters) == set(xgpu_runtime.OFFLINE_ENV_ARCHIVES)
    assert all(path.is_file() for path in interpreters.values())


def test_one_failed_runtime_does_not_withhold_the_others(tmp_path, monkeypatch):
    """A slow or broken runtime must not keep every worker offline."""
    workspace = tmp_path / "workspace"
    models = workspace / "models"
    (models / "offline").mkdir(parents=True)
    checksums = []
    for relative in xgpu_runtime.OFFLINE_ENV_ARCHIVES.values():
        archive = models / relative
        archive.write_bytes(b"payload")
        checksums.append(f"{xgpu_runtime._sha256(archive)}  {archive.name}")
    (models / "offline" / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )

    def failing_install(prefix, archive, expected_sha256, label=""):
        if label == "changeos":
            raise RuntimeError("conda-unpack exploded")
        (prefix / "bin").mkdir(parents=True, exist_ok=True)
        (prefix / "bin" / "python").write_text("", encoding="utf-8")

    monkeypatch.setattr(xgpu_runtime, "_install_archive", failing_install)
    interpreters, failures = xgpu_runtime.ensure_xgpu_runtime_envs(workspace, models)

    assert set(failures) == {"changeos"}
    assert "conda-unpack exploded" in failures["changeos"]
    assert set(interpreters) == set(xgpu_runtime.OFFLINE_ENV_ARCHIVES) - {"changeos"}


def test_only_ready_runtimes_get_their_interpreter_variable(monkeypatch):
    for variable in xgpu_runtime.RUNTIME_PYTHON_VARIABLES.values():
        monkeypatch.delenv(variable, raising=False)

    xgpu_runtime.configure_runtime_environment({"sam2": Path("/envs/sam2/bin/python")})

    assert os.environ["RSMEM_EARTHBENCH_PYTHON"] == "/envs/sam2/bin/python"
    assert "RSMEM_OEA_E1_PYTHON" not in os.environ


def test_extraction_rejects_paths_outside_the_prefix(tmp_path):
    archive = tmp_path / "hostile.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        payload = b"x"
        info = tarfile.TarInfo("../escaped")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    with pytest.raises(RuntimeError, match="Unsafe path"):
        xgpu_runtime._safe_extract(archive, tmp_path / "prefix")
    assert not (tmp_path / "escaped").exists()


def test_archive_download_retries_after_a_stalled_connection(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    relative = "offline/sam2-py310-cu121-v1.tar.gz"
    attempts = []

    def fake_run(command, **kwargs):
        attempts.append(command)
        if len(attempts) == 1:
            raise xgpu_runtime.subprocess.TimeoutExpired(command, kwargs["timeout"])
        # command[-1] is the private staging directory this call created;
        # a shared, hardcoded path here would defeat the thing under test.
        target = Path(command[-1]) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"payload")
        return None

    monkeypatch.setattr(xgpu_runtime.subprocess, "run", fake_run)
    monkeypatch.setattr(xgpu_runtime.time, "sleep", lambda _seconds: None)
    resolved = xgpu_runtime._download_archive(tmp_path / "models", relative, workspace)
    assert len(attempts) == 2
    assert resolved.read_bytes() == b"payload"


def test_archive_download_reports_every_failed_attempt(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    relative = "offline/oea-e3-py310-cu124-v1.tar.gz"
    attempts = []

    def always_stall(command, **kwargs):
        attempts.append(command)
        raise xgpu_runtime.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(xgpu_runtime.subprocess, "run", always_stall)
    monkeypatch.setattr(xgpu_runtime.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError) as failure:
        xgpu_runtime._download_archive(tmp_path / "models", relative, workspace)
    assert len(attempts) == xgpu_runtime.DOWNLOAD_ATTEMPTS
    assert relative in str(failure.value)
    assert "stalled" in str(failure.value)


def test_archive_download_rejects_a_silent_success(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    relative = "offline/bright-py310-cu121-v1.tar.gz"

    monkeypatch.setattr(xgpu_runtime.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(xgpu_runtime.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError) as failure:
        xgpu_runtime._download_archive(tmp_path / "models", relative, workspace)
    assert "produced no file" in str(failure.value)


def test_download_archive_uses_a_private_staging_directory_each_call(tmp_path, monkeypatch):
    """Two processes (ModelScope's "double instance" quirk) downloading the
    same archive at the same time must never write into the same directory;
    that shared, mutable path is exactly what let one instance's cleanup
    delete a file the other was still reading."""
    workspace = tmp_path / "workspace"
    relative = "offline/sam2-py310-cu121-v1.tar.gz"
    seen_roots = []

    def fake_run(command, **kwargs):
        staging = Path(command[-1])
        seen_roots.append(staging)
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"payload")

    monkeypatch.setattr(xgpu_runtime.subprocess, "run", fake_run)

    first = xgpu_runtime._download_archive(tmp_path / "models", relative, workspace)
    second = xgpu_runtime._download_archive(tmp_path / "models", relative, workspace)

    assert seen_roots[0] != seen_roots[1]
    assert first.parent != second.parent
    assert first.is_file() and second.is_file()
    # Both staging directories still live under workspace: publishing relies
    # on os.replace, which requires source and destination to share a
    # filesystem.
    assert str(first.parent).startswith(str(workspace / ".rsmem-runtime-downloads"))


def _service(name: str, path: Path) -> manager.ServiceSpec:
    source = path / f"{name}.py"
    source.write_text("pass\n", encoding="utf-8")
    return manager.ServiceSpec(
        name,
        (sys.executable, str(source)),
        path,
        f"http://127.0.0.1:9/{name}",
    )


def test_controller_failure_skips_oea_workers(tmp_path, monkeypatch):
    specs = [
        _service("earthbench-sam2", tmp_path),
        _service("oea-controller", tmp_path),
        _service("oea-ocr", tmp_path),
        _service("oea-qwen3-vl", tmp_path),
    ]
    started = []

    def fake_start(self, spec):
        started.append(spec.name)
        if spec.name == "oea-controller":
            self._mark_skipped(spec.name, "startup timed out")
            return False
        self._record_timing(spec.name, 0.1)
        return True

    monkeypatch.setattr(manager, "service_specs", lambda: specs)
    monkeypatch.setattr(manager, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(manager, "prepare_xgpu_assets", lambda *args, **kwargs: (args[0], args[1]))
    monkeypatch.setenv("RSMEM_GPU_BOOTSTRAP_ENVS", "0")
    monkeypatch.setenv("RSMEM_ENABLE_GPU_PERCEPTION", "0")
    monkeypatch.setattr(manager.GPUServiceManager, "_start_one", fake_start)

    instance = manager.GPUServiceManager()
    instance._start_selected_services()
    assert "oea-controller" in started
    assert "earthbench-sam2" in started
    assert "oea-ocr" not in started
    assert "oea-qwen3-vl" not in started
    assert instance.skipped["oea-ocr"] == "controller unavailable"
    assert instance.skipped["oea-qwen3-vl"] == "controller unavailable"


def test_phase_b_starts_workers_concurrently(tmp_path, monkeypatch):
    import threading

    specs = [
        _service("oea-controller", tmp_path),
        _service("earthbench-sam2", tmp_path),
        _service("earthbench-remoteclip", tmp_path),
        _service("oea-ocr", tmp_path),
        _service("oea-qwen3-vl", tmp_path),
    ]
    order = []
    barrier = threading.Barrier(3, timeout=5)

    def fake_start(self, spec):
        if spec.name == "oea-controller":
            order.append(("controller", spec.name))
            return True
        if spec.name == "oea-qwen3-vl":
            order.append(("qwen", spec.name))
            return True
        barrier.wait()
        order.append(("phase_b", spec.name))
        return True

    monkeypatch.setattr(manager, "service_specs", lambda: specs)
    monkeypatch.setattr(manager, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(manager, "prepare_xgpu_assets", lambda *args, **kwargs: (args[0], args[1]))
    monkeypatch.setenv("RSMEM_GPU_BOOTSTRAP_ENVS", "0")
    monkeypatch.setenv("RSMEM_ENABLE_GPU_PERCEPTION", "0")
    monkeypatch.setenv("RSMEM_GPU_STARTUP_CONCURRENCY", "3")
    monkeypatch.setattr(manager.GPUServiceManager, "_start_one", fake_start)

    instance = manager.GPUServiceManager()
    instance._start_selected_services()
    assert order[0] == ("controller", "oea-controller")
    assert order[-1] == ("qwen", "oea-qwen3-vl")
    assert {item[1] for item in order[1:-1]} == {
        "earthbench-sam2", "earthbench-remoteclip", "oea-ocr"
    }


def test_single_worker_failure_does_not_abort_siblings(tmp_path, monkeypatch):
    specs = [
        _service("oea-controller", tmp_path),
        _service("earthbench-sam2", tmp_path),
        _service("earthbench-changeos", tmp_path),
        _service("oea-qwen3-vl", tmp_path),
    ]

    def fake_start(self, spec):
        if spec.name == "earthbench-changeos":
            self._mark_skipped(spec.name, "startup timed out")
            return False
        self._record_timing(spec.name, 0.2)
        return True

    monkeypatch.setattr(manager, "service_specs", lambda: specs)
    monkeypatch.setattr(manager, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(manager, "prepare_xgpu_assets", lambda *args, **kwargs: (args[0], args[1]))
    monkeypatch.setenv("RSMEM_GPU_BOOTSTRAP_ENVS", "0")
    monkeypatch.setenv("RSMEM_ENABLE_GPU_PERCEPTION", "0")
    monkeypatch.setenv("RSMEM_GPU_STARTUP_CONCURRENCY", "2")
    monkeypatch.setattr(manager.GPUServiceManager, "_start_one", fake_start)

    instance = manager.GPUServiceManager()
    instance._start_selected_services()
    assert "startup timed out" in instance.skipped["earthbench-changeos"]
    assert "earthbench-sam2" in instance.startup_timings
    assert "oea-qwen3-vl" in instance.startup_timings


def test_eio_reading_the_worker_script_does_not_escape_start_one(tmp_path, monkeypatch):
    """Path.is_file does not swallow EIO the way it swallows ENOENT: before the
    top-level guard in _start_one, this OSError propagated straight out of
    _start_one, and Phase A/Phase C call it with no try/except of their own."""
    spec = manager.ServiceSpec(
        "flaky-worker", (sys.executable, "/mnt/workspace/flaky_worker.py"), tmp_path,
        "http://127.0.0.1:9/health",
    )
    monkeypatch.setattr(
        manager.Path, "is_file",
        lambda self: (_ for _ in ()).throw(OSError(5, "Input/output error")),
    )
    instance = manager.GPUServiceManager()
    assert instance._start_one(spec) is False
    assert "unexpected error: OSError" in instance.skipped["flaky-worker"]


def test_eio_checking_required_assets_does_not_escape_start_one(tmp_path, monkeypatch):
    source = tmp_path / "worker.py"
    source.write_text("pass\n", encoding="utf-8")
    spec = manager.ServiceSpec(
        "flaky-worker", (sys.executable, str(source)), tmp_path,
        "http://127.0.0.1:9/health",
        required_paths=(tmp_path / "checkpoint.pt",),
    )
    monkeypatch.setattr(
        manager.Path, "exists",
        lambda self: (_ for _ in ()).throw(OSError(5, "Input/output error")),
    )
    instance = manager.GPUServiceManager()
    assert instance._start_one(spec) is False
    assert "unexpected error: OSError" in instance.skipped["flaky-worker"]


def test_log_open_failure_falls_back_to_devnull_and_the_worker_still_starts(tmp_path, monkeypatch):
    """A log we cannot open is not a reason to skip a worker that would
    otherwise run; only the log itself should be lost."""
    source = tmp_path / "worker.py"
    source.write_text("pass\n", encoding="utf-8")
    spec = manager.ServiceSpec(
        "flaky-worker", (sys.executable, str(source)), tmp_path,
        "http://127.0.0.1:9/health",
    )
    fake_log_dir = tmp_path / "unreadable-logs"
    monkeypatch.setattr(manager, "LOG_DIR", fake_log_dir)

    real_open = Path.open

    def failing_open(self, *args, **kwargs):
        if self == fake_log_dir / "flaky-worker.log":
            raise OSError(5, "Input/output error")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(manager.Path, "open", failing_open)

    calls = {"n": 0}

    def fake_reachable(_spec):
        calls["n"] += 1
        return calls["n"] > 1  # not running yet, then healthy once polled

    monkeypatch.setattr(manager, "_reachable", fake_reachable)

    class FakeProcess:
        pid = 4242

        def poll(self):
            return None

    monkeypatch.setattr(manager.subprocess, "Popen", lambda *a, **k: FakeProcess())

    instance = manager.GPUServiceManager()
    assert instance._start_one(spec) is True
    assert "flaky-worker" not in instance.skipped


def test_phase_c_volume_failure_does_not_kill_the_startup_thread(tmp_path, monkeypatch):
    """Regression for the incident where oea-qwen3-vl vanished from both
    started and skipped. Phase C called _start_one directly with no
    try/except, so an uncaught OSError from script.is_file() silently killed
    the rsmem-gpu-startup thread with Phase C never having run at all -- not
    a slow load, not a timeout, just gone with no record anywhere."""
    controller = _service("oea-controller", tmp_path)
    qwen = _service("oea-qwen3-vl", tmp_path)
    calls = {"qwen": 0}

    def fake_unguarded(self, spec):
        if spec.name == "oea-controller":
            self._record_timing(spec.name, 0.1)
            return True
        calls["qwen"] += 1
        raise OSError(5, "Input/output error")  # what script.is_file() raised in production

    monkeypatch.setattr(manager, "service_specs", lambda: [controller, qwen])
    monkeypatch.setattr(manager, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(manager, "prepare_xgpu_assets", lambda *args, **kwargs: (args[0], args[1]))
    monkeypatch.setenv("RSMEM_GPU_BOOTSTRAP_ENVS", "0")
    monkeypatch.setenv("RSMEM_ENABLE_GPU_PERCEPTION", "0")
    monkeypatch.setattr(manager.GPUServiceManager, "_start_one_unguarded", fake_unguarded)

    instance = manager.GPUServiceManager()
    instance._start_selected_services()  # must return, not raise

    assert "oea-controller" in instance.startup_timings
    assert "unexpected error: OSError" in instance.skipped["oea-qwen3-vl"]
    # Retried once (DEFAULT_START_RETRIES), not left at a single silent attempt.
    assert calls["qwen"] == manager.DEFAULT_START_RETRIES + 1


def test_retry_eligibility_matches_crash_and_exception_only():
    assert manager._is_retryable_failure("exited during startup; see x.log")
    assert manager._is_retryable_failure("unexpected error: OSError: [Errno 5] Input/output error")
    assert not manager._is_retryable_failure("startup timed out after 900s; see x.log")
    assert not manager._is_retryable_failure("missing assets: /a/b.pt")
    assert not manager._is_retryable_failure("missing source: /a/b.py")
    assert not manager._is_retryable_failure("already running")
    assert not manager._is_retryable_failure("not selected")
    assert not manager._is_retryable_failure("controller unavailable")


def test_a_crash_is_retried_once_and_can_recover(tmp_path, monkeypatch):
    spec = _service("earthbench-sam2", tmp_path)
    attempts = {"n": 0}

    def flaky_start(self, target_spec):
        attempts["n"] += 1
        if attempts["n"] == 1:
            self._mark_skipped(target_spec.name, "exited during startup; see x.log")
            return False
        self._record_timing(target_spec.name, 0.1)
        return True

    monkeypatch.setattr(manager.GPUServiceManager, "_start_one", flaky_start)
    instance = manager.GPUServiceManager()
    assert instance._start_with_retries(spec, manager.DEFAULT_START_RETRIES) is True
    assert attempts["n"] == 2
    assert "earthbench-sam2" in instance.startup_timings


def test_a_timeout_is_not_retried(tmp_path, monkeypatch):
    """A timeout already spent its full budget; doubling it has no evidence of
    helping when the process is still alive and simply never registered."""
    spec = _service("oea-grounding-dino", tmp_path)
    attempts = {"n": 0}

    def always_times_out(self, target_spec):
        attempts["n"] += 1
        self._mark_skipped(target_spec.name, "startup timed out after 900s; see x.log")
        return False

    monkeypatch.setattr(manager.GPUServiceManager, "_start_one", always_times_out)
    instance = manager.GPUServiceManager()
    assert instance._start_with_retries(spec, manager.DEFAULT_START_RETRIES) is False
    assert attempts["n"] == 1


def test_controller_gets_more_retries_than_a_regular_worker(tmp_path, monkeypatch):
    """The controller gates every OEA-dependent worker, so losing it is the
    highest-leverage failure in the whole startup."""
    controller = _service("oea-controller", tmp_path)
    attempts = {"n": 0}

    def always_crashes(self, target_spec):
        attempts["n"] += 1
        self._mark_skipped(target_spec.name, "exited during startup; see x.log")
        return False

    monkeypatch.setattr(manager, "service_specs", lambda: [controller])
    monkeypatch.setattr(manager, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(manager, "prepare_xgpu_assets", lambda *args, **kwargs: (args[0], args[1]))
    monkeypatch.setenv("RSMEM_GPU_BOOTSTRAP_ENVS", "0")
    monkeypatch.setenv("RSMEM_ENABLE_GPU_PERCEPTION", "0")
    monkeypatch.setattr(manager.GPUServiceManager, "_start_one", always_crashes)

    instance = manager.GPUServiceManager()
    instance._start_selected_services()
    assert attempts["n"] == manager.CONTROLLER_START_RETRIES + 1


def test_extraction_progress_is_time_driven(tmp_path, monkeypatch, capsys):
    archive = tmp_path / "demo.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for index in range(3):
            payload = b"x"
            info = tarfile.TarInfo(f"file-{index}")
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))

    clock = {"now": 1000.0}

    def fake_time():
        clock["now"] += 30.0
        return clock["now"]

    monkeypatch.setattr(xgpu_runtime, "EXTRACT_PROGRESS_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(xgpu_runtime.time, "time", fake_time)
    extracted = xgpu_runtime._safe_extract(archive, tmp_path / "out", label="demo")
    assert extracted == 3
    captured = capsys.readouterr().out
    assert "files/s" in captured
    assert "demo: extracted" in captured


def _single_file_archive(path: Path, name: str = "bin/python") -> None:
    with tarfile.open(path, "w:gz") as bundle:
        payload = b"#!/bin/sh\n"
        info = tarfile.TarInfo(name)
        info.mode = 0o755
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))


def test_install_writes_into_the_final_path_without_moving_a_tree(
    tmp_path, monkeypatch
):
    """Publishing by rename costs ~586 ms per file on the target volume."""
    archive = tmp_path / "demo.tar.gz"
    _single_file_archive(archive)
    prefix = tmp_path / "envs" / "sam2"

    def unforgiving_move(*args, **kwargs):
        raise AssertionError("the install path must not move whole trees")

    monkeypatch.setattr(xgpu_runtime.Path, "rename", unforgiving_move)
    monkeypatch.setattr(xgpu_runtime.subprocess, "run", lambda *args, **kwargs: None)
    xgpu_runtime._install_archive(prefix, archive, "sha", label="sam2")

    assert (prefix / "bin" / "python").is_file()
    assert not list((tmp_path / "envs").glob(".rsmem-staging-*"))


def test_install_keeps_an_existing_tree_and_clears_the_ready_marker(
    tmp_path, monkeypatch
):
    """Clearing the prefix up front destroyed a concurrent instance's work.

    Deleting it is also unaffordable here, so the half-written state is instead
    signalled by dropping the marker until the install completes.
    """
    archive = tmp_path / "demo.tar.gz"
    _single_file_archive(archive)
    prefix = tmp_path / "envs" / "sam2"
    in_flight = prefix / "lib" / "python3.10"
    in_flight.mkdir(parents=True)
    (in_flight / "partial").write_text("still extracting", encoding="utf-8")
    marker = prefix / ".rsmem-runtime.json"
    marker.write_text('{"schema": 1}', encoding="utf-8")

    seen = []

    def watching_extract(source, destination, label="", resume=False):
        seen.append((in_flight / "partial").exists() and not marker.exists())
        return real_extract(source, destination, label, resume)

    real_extract = xgpu_runtime._safe_extract
    monkeypatch.setattr(xgpu_runtime, "_safe_extract", watching_extract)

    def refuse_delete(path, *args, **kwargs):
        raise AssertionError(f"refused to delete {path} on the critical path")

    monkeypatch.setattr(xgpu_runtime.shutil, "rmtree", refuse_delete)
    monkeypatch.setattr(xgpu_runtime.subprocess, "run", lambda *args, **kwargs: None)
    xgpu_runtime._install_archive(prefix, archive, "sha", label="sam2")

    assert seen == [True], "existing files must survive and the marker must be cleared"
    assert (in_flight / "partial").is_file()
    assert json.loads(marker.read_text(encoding="utf-8"))["archive"] == "demo.tar.gz"


def test_failed_install_leaves_no_ready_marker(tmp_path, monkeypatch):
    archive = tmp_path / "demo.tar.gz"
    _single_file_archive(archive)
    prefix = tmp_path / "envs" / "sam2"
    prefix.mkdir(parents=True)
    (prefix / ".rsmem-runtime.json").write_text('{"schema": 1}', encoding="utf-8")

    def exploding_extract(source, destination, label="", resume=False):
        raise RuntimeError("volume went away")

    monkeypatch.setattr(xgpu_runtime, "_safe_extract", exploding_extract)
    with pytest.raises(RuntimeError, match="volume went away"):
        xgpu_runtime._install_archive(prefix, archive, "sha", label="sam2")
    assert not (prefix / ".rsmem-runtime.json").exists()


def _build_archive(path: Path, members: list[tuple[str, bytes, int]], links=()) -> None:
    # Links go first so a test can express "a later member traverses this link".
    with tarfile.open(path, "w:gz") as bundle:
        for name, linkname in links:
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = linkname
            bundle.addfile(info)
        for name, payload, mode in members:
            info = tarfile.TarInfo(name)
            info.mode = mode
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))


def test_concurrent_extraction_restores_contents_links_and_exec_bits(tmp_path):
    archive = tmp_path / "env.tar.gz"
    members = [(f"lib/python3.10/mod{i}.py", f"payload-{i}".encode(), 0o644) for i in range(50)]
    members.append(("bin/python", b"#!/bin/sh\n", 0o755))
    members.append(("share/big.bin", b"z" * 4096, 0o644))
    _build_archive(archive, members, links=[("bin/python3", "python")])
    out = tmp_path / "out"

    extracted = xgpu_runtime._safe_extract(archive, out)

    assert extracted == len(members) + 1
    for index in range(50):
        assert (out / "lib" / "python3.10" / f"mod{index}.py").read_bytes() == f"payload-{index}".encode()
    assert (out / "bin" / "python").stat().st_mode & 0o111
    assert not (out / "lib" / "python3.10" / "mod0.py").stat().st_mode & 0o111
    assert os.readlink(out / "bin" / "python3") == "python"


def test_extraction_replaces_links_left_by_an_interrupted_install(tmp_path):
    """Re-running over a partial prefix is the normal case, not the exception."""
    archive = tmp_path / "env.tar.gz"
    _build_archive(
        archive,
        [("bin/python3.10", b"#!/bin/sh\n", 0o755)],
        links=[("bin/python", "python3.10")],
    )
    out = tmp_path / "out"
    (out / "bin").mkdir(parents=True)
    # Left behind pointing somewhere else by the previous attempt.
    os.symlink("python3.9", out / "bin" / "python")

    xgpu_runtime._safe_extract(archive, out)

    assert os.readlink(out / "bin" / "python") == "python3.10"


def test_members_are_written_concurrently(tmp_path, monkeypatch):
    """The whole speedup rests on overlapping the volume's per-file latency."""
    import threading

    archive = tmp_path / "env.tar.gz"
    _build_archive(archive, [(f"lib/mod{i}.py", b"x", 0o644) for i in range(8)])
    barrier = threading.Barrier(8, timeout=10)
    real_write = xgpu_runtime._write_regular

    def blocking_write(target, data, mode, dirs, resume=False):
        # Serial extraction never reaches eight simultaneous writes.
        barrier.wait()
        return real_write(target, data, mode, dirs, resume)

    monkeypatch.setattr(xgpu_runtime, "_write_regular", blocking_write)
    assert xgpu_runtime._safe_extract(archive, tmp_path / "out") == 8


def test_large_members_are_written_without_buffering_them_whole(tmp_path, monkeypatch):
    archive = tmp_path / "env.tar.gz"
    payload = b"q" * 8192
    _build_archive(archive, [("lib/big.so", payload, 0o755), ("lib/small.py", b"s", 0o644)])
    monkeypatch.setattr(xgpu_runtime, "LARGE_FILE_BYTES", 1024)
    out = tmp_path / "out"

    xgpu_runtime._safe_extract(archive, out)

    assert (out / "lib" / "big.so").read_bytes() == payload
    assert (out / "lib" / "big.so").stat().st_mode & 0o111
    assert (out / "lib" / "small.py").read_bytes() == b"s"


def test_extraction_skips_owner_and_mtime_round_trips(tmp_path, monkeypatch):
    """Each avoided metadata call is one fewer network round trip per file."""
    archive = tmp_path / "env.tar.gz"
    _build_archive(archive, [(f"lib/mod{i}.py", b"x", 0o644) for i in range(20)])
    chmod_calls = []
    monkeypatch.setattr(
        xgpu_runtime.os, "chmod", lambda *args, **kwargs: chmod_calls.append(args)
    )

    def forbidden(name):
        def fail(*args, **kwargs):
            raise AssertionError(f"{name} costs a round trip and is not needed")
        return fail

    monkeypatch.setattr(xgpu_runtime.os, "chown", forbidden("chown"), raising=False)
    monkeypatch.setattr(xgpu_runtime.os, "utime", forbidden("utime"))

    xgpu_runtime._safe_extract(archive, tmp_path / "out")

    assert chmod_calls == [], "non-executable files must not be chmod'ed"


def test_absolute_symlinks_are_preserved(tmp_path):
    """Conda environments ship links to build-machine paths; they are not attacks.

    Rejecting them aborted oea-e1 at 86% on a GDB auto-load script.
    """
    archive = tmp_path / "env.tar.gz"
    _build_archive(
        archive,
        [],
        links=[("share/gdb/auto-load/libarrow.so-gdb.py", "/build/miniconda3/lib/x.py")],
    )
    out = tmp_path / "out"

    xgpu_runtime._safe_extract(archive, out)

    link = out / "share" / "gdb" / "auto-load" / "libarrow.so-gdb.py"
    assert os.readlink(link) == "/build/miniconda3/lib/x.py"


def test_members_may_not_be_written_through_a_symlink(tmp_path):
    """The real risk is a later member traversing a link, not the link itself."""
    archive = tmp_path / "hostile.tar.gz"
    outside = tmp_path / "outside"
    outside.mkdir()
    _build_archive(
        archive,
        [("lib/escape/pwned", b"owned", 0o644)],
        links=[("lib/escape", str(outside))],
    )

    with pytest.raises(RuntimeError, match="through a symlink"):
        xgpu_runtime._safe_extract(archive, tmp_path / "out")
    assert not (outside / "pwned").exists()


def test_resume_skips_members_already_written(tmp_path):
    """A retry after an interruption must not rewrite the whole tree."""
    archive = tmp_path / "env.tar.gz"
    _build_archive(archive, [(f"lib/mod{i}.py", b"payload", 0o644) for i in range(10)])
    out = tmp_path / "out"
    xgpu_runtime._safe_extract(archive, out)

    written = []
    real_open = builtins.open

    def counting_open(file, *args, **kwargs):
        if "lib/mod" in str(file) and "w" in str(args[0] if args else kwargs.get("mode", "")):
            written.append(str(file))
        return real_open(file, *args, **kwargs)

    import unittest.mock as mock
    with mock.patch.object(builtins, "open", counting_open):
        xgpu_runtime._safe_extract(archive, out, resume=True)

    assert written == [], "identical members should not be rewritten on resume"
    assert (out / "lib" / "mod3.py").read_bytes() == b"payload"


def test_resume_still_repairs_a_truncated_member(tmp_path):
    archive = tmp_path / "env.tar.gz"
    _build_archive(archive, [("lib/mod.py", b"full-payload", 0o644)])
    out = tmp_path / "out"
    xgpu_runtime._safe_extract(archive, out)
    (out / "lib" / "mod.py").write_bytes(b"trunc")

    xgpu_runtime._safe_extract(archive, out, resume=True)

    assert (out / "lib" / "mod.py").read_bytes() == b"full-payload"


def test_write_failures_surface_from_the_pool(tmp_path, monkeypatch):
    """A writer thread's error must not be swallowed into a silent success."""
    archive = tmp_path / "env.tar.gz"
    _build_archive(archive, [(f"lib/mod{i}.py", b"x", 0o644) for i in range(10)])
    real_open = builtins.open

    def failing_on_one_member(file, *args, **kwargs):
        if str(file).endswith("mod7.py"):
            raise OSError(28, "No space left on device")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing_on_one_member)
    with pytest.raises(OSError, match="No space left"):
        xgpu_runtime._safe_extract(archive, tmp_path / "out")


def test_a_rejected_path_is_reported_not_a_writer_error(tmp_path, monkeypatch):
    """The escape must surface even when queued writes are also failing."""
    archive = tmp_path / "hostile.tar.gz"
    members = [(f"lib/mod{i}.py", b"x", 0o644) for i in range(5)]
    members.append(("../escaped", b"x", 0o644))
    _build_archive(archive, members)
    real_open = builtins.open

    def failing_writes(file, *args, **kwargs):
        if "lib/mod" in str(file):
            raise OSError(28, "No space left on device")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing_writes)
    with pytest.raises(RuntimeError, match="Unsafe path"):
        xgpu_runtime._safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escaped").exists()


def test_startup_budget_is_wall_clock_not_iterations(tmp_path, monkeypatch):
    """Each pass also spends up to 1.5 s probing, so counting loops overshot."""
    script = tmp_path / "worker.py"
    script.write_text("", encoding="utf-8")
    spec = manager.ServiceSpec(
        "earthbench-sam2", ("/bin/true", str(script)), tmp_path,
        "http://127.0.0.1:1/health", runtime="sam2",
    )
    clock = {"now": 0.0}
    monkeypatch.setattr(manager.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(manager.time, "sleep", lambda seconds: None)

    def slow_probe(_spec):
        # A real probe costs wall-clock time even when it fails.
        clock["now"] += 2.5
        return False

    monkeypatch.setattr(manager, "_reachable", slow_probe)
    monkeypatch.setattr(manager, "_startup_timeout_sec", lambda: 10)
    monkeypatch.setattr(manager.Path, "is_file", lambda self: True)
    monkeypatch.setattr(manager, "LOG_DIR", tmp_path)

    class NeverExits:
        pid = 1
        def poll(self):
            return None

    monkeypatch.setattr(manager.subprocess, "Popen", lambda *a, **k: NeverExits())
    instance = manager.GPUServiceManager()
    assert instance._start_one(spec) is False
    # Ten seconds of budget, 2.5 s per pass: four passes, not ten.
    assert clock["now"] == pytest.approx(10.0, abs=2.5)
    assert "10s" in instance.skipped["earthbench-sam2"]


def test_the_vlm_gets_a_longer_budget_than_the_light_workers():
    specs = {spec.name: spec for spec in manager.service_specs()}
    vlm = specs["oea-qwen3-vl"]
    assert vlm.startup_timeout_sec > manager._startup_timeout_sec()
    assert specs["earthbench-sam2"].startup_timeout_sec == 0


def test_worker_log_is_readable_only_for_known_services(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "LOG_DIR", tmp_path)
    (tmp_path / "earthbench-remoteclip.log").write_text(
        "Traceback\nImportError: no module named torch\n", encoding="utf-8"
    )

    assert "ImportError" in manager.read_worker_log("earthbench-remoteclip")
    with pytest.raises(KeyError):
        manager.read_worker_log("../../etc/passwd")


def test_changeos_override_refuses_silent_imwrite_failure():
    source = Path(__file__).resolve().parents[1] / "xgpu_workers" / "changeos_api.py"
    text = source.read_text(encoding="utf-8")
    assert "if not cv2.imwrite" in text
    assert "refusing to report success" in text


def test_sam2_override_paints_only_the_saved_mask():
    """The QY_11 worker visualizes all three multimask candidates and dims
    the whole frame with addWeighted. The Studio override must do neither,
    and a box prompt must use SAM2's single-mask token."""
    source = Path(__file__).resolve().parents[1] / "xgpu_workers" / "sam2_api.py"
    text = source.read_text(encoding="utf-8")
    assert "multimask_output = bbox is None" in text
    assert "_save_visualization(image_rgb, best_mask, bboxes, vis_path)" in text
    assert "cv2.addWeighted" not in text
    assert "fill_alpha = 0.35" in text


def test_status_exposes_instance_identity_for_double_instance_diagnosis():
    """Whether the "double instance" quirk is two processes on one host or
    two separate hosts changes which fix actually matters (the host-local
    lock vs. the cross-process archive publishing); comparing
    hostname/boot_id/pid across repeated /rsmem/gpu-status polls is how that
    gets told apart on a platform with no shell access."""
    instance = manager.GPUServiceManager()
    identity = instance.status()["identity"]
    assert identity["hostname"] == manager.socket.gethostname()
    assert identity["pid"] == os.getpid()
    assert isinstance(identity["instance_id"], str) and identity["instance_id"]
    assert "boot_id" in identity


def test_status_exposes_storage_snapshot_for_concurrency_probes():
    snapshot = manager._storage_snapshot()
    assert set(snapshot) == {"tmp", "root", "workspace"}
    assert snapshot["tmp"]["path"] == "/tmp"
    assert "free_gib" in snapshot["tmp"]
    assert snapshot["tmp"]["total_gib"] > 0
    storage = manager.GPUServiceManager().status()["storage"]
    assert set(storage) == {"tmp", "root", "workspace"}
    for name in ("tmp", "root", "workspace"):
        assert "free_gib" in storage[name] or "error" in storage[name]


def test_gpu_status_endpoint_exposes_manager_state(monkeypatch):
    import app as studio_app
    from fastapi.testclient import TestClient

    class FakeManager:
        def status(self):
            return {
                "workspace": "/mnt/workspace",
                "started": {"oea-controller": {"pid": 1, "running": True}},
                "skipped": {},
                "startup_timings": {"oea-controller": 1.5},
                "log_dir": "/tmp/logs",
                "startup_running": False,
            }

    monkeypatch.setattr(studio_app, "get_gpu_service_manager", lambda: FakeManager())
    monkeypatch.setenv("RSMEM_GPU_AUTOSTART", "0")
    client = TestClient(studio_app.create_app())
    response = client.get("/rsmem/gpu-status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["started"]["oea-controller"]["running"] is True
    assert payload["startup_timings"]["oea-controller"] == 1.5
