from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

import mcp_runtime


def _write_raster(path: Path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        transform=from_origin(0, 100, 10, 10),
        nodata=-9999,
    ) as dst:
        dst.write(values.astype("float32"), 1)


def test_subprocess_environment_removes_all_credentials():
    clean = mcp_runtime.sanitized_subprocess_environment(
        {
            "PATH": "/usr/bin",
            "LANG": "C.UTF-8",
            "OPENAI_API_KEY": "must-not-leak",
            "MODELSCOPE_ACCESS_TOKEN": "must-not-leak",
            "SERVICE_SECRET": "must-not-leak",
            "PROXY_PASSWORD": "must-not-leak",
        }
    )
    assert clean == {"PATH": "/usr/bin", "LANG": "C.UTF-8"}


def test_runtime_starts_four_cpu_servers_and_discovers_extensions(tmp_path):
    runtime = mcp_runtime.MCPRuntime(tmp_path, timeout=30).start()
    try:
        health = runtime.health()
        assert health["transport"] == "stdio"
        assert health["enabled_servers"] == ["Analysis", "Index", "Inversion", "Statistics"]
        assert health["disabled_servers"] == ["Perception", "OEA_Perception"]
        assert health["discovered_tools"] >= 100
        assert "Analysis.studio_linear_trend" in runtime.catalog
        assert "Index.studio_calculate_ndvi" in runtime.catalog
        assert "Inversion.studio_lst_single_channel" in runtime.catalog
        assert "Statistics.studio_inspect_rasters" in runtime.catalog
        assert not any(key.startswith("Perception.") for key in runtime.catalog)
        assert not any(key.startswith("OEA_Perception.") for key in runtime.catalog)
    finally:
        runtime.close()


def test_real_mcp_call_creates_request_scoped_geotiff(tmp_path):
    nir = tmp_path / "sample_b5.tif"
    red = tmp_path / "sample_b4.tif"
    output_dir = tmp_path / "request-output"
    output = output_dir / "ndvi.tif"
    _write_raster(nir, np.array([[0.8, 0.6], [0.3, 0.1]]))
    _write_raster(red, np.array([[0.2, 0.3], [0.2, 0.1]]))

    runtime = mcp_runtime.MCPRuntime(output_dir, timeout=30).start()
    try:
        result = runtime.call(
            "Index",
            "studio_calculate_ndvi",
            {"nir_path": str(nir), "red_path": str(red), "output_path": str(output)},
        )
    finally:
        runtime.close()

    assert result == str(output)
    assert output.is_file()
    assert output.resolve().is_relative_to(output_dir.resolve())
    with rasterio.open(output) as src:
        actual = src.read(1)
    np.testing.assert_allclose(actual[0, 0], 0.6, atol=1e-5)


def test_sync_facade_routes_calls_to_discovered_mcp_tool(monkeypatch):
    calls = []

    class FakeRuntime:
        def call(self, service, tool, arguments):
            calls.append((service, tool, arguments))
            return {"slope": 2.0, "direction": "increasing"}

    token = mcp_runtime._ACTIVE_RUNTIME.set(FakeRuntime())
    try:
        result = mcp_runtime.linear_trend([1.0, 3.0], [0.0, 1.0])
    finally:
        mcp_runtime._ACTIVE_RUNTIME.reset(token)

    assert result["slope"] == 2.0
    assert calls == [
        (
            "Analysis",
            "studio_linear_trend",
            {"values": [1.0, 3.0], "x": [0.0, 1.0]},
        )
    ]


def test_gpu_servers_are_opt_in_and_unhealthy_workers_are_not_planner_visible(tmp_path, monkeypatch):
    monkeypatch.setenv("RSMEM_ENABLE_GPU_PERCEPTION", "1")
    runtime = mcp_runtime.MCPRuntime(tmp_path, timeout=30).start()
    try:
        health = runtime.health()
        aliases = runtime.workflow_aliases()
        assert health["enabled_servers"] == [
            "Analysis", "Index", "Inversion", "Statistics", "Perception", "OEA_Perception"
        ]
        assert "Perception.available_tools" in runtime.catalog
        assert "OEA_Perception.available_tools" in runtime.catalog
        assert not (mcp_runtime.GPU_NATIVE_TOOLS["Perception"] & runtime.live_gpu_tools["Perception"])
        assert not (mcp_runtime.GPU_NATIVE_TOOLS["OEA_Perception"] & runtime.live_gpu_tools["OEA_Perception"])
        assert not ({
            "remoteclip_classify", "sam2_segment", "earthbench_change_detection",
            "earthbench_damage_assessment", "text_to_bbox", "object_detection",
            "image_description", "region_attribute_description", "count_given_object", "ocr_image",
        } & aliases)
    finally:
        runtime.close()


def test_call_workflow_tool_uses_binding(monkeypatch):
    calls = []

    class FakeRuntime:
        def call(self, service, tool, arguments):
            calls.append((service, tool, arguments))
            return {"predicted_class": "Forest"}

    token = mcp_runtime._ACTIVE_RUNTIME.set(FakeRuntime())
    try:
        result = mcp_runtime.call_workflow_tool(
            "remoteclip_classify", {"input_image_path": "/tmp/example.tif"}
        )
    finally:
        mcp_runtime._ACTIVE_RUNTIME.reset(token)

    assert result["predicted_class"] == "Forest"
    assert calls == [
        ("Perception", "RemoteCLIP", {"input_image_path": "/tmp/example.tif"})
    ]
