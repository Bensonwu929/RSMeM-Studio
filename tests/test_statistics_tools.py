"""Raster maths that must work without the system GDAL Python binding.

The Studio container ships rasterio but not `osgeo`, so tools importing it
failed on every call. Because a failing tool consumes one of the attempt's
thirty model calls, that waste is what pushed cases past the gateway's
thirty-minute ceiling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import utils


def _write_png(path: Path, data: np.ndarray) -> None:
    with rasterio.open(
        path, "w", driver="PNG", height=data.shape[0], width=data.shape[1],
        count=1, dtype="uint8",
    ) as dst:
        dst.write(data, 1)


def _load_statistics(tmp_path):
    """Import Statistics.py, which parses --temp_dir at module scope."""
    import importlib.util

    saved = sys.argv
    sys.argv = ["Statistics.py", "--temp_dir", str(tmp_path / "temp")]
    try:
        spec = importlib.util.spec_from_file_location(
            "statistics_tools", TOOLS / "Statistics.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = saved


def test_write_image_round_trips_through_read_image(tmp_path):
    target = tmp_path / "out.tif"
    payload = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)

    utils.write_image(target, payload)

    assert np.allclose(utils.read_image(target), payload)


def test_stretch_to_uint8_handles_a_flat_band(tmp_path):
    flat = np.full((2, 2), 7.0)
    assert utils.stretch_to_uint8(flat).tolist() == [[0, 0], [0, 0]]


def test_tif_difference_works_on_png_without_osgeo(tmp_path):
    module = _load_statistics(tmp_path)
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    _write_png(first, np.array([[10, 20]], dtype=np.uint8))
    _write_png(second, np.array([[40, 25]], dtype=np.uint8))

    message = module.calculate_tif_difference(str(first), str(second), "diff.tif")

    produced = Path(message.split("Result save at ")[-1].strip())
    assert produced.is_file()
    assert utils.read_image(produced).tolist() == [[30.0, 5.0]]


def test_tif_average_works_on_png_without_osgeo(tmp_path):
    module = _load_statistics(tmp_path)
    paths = []
    for index, values in enumerate([[10, 20], [30, 40]]):
        path = tmp_path / f"m{index}.png"
        _write_png(path, np.array([values], dtype=np.uint8))
        paths.append(str(path))

    message = module.calculate_tif_average(paths, "avg.tif")

    produced = Path(message.split("Result save at ")[-1].strip())
    assert utils.read_image(produced).tolist() == [[20.0, 30.0]]


def test_tif_average_rejects_mismatched_shapes(tmp_path):
    module = _load_statistics(tmp_path)
    small = tmp_path / "s.png"
    large = tmp_path / "l.png"
    _write_png(small, np.array([[1, 2]], dtype=np.uint8))
    _write_png(large, np.array([[1, 2, 3]], dtype=np.uint8))

    with pytest.raises(ValueError, match="same dimensions"):
        module.calculate_tif_average([str(small), str(large)], "avg.tif")


def _load_tool_module(filename: str, module_name: str, tmp_path):
    """Import a tools/*.py module that parses --temp_dir at import time."""
    import importlib.util

    saved = sys.argv
    sys.argv = [filename, "--temp_dir", str(tmp_path / "temp")]
    try:
        spec = importlib.util.spec_from_file_location(module_name, TOOLS / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = saved


def _write_geotiff(path: Path, data: np.ndarray) -> None:
    data = np.asarray(data)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype.name,
        transform=rasterio.Affine(0.1, 0, 32.0, 0, -0.1, 35.0),
    ) as dst:
        dst.write(data, 1)


def test_write_image_records_nodata(tmp_path):
    target = tmp_path / "out.tif"
    payload = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)

    utils.write_image(target, payload, nodata=np.nan)

    with rasterio.open(target) as src:
        assert np.isnan(src.nodata)


def test_no_tool_module_imports_osgeo():
    import ast

    offenders = []
    for path in sorted(TOOLS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            if "osgeo" in names:
                offenders.append(path.name)
                break
    assert offenders == [], (
        f"{offenders} import osgeo, which the Studio container does not ship"
    )


def test_ati_resizes_inputs_to_the_daytime_grid(tmp_path):
    module = _load_tool_module("Inversion.py", "inversion_tools", tmp_path)
    day = tmp_path / "day.tif"
    night = tmp_path / "night.tif"
    albedo = tmp_path / "albedo.tif"
    _write_geotiff(day, np.full((4, 3), 310.0, dtype=np.float32))
    _write_geotiff(night, np.full((6, 5), 300.0, dtype=np.float32))
    _write_geotiff(albedo, np.full((5, 4), 0.2, dtype=np.float32))

    message = module.ATI(str(day), str(night), str(albedo), "ati.tif")

    produced = Path(message.split("Result saved at ")[-1].strip())
    assert produced.is_file()
    with rasterio.open(produced) as src:
        assert src.shape == (4, 3)
        assert src.read(1) == pytest.approx(0.08, abs=1e-5)


def test_ati_resamples_integer_inputs_at_float_precision(tmp_path):
    """The GDAL path staged every input as float32 before warping.

    GDAL rounds resampled values to the band's own dtype, so resampling an
    int16 albedo raster directly would quantise it and silently shift ATI.
    """
    module = _load_tool_module("Inversion.py", "inversion_tools", tmp_path)
    day = tmp_path / "day.tif"
    night = tmp_path / "night.tif"
    _write_geotiff(day, np.full((5, 5), 310.0, dtype=np.float32))
    _write_geotiff(night, np.full((5, 5), 300.0, dtype=np.float32))

    # Negative albedo keeps (1 - albedo) / 10 inside the [0, 10] clip, so the
    # comparison is not masked by saturation.
    gradient = np.arange(-9, 0).reshape(3, 3)
    as_int = tmp_path / "albedo_i16.tif"
    as_float = tmp_path / "albedo_f32.tif"
    _write_geotiff(as_int, gradient.astype(np.int16))
    _write_geotiff(as_float, gradient.astype(np.float32))

    def run(albedo_path: Path, name: str) -> np.ndarray:
        message = module.ATI(str(day), str(night), str(albedo_path), name)
        with rasterio.open(message.split("Result saved at ")[-1].strip()) as src:
            return src.read(1)

    from_int = run(as_int, "from_int.tif")
    from_float = run(as_float, "from_float.tif")

    assert from_int == pytest.approx(from_float, abs=1e-6)
    # Guard against a vacuous pass: a quantised run lands on multiples of 0.1.
    assert np.any(np.abs(from_int * 10 - np.round(from_int * 10)) > 1e-3)


def test_getis_ord_gi_star_writes_a_matching_geotiff(tmp_path):
    module = _load_tool_module("Analysis.py", "analysis_tools", tmp_path)
    source = tmp_path / "ndvi.tif"
    _write_geotiff(
        source,
        np.array(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 9.0, 3.0, 1.0], [1.0, 2.0, 8.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=np.float32,
        ),
    )

    message = module.getis_ord_gi_star(
        str(source),
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
        "gi_star.tif",
    )

    produced = Path(message.split("Result save at ")[-1].strip())
    assert produced.is_file()
    with rasterio.open(produced) as src:
        assert src.shape == (4, 4)
        assert src.transform == rasterio.Affine(0.1, 0, 32.0, 0, -0.1, 35.0)
        assert np.isfinite(src.read(1)).all()


def test_hotspot_tif_preserves_georeference_without_osgeo(tmp_path):
    module = _load_statistics(tmp_path)
    source = tmp_path / "lst.tif"
    _write_geotiff(source, np.array([[10.0, 30.0], [40.0, 20.0]], dtype=np.float32))

    message = module.calc_single_image_hotspot_tif(str(source), 25.0, "hotspot.tif")

    produced = Path(message.split("Result save at ")[-1].strip())
    assert produced.is_file()
    with rasterio.open(produced) as src:
        assert src.shape == (2, 2)
        assert src.transform == rasterio.Affine(0.1, 0, 32.0, 0, -0.1, 35.0)
        assert src.read(1).tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_grayscale_to_colormap_writes_png_without_cv2(tmp_path):
    module = _load_statistics(tmp_path)
    source = tmp_path / "gray.tif"
    _write_geotiff(source, np.array([[0.0, 50.0], [100.0, 150.0]], dtype=np.float32))

    message = module.grayscale_to_colormap(str(source), "color.png")

    produced = Path(message.split("Result save at ")[-1].strip())
    assert produced.is_file()
    assert produced.suffix == ".png"
    assert produced.stat().st_size > 0
