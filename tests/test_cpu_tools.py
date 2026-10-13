"""Raster I/O for the formats the benchmark actually ships.

Every selected EarthBench case delivers PNG or JPEG; none ship GeoTIFF. Tools
that accepted only GeoTIFF aborted whole attempts with "Unsupported raster
type: .png".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cpu_tools


def _write_png(path: Path, data: np.ndarray) -> None:
    with rasterio.open(
        path, "w", driver="PNG", height=data.shape[0], width=data.shape[1],
        count=1, dtype="uint8",
    ) as dst:
        dst.write(data, 1)


def test_png_is_readable(tmp_path):
    source = tmp_path / "t1.png"
    _write_png(source, np.array([[10, 20], [30, 40]], dtype=np.uint8))

    data, profile = cpu_tools._read_band(source)

    assert data.shape == (2, 2)
    assert data[0, 0] == pytest.approx(10.0)
    assert profile["height"] == 2


def test_zero_pixels_in_a_png_survive(tmp_path):
    """GDAL reports nodata=0 for PNG, but zero is a legitimate dark pixel."""
    source = tmp_path / "dark.png"
    _write_png(source, np.array([[0, 20], [30, 40]], dtype=np.uint8))

    data, _ = cpu_tools._read_band(source)

    assert not np.isnan(data).any(), "a real dark pixel was mistaken for a gap"
    assert data[0, 0] == pytest.approx(0.0)


def test_results_from_png_inputs_are_written_as_geotiff(tmp_path):
    """Float32 output cannot live in a PNG, whatever the input format was."""
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    _write_png(first, np.array([[50, 60]], dtype=np.uint8))
    _write_png(second, np.array([[20, 10]], dtype=np.uint8))
    output = tmp_path / "diff.tif"

    cpu_tools.subtract_rasters(str(first), str(second), str(output))

    with rasterio.open(output) as src:
        assert src.driver == "GTiff"
        assert src.dtypes[0] == "float32"
        assert src.read(1).tolist() == [[30.0, 50.0]]


def test_unknown_suffixes_are_still_rejected(tmp_path):
    bogus = tmp_path / "notes.txt"
    bogus.write_text("not a raster", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported raster type"):
        cpu_tools._read_band(bogus)
