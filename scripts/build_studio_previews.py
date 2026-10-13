"""Rebuild display-only assets from the bundled Black Forest rasters (Apache-2.0)."""

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "examples" / "q10_black_forest"

NDVI_COLORS = [(10, 24, 47), (25, 69, 102), (51, 140, 159), (161, 228, 200)]
LST_COLORS = [(24, 38, 62), (102, 81, 106), (198, 131, 116), (252, 218, 164)]


def _read_resampled(path: Path) -> np.ndarray:
    with rasterio.open(path) as raster:
        width = min(raster.width, 1200)
        height = max(1, round(raster.height * width / raster.width))
        return raster.read(1, out_shape=(height, width), masked=True,
                            resampling=Resampling.bilinear).astype(float).filled(np.nan)


def _colorize(values: np.ndarray, colors: list[tuple[int, int, int]]) -> np.ndarray:
    valid = np.isfinite(values)
    low, high = np.percentile(values[valid], [2, 98])
    normalized = np.nan_to_num(np.clip((values - low) / max(high - low, 1e-9), 0, 1))
    positions = np.linspace(0, 1, len(colors))
    rgb = np.stack([np.interp(normalized, positions, np.array(colors)[:, i])
                    for i in range(3)], axis=-1).astype(np.uint8)
    rgb[~valid] = colors[0]
    return rgb


def main():
    red = _read_resampled(SOURCE_DIR / "Germany_2021-07-29_b4.tif")
    nir = _read_resampled(SOURCE_DIR / "Germany_2021-07-29_b5.tif")
    ndvi = (nir - red) / np.where((nir + red) == 0, np.nan, (nir + red))
    Image.fromarray(_colorize(ndvi, NDVI_COLORS)).save(
        ROOT / "assets" / "blackforest-ndvi.webp", quality=88
    )

    brightness_temp = _read_resampled(SOURCE_DIR / "Germany_2021-07-29_BT10.tif")
    Image.fromarray(_colorize(brightness_temp, LST_COLORS)).save(
        ROOT / "assets" / "blackforest-lst.webp", quality=88
    )


if __name__ == "__main__":
    main()
