"""Rebuild display-only assets from the bundled Sichuan rasters (Apache-2.0)."""

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parents[1]


def main():
    for band, colors in (
        ("NDVI", [(10, 24, 47), (25, 69, 102), (51, 140, 159), (161, 228, 200)]),
        ("LST", [(24, 38, 62), (102, 81, 106), (198, 131, 116), (252, 218, 164)]),
    ):
        source = ROOT / "examples" / "tvdi_sichuan" / f"Sichuan_2021-07-12_{band}.tif"
        with rasterio.open(source) as raster:
            width = min(raster.width, 1200)
            height = max(1, round(raster.height * width / raster.width))
            values = raster.read(1, out_shape=(height, width), masked=True,
                                 resampling=Resampling.bilinear).astype(float).filled(np.nan)
        valid = np.isfinite(values)
        low, high = np.percentile(values[valid], [2, 98])
        normalized = np.nan_to_num(np.clip((values - low) / max(high - low, 1e-9), 0, 1))
        positions = np.linspace(0, 1, len(colors))
        rgb = np.stack([np.interp(normalized, positions, np.array(colors)[:, i])
                        for i in range(3)], axis=-1).astype(np.uint8)
        rgb[~valid] = colors[0]
        Image.fromarray(rgb).save(ROOT / "assets" / f"sichuan-{band.lower()}.webp", quality=88)


if __name__ == "__main__":
    main()
