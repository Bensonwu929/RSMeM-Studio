"""CPU-only raster tools used by the public RSMeM Studio.

The functions mirror the lightweight Index/Statistics tool semantics in the
released RSMeM repository, but avoid the GPU perception services and the
system-level GDAL Python binding.  Rasterio supplies the GeoTIFF I/O layer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


RASTER_SUFFIXES = {".tif", ".tiff"}


def _read_band(path: str | Path) -> tuple[np.ndarray, dict]:
    path = Path(path)
    if path.suffix.lower() not in RASTER_SUFFIXES:
        raise ValueError(f"Unsupported raster type: {path.suffix}")
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        profile = src.profile.copy()
    return data, profile


def _write_band(path: str | Path, data: np.ndarray, profile: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output_profile = profile.copy()
    output_profile.update(count=1, dtype="float32", nodata=-9999.0, compress="lzw")
    with rasterio.open(path, "w", **output_profile) as dst:
        dst.write(np.nan_to_num(data, nan=-9999.0).astype(np.float32), 1)
    return str(path)


def _normalized_difference(a_path: str, b_path: str, output_path: str) -> str:
    a, profile = _read_band(a_path)
    b, _ = _read_band(b_path)
    if a.shape != b.shape:
        raise ValueError(f"Raster shape mismatch: {a.shape} versus {b.shape}")
    denominator = a + b
    result = np.divide(a - b, denominator, out=np.zeros_like(a), where=np.abs(denominator) > 1e-6)
    return _write_band(output_path, result, profile)


def calculate_ndvi(nir_path: str, red_path: str, output_path: str) -> str:
    return _normalized_difference(nir_path, red_path, output_path)


def calculate_ndwi(nir_path: str, swir_path: str, output_path: str) -> str:
    """Match the NIR/SWIR NDWI implementation released in RSMeM."""
    return _normalized_difference(nir_path, swir_path, output_path)


def calculate_nbr(nir_path: str, swir2_path: str, output_path: str) -> str:
    return _normalized_difference(nir_path, swir2_path, output_path)


def compute_tvdi(ndvi_path: str, lst_path: str, output_path: str) -> str:
    """Compute TVDI with the trapezoidal NDVI-LST feature-space method."""
    ndvi_raw, profile = _read_band(ndvi_path)
    lst_raw, _ = _read_band(lst_path)
    if ndvi_raw.shape != lst_raw.shape:
        raise ValueError(f"Raster shape mismatch: {ndvi_raw.shape} versus {lst_raw.shape}")

    # Match the scale factors used by the released RSMeM Index tool.
    ndvi = ndvi_raw * 0.0001 if np.nanpercentile(np.abs(ndvi_raw), 95) > 2 else ndvi_raw
    lst = lst_raw * 0.02 if np.nanpercentile(lst_raw, 95) > 1000 else lst_raw
    valid = np.isfinite(ndvi) & np.isfinite(lst) & (ndvi >= 0) & (ndvi <= 1) & (lst > 0)
    if np.count_nonzero(valid) < 100:
        raise ValueError("Too few valid NDVI/LST pixels for TVDI regression.")

    ndvi_valid, lst_valid = ndvi[valid], lst[valid]
    bins = np.linspace(float(ndvi_valid.min()), float(ndvi_valid.max()), 101)
    centers, dry_edge, wet_edge = [], [], []
    for index in range(100):
        selected = (ndvi_valid >= bins[index]) & (ndvi_valid < bins[index + 1])
        if np.any(selected):
            centers.append((bins[index] + bins[index + 1]) / 2)
            dry_edge.append(float(np.max(lst_valid[selected])))
            wet_edge.append(float(np.min(lst_valid[selected])))
    if len(centers) < 2:
        raise ValueError("Not enough occupied NDVI bins for TVDI regression.")

    dry_slope, dry_intercept = np.polyfit(centers, dry_edge, 1)
    wet_slope, wet_intercept = np.polyfit(centers, wet_edge, 1)
    maximum = ndvi * dry_slope + dry_intercept
    minimum = ndvi * wet_slope + wet_intercept
    denominator = maximum - minimum
    tvdi = np.divide(lst - minimum, denominator, out=np.full_like(lst, np.nan), where=np.abs(denominator) > 1e-6)
    tvdi = np.clip(tvdi, 0, 1)
    tvdi[~valid] = np.nan
    return _write_band(output_path, tvdi, profile)


def sens_slope(values: list[float]) -> dict:
    if len(values) < 2:
        raise ValueError("At least two time-series values are required for Sen's slope.")
    slopes = [
        (values[j] - values[i]) / (j - i)
        for i in range(len(values) - 1)
        for j in range(i + 1, len(values))
    ]
    slope = float(np.median(slopes))
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
    return {"slope_per_observation": slope, "direction": direction, "observations": len(values)}


def calculate_threshold_ratio(image_path: str, threshold: float = 0.0) -> float:
    image, _ = _read_band(image_path)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("Raster contains no finite pixels.")
    return float(np.mean(finite > threshold))


def create_threshold_mask(image_path: str, threshold: float, output_path: str) -> str:
    image, profile = _read_band(image_path)
    mask = (np.isfinite(image) & (image > threshold)).astype(np.float32)
    return _write_band(output_path, mask, profile)


def calculate_area(mask_path: str, gsd: float | None = None) -> dict:
    mask, _ = _read_band(mask_path)
    count = int(np.count_nonzero(mask > 0))
    if gsd is None:
        with rasterio.open(mask_path) as src:
            transform = src.transform
            if src.crs and src.crs.is_projected and not transform.is_identity:
                pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
                return {"pixels": count, "area_m2": float(count * pixel_area), "source": "GeoTIFF transform"}
        return {"pixels": count, "area_m2": None, "source": "pixel count only; no projected transform"}
    return {"pixels": count, "area_m2": float(count * gsd * gsd), "source": "user/default GSD"}


def calc_batch_image_mean(paths: list[str]) -> list[float]:
    means = []
    for path in paths:
        image, _ = _read_band(path)
        values = image[np.isfinite(image)]
        means.append(float(np.mean(values)) if values.size else float("nan"))
    return means


def inspect_rasters(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with rasterio.open(path) as src:
            rows.append(
                {
                    "name": Path(path).name,
                    "shape": [src.height, src.width],
                    "bands": src.count,
                    "dtype": src.dtypes[0],
                    "crs": str(src.crs) if src.crs else None,
                    "pixel_size": [abs(src.transform.a), abs(src.transform.e)],
                }
            )
    return rows


def infer_band_roles(paths: list[str]) -> dict[str, str]:
    """Infer common Landsat/Sentinel band roles from safe uploaded filenames."""
    patterns = {
        "green": (r"(^|[_-])green([_.-]|$)", r"(^|[_-])b0?3([_.-]|$)"),
        "red": (r"(^|[_-])red([_.-]|$)", r"(^|[_-])b0?4([_.-]|$)"),
        "nir": (r"(^|[_-])nir([_.-]|$)", r"(^|[_-])b0?(5|8|8a)([_.-]|$)"),
        "swir1": (r"(^|[_-])swir1?([_.-]|$)", r"(^|[_-])b0?(6|11)([_.-]|$)"),
        "swir2": (r"(^|[_-])swir2([_.-]|$)", r"(^|[_-])b0?(7|12)([_.-]|$)"),
    }
    roles: dict[str, str] = {}
    for path in paths:
        name = Path(path).name.lower()
        for role, role_patterns in patterns.items():
            if role not in roles and any(re.search(pattern, name) for pattern in role_patterns):
                roles[role] = path
    return roles


def create_demo_scene(directory: str | Path) -> list[str]:
    """Create a deterministic CPU demo scene when the visitor uploads no data."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:128, :128]
    water = ((x - 64) ** 2 / 40**2 + (y - 66) ** 2 / 27**2) < 1
    texture = 0.015 * np.sin(x / 7) * np.cos(y / 9)
    bands = {
        "scene_B3_green.tif": np.where(water, 0.13, 0.16) + texture,
        "scene_B4_red.tif": np.where(water, 0.07, 0.20) + texture,
        "scene_B5_nir.tif": np.where(water, 0.05, 0.42) + texture,
        "scene_B6_swir1.tif": np.where(water, 0.02, 0.38) + texture,
        "scene_B7_swir2.tif": np.where(water, 0.015, 0.23) + texture,
    }
    profile = {
        "driver": "GTiff",
        "height": 128,
        "width": 128,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32650",
        "transform": from_origin(500000, 4000000, 10, 10),
        "compress": "lzw",
    }
    paths = []
    for name, data in bands.items():
        path = directory / name
        _write_band(path, data.astype(np.float32), profile)
        paths.append(str(path))
    (directory / "scene_manifest.json").write_text(
        json.dumps({"source": "synthetic demonstration", "bands": [Path(p).name for p in paths]}, indent=2),
        encoding="utf-8",
    )
    return paths
