import numpy as np
import rasterio


def read_image(file_path: str) -> np.ndarray:
    with rasterio.open(file_path) as src:
        data = src.read()
    return data[0] if data.shape[0] == 1 else np.moveaxis(data, 0, -1)


def read_image_uint8(file_path: str) -> np.ndarray:
    img = read_image(file_path)

    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val) * 255
    else:
        img = np.zeros_like(img)

    return img.astype(np.uint8)


def write_image(file_path, data: np.ndarray, geotransform=None, projection=None, nodata=None) -> str:
    """Save a 2-D or (H, W, C) array as a GeoTIFF.

    Mirrors :func:`read_image` so tools can round-trip without the system GDAL
    Python binding, which is not installed in the Studio container.
    """
    data = np.asarray(data)
    stacked = data[np.newaxis, :, :] if data.ndim == 2 else np.moveaxis(data, -1, 0)
    profile = {
        "driver": "GTiff",
        "height": stacked.shape[1],
        "width": stacked.shape[2],
        "count": stacked.shape[0],
        "dtype": stacked.dtype.name,
    }
    if geotransform:
        profile["transform"] = rasterio.Affine.from_gdal(*geotransform)
    if projection:
        profile["crs"] = projection
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(str(file_path), "w", **profile) as dst:
        dst.write(stacked)
    return str(file_path)


def stretch_to_uint8(data: np.ndarray) -> np.ndarray:
    """Rescale each band to 0-255 independently."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2:
        return read_image_uint8_from_array(data)
    bands = [read_image_uint8_from_array(data[:, :, index]) for index in range(data.shape[2])]
    return np.stack(bands, axis=-1)


def read_image_uint8_from_array(band: np.ndarray) -> np.ndarray:
    band = np.asarray(band, dtype=np.float32)
    low, high = np.min(band), np.max(band)
    if high <= low:
        return np.zeros_like(band, dtype=np.uint8)
    return ((band - low) / (high - low) * 255).astype(np.uint8)


def get_geotransform(file_path) -> tuple:
    with rasterio.open(file_path) as src:
        transform = src.transform
        proj = src.crs.to_wkt() if src.crs else ""
    geo = transform.to_gdal()
    if geo == (0, 1.0, 0, 0, 0, 1.0):
        return None, None
    else:
        return geo, proj
