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


def get_geotransform(file_path) -> tuple:
    with rasterio.open(file_path) as src:
        transform = src.transform
        proj = src.crs.to_wkt() if src.crs else ""
    geo = transform.to_gdal()
    if geo == (0, 1.0, 0, 0, 0, 1.0):
        return None, None
    else:
        return geo, proj
