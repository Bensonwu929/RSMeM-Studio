"""CPU-safe Studio extensions registered on the released EarthAgent MCP servers.

The public Studio keeps the complete released CPU tool catalog and adds a small
set of rasterio-only adapters used by its prepared cases.  These adapters live
inside the MCP processes: the Gradio process never imports or invokes their
implementations directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cpu_tools as cpu  # noqa: E402


def register_analysis_extensions(mcp) -> None:
    @mcp.tool(name="studio_linear_trend")
    def studio_linear_trend(values: list[float], x: list[float] | None = None) -> dict:
        return cpu.linear_trend(values, x)

    @mcp.tool(name="studio_sens_slope")
    def studio_sens_slope(values: list[float]) -> dict:
        return cpu.sens_slope(values)

    @mcp.tool(name="studio_hotspot_direction")
    def studio_hotspot_direction(mask_path: str) -> dict:
        return cpu.hotspot_direction(mask_path)

    @mcp.tool(name="studio_consecutive_percentage_changes")
    def studio_consecutive_percentage_changes(values: list[float]) -> list[float]:
        return cpu.consecutive_percentage_changes(values)


def register_index_extensions(mcp) -> None:
    @mcp.tool(name="studio_calculate_ndvi")
    def studio_calculate_ndvi(nir_path: str, red_path: str, output_path: str) -> str:
        return cpu.calculate_ndvi(nir_path, red_path, output_path)

    @mcp.tool(name="studio_calculate_ndwi")
    def studio_calculate_ndwi(nir_path: str, swir_path: str, output_path: str) -> str:
        return cpu.calculate_ndwi(nir_path, swir_path, output_path)

    @mcp.tool(name="studio_calculate_nbr")
    def studio_calculate_nbr(nir_path: str, swir2_path: str, output_path: str) -> str:
        return cpu.calculate_nbr(nir_path, swir2_path, output_path)

    @mcp.tool(name="studio_compute_tvdi")
    def studio_compute_tvdi(ndvi_path: str, lst_path: str, output_path: str) -> str:
        return cpu.compute_tvdi(ndvi_path, lst_path, output_path)

    @mcp.tool(name="studio_create_threshold_mask")
    def studio_create_threshold_mask(
        image_path: str, threshold: float, output_path: str
    ) -> str:
        return cpu.create_threshold_mask(image_path, threshold, output_path)

    @mcp.tool(name="studio_create_below_threshold_mask")
    def studio_create_below_threshold_mask(
        image_path: str, threshold: float, output_path: str
    ) -> str:
        return cpu.create_below_threshold_mask(image_path, threshold, output_path)

    @mcp.tool(name="studio_combine_binary_masks")
    def studio_combine_binary_masks(mask_paths: list[str], output_path: str) -> str:
        return cpu.combine_binary_masks(mask_paths, output_path)


def register_inversion_extensions(mcp) -> None:
    @mcp.tool(name="studio_lst_single_channel")
    def studio_lst_single_channel(
        bt_path: str, red_path: str, nir_path: str, output_path: str
    ) -> str:
        return cpu.lst_single_channel(bt_path, red_path, nir_path, output_path)

    @mcp.tool(name="studio_mean_lst_by_ndvi")
    def studio_mean_lst_by_ndvi(
        red_path: str, nir_path: str, lst_path: str, ndvi_output_path: str
    ) -> dict:
        return cpu.mean_lst_by_ndvi(red_path, nir_path, lst_path, ndvi_output_path)

    @mcp.tool(name="studio_apparent_thermal_inertia")
    def studio_apparent_thermal_inertia(
        day_temp_path: str,
        night_temp_path: str,
        albedo_path: str,
        output_path: str,
    ) -> str:
        return cpu.apparent_thermal_inertia(
            day_temp_path, night_temp_path, albedo_path, output_path
        )

    @mcp.tool(name="studio_calculate_turbidity")
    def studio_calculate_turbidity(input_red_path: str, output_path: str) -> str:
        return cpu.calculate_turbidity(input_red_path, output_path)


def register_statistics_extensions(mcp) -> None:
    @mcp.tool(name="studio_validate_inputs")
    def studio_validate_inputs(paths: list[str]) -> dict:
        inspected = cpu.inspect_rasters(paths)
        if not inspected:
            raise ValueError("No readable GeoTIFF inputs are available.")
        return {"valid": True, "count": len(inspected)}

    @mcp.tool(name="studio_inspect_rasters")
    def studio_inspect_rasters(paths: list[str]) -> list[dict]:
        return cpu.inspect_rasters(paths)

    @mcp.tool(name="studio_calc_batch_image_mean")
    def studio_calc_batch_image_mean(paths: list[str]) -> list[float]:
        return cpu.calc_batch_image_mean(paths)

    @mcp.tool(name="studio_raster_skewness")
    def studio_raster_skewness(path: str) -> float:
        return cpu.raster_skewness(path)

    @mcp.tool(name="studio_subtract_rasters")
    def studio_subtract_rasters(
        first_path: str, second_path: str, output_path: str
    ) -> str:
        return cpu.subtract_rasters(first_path, second_path, output_path)

    @mcp.tool(name="studio_maximum_composite")
    def studio_maximum_composite(paths: list[str], output_path: str) -> str:
        return cpu.maximum_composite(paths, output_path)

    @mcp.tool(name="studio_fire_prone_mask")
    def studio_fire_prone_mask(
        image_path: str, output_path: str, percentile: float = 75
    ) -> list:
        path, threshold = cpu.fire_prone_mask(image_path, output_path, percentile)
        return [path, threshold]

    @mcp.tool(name="studio_calculate_threshold_ratio")
    def studio_calculate_threshold_ratio(image_path: str, threshold: float) -> float:
        return cpu.calculate_threshold_ratio(image_path, threshold)

    @mcp.tool(name="studio_calculate_area")
    def studio_calculate_area(mask_path: str) -> dict:
        return cpu.calculate_area(mask_path)

    @mcp.tool(name="studio_difference")
    def studio_difference(a: float, b: float) -> dict:
        return {
            "first": float(a),
            "second": float(b),
            "absolute_difference": abs(float(b) - float(a)),
            "direction": "second period is higher" if b > a else "first period is higher",
        }

    @mcp.tool(name="studio_max_value_and_index")
    def studio_max_value_and_index(values: list[float]) -> dict:
        finite = [(index, value) for index, value in enumerate(values) if np.isfinite(value)]
        if not finite:
            raise ValueError("No finite values are available for maximum selection.")
        index, value = max(finite, key=lambda row: row[1])
        return {"index": index, "maximum": float(value)}

    @mcp.tool(name="studio_max_three_day_window")
    def studio_max_three_day_window(values: list[float]) -> dict:
        if len(values) < 3:
            raise ValueError("At least three daily values are required.")
        windows = [float(np.mean(values[i:i + 3])) for i in range(len(values) - 2)]
        index = int(np.nanargmax(windows))
        return {"start_index": index, "end_index": index + 2, "three_day_mean": windows[index]}

    @mcp.tool(name="studio_validate_result")
    def studio_validate_result(input_paths: list[str], artifact_paths: list[str]) -> dict:
        candidates = artifact_paths or input_paths
        inspected = cpu.inspect_rasters(candidates)
        if len(inspected) != len(candidates):
            raise ValueError("One or more result rasters are unreadable.")
        return {"valid": True, "checked": len(inspected)}
