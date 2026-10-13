"""Synchronous facade over the RSMeM stdio MCP tool processes."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import sys
import tempfile
import threading
from contextlib import AsyncExitStack, contextmanager
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "agent_mcp.json"
CPU_SERVERS = ("Analysis", "Index", "Inversion", "Statistics")
GPU_SERVERS = ("Perception", "OEA_Perception")
CODE_SERVERS = ("CodeExec",)

WORKFLOW_BINDINGS = {
    "inspect_rasters": ("Statistics", "studio_inspect_rasters"),
    "get_filelist": ("Statistics", "studio_inspect_rasters"),
    "validate_inputs": ("Statistics", "studio_validate_inputs"),
    "calculate_ndwi": ("Index", "studio_calculate_ndwi"),
    "calculate_ndvi": ("Index", "studio_calculate_ndvi"),
    "calculate_nbr": ("Index", "studio_calculate_nbr"),
    "compute_tvdi": ("Index", "studio_compute_tvdi"),
    "calculate_threshold_ratio": ("Statistics", "studio_calculate_threshold_ratio"),
    "create_threshold_mask": ("Index", "studio_create_threshold_mask"),
    "calculate_area": ("Statistics", "studio_calculate_area"),
    "calculate_mask_area": ("Statistics", "studio_calculate_area"),
    "calc_batch_image_mean": ("Statistics", "studio_calc_batch_image_mean"),
    "sens_slope": ("Analysis", "studio_sens_slope"),
    "difference": ("Statistics", "studio_difference"),
    "validate_result": ("Statistics", "studio_validate_result"),
    "lst_single_channel": ("Inversion", "studio_lst_single_channel"),
    "calculate_mean_lst_by_ndvi": ("Inversion", "studio_mean_lst_by_ndvi"),
    "ATI": ("Inversion", "studio_apparent_thermal_inertia"),
    "compute_linear_trend": ("Analysis", "studio_linear_trend"),
    "percentage_change": ("Analysis", "studio_consecutive_percentage_changes"),
    "max_value_and_index": ("Statistics", "studio_max_value_and_index"),
    "calculate_water_turbidity_ntu": ("Inversion", "studio_calculate_turbidity"),
    "max_three_day_window": ("Statistics", "studio_max_three_day_window"),
    "calc_batch_image_skewness": ("Statistics", "studio_raster_skewness"),
    "create_hotspot_masks": ("Index", "studio_create_below_threshold_mask"),
    "analyze_hotspot_direction": ("Analysis", "studio_hotspot_direction"),
    "maximum_composite": ("Statistics", "studio_maximum_composite"),
    "identify_fire_prone_areas": ("Statistics", "studio_fire_prone_mask"),
    "subtract": ("Statistics", "studio_subtract_rasters"),
    "remoteclip_classify": ("Perception", "RemoteCLIP"),
    "sam2_segment": ("Perception", "SAM2"),
    "earthbench_change_detection": ("Perception", "ChangeOS"),
    "earthbench_damage_assessment": ("Perception", "BRIGHT_DamageAssessment"),
    "text_to_bbox": ("OEA_Perception", "TextToBbox"),
    "object_detection": ("OEA_Perception", "ObjectDetection"),
    "image_description": ("OEA_Perception", "ImageDescription"),
    "region_attribute_description": ("OEA_Perception", "RegionAttributeDescription"),
    "count_given_object": ("OEA_Perception", "CountGivenObject"),
    "ocr_image": ("OEA_Perception", "OCR"),
}

# Tools that exist on a CPU MCP server but should not be offered to ReAct.
# get_filelist is a bare os.listdir; writing tools already return output_path,
# and the workflow alias of the same name is studio_inspect_rasters, not this.
WITHHELD_TOOLS = {("Statistics", "get_filelist")}

GPU_NATIVE_TOOLS = {
    "Perception": {
        "RemoteCLIP", "SAM2", "ChangeOS", "BRIGHT_DamageAssessment",
    },
    "OEA_Perception": {
        "TextToBbox", "ObjectDetection", "ImageDescription",
        "RegionAttributeDescription", "CountGivenObject", "OCR",
    },
}


def gpu_perception_enabled(environment: dict[str, str] | None = None) -> bool:
    source = os.environ if environment is None else environment
    return source.get("RSMEM_ENABLE_GPU_PERCEPTION", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def code_tools_enabled(environment: dict[str, str] | None = None) -> bool:
    """Whether the agent may write and run its own Python.

    Off by default. The Studio serves every visitor from one process, so
    turning arbitrary execution on is a deployment decision rather than a
    development one, and it should be made explicitly per environment.
    ``app.create_app`` is that decision for the hosted Studio and sets the
    flag; nothing else in the package does, so imports and tests stay off.
    """
    source = os.environ if environment is None else environment
    return source.get("RSMEM_ENABLE_CODE_TOOLS", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def configured_servers(environment: dict[str, str] | None = None) -> tuple[str, ...]:
    return (
        CPU_SERVERS
        + (GPU_SERVERS if gpu_perception_enabled(environment) else ())
        + (CODE_SERVERS if code_tools_enabled(environment) else ())
    )


class MCPRuntimeError(RuntimeError):
    pass


def sanitized_subprocess_environment(environment: dict[str, str] | None = None) -> dict[str, str]:
    """Inherit runtime settings without forwarding credentials to tool servers."""
    source = os.environ if environment is None else environment
    return {
        key: value
        for key, value in source.items()
        if not any(secret in key.upper() for secret in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))
    }


def _decode_result(result: Any) -> Any:
    if getattr(result, "isError", False):
        messages = [getattr(item, "text", str(item)) for item in result.content]
        raise MCPRuntimeError("\n".join(messages))
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        return structured
    texts = [getattr(item, "text", str(item)) for item in result.content]
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except (json.JSONDecodeError, TypeError):
            return texts[0]
    return texts


def discover_workflow_tools(timeout: float = 90.0) -> frozenset[str]:
    """Discover the planner-visible aliases from live MCP list_tools responses."""
    active = _ACTIVE_RUNTIME.get()
    if active is not None:
        return active.workflow_aliases()
    with tempfile.TemporaryDirectory(prefix="rsmem_mcp_discovery_") as directory:
        runtime = MCPRuntime(Path(directory), timeout=timeout).start()
        try:
            return runtime.workflow_aliases()
        finally:
            runtime.close()


class MCPRuntime:
    """Own one background event loop and four persistent stdio MCP processes."""

    def __init__(
        self,
        output_dir: Path,
        timeout: float = 90.0,
        servers: tuple[str, ...] | None = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.timeout = timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._startup_error: BaseException | None = None
        self._sessions: dict[str, ClientSession] = {}
        self.catalog: dict[str, dict] = {}
        self.servers = configured_servers() if servers is None else tuple(servers)
        self.live_gpu_tools: dict[str, set[str]] = {}

    def start(self) -> "MCPRuntime":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        if not self._ready.wait(self.timeout):
            raise MCPRuntimeError("Timed out while starting the MCP servers.")
        if self._startup_error:
            raise MCPRuntimeError(f"MCP startup failed: {self._startup_error}")
        return self

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:  # make startup failures visible to Gradio
            self._startup_error = exc
            self._ready.set()

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        clean_env = sanitized_subprocess_environment()
        configuration = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["mcpServers"]
        async with AsyncExitStack() as stack:
            for service in self.servers:
                server = configuration[service]
                args = list(server["args"])
                if "--temp_dir" in args:
                    args[args.index("--temp_dir") + 1] = str(self.output_dir)
                args[0] = str((ROOT / args[0]).resolve())
                params = StdioServerParameters(
                    command=sys.executable if server["command"] in {"python", "python3"} else server["command"],
                    args=args,
                    cwd=str(ROOT),
                    env={**clean_env, **server.get("env", {})},
                )
                read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                await session.initialize()
                listed = await session.list_tools()
                self._sessions[service] = session
                for tool in listed.tools:
                    self.catalog[f"{service}.{tool.name}"] = {
                        "service": service,
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": tool.inputSchema,
                    }
                if service in GPU_SERVERS and f"{service}.available_tools" in self.catalog:
                    available = await session.call_tool("available_tools", {})
                    decoded = _decode_result(available)
                    self.live_gpu_tools[service] = set(decoded if isinstance(decoded, list) else [])
            self._ready.set()
            while not self._stop.is_set():
                await asyncio.sleep(0.05)

    def call(self, service: str, tool: str, arguments: dict) -> Any:
        if service not in self._sessions or not self._loop:
            raise MCPRuntimeError(f"MCP service is not available: {service}")
        key = f"{service}.{tool}"
        if key not in self.catalog:
            raise MCPRuntimeError(f"MCP tool was not discovered: {key}")
        future = asyncio.run_coroutine_threadsafe(
            self._sessions[service].call_tool(tool, arguments), self._loop
        )
        return _decode_result(future.result(timeout=self.timeout))

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def health(self) -> dict:
        return {
            "transport": "stdio",
            "enabled_servers": list(self.servers),
            "disabled_servers": [
                name
                for name in GPU_SERVERS + CODE_SERVERS
                if name not in self.servers
            ],
            "live_gpu_tools": {
                service: sorted(tools) for service, tools in self.live_gpu_tools.items()
            },
            "discovered_tools": len(self.catalog),
            "config": str(CONFIG_PATH),
        }

    def workflow_aliases(self) -> frozenset[str]:
        aliases = []
        for alias, (service, tool) in WORKFLOW_BINDINGS.items():
            if f"{service}.{tool}" not in self.catalog:
                continue
            if service in GPU_SERVERS and tool not in self.live_gpu_tools.get(service, set()):
                continue
            aliases.append(alias)
        return frozenset(aliases)

    def is_tool_available(self, service: str, tool: str) -> bool:
        """Whether a discovered tool may be exposed to the planning agent."""
        if (service, tool) in WITHHELD_TOOLS:
            return False
        if f"{service}.{tool}" not in self.catalog:
            return False
        if service not in GPU_SERVERS:
            return True
        return tool in self.live_gpu_tools.get(service, set())


_ACTIVE_RUNTIME: contextvars.ContextVar[MCPRuntime | None] = contextvars.ContextVar(
    "active_mcp_runtime", default=None
)


@contextmanager
def workflow_mcp(output_dir: Path):
    runtime = MCPRuntime(output_dir).start()
    token = _ACTIVE_RUNTIME.set(runtime)
    try:
        yield runtime
    finally:
        _ACTIVE_RUNTIME.reset(token)
        runtime.close()


def _call(service: str, tool: str, **arguments):
    runtime = _ACTIVE_RUNTIME.get()
    if runtime is None:
        raise MCPRuntimeError("Tool call attempted outside an MCP workflow session.")
    return runtime.call(service, tool, arguments)


def call_workflow_tool(alias: str, arguments: dict):
    if alias not in WORKFLOW_BINDINGS:
        raise MCPRuntimeError(f"Unknown workflow tool alias: {alias}")
    service, tool = WORKFLOW_BINDINGS[alias]
    return _call(service, tool, **arguments)


def inspect_rasters(paths):
    return _call("Statistics", "studio_inspect_rasters", paths=list(paths))


def calculate_ndvi(nir_path, red_path, output_path):
    return _call("Index", "studio_calculate_ndvi", nir_path=nir_path, red_path=red_path, output_path=output_path)


def calculate_ndwi(nir_path, swir_path, output_path):
    return _call("Index", "studio_calculate_ndwi", nir_path=nir_path, swir_path=swir_path, output_path=output_path)


def calculate_nbr(nir_path, swir2_path, output_path):
    return _call("Index", "studio_calculate_nbr", nir_path=nir_path, swir2_path=swir2_path, output_path=output_path)


def compute_tvdi(ndvi_path, lst_path, output_path):
    return _call("Index", "studio_compute_tvdi", ndvi_path=ndvi_path, lst_path=lst_path, output_path=output_path)


def create_threshold_mask(image_path, threshold, output_path):
    return _call("Index", "studio_create_threshold_mask", image_path=image_path, threshold=threshold, output_path=output_path)


def create_below_threshold_mask(image_path, threshold, output_path):
    return _call("Index", "studio_create_below_threshold_mask", image_path=image_path, threshold=threshold, output_path=output_path)


def combine_binary_masks(mask_paths, output_path):
    return _call("Index", "studio_combine_binary_masks", mask_paths=list(mask_paths), output_path=output_path)


def lst_single_channel(bt_path, red_path, nir_path, output_path):
    return _call("Inversion", "studio_lst_single_channel", bt_path=bt_path, red_path=red_path, nir_path=nir_path, output_path=output_path)


def mean_lst_by_ndvi(red_path, nir_path, lst_path, ndvi_output_path):
    return _call("Inversion", "studio_mean_lst_by_ndvi", red_path=red_path, nir_path=nir_path, lst_path=lst_path, ndvi_output_path=ndvi_output_path)


def apparent_thermal_inertia(day_temp_path, night_temp_path, albedo_path, output_path):
    return _call("Inversion", "studio_apparent_thermal_inertia", day_temp_path=day_temp_path, night_temp_path=night_temp_path, albedo_path=albedo_path, output_path=output_path)


def calculate_turbidity(input_red_path, output_path):
    return _call("Inversion", "studio_calculate_turbidity", input_red_path=input_red_path, output_path=output_path)


def calc_batch_image_mean(paths):
    return _call("Statistics", "studio_calc_batch_image_mean", paths=list(paths))


def raster_skewness(path):
    return _call("Statistics", "studio_raster_skewness", path=path)


def subtract_rasters(first_path, second_path, output_path):
    return _call("Statistics", "studio_subtract_rasters", first_path=first_path, second_path=second_path, output_path=output_path)


def maximum_composite(paths, output_path):
    return _call("Statistics", "studio_maximum_composite", paths=list(paths), output_path=output_path)


def fire_prone_mask(image_path, output_path, percentile=75):
    result = _call("Statistics", "studio_fire_prone_mask", image_path=image_path, output_path=output_path, percentile=percentile)
    return result[0], float(result[1])


def calculate_threshold_ratio(image_path, threshold):
    return float(_call("Statistics", "studio_calculate_threshold_ratio", image_path=image_path, threshold=threshold))


def calculate_area(mask_path):
    return _call("Statistics", "studio_calculate_area", mask_path=mask_path)


def validate_inputs(paths):
    return _call("Statistics", "studio_validate_inputs", paths=list(paths))


def difference(a, b):
    return _call("Statistics", "studio_difference", a=float(a), b=float(b))


def max_value_and_index(values):
    return _call("Statistics", "studio_max_value_and_index", values=list(values))


def max_three_day_window(values):
    return _call("Statistics", "studio_max_three_day_window", values=list(values))


def validate_result(input_paths, artifact_paths):
    return _call(
        "Statistics",
        "studio_validate_result",
        input_paths=list(input_paths),
        artifact_paths=list(artifact_paths),
    )


def linear_trend(values, x=None):
    return _call("Analysis", "studio_linear_trend", values=list(values), x=None if x is None else list(x))


def sens_slope(values):
    return _call("Analysis", "studio_sens_slope", values=list(values))


def hotspot_direction(mask_path):
    return _call("Analysis", "studio_hotspot_direction", mask_path=mask_path)


def consecutive_percentage_changes(values):
    return _call("Analysis", "studio_consecutive_percentage_changes", values=list(values))
