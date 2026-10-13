"""Public RSMeM agent with model-authored planning and MCP geospatial tools."""

from __future__ import annotations

import ast
import copy
import functools
import html as html_lib
import ipaddress
import json
import logging
import os
import re
import secrets
import socket
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import gradio as gr
import numpy as np
import rasterio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from openai import OpenAI
from PIL import Image
from rasterio.enums import ColorInterp, Resampling

import modelscope_oauth as msoauth
from cpu_tools import infer_band_roles
from deterministic_evaluator import evaluate_case_deterministically, step_succeeded
from gpu_service_manager import (
    get_gpu_service_manager,
    read_worker_log,
    start_configured_gpu_services,
)
from mcp_runtime import (
    apparent_thermal_inertia,
    calc_batch_image_mean,
    calculate_area,
    calculate_nbr,
    calculate_ndvi,
    calculate_ndwi,
    calculate_threshold_ratio,
    calculate_turbidity,
    call_workflow_tool,
    combine_binary_masks,
    compute_tvdi,
    consecutive_percentage_changes,
    create_below_threshold_mask,
    create_threshold_mask,
    discover_workflow_tools,
    fire_prone_mask,
    hotspot_direction,
    inspect_rasters,
    linear_trend,
    lst_single_channel,
    maximum_composite,
    mean_lst_by_ndvi,
    raster_skewness,
    sens_slope,
    subtract_rasters,
    workflow_mcp,
)
from mcp_runtime import (
    difference as mcp_difference,
)
from mcp_runtime import (
    max_three_day_window as mcp_max_three_day_window,
)
from mcp_runtime import (
    max_value_and_index as mcp_max_value_and_index,
)
from mcp_runtime import (
    validate_inputs as mcp_validate_inputs,
)
from mcp_runtime import (
    validate_result as mcp_validate_result,
)
import run_storage
from react_harness import MAX_MODEL_CALLS_PER_ATTEMPT, run_react_attempt
from studio_ui import (
    CSS,
    ENGLISH_CONTROLS,
    FOOTER,
    NAV,
    RUN_TIMER_JS,
    WALKTHROUGH_JS,
    WORKSPACE_HEADING,
    hero_html,
    language_script,
    walkthrough_html,
)
from workflow_logic import (
    generate_reflection_candidates,
    judge_builtin_case,
    synthesize_answer,
    teacher_failure_gate,
)

LOGGER = logging.getLogger("rsmem.oauth")

ROOT = Path(__file__).resolve().parent
GPU_SERVICE_MANAGER = start_configured_gpu_services()
KB_PATH = ROOT / "knowledge_base.json"
CASE_REFERENCE_PATH = ROOT / "case_references.json"

PROJECT_TITLE = "RSMeM: Knowledge-Enhanced Memory Evolution for Remote Sensing Agents"
PROJECT_ABSTRACT = (
    "RSMeM is a knowledge-enhanced memory evolution mechanism that bootstraps remote-sensing "
    "agents with pre-distilled domain knowledge and iteratively integrates online experience "
    "for robust multi-step tool execution. Hierarchical Knowledge Grounding performs "
    "taxonomy-aware retrieval over a hierarchical domain corpus to guide planning and tool "
    "selection; Failure-Aware Experience Refinement distills failure-annotated tool-use traces "
    "into reusable constraints for the next round of execution. On EarthBench, RSMeM "
    "consistently improves tool use and end-to-end answer accuracy across diverse LLM backbones."
)
PROJECT_ABSTRACT_WORDS = len(PROJECT_ABSTRACT.split())
GITHUB_URL = "https://github.com/AI9Stars/RSMeM"
LEARN_URL = "https://modelscope.cn/learn/435887"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-flash"
MODE_MODELSCOPE = "ModelScope quota"
MODE_OPENAI = "API"
MODELSCOPE_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro-0813"
MODELSCOPE_MODEL_CHOICES = [
    "deepseek-ai/DeepSeek-V4-Pro-0813",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "moonshotai/Kimi-K3",
    "ZhipuAI/GLM-5.2",
    "Qwen/Qwen3.8-27B",
]
STUDIO_DIRECT_URL = "https://wbx929-rsmem.ms.show"

EXAMPLES = {
    "🌾 Sichuan drought severity — derive TVDI and quantify severe-dryness pixels": {
        "directory": "tvdi_sichuan",
        "query": "Calculate TVDI for the Sichuan Plain agricultural area and determine the percentage of valid pixels where TVDI exceeds 0.75.",
        "description": "2 GeoTIFFs · NDVI + LST → TVDI → threshold ratio → affected area",
    },
    "🔥 California burn-severity evolution — build an NBR series and estimate its trend": {
        "directory": "nbr_california",
        "query": "Using California B02 and B07 surface-reflectance images from January to March 2025, calculate NBR for every date, summarize the daily mean NBR series, and estimate its trend with Sen's slope.",
        "description": "22 GeoTIFFs · 11 paired dates → NBR series → regional means → Sen's slope",
    },
    "🌃 Bhutan night-light change — compare June and August regional intensity": {
        "directory": "nightlights_bhutan",
        "query": "Compare Bhutan nighttime-light intensity in June and August 2024 by computing the spatial mean of each GeoTIFF, identify the lower month, and report the absolute difference.",
        "description": "2 GeoTIFFs · regional means → temporal comparison → absolute difference",
    },
    "🌲 Black Forest thermal contrast — compare forest and non-vegetated land temperatures": {
        "directory": "q10_black_forest",
        "query": "Estimate land surface temperature over Germany's Black Forest from Landsat 8 Band 10, Red, and NIR; compare forest pixels (NDVI > 0.7) with non-vegetated pixels (NDVI < 0.2), quantify the temperature contrast, and determine whether a cooling effect is present.",
        "description": "3 GeoTIFFs · NDVI emissivity → single-channel LST → forest/non-vegetated temperature contrast",
    },
    "🌡️ Cyprus surface thermal inertia — compare two July ATI observations": {
        "directory": "q41_cyprus_ati",
        "query": "Calculate Apparent Thermal Inertia (ATI) for Cyprus on 1 and 15 July 2020 from paired daytime/nighttime brightness temperature and albedo, then report the change.",
        "description": "6 GeoTIFFs · day/night BT + albedo → ATI maps → temporal difference",
    },
    "🌃 Mumbai urban-light growth — estimate the 2013–2024 nighttime-light trend": {
        "directory": "q102_mumbai_lights",
        "query": "Using Mumbai January nighttime-light rasters from 2013 to 2024, compute the annual spatial means and estimate the long-term linear trend.",
        "description": "12 GeoTIFFs · annual regional means → linear trend",
    },
    "🌱 Rosario vegetation dynamics — find the largest consecutive NDVI increase": {
        "directory": "q108_rosario_ndvi",
        "query": "Using Rosario vegetation-coverage rasters from January to June 2021, compute consecutive percentage changes and report the maximum increase and its date interval.",
        "description": "17 GeoTIFFs · NDVI means → consecutive percentage changes → maximum growth interval",
    },
    "🌊 Dongting Lake water quality — locate the most turbid three-day period": {
        "directory": "q124_dongting_turbidity",
        "query": "Calculate daily Dongting Lake turbidity from red-band surface reflectance in September 2022 and identify the most polluted consecutive three-day period.",
        "description": "29 GeoTIFFs · red reflectance → turbidity → daily means → rolling three-day maximum",
    },
    "🌧️ Congo Rainforest precipitation — estimate May rainfall level and trend": {
        "directory": "q145_congo_rainfall",
        "query": "Calculate daily mean precipitation over the Congo Rainforest during May 2025 and estimate whether rainfall increased or decreased over the month.",
        "description": "30 GeoTIFFs · daily precipitation means → monthly average → linear trend",
    },
    "🧪 Lake Urmia turbidity anomalies — diagnose right-skewed pollution events": {
        "directory": "q156_urmia_turbidity",
        "query": "Calculate Lake Urmia turbidity throughout 2022 and use distribution skewness to assess whether the time series contains asymmetric high-turbidity anomalies.",
        "description": "24 GeoTIFFs · red reflectance → turbidity maps → means and pixel-distribution skewness",
    },
    "🔥 California fire-hotspot geography — determine the dominant hotspot direction": {
        "directory": "q161_california_hotspots",
        "query": "Calculate California NBR from January to March 2025, identify fire hotspots where NBR is below -0.3, and determine their dominant general direction.",
        "description": "22 GeoTIFFs · paired NIR/SWIR → NBR → low-NBR masks → hotspot direction",
    },
    "🏙️ Shanghai residential development — derive and trend building volume": {
        "directory": "q187_shanghai_volume",
        "query": "Derive Shanghai residential building volume as total minus non-residential volume for 1985–2020, then estimate its long-term linear trend.",
        "description": "16 GeoTIFFs · year-paired subtraction → residential-volume maps → regional means → trend",
    },
    # Open-ended perception cases: no reference answer, so they run like an upload.
    "🏚️ Wildfire building loss — segment buildings before and after the fire and total the lost area": {
        "directory": "oea_change_building_loss",
        "freeform": True,
        "query": "These satellite images show the same area before (01_pre_disaster.png) and after (02_post_disaster.png) a wildfire. Detect and segment the buildings in both images, identify where buildings disappeared or shrank, and report the total reduction in building area in pixels.",
        "description": "2 PNGs · pre/post building segmentation → lost building area",
    },
    "🏚️ Two-region wildfire triage — decide which area suffered the heavier building damage": {
        "directory": "oea_change_region_triage",
        "freeform": True,
        "query": "Region A and region B were both hit by a wildfire, and each has a pre-event and a post-event image. Compare the building damage in each region and determine which area suffered the more severe impact.",
        "description": "4 PNGs · two pre/post pairs → per-region building damage → comparison",
    },
    "🏘️ District damage comparison — quantify building-area loss in two burned districts": {
        "directory": "oea_change_urban_loss",
        "freeform": True,
        "query": "A wildfire struck two urban areas, region A and region B. Using the pre- and post-event images of each region, detect and segment the buildings, calculate the reduction in total building area for each region, and identify which region experienced the greater loss.",
        "description": "4 PNGs · building segmentation per date → area reduction per region → larger loss",
    },
    "🚑 Relief prioritization — rank two fire-hit neighbourhoods by destroyed building area": {
        "directory": "oea_change_relief_rank",
        "freeform": True,
        "query": "A wildfire struck two neighbouring areas, region A and region B, and responders must prioritise relief. Using the pre- and post-event images of both regions, detect the buildings at each time point, identify buildings that were destroyed or significantly damaged, calculate the reduction in building area for each region, and determine which region was more severely affected.",
        "description": "4 PNGs · destroyed-building detection → area reduction → relief priority",
    },
    "🔥 Santa Rosa wildfire damage — count buildings destroyed between pre- and post-fire images": {
        "directory": "oea_p_santarosa_change",
        "freeform": True,
        "query": "How many buildings destroyed in the post image compared to pre image?",
        "description": "2 PNGs · pre/post change detection → destroyed-building count",
    },
    "🧱 Collapsed-building survey — separate fully and partially destroyed buildings and give the ratio": {
        "directory": "oea_p_collapse_ratio",
        "freeform": True,
        "query": "Identify the destroyed building and estimate the ratio of partially destroyed and fully destroyed",
        "description": "1 JPEG · damaged-building grounding → collapse attribute → ratio",
    },
    "🌊 Flood impact summary — describe the flooded scene and count the affected houses and cars": {
        "directory": "oea_p_flood_summary",
        "freeform": True,
        "query": "Describe the situation by sharing number of houses and cars in the flood.",
        "description": "1 JPEG · scene captioning → house and vehicle counts",
    },
    "🌉 Highway corridor inventory — count the human-made structures besides roads and vehicles": {
        "directory": "oea_p_structures_highway",
        "freeform": True,
        "query": "Count and list the human-made structures other than the roads and the vehicle?",
        "description": "1 JPEG · open-vocabulary detection → structure inventory",
    },
    "🅿️ Parking-lot occupancy — count the cars parked in the top-left lot of the scene": {
        "directory": "oea_p_parking_count",
        "freeform": True,
        "query": "how many cars are parked in parking lot at top left",
        "description": "1 PNG · region grounding → vehicle count",
    },
    "🛬 Runway identification — read the runway number and check whether any aircraft sit on it": {
        "directory": "oea_p_runway_ocr",
        "freeform": True,
        "query": "Extract the visible runway number, and determine planes present on that runway.",
        "description": "1 PNG · OCR → runway localization → aircraft count",
    },
    "🎾 Occluded parking survey — judge visibility south of the tennis courts and count vehicles": {
        "directory": "oea_p_tennis_parking",
        "freeform": True,
        "query": "Is the parking area south of tennis courts fully visible or partially occluded in the given imagery, and can you provide a count of the parked vehicles present within the visible area?",
        "description": "1 PNG · grounding → occlusion attribute → vehicle count",
    },
    "🚢 Harbour vessel alignment — decide whether every ship in the scene faces the same way": {
        "directory": "oea_p_ship_heading",
        "freeform": True,
        "query": "Are all ships facing the same direction?",
        "description": "1 JPEG · ship grounding → per-ship heading attribute",
    },
    "✈️ Terminal apron activity — count aircraft at the terminal and those docked at jet bridges": {
        "directory": "oea_p_terminal_aircraft",
        "freeform": True,
        "query": "How many aircraft are currently positioned near the main terminal building, and of those, how many are connected to a jet bridge?",
        "description": "1 PNG · terminal grounding → aircraft count → jet-bridge attribute",
    },
    "🏟️ Turf lettering survey — read the field markings and estimate their bearing from north": {
        "directory": "oea_p_turf_ocr_bearing",
        "freeform": True,
        "query": "What is the exact wording, and orientation (degree from north) of the large field markings painted on the turf inside the central track?",
        "description": "1 JPEG · OCR → marking orientation",
    },
    "🏊 Residential pool condition — locate each swimming pool and rate its maintenance state": {
        "directory": "oea_p_pool_condition",
        "freeform": True,
        "query": "Can you rate the condition of the swimming pools in this residential area?",
        "description": "1 JPEG · pool grounding → per-pool condition attribute",
    },
    "📤 Custom GeoTIFF upload": {
        "directory": None,
        "query": "Analyze the uploaded remote-sensing GeoTIFFs with an appropriate CPU tool chain.",
        "description": "Upload your own GeoTIFF bands; recognizable band names improve automatic pairing.",
    },
}
DEFAULT_EXAMPLE = next(iter(EXAMPLES))

PAPER_BIBTEX = r"""@article{wu2026rsmem,
  title   = {RSMeM: Knowledge-Enhanced Memory Evolution for Remote Sensing Agents with Systematic Evaluation},
  author  = {Wu, Bingxian and Zhang, Yu and Guo, Zonghao and Liu, Tang and Qian, Chen and Lu, Yuxiang and Du, Xingbo and Li, Yanghao and Zhang, Yidan and Chen, Chi and Yao, Ling and Sun, Maosong},
  journal = {arXiv preprint arXiv:2607.24772},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.24772},
  url     = {https://arxiv.org/abs/2607.24772}
}"""

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "of", "on", "or", "the", "to", "use", "using", "with",
}

APPLICATION_ALIASES = {
    "Land Cover Mapping": "classification segmentation scene mapping land use landcover",
    "Land cover change detection": "change changed temporal difference bi-temporal",
    "Water resources management": "flood flooded flooding inundation water river lake reservoir watershed",
    "Land surface and air temperature": "temperature thermal heat urban heat island lst",
    "Atmospheric Parameter Estimation": "atmosphere atmospheric aerosol cloud humidity",
    "Vegetation parameters": "vegetation crop forest biomass ndvi lai chlorophyll",
    "Soil property estimation": "soil moisture salinity organic carbon",
    "Building Extraction": "building footprint roof extraction urban structure",
    "Nighttime Light Dynamics Monitoring": "night nighttime light illumination economy",
    "Infrastructure Monitoring": "road bridge railway airport infrastructure",
    "Building Damage Assesement": "damage damaged earthquake collapse disaster building",
    "Wildfire": "wildfire fire burned burning smoke burn scar",
    "Drought Monitoring": "drought dry arid precipitation water stress",
}


def load_entries() -> list[dict]:
    with KB_PATH.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    entries = []
    for domain, items in raw.items():
        for item in items:
            entries.append(
                {
                    "domain": domain,
                    "application": item.get("application_domain", "Remote sensing task"),
                    "keywords": item.get("key_words", []),
                    "summary": item.get("summary", ""),
                }
            )
    return entries


KB_ENTRIES = load_entries()
with CASE_REFERENCE_PATH.open("r", encoding="utf-8") as handle:
    CASE_REFERENCES = json.load(handle)


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def retrieve_knowledge(query: str, top_k: int = 3) -> list[tuple[int, dict]]:
    query_tokens = tokens(query)
    ranked = []
    for entry in KB_ENTRIES:
        title_tokens = tokens(f"{entry['domain']} {entry['application']}")
        keyword_tokens = tokens(" ".join(entry["keywords"]))
        alias_tokens = tokens(APPLICATION_ALIASES.get(entry["application"], ""))
        score = (
            4 * len(query_tokens & title_tokens)
            + 3 * len(query_tokens & alias_tokens)
            + len(query_tokens & keyword_tokens)
        )
        if score:
            ranked.append((score, entry))

    if not ranked:
        ranked = [(0, entry) for entry in KB_ENTRIES[:top_k]]
    return sorted(ranked, key=lambda row: row[0], reverse=True)[:top_k]


def summarize_failure(trajectory: str) -> tuple[str, str]:
    text = trajectory.strip().lower()
    if not text:
        return (
            "No failure trajectory was provided.",
            "Record the tool name, inputs, observation, exception, and expected output before retrying.",
        )

    patterns = [
        (("file not found", "missing file", "path"), "Input-path failure",
         "Validate paths and file existence before invoking the remote-sensing tool."),
        (("invalid", "parameter", "argument"), "Parameter-contract failure",
         "Check the tool schema, units, value ranges, CRS, and required parameters before retrying."),
        (("timeout", "unavailable", "connection"), "Service-availability failure",
         "Verify service health, use bounded retries, and preserve intermediate outputs."),
        (("empty", "none", "no result"), "Empty-output failure",
         "Validate upstream data and add an explicit non-empty output check before downstream steps."),
        (("tool not found", "unknown tool"), "Tool-selection failure",
         "Retrieve the tool catalogue and select a supported operation with a matching input contract."),
    ]
    for cues, diagnosis, correction in patterns:
        if any(cue in text for cue in cues):
            return diagnosis, correction
    return (
        "Execution-contract failure",
        "Inspect the last successful observation, isolate the failing step, verify its input contract, and rerun only that step.",
    )


def validate_base_url(value: str) -> str:
    """Allow public HTTPS OpenAI-compatible endpoints while blocking SSRF targets."""
    base_url = value.strip().rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("Base URL must be a valid public HTTPS URL.")
    if parsed.username or parsed.password:
        raise ValueError("Credentials must not be embedded in the Base URL.")

    if not parsed.path:
        raise ValueError("The Base URL must include the API path, not only the host.")

    hostname = parsed.hostname.lower().rstrip(".")
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Local and internal endpoints are not allowed in the public Studio.")

    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        try:
            records = socket.getaddrinfo(
                hostname,
                parsed.port or 443,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise ValueError("The Base URL hostname could not be resolved.") from exc
        addresses = {ipaddress.ip_address(record[4][0]) for record in records}

    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Private, loopback, link-local, and reserved endpoints are blocked.")
    return base_url


TOOL_REFERENCE = """
- inspect_rasters: inspect uploaded GeoTIFF dimensions, CRS, pixel size, and band count.
- calculate_ndwi: calculate the released RSMeM NIR/SWIR water-moisture index.
- calculate_ndvi: estimate vegetation greenness from NIR and Red bands.
- calculate_nbr: assess burned areas from NIR and SWIR2 bands.
- compute_tvdi: derive agricultural dryness from aligned NDVI and LST rasters.
- calculate_threshold_ratio: summarize the fraction of raster pixels satisfying a threshold.
- calc_batch_image_mean: produce one scalar mean per raster/date.
- sens_slope: estimate a robust monotonic trend from a raster-derived time series.
- difference: compute an absolute difference between two regional statistics.
- calculate_area: convert a validated binary mask into physical area.
- lst_single_channel: correct Landsat brightness temperature with NDVI-derived emissivity.
- ATI: derive apparent thermal inertia from paired daytime/nighttime temperature and albedo.
- calculate_water_turbidity_ntu: convert red-band reflectance into an auditable turbidity raster.
- compute_linear_trend: fit a least-squares trend to date-ordered raster statistics.
- percentage_change: calculate consecutive relative changes in a vegetation series.
- analyze_hotspot_direction: summarize the spatial concentration of low-NBR fire masks.
- identify_fire_prone_areas: map the upper quartile of positive MaxFRP observations.
- subtract: derive residential building volume from total and non-residential rasters.
- remoteclip_classify: classify land-use or scene type from a remote-sensing image with RemoteCLIP.
- sam2_segment: segment an object with EarthBench SAM2 from a pixel bbox=[x1, y1, x2, y2] (top-left then bottom-right corner; reuse a detector's box unmodified).
- earthbench_change_detection: map semantic change between aligned pre-event and post-event images with ChangeOS.
- earthbench_damage_assessment: classify building damage from aligned pre/post disaster images with BRIGHT.
- text_to_bbox: ground an English object phrase to pixel bounding boxes with OEA Grounding DINO.
- object_detection: detect remote-sensing objects with OEA LAE-DINO.
- image_description: describe visible content with the OEA vision-language model.
- region_attribute_description: describe a requested attribute in an optional image region.
- count_given_object: count visible instances of a requested object.
- ocr_image: recognize text visible in a remote-sensing image.
""".strip()

CPU_SUPPORTED_TOOLS = {
    "inspect_rasters", "get_filelist", "validate_inputs", "calculate_ndwi",
    "calculate_ndvi", "calculate_nbr", "compute_tvdi", "calculate_threshold_ratio",
    "create_threshold_mask", "calculate_area", "calculate_mask_area",
    "calc_batch_image_mean", "sens_slope", "difference", "validate_result",
    "lst_single_channel", "calculate_mean_lst_by_ndvi", "ATI",
    "compute_linear_trend", "percentage_change", "max_value_and_index",
    "calculate_water_turbidity_ntu", "max_three_day_window",
    "calc_batch_image_skewness", "create_hotspot_masks",
    "analyze_hotspot_direction", "maximum_composite",
    "identify_fire_prone_areas", "subtract",
}
GPU_SUPPORTED_TOOLS = {
    "remoteclip_classify", "sam2_segment", "earthbench_change_detection",
    "earthbench_damage_assessment", "text_to_bbox", "object_detection",
    "image_description", "region_attribute_description", "count_given_object",
    "ocr_image",
}
SUPPORTED_TOOLS = CPU_SUPPORTED_TOOLS | GPU_SUPPORTED_TOOLS


def _strip_reasoning(text: str) -> str:
    """Drop <think> blocks emitted by reasoning models; an unclosed block
    means the reply was truncated mid-reasoning, so nothing usable follows."""
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text)
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


def api_completion(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 16384,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=180.0)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return _strip_reasoning(response.choices[0].message.content or "")


def extract_json_object(text: str) -> dict | list | None:
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    start_obj, end_obj = text.find("{"), text.rfind("}")
    start_arr, end_arr = text.find("["), text.rfind("]")
    if start_obj >= 0 and end_obj > start_obj:
        candidates.append(text[start_obj : end_obj + 1])
    if start_arr >= 0 and end_arr > start_arr:
        candidates.append(text[start_arr : end_arr + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def normalize_plan(
    data: dict | list | None,
    supported_tools: set[str] | frozenset[str] | None = None,
) -> list[dict]:
    if isinstance(data, dict):
        data = data.get("tool_chain") or data.get("tools") or data.get("plan")
    if not isinstance(data, list):
        raise ValueError("The planner did not return a JSON tool_chain array.")

    supported_tools = SUPPORTED_TOOLS if supported_tools is None else supported_tools
    normalized = []
    for item in data[:5]:
        if isinstance(item, str):
            tool = item
            purpose = "Planned by the agent."
            parameters = {}
        elif isinstance(item, dict) and item.get("tool"):
            tool = str(item["tool"])
            purpose = str(item.get("purpose", "Planned by the agent."))
            parameters = item.get("parameters", {})
            if not isinstance(parameters, dict):
                raise ValueError(f"Invalid parameters for tool: {tool}")
        else:
            raise ValueError("Every tool-chain item must name a supported tool.")

        if tool not in supported_tools:
            raise ValueError(f"The planner selected an unsupported tool: {tool}")
        normalized.append({"tool": tool, "purpose": purpose, "parameters": parameters})

    if not normalized:
        raise ValueError("The planner returned an empty executable tool chain.")
    return normalized


def plan_toolchain(
    query: str,
    knowledge_context: str,
    reflection: str,
    api_key: str,
    base_url: str,
    model: str,
) -> list[dict]:
    if not api_key:
        raise ValueError("An API key is required for RSMeM planning.")
    live_tools = discover_workflow_tools()
    missing = CPU_SUPPORTED_TOOLS - set(live_tools)
    if missing:
        raise ValueError(f"Required MCP tools were not discovered: {', '.join(sorted(missing))}")
    live_reference = "\n".join(
        line for line in TOOL_REFERENCE.splitlines()
        if line.removeprefix("- ").split(":", 1)[0] in live_tools
    )
    system = """You are the planner inside RSMeM. Select a short remote-sensing tool chain.
Return ONLY JSON with a tool_chain array. Each item must contain tool, purpose, and parameters.
Use only tools in the supplied reference. Always inspect inputs before band-dependent tools.
Use raster tools for GeoTIFF values and GPU perception tools for visual interpretation of TIFF/JPEG/PNG inputs.
Do not claim execution has happened. If prior failure-aware memory is present, revise the plan."""
    user = f"""Query: {query[:3000]}

Retrieved HKG knowledge:
{knowledge_context[:4000]}

Failure-aware memory from the previous attempt:
{reflection[:2000] or "(none; this is the first attempt)"}

Available executable MCP tool reference (GPU tools appear only when their worker is healthy):
{live_reference}
"""
    raw = api_completion(api_key, base_url, model, system, user, max_tokens=16384)
    for attempt in range(2):
        if not raw.strip():
            raise ValueError(
                "The planner model returned an empty reply — reasoning models can "
                "spend the whole token budget on hidden thinking. Try again or "
                "select a different model."
            )
        try:
            return normalize_plan(extract_json_object(raw), live_tools)
        except ValueError as exc:
            if attempt == 0:
                repair = (
                    f"\n\nYour previous reply was invalid: {exc} "
                    'Reply again with ONLY one JSON object shaped like '
                    '{"tool_chain": [{"tool": "...", "purpose": "...", "parameters": {}}]}.'
                )
                raw = api_completion(
                    api_key, base_url, model, system, user + repair, max_tokens=16384
                )
            else:
                snippet = raw.strip()[:400]
                raise ValueError(
                    f"{exc} Raw model reply (first 400 chars): {snippet!r}"
                ) from exc


def _normalize_preview_band(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.nanpercentile(values[finite], [2, 98])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((values - low) / (high - low), 0, 1)
    scaled[~finite] = 0
    return np.round(scaled * 255).astype(np.uint8)


def raster_preview(paths) -> np.ndarray | None:
    """Fit the first image to a display thumbnail without mixing separate rasters."""
    return raster_preview_at(paths, 0)


def _preview_index(index, paths: list[str]) -> int:
    if isinstance(index, (list, tuple)) and index:
        index = index[0]
    if isinstance(index, str):
        if index.isdigit():
            index = int(index)
        else:
            names = [Path(path).name for path in paths]
            index = names.index(index) if index in names else 0
    try:
        index = int(index)
    except (TypeError, ValueError):
        index = 0
    if not paths:
        return 0
    return max(0, min(index, len(paths) - 1))


def raster_preview_at(paths, index: int = 0) -> np.ndarray | None:
    """Fit one selected image to a display thumbnail."""
    raster_paths = [
        Path(item) if isinstance(item, (str, os.PathLike)) else Path(item.name)
        for item in (paths or [])
    ]
    raster_paths = [
        path for path in raster_paths
        if path.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"} and path.is_file()
    ]
    if not raster_paths:
        return None
    selected = raster_paths[_preview_index(index, [str(path) for path in raster_paths])]

    if selected.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        image = Image.open(selected).convert("RGB")
        image.thumbnail((640, 640))
        return np.asarray(image)

    with rasterio.open(selected) as src:
        scale = min(1.0, 640 / max(src.height, src.width))
        height = max(1, round(src.height * scale))
        width = max(1, round(src.width * scale))
        rgb_roles = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
        # Use true RGB metadata when available, otherwise show band 1 in gray.
        bands = (
            [src.colorinterp.index(role) + 1 for role in rgb_roles]
            if all(role in src.colorinterp for role in rgb_roles) else [1]
        )
        channels = []
        for band_index in bands:
            band = src.read(
                band_index,
                out_shape=(height, width),
                masked=True,
                resampling=Resampling.bilinear,
            ).astype(np.float32).filled(np.nan)
            channels.append(_normalize_preview_band(band))

    if len(channels) == 1:
        return np.repeat(channels[0][..., None], 3, axis=2)
    return np.stack(channels, axis=2)


def preview_uploaded_rasters(uploaded_files):
    return raster_preview(uploaded_files)


BUNDLED_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _bundled_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.suffix.lower() in BUNDLED_IMAGE_SUFFIXES and path.is_file()
    )


def load_example(example_name: str):
    if not example_name:
        return ("", None)
    example = EXAMPLES.get(example_name, EXAMPLES[DEFAULT_EXAMPLE])
    directory = ROOT / "examples" / example["directory"] if example["directory"] else None
    paths = _bundled_files(directory) if directory and directory.is_dir() else []
    return (example["query"], raster_preview(paths))


def _bundled_raster_paths(example_name: str) -> list[str]:
    if not example_name:
        return []
    example = EXAMPLES.get(example_name) or {}
    directory = example.get("directory")
    if not directory:
        return []
    folder = ROOT / "examples" / directory
    if not folder.is_dir():
        return []
    return [str(path.resolve()) for path in _bundled_files(folder)]


def _hidden_case_row():
    return gr.update(choices=[], value=None, visible=False), []


def _visible_case_row(example_name: str):
    paths = _bundled_raster_paths(example_name)
    choices = [(Path(path).name, index) for index, path in enumerate(paths)]
    return gr.update(
        choices=choices,
        value=0 if choices else None,
        visible=bool(choices),
    ), paths


def load_example_for_ui(example_name: str):
    """A deliberate case selection replaces any previously uploaded source."""
    query, preview = load_example(example_name)
    if not example_name:
        return query, preview, gr.update(value=None, visible=False), *_hidden_case_row()
    case_update, paths = _visible_case_row(example_name)
    return query, preview, gr.update(value=None, visible=False), case_update, paths


def select_uploaded_source(uploaded_files, example_name: str):
    """Keep the preview and execution source in sync on upload and clear."""
    if uploaded_files:
        return (
            preview_uploaded_rasters(uploaded_files),
            gr.update(value=None),
            gr.update(visible=True),
            *_hidden_case_row(),
        )
    if example_name:
        case_update, paths = _visible_case_row(example_name)
        return (
            raster_preview_at(paths, 0),
            gr.skip(),
            gr.update(value=None, visible=False),
            case_update,
            paths,
        )
    return None, gr.skip(), gr.update(visible=False), *_hidden_case_row()


def _file_paths(items) -> list[str]:
    if not items:
        return []
    if not isinstance(items, (list, tuple)):
        items = [items]
    paths = []
    for item in items:
        path = getattr(item, "name", item)
        if path:
            paths.append(str(path))
    return paths


def append_uploaded_sources(added, current, example_choice):
    """UploadButton is the only way to add files once gr.File holds a value.

    Gradio drops the drop-target and the file input from the DOM as soon as the
    component has files, so the button carries every upload after the first.
    """
    merged = _file_paths(current)
    seen = set(merged)
    for path in _file_paths(added):
        if path not in seen:
            merged.append(path)
            seen.add(path)
    preview, cleared_choice, _, case_update, case_paths = select_uploaded_source(
        merged, example_choice
    )
    return (
        gr.update(value=merged, visible=bool(merged)),
        preview,
        cleared_choice,
        case_update,
        case_paths,
    )


def drop_uploaded_sources(added, current, example_choice):
    """Keep the overlay file input empty so it stays a drop target."""
    upload_row, preview, cleared_choice, case_update, case_paths = append_uploaded_sources(
        added, current, example_choice
    )
    return None, upload_row, preview, cleared_choice, case_update, case_paths


def preview_selected_upload(uploaded_files, evt: gr.SelectData):
    return raster_preview_at(uploaded_files, getattr(evt, "index", 0))


def preview_selected_case(index, paths):
    return raster_preview_at(paths, index)


def prepare_rasters(uploaded_files, run_dir: Path, example_name: str) -> tuple[list[str], str]:
    paths = []
    for item in uploaded_files or []:
        source = Path(getattr(item, "name", item))
        if source.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"} and source.is_file():
            paths.append(str(source.resolve()))
    if paths:
        return paths, "visitor-uploaded remote-sensing imagery"
    example = EXAMPLES.get(example_name, EXAMPLES[DEFAULT_EXAMPLE])
    if example["directory"]:
        directory = ROOT / "examples" / example["directory"]
        bundled = [str(path.resolve()) for path in _bundled_files(directory)]
        if bundled:
            return bundled, f"bundled remote-sensing case: {example_name}"
    raise gr.Error("Custom mode requires at least one .tif, .tiff, .png, .jpg, or .jpeg image.")


def _execute_cpu_tools_over_mcp(
    plan: list[dict],
    raster_paths: list[str],
    run_dir: Path,
    attempt: int,
) -> tuple[list[dict], bool, list[str], str, list[str]]:
    """Execute a constrained CPU/GPU tool chain through the active MCP runtime."""
    trajectory = []
    artifacts: list[str] = []
    roles = infer_band_roles(raster_paths)
    last_index: str | None = None
    last_mask: str | None = None
    derived_for_stats: list[str] = []
    last_values: list[float] = []
    last_labels: list[str] = []
    trend_x: list[float] = []
    last_index_kind = ""
    last_perception_result: dict | list | str | None = None

    def selected_image(parameters: dict, position: int = 0) -> str:
        requested = str(
            parameters.get("image")
            or parameters.get("image_path")
            or parameters.get("input_image_path")
            or ""
        ).strip()
        if requested:
            match = next(
                (path for path in raster_paths if Path(path).name == Path(requested).name),
                None,
            )
            if match:
                return match
        if position >= len(raster_paths):
            raise ValueError("This perception tool needs more input images.")
        return raster_paths[position]

    for index, step in enumerate(plan):
        tool = step["tool"]
        parameters = dict(step.get("parameters", {}))
        status = "success"
        try:
            if tool in GPU_SUPPORTED_TOOLS:
                if tool == "remoteclip_classify":
                    arguments = {"input_image_path": selected_image(parameters)}
                elif tool == "sam2_segment":
                    bbox = parameters.get("bbox")
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        raise ValueError("sam2_segment requires bbox=[x1, y1, x2, y2].")
                    arguments = {
                        "input_image_path": selected_image(parameters),
                        "bbox": bbox,
                        "output_path": f"attempt_{attempt}_sam2_mask.png",
                    }
                elif tool in {"earthbench_change_detection", "earthbench_damage_assessment"}:
                    arguments = {
                        "pre_image_path": selected_image(parameters, 0),
                        "post_image_path": selected_image(parameters, 1),
                        "output_path": (
                            f"attempt_{attempt}_changeos.png"
                            if tool == "earthbench_change_detection"
                            else f"attempt_{attempt}_bright_damage.png"
                        ),
                    }
                else:
                    arguments = {"image": selected_image(parameters)}
                    if tool in {"text_to_bbox", "object_detection", "count_given_object"}:
                        arguments["text"] = str(
                            parameters.get("text") or parameters.get("object") or step["purpose"]
                        )
                    if tool == "region_attribute_description":
                        arguments["attribute"] = str(
                            parameters.get("attribute") or step["purpose"]
                        )
                    if parameters.get("bbox") is not None and tool in {
                        "region_attribute_description", "count_given_object"
                    }:
                        bbox = parameters["bbox"]
                        arguments["bbox"] = (
                            str(bbox) if isinstance(bbox, str)
                            else "(" + ", ".join(str(value) for value in bbox) + ")"
                        )
                last_perception_result = call_workflow_tool(tool, arguments)
                observation = json.dumps(last_perception_result, ensure_ascii=False, default=str)
                parameters = {
                    key: Path(value).name if key.endswith("path") and isinstance(value, str) else value
                    for key, value in arguments.items()
                }
                if isinstance(last_perception_result, dict):
                    output_candidate = last_perception_result.get("output_path")
                    if output_candidate and Path(output_candidate).is_file():
                        artifacts.append(str(Path(output_candidate).resolve()))
            elif tool in {"inspect_rasters", "get_filelist"}:
                result = inspect_rasters(raster_paths)
                observation = json.dumps(
                    {"count": len(result), "sample": result[:6], "truncated": len(result) > 6},
                    ensure_ascii=False,
                )
                parameters = {"files": [Path(path).name for path in raster_paths[:8]], "total": len(raster_paths)}
            elif tool == "validate_inputs":
                result = mcp_validate_inputs(raster_paths)
                observation = f"Validated {result['count']} readable GeoTIFF inputs and isolated output paths."
            elif tool == "lst_single_channel":
                bt = next((p for p in raster_paths if "bt10" in Path(p).name.lower()), None)
                red = next((p for p in raster_paths if re.search(r"(^|[_-])b4([_.-]|$)", Path(p).name.lower())), None)
                nir = next((p for p in raster_paths if re.search(r"(^|[_-])b5([_.-]|$)", Path(p).name.lower())), None)
                if not all((bt, red, nir)):
                    raise ValueError("Single-channel LST requires filename-identifiable BT10, B4, and B5 rasters.")
                output = run_dir / f"attempt_{attempt}_black_forest_lst.tif"
                last_index = lst_single_channel(bt, red, nir, str(output))
                derived_for_stats = [last_index]
                artifacts.append(last_index)
                observation = f"NDVI-derived emissivity correction produced {Path(last_index).name}."
                parameters = {"bt": Path(bt).name, "red": Path(red).name, "nir": Path(nir).name}
            elif tool == "calculate_mean_lst_by_ndvi":
                if not last_index:
                    raise ValueError("An emissivity-corrected LST raster is required.")
                red = next((p for p in raster_paths if re.search(r"(^|[_-])b4([_.-]|$)", Path(p).name.lower())), None)
                nir = next((p for p in raster_paths if re.search(r"(^|[_-])b5([_.-]|$)", Path(p).name.lower())), None)
                if not red or not nir:
                    raise ValueError("NDVI masking requires Red and NIR rasters.")
                ndvi_output = run_dir / f"attempt_{attempt}_black_forest_ndvi.tif"
                comparison = mean_lst_by_ndvi(red, nir, last_index, str(ndvi_output))
                artifacts.append(str(ndvi_output))
                last_values = [comparison["forest_mean_k"], comparison["nonvegetated_mean_k"]]
                last_labels = ["forest NDVI > 0.7", "non-vegetated NDVI < 0.2"]
                observation = json.dumps(comparison, ensure_ascii=False)
                parameters = {"forest_threshold": 0.7, "nonvegetated_threshold": 0.2}
            elif tool == "ATI":
                outputs = []
                for date in ("2020_07_01", "2020_07_15"):
                    day = next((p for p in raster_paths if date in Path(p).name and "day" in Path(p).name.lower()), None)
                    night = next((p for p in raster_paths if date in Path(p).name and "night" in Path(p).name.lower()), None)
                    iso_date = date.replace("_", "-")
                    albedo = next((p for p in raster_paths if iso_date in Path(p).name and "albedo" in Path(p).name.lower()), None)
                    if not all((day, night, albedo)):
                        raise ValueError(f"Incomplete day/night/albedo triplet for {iso_date}.")
                    output = run_dir / f"attempt_{attempt}_ati_{iso_date}.tif"
                    outputs.append(apparent_thermal_inertia(day, night, albedo, str(output)))
                derived_for_stats = outputs
                artifacts.extend(outputs)
                observation = f"Produced ATI maps for {len(outputs)} correctly paired dates."
                parameters = {"dates": ["2020-07-01", "2020-07-15"]}
            elif tool == "calculate_water_turbidity_ntu":
                outputs = []
                for path in raster_paths:
                    match = re.search(r"\d{4}-\d{2}-\d{2}", Path(path).name)
                    label = match.group() if match else f"{len(outputs) + 1:02d}"
                    output = run_dir / f"attempt_{attempt}_turbidity_{label}.tif"
                    outputs.append(calculate_turbidity(path, str(output)))
                derived_for_stats = outputs
                artifacts.extend(outputs)
                observation = f"Converted {len(outputs)} red-band rasters into non-negative turbidity maps."
                parameters = {"method": "linear", "a": 1.0, "b": 0.0}
            elif tool == "maximum_composite":
                output = run_dir / f"attempt_{attempt}_maxfrp_composite.tif"
                last_index = maximum_composite(raster_paths, str(output))
                artifacts.append(last_index)
                derived_for_stats = [last_index]
                observation = f"Created a maximum-FRP composite from {len(raster_paths)} daily rasters."
                parameters = {"dates": len(raster_paths)}
            elif tool == "identify_fire_prone_areas":
                if not last_index:
                    raise ValueError("A MaxFRP composite is required before fire-prone thresholding.")
                output = run_dir / f"attempt_{attempt}_fire_prone_mask.tif"
                last_mask, threshold = fire_prone_mask(last_index, str(output), 75)
                artifacts.append(last_mask)
                observation = f"Mapped the upper quartile of positive MaxFRP pixels using threshold {threshold:.6g}."
                parameters = {"positive_pixel_percentile": 75, "threshold": threshold}
            elif tool == "create_hotspot_masks":
                if not derived_for_stats:
                    raise ValueError("A dated NBR series is required before hotspot extraction.")
                masks = []
                for nbr_path in derived_for_stats:
                    output = run_dir / f"attempt_{attempt}_hotspot_{Path(nbr_path).stem}.tif"
                    masks.append(create_below_threshold_mask(nbr_path, -0.3, str(output)))
                combined = run_dir / f"attempt_{attempt}_combined_fire_hotspots.tif"
                last_mask = combine_binary_masks(masks, str(combined))
                artifacts.extend(masks + [last_mask])
                observation = f"Created {len(masks)} low-NBR masks and a combined hotspot map."
                parameters = {"condition": "NBR < -0.3"}
            elif tool == "analyze_hotspot_direction":
                if not last_mask:
                    raise ValueError("A combined hotspot mask is required for directional analysis.")
                direction = hotspot_direction(last_mask)
                observation = json.dumps(direction, ensure_ascii=False)
                parameters = {"mask": Path(last_mask).name, "frame": "image north/south/east/west halves"}
            elif tool == "subtract":
                dated: dict[str, dict[str, str]] = {}
                for path in raster_paths:
                    match = re.search(r"(19|20)\d{2}", Path(path).name)
                    if not match:
                        continue
                    year = match.group()
                    role = "nres" if "_nres_" in Path(path).name.lower() else "total" if "_total_" in Path(path).name.lower() else None
                    if role:
                        dated.setdefault(year, {})[role] = path
                outputs = []
                years = []
                for year, pair in sorted(dated.items()):
                    if year < "1985" or year > "2020" or not {"total", "nres"} <= pair.keys():
                        continue
                    output = run_dir / f"attempt_{attempt}_residential_volume_{year}.tif"
                    outputs.append(subtract_rasters(pair["total"], pair["nres"], str(output)))
                    years.append(float(year))
                if not outputs:
                    raise ValueError("No correctly year-paired total/non-residential rasters were found.")
                derived_for_stats = outputs
                trend_x = years
                artifacts.extend(outputs)
                observation = f"Produced {len(outputs)} residential-volume maps for 1985–2020."
                parameters = {"years": [int(year) for year in years], "formula": "total - non-residential"}
            elif tool == "compute_tvdi":
                ndvi_paths = [path for path in raster_paths if "ndvi" in Path(path).name.lower()]
                lst_paths = [path for path in raster_paths if "lst" in Path(path).name.lower()]
                if len(ndvi_paths) != 1 or len(lst_paths) != 1:
                    raise ValueError("TVDI requires exactly one filename-identifiable NDVI and one LST raster.")
                output = run_dir / f"attempt_{attempt}_tvdi.tif"
                last_index = compute_tvdi(ndvi_paths[0], lst_paths[0], str(output))
                last_index_kind = "tvdi"
                derived_for_stats = [last_index]
                artifacts.append(last_index)
                observation = f"Real CPU TVDI regression produced {Path(last_index).name}."
                parameters = {"ndvi": Path(ndvi_paths[0]).name, "lst": Path(lst_paths[0]).name}
            elif tool in {"calculate_ndwi", "calculate_ndvi", "calculate_nbr"}:
                if tool == "calculate_nbr":
                    dated = {}
                    for path in raster_paths:
                        name = Path(path).name.lower()
                        match = re.search(r"\d{4}-\d{2}-\d{2}", name)
                        if match and ("b02" in name or "b07" in name):
                            dated.setdefault(match.group(), {})["nir" if "b02" in name else "swir2"] = path
                    pairs = [(date, pair) for date, pair in sorted(dated.items()) if {"nir", "swir2"} <= pair.keys()]
                    if pairs:
                        outputs = []
                        for date, pair in pairs:
                            output = run_dir / f"attempt_{attempt}_nbr_{date}.tif"
                            outputs.append(calculate_nbr(pair["nir"], pair["swir2"], str(output)))
                        last_index = outputs[-1]
                        last_index_kind = "nbr"
                        derived_for_stats = outputs
                        artifacts.extend(outputs)
                        observation = f"Paired {len(pairs)} dates and produced a real NBR GeoTIFF time series."
                        parameters = {"dates": [date for date, _ in pairs]}
                        trajectory.append(
                            {
                                "step": index + 1,
                                "tool": tool,
                                "purpose": step["purpose"],
                                "parameters": parameters,
                                "status": "success",
                                "observation": observation,
                            }
                        )
                        continue
                required = {
                    "calculate_ndwi": ("nir", "swir1"),
                    "calculate_ndvi": ("nir", "red"),
                    "calculate_nbr": ("nir", "swir2"),
                }[tool]
                missing = [role for role in required if role not in roles]
                if missing:
                    raise ValueError(f"Missing filename-identifiable bands: {', '.join(missing)}")
                output = run_dir / f"attempt_{attempt}_{tool}.tif"
                function = {
                    "calculate_ndwi": calculate_ndwi,
                    "calculate_ndvi": calculate_ndvi,
                    "calculate_nbr": calculate_nbr,
                }[tool]
                last_index = function(roles[required[0]], roles[required[1]], str(output))
                last_index_kind = tool.replace("calculate_", "")
                derived_for_stats = [last_index]
                artifacts.append(last_index)
                observation = f"Real CPU execution produced {Path(last_index).name}."
                parameters = {required[0]: Path(roles[required[0]]).name, required[1]: Path(roles[required[1]]).name}
            elif tool == "calculate_threshold_ratio":
                if not last_index:
                    raise ValueError("No index raster is available for thresholding.")
                threshold_value = parameters.get("threshold")
                if not isinstance(threshold_value, (int, float)):
                    raise ValueError("calculate_threshold_ratio requires a numeric threshold parameter.")
                threshold = float(threshold_value)
                ratio = calculate_threshold_ratio(last_index, threshold)
                mask = run_dir / f"attempt_{attempt}_threshold_mask.tif"
                last_mask = create_threshold_mask(last_index, threshold, str(mask))
                artifacts.append(last_mask)
                observation = f"{ratio:.2%} of valid pixels exceed threshold {threshold}; mask artifact created."
                parameters = {"image": Path(last_index).name, "threshold": threshold}
            elif tool in {"calculate_area", "calculate_mask_area"}:
                if not last_mask:
                    raise ValueError("No validated binary mask is available for area calculation.")
                area = calculate_area(last_mask)
                observation = json.dumps(area, ensure_ascii=False)
                parameters = {"mask": Path(last_mask).name, "area_source": area["source"]}
            elif tool == "calc_batch_image_mean":
                targets = derived_for_stats or raster_paths
                last_values = calc_batch_image_mean(targets)
                last_labels = [Path(path).name for path in targets]
                parsed_years = [re.search(r"(19|20)\d{2}", label) for label in last_labels]
                if not trend_x and all(parsed_years):
                    years = [float(match.group()) for match in parsed_years]
                    trend_x = years if len(set(years)) == len(years) else list(range(len(last_values)))
                elif not trend_x:
                    trend_x = list(range(len(last_values)))
                observation = json.dumps(dict(zip([Path(p).name for p in targets], last_values)), ensure_ascii=False)
                parameters = {"count": len(targets), "source": "derived rasters" if derived_for_stats else "input rasters"}
            elif tool == "compute_linear_trend":
                trend = linear_trend(last_values, trend_x or None)
                trend["series_mean"] = float(np.mean(last_values))
                observation = json.dumps(trend, ensure_ascii=False)
                parameters = {"observations": len(last_values), "x": trend_x}
            elif tool == "percentage_change":
                changes = consecutive_percentage_changes(last_values)
                last_values = changes
                last_labels = [f"{last_labels[i]} → {last_labels[i + 1]}" for i in range(len(changes))]
                observation = json.dumps(dict(zip(last_labels, changes)), ensure_ascii=False)
                parameters = {"formula": "(current - previous) / previous × 100"}
            elif tool == "max_value_and_index":
                maximum = mcp_max_value_and_index(last_values)
                max_index, max_value = int(maximum["index"]), float(maximum["maximum"])
                observation = json.dumps(
                    {"maximum": max_value, "interval": last_labels[max_index] if last_labels else max_index},
                    ensure_ascii=False,
                )
                parameters = {"values": len(last_values)}
            elif tool == "max_three_day_window":
                window = mcp_max_three_day_window(last_values)
                best = int(window["start_index"])
                observation = json.dumps(
                    {
                        "start": last_labels[best],
                        "end": last_labels[best + 2],
                        "three_day_mean": window["three_day_mean"],
                    },
                    ensure_ascii=False,
                )
                parameters = {"window_days": 3, "aggregation": "mean"}
            elif tool == "calc_batch_image_skewness":
                targets = derived_for_stats or raster_paths
                skewness_values = [raster_skewness(path) for path in targets]
                observation = json.dumps(
                    {
                        "mean_skewness": float(np.mean(skewness_values)),
                        "maximum_skewness": float(np.max(skewness_values)),
                        "interpretation": "right-skewed high-turbidity anomalies" if np.mean(skewness_values) > 0 else "no persistent right skew",
                    },
                    ensure_ascii=False,
                )
                parameters = {"rasters": len(targets)}
            elif tool == "sens_slope":
                trend = sens_slope(last_values)
                observation = json.dumps(trend, ensure_ascii=False)
                parameters = {"values": last_values}
            elif tool == "difference":
                if len(last_values) < 2:
                    raise ValueError("Difference requires at least two preceding regional statistics.")
                comparison = mcp_difference(last_values[0], last_values[1])
                observation = json.dumps(comparison, ensure_ascii=False)
                parameters = {"a": last_values[0], "b": last_values[1]}
            elif tool == "validate_result":
                validation = mcp_validate_result(raster_paths, artifacts)
                observation = f"Validated {validation['checked']} readable input/result raster(s)."
            else:
                raise ValueError(f"Unsupported MCP tool: {tool}")
        except Exception as exc:
            status = "error"
            observation = str(exc)

        trajectory.append(
            {
                "step": index + 1,
                "tool": tool,
                "purpose": step["purpose"],
                "parameters": parameters,
                "status": status,
                "observation": observation,
            }
        )
        if status == "error":
            break

    failed_steps = [step for step in trajectory if step["status"] == "error"]
    if failed_steps:
        diagnosis, _ = summarize_failure(trajectory[-1]["observation"])
        return trajectory, False, [diagnosis], trajectory[-1]["observation"], artifacts
    return (
        trajectory,
        True,
        [],
        f"The real MCP tool chain completed and produced {len(artifacts)} downloadable artifact(s).",
        artifacts,
    )


def execute_cpu_tools(
    plan: list[dict],
    raster_paths: list[str],
    run_dir: Path,
    attempt: int,
) -> tuple[list[dict], bool, list[str], str, list[str]]:
    """Run the full tool chain through discovered stdio MCP servers."""
    with workflow_mcp(run_dir):
        return _execute_cpu_tools_over_mcp(plan, raster_paths, run_dir, attempt)


_GEOTRACE_ERROR_KEYWORDS = (
    "error", "exception", "failed", "traceback", "invalid", "not found", "cannot", "unable"
)


def _truncate_geospatial_io(text: str, max_chars: int = 300) -> str:
    """Paper-aligned GeoTraceCompress for verbose geospatial tool I/O."""
    if len(text) <= max_chars:
        return text

    low = text.lower()
    if any(keyword in low for keyword in _GEOTRACE_ERROR_KEYWORDS):
        elevated_budget = max_chars * 6
        if len(text) <= elevated_budget:
            return text
        half = elevated_budget // 2
        return text[:half] + f" ... [TRUNCATED: {len(text) - elevated_budget} chars] ... " + text[-half:]

    parsed = None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError, TypeError, MemoryError):
            parsed = None

    if isinstance(parsed, list) and len(parsed) > 5:
        parsed = parsed[:3] + [f"... [{len(parsed) - 5} items omitted] ..."] + parsed[-2:]
        compact = json.dumps(parsed, ensure_ascii=False)
        return compact[:max_chars] + (f" [TRUNCATED: {len(compact) - max_chars} chars]" if len(compact) > max_chars else "")

    if isinstance(parsed, dict):
        keys = list(parsed)[:2]
        sample = {key: parsed[key] for key in keys}
        compact = json.dumps(sample, ensure_ascii=False)
        suffix = f" ... [TRUNCATED: {len(parsed) - 2} more keys]" if len(parsed) > 2 else ""
        compact += suffix
        return compact[:max_chars] + (f" [TRUNCATED: {len(compact) - max_chars} chars]" if len(compact) > max_chars else "")

    half = max_chars // 2
    return text[:half] + f" ... [TRUNCATED: {len(text) - max_chars} chars] ... " + text[-half:]


def geotrace_compress(trajectory: list[dict]) -> dict:
    """Convert execution steps into compact tool-use/tool-result messages."""
    compressed = []
    for step in trajectory:
        parameter_text = json.dumps(step.get("parameters", {}), ensure_ascii=False)
        compressed.append(
            {
                "type": "tool_use",
                "name": step.get("tool", ""),
                "purpose": step.get("purpose", ""),
                "input": _truncate_geospatial_io(parameter_text),
            }
        )
        compressed.append(
            {
                "type": "tool_result",
                "name": step.get("tool", ""),
                "status": step.get("status", ""),
                "output": _truncate_geospatial_io(str(step.get("observation", ""))),
            }
        )

    raw_text = json.dumps(trajectory, ensure_ascii=False)
    compressed_text = json.dumps(compressed, ensure_ascii=False)
    return {
        "strategy": "GeoTraceCompress/type-aware-v1",
        "raw_chars": len(raw_text),
        "compressed_chars": len(compressed_text),
        "compression_ratio": round(len(compressed_text) / max(len(raw_text), 1), 3),
        "trajectory": compressed,
    }


def generate_reflection(
    query: str,
    knowledge_context: str,
    trajectory: list[dict],
    critic_reasons: list[str],
    api_key: str,
    base_url: str,
    model: str,
) -> tuple[str, dict]:
    if not api_key:
        raise ValueError("An API key is required for failure-aware reflection.")
    compression = geotrace_compress(trajectory)
    system = """You are RSMeM's Failure-Aware Experience Refinement component.
Diagnose the failed tool trajectory and produce a concise, reusable constraint for the next attempt.
Return 3-5 sentences only. Do not claim a rerun has already succeeded."""
    user = f"""Query: {query[:3000]}
Retrieved knowledge: {knowledge_context[:3000]}
Symbolic critic: {json.dumps(critic_reasons, ensure_ascii=False)}
GeoTraceCompress output: {json.dumps(compression['trajectory'], ensure_ascii=False)}
"""
    reflection = api_completion(api_key, base_url, model, system, user, max_tokens=16384).strip()
    if not reflection:
        raise ValueError("The reflection model returned an empty failure-aware memory.")
    return reflection, compression


def build_timeline(attempts: list[dict], used_reflection: bool) -> str:
    stages = [
        ("1", "HKG", "Knowledge retrieved", "done"),
        ("2", "Plan", "Tool chain built", "done"),
        ("3", "Execute", "Real CPU tools", "done"),
        ("4", "Critic", "Result evaluated", "done"),
        ("5", "GeoTrace + FAR", "Compressed and reflected" if used_reflection else "Not needed", "done" if used_reflection else "skip"),
        ("6", "Retry", "Attempt 2 executed" if len(attempts) > 1 else "Stopped after success", "done" if len(attempts) > 1 else "skip"),
    ]
    cards = "".join(
        f'<div class="stage {state}"><span>{number}</span><b>{name}</b><small>{detail}</small></div>'
        for number, name, detail, state in stages
    )
    attempt_cards = []
    for attempt in attempts:
        verdict = "PASS" if attempt["critic"]["success"] else "REFLECT & RETRY"
        cls = "pass" if attempt["critic"]["success"] else "fail"
        attempt_cards.append(
            f'<div class="attempt {cls}"><b>Attempt {attempt["attempt"]}</b>'
            f'<span>{len(attempt["tool_plan"])} planned tools</span>'
            f'<strong>{verdict}</strong></div>'
        )
    return f'<div class="pipeline">{cards}</div><div class="attempts">{"".join(attempt_cards)}</div>'


def attempts_to_markdown(attempts: list[dict]) -> str:
    sections = []
    for attempt in attempts:
        success = attempt["critic"]["success"]
        far_used = attempt["context"]["far_memory"]
        section = [
            f"### Attempt {attempt['attempt']} — {'✅ critic passed' if success else '❌ critic found an issue'}",
            f"**Injected context:** HKG = yes · FAR memory = {'yes' if far_used else 'no'}",
            "\n#### Planned tool chain",
            "| Step | Tool | Purpose |",
            "|---:|---|---|",
        ]
        for index, step in enumerate(attempt["tool_plan"], start=1):
            purpose = str(step["purpose"]).replace("|", "\\|")
            section.append(f"| {index} | `{step['tool']}` | {purpose} |")
        section.extend(
            [
                "\n#### Execution trajectory",
                "| Step | Tool | Status | Observation |",
                "|---:|---|---|---|",
            ]
        )
        for step in attempt["trajectory"]:
            status = "✅ success" if step["status"] == "success" else "❌ error"
            observation = str(step["observation"]).replace("|", "\\|")
            section.append(
                f"| {step['step']} | `{step['tool']}` | {status} | {observation} |"
            )
        reasons = ", ".join(attempt["critic"]["reasons"]) or "none"
        section.extend(
            [
                "\n#### Critic verdict",
                f"**{'PASS' if success else 'FAIL'}** · reasons: `{reasons}`",
            ]
        )
        sections.append("\n".join(section))
    return "\n\n---\n\n".join(sections)


def run_rsmem_loop(
    query: str,
    example_name: str,
    uploaded_files,
    requested_base_url: str,
    requested_model: str,
    user_api_key: str,
):
    query = query.strip() or "Map flooded areas after a major storm using satellite imagery."
    api_key = user_api_key.strip() or os.getenv("RSMEM_API_KEY", "")
    if not api_key:
        raise gr.Error("API key is required. RSMeM will not use a preset or deterministic planner.")
    model = requested_model.strip() or os.getenv("RSMEM_MODEL", DEFAULT_MODEL)
    try:
        base_url = validate_base_url(
            requested_base_url or os.getenv("RSMEM_BASE_URL", DEFAULT_BASE_URL)
        )
    except ValueError as exc:
        raise gr.Error(f"Invalid Base URL: {exc}") from exc

    retrieved = retrieve_knowledge(query)
    if retrieved and max(score for score, _ in retrieved) <= 1:
        retrieved = []
    knowledge_lines = []
    knowledge_context_lines = []
    for index, (score, entry) in enumerate(retrieved, start=1):
        excerpt = re.sub(r"\s+", " ", entry["summary"]).strip()[:420]
        knowledge_lines.append(
            f"**{index}. {entry['domain']} → {entry['application']}**  \n"
            f"Retrieval score: `{score}`  \n{excerpt}..."
        )
        knowledge_context_lines.append(
            f"Domain: {entry['domain']} | Task: {entry['application']} | Guidance: {excerpt}"
        )
    knowledge_md = "\n\n".join(knowledge_lines)
    knowledge_context = "\n".join(knowledge_context_lines)

    run_dir = Path(tempfile.mkdtemp(prefix="rsmem_cpu_"))
    raster_paths, data_source = prepare_rasters(uploaded_files, run_dir, example_name)

    attempts = []
    all_artifacts: list[str] = []
    try:
        plan1 = plan_toolchain(query, knowledge_context, "", api_key, base_url, model)
    except Exception as exc:
        raise gr.Error(f"Model planning failed: {exc}") from exc
    trajectory1, success1, reasons1, result1, artifacts1 = execute_cpu_tools(
        plan1, raster_paths, run_dir, 1
    )
    all_artifacts.extend(artifacts1)
    attempts.append(
        {
            "attempt": 1,
            "context": {"hkg": True, "far_memory": False},
            "tool_plan": plan1,
            "trajectory": trajectory1,
            "result": result1,
            "critic": {"success": success1, "reasons": reasons1},
        }
    )

    reflection = ""
    compression = None
    if not success1:
        try:
            reflection, compression = generate_reflection(
                query, knowledge_context, trajectory1, reasons1, api_key, base_url, model
            )
            plan2 = plan_toolchain(
                query, knowledge_context, reflection, api_key, base_url, model
            )
        except Exception as exc:
            raise gr.Error(f"Failure-aware reflection or replanning failed: {exc}") from exc
        trajectory2, success2, reasons2, result2, artifacts2 = execute_cpu_tools(
            plan2, raster_paths, run_dir, 2
        )
        all_artifacts.extend(artifacts2)
        attempts.append(
            {
                "attempt": 2,
                "context": {"hkg": True, "far_memory": True},
                "tool_plan": plan2,
                "trajectory": trajectory2,
                "result": result2,
                "critic": {"success": success2, "reasons": reasons2},
            }
        )

    final_success = attempts[-1]["critic"]["success"]
    trace = {
        "mode": "real MCP execution with health-gated CPU and GPU tools",
        "data_source": data_source,
        "query": query,
        "retrieved_domains": [
            {"domain": entry["domain"], "task": entry["application"], "score": score}
            for score, entry in retrieved
        ],
        "geotrace_compress": compression,
        "attempts": attempts,
    }
    compression_payload = compression or {
        "status": "not_needed",
        "reason": "Attempt 1 passed the symbolic critic; reflection was not invoked.",
        "trajectory": [],
    }
    memory_md = (
        "### GeoTraceCompress\n\n"
        f"- Strategy: `{compression['strategy']}`\n"
        f"- Raw trajectory: `{compression['raw_chars']}` characters\n"
        f"- Compressed trajectory: `{compression['compressed_chars']}` characters\n"
        f"- Compression ratio: `{compression['compression_ratio']}`\n\n"
        "The Reflection Generator receives only this type-aware compressed trajectory, "
        "not the complete raw execution log.\n\n"
        "### Failure-Aware Memory written after Attempt 1\n\n"
        f"> {reflection}\n\n"
        "This compact reflection is injected into Attempt 2 together with the original HKG context."
        if reflection
        else "### GeoTraceCompress and Failure-Aware Memory\n\n"
        "Attempt 1 passed the symbolic critic, so no trajectory compression or reflection was needed."
    )
    final_md = (
        f"### Final outcome: {'✅ critic passed' if final_success else '⚠️ max attempts reached'}\n\n"
        f"**Attempts executed:** {len(attempts)}  \n"
        f"**Input source:** {data_source}  \n"
        f"**Final result:** {attempts[-1]['result']}\n\n"
        "The observations above come from real MCP tool calls. GPU perception is offered only when "
        "the corresponding xGPU worker passes its health check."
    )
    api_status = f"🟢 API planner active · model: `{model}` · no preset fallback"
    return (
        api_status,
        build_timeline(attempts, bool(reflection)),
        knowledge_md,
        attempts_to_markdown(attempts),
        trace,
        compression_payload,
        memory_md,
        final_md,
        sorted(set(all_artifacts)),
    )


def _retrieve_for_session(query: str) -> tuple[str, str, list[dict]]:
    retrieved = retrieve_knowledge(query)
    if not retrieved or max(score for score, _ in retrieved) <= 1:
        return "", "", []
    markdown, context, records = [], [], []
    for index, (score, entry) in enumerate(retrieved, start=1):
        excerpt = re.sub(r"\s+", " ", entry["summary"]).strip()[:420]
        markdown.append(
            f"**{index}. {entry['domain']} → {entry['application']}**  \n"
            f"Retrieval score: `{score}`  \n{excerpt}..."
        )
        context.append(
            f"Domain: {entry['domain']} | Task: {entry['application']} | Guidance: {excerpt}"
        )
        records.append({"domain": entry["domain"], "task": entry["application"], "score": score})
    return "\n\n".join(markdown), "\n".join(context), records


def _ftro_candidate_view(trajectory: list[dict]) -> dict:
    return geotrace_compress(trajectory)


def _reflection_trace(attempt: dict) -> list[dict]:
    """Include the answer-synthesis decision in every reflector evidence arm."""
    trajectory = copy.deepcopy(attempt["trajectory"])
    trajectory.append(
        {
            "step": len(trajectory) + 1,
            "tool": "answer_synthesizer",
            "purpose": "Form the final answer using only verified tool evidence.",
            "parameters": {},
            "status": "success",
            "observation": attempt["answer"],
        }
    )
    return trajectory


def _stream_progress_enabled() -> bool:
    return os.getenv("RSMEM_STREAM_PROGRESS", "1").strip().lower() not in {"0", "false", "no"}


def _stream_progress_interval_s() -> float:
    raw = os.getenv("RSMEM_STREAM_PROGRESS_INTERVAL_S", "").strip()
    if not raw:
        return 5.0
    try:
        return max(0.1, float(raw))
    except ValueError:
        return 5.0


def _run_react_with_progress(state: dict, api_key: str, attempt_dir: Path, memory: str, attempt_number: int):
    """Run one ReAct attempt on a worker thread, yielding a progress frame
    every few seconds so the SSE stream never goes silent.

    The agent loop routinely runs for minutes. Calling run_react_attempt
    inline produces no frames for that whole span, which leaves the UI frozen
    on the planning card and gives any intermediary on the connection nothing
    to observe.
    """
    call = functools.partial(
        run_react_attempt,
        query=state["query"],
        raster_paths=state["raster_paths"],
        output_dir=attempt_dir,
        api_key=api_key,
        base_url=state["base_url"],
        model=state["model"],
        knowledge_context=state["knowledge_context"],
        failure_memory=memory,
        # Each RSMeM attempt gets its own checkpoint namespace. Cross-attempt
        # learning is carried only by the user-confirmed FAM reflection, not by
        # silently replaying the complete previous transcript.
        thread_id=f"{state['agent_thread_id']}:attempt:{attempt_number}",
        checkpoint_path=Path(state["checkpoint_path"]),
    )
    if not _stream_progress_enabled():
        return call()

    started = time.monotonic()
    interval = _stream_progress_interval_s()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rsmem-react")
    try:
        future = pool.submit(call)
        while True:
            try:
                return future.result(timeout=interval)
            except FuturesTimeout:
                # future.result(timeout=) raises TimeoutError both when the
                # wait expired and when the worker itself raised TimeoutError.
                # Only the former should become a progress frame.
                if future.done():
                    raise
                elapsed = int(time.monotonic() - started)
                yield "progress", (
                    "The ReAct agent is reasoning and calling MCP tools "
                    f"({elapsed // 60}m {elapsed % 60:02d}s elapsed)."
                )
    finally:
        # wait=False: if the client disconnects, Gradio throws GeneratorExit
        # into this frame. Joining the worker here would block the server
        # thread for the rest of the agent loop; let it finish orphaned.
        pool.shutdown(wait=False)


def _execute_interactive_attempt_steps(state: dict, api_key: str):
    attempt_number = len(state["attempts"]) + 1
    memory = state["memory"][-1]["reflection"] if state["memory"] else ""
    yield "plan", "Selecting the next action from the current evidence."
    attempt_dir = Path(state["run_dir"]) / f"attempt_{attempt_number}"
    react_result = yield from _run_react_with_progress(
        state, api_key, attempt_dir, memory, attempt_number
    )
    total_model_calls = react_result["model_calls"]

    def budgeted_api_completion(*args, **kwargs):
        nonlocal total_model_calls
        if total_model_calls >= MAX_MODEL_CALLS_PER_ATTEMPT:
            raise RuntimeError(
                f"Attempt reached the {MAX_MODEL_CALLS_PER_ATTEMPT}-model-call budget."
            )
        total_model_calls += 1
        return api_completion(*args, **kwargs)
    plan = react_result["tool_plan"]
    trajectory = react_result["trajectory"]
    answer = react_result["answer"]
    artifacts = react_result["artifacts"]
    # step_succeeded, not status alone: a code step reports a crashed script
    # through its exit code while the call itself succeeds, so status would
    # otherwise record a run as clean when the agent's own code failed.
    executor_success = bool(trajectory) and all(
        step_succeeded(step) for step in trajectory
    )
    executor_reasons = [
        str(step.get("observation", ""))
        for step in trajectory if not step_succeeded(step)
    ]
    executor_result = (
        f"ReAct completed with {react_result['model_calls']} model call(s) and "
        f"{len(trajectory)} MCP tool call(s)."
    )
    _trace_event(
        state, "plan",
        f"ReAct selected {len(plan)} tool call(s): "
        + ", ".join(step["tool"] for step in plan),
        model_calls=react_result["model_calls"],
        mcp_health=react_result["mcp_health"],
    )
    yield "execute", (
        f"ReAct executed {len(trajectory)} MCP call(s), observing each result before its next action."
    )
    _trace_event(
        state, "execute",
        "Execution finished · " + ", ".join(
            f"{step['tool']}={step['status']}" for step in trajectory
        ),
        executor_success=executor_success,
        executor_reasons=executor_reasons,
        observations=[
            {
                "tool": step["tool"],
                "status": step["status"],
                "observation": str(step["observation"])[:600],
            }
            for step in trajectory
        ],
    )
    yield "answer", "The ReAct agent produced its final answer from the observed MCP evidence."

    yield "evaluate", (
        "Comparing the result with the case reference."
        if state["mode"] == "case"
        else "Running the Teacher Failure Gate; the user remains the final judge."
    )
    if state["mode"] == "case":
        evaluation = evaluate_case_deterministically(
            state["case_id"], trajectory, answer
        )
        if evaluation is None:
            evaluation = judge_builtin_case(
                api_call=budgeted_api_completion, api_key=api_key,
                base_url=state["base_url"], model=state["model"],
                query=state["query"], reference=state["reference"],
                trajectory=trajectory, final_answer=answer,
            )
            evaluation["source"] = "llm_judge"
        correct = evaluation["verdict"] == "correct"
    else:
        evaluation = teacher_failure_gate(
            api_call=budgeted_api_completion, api_key=api_key,
            base_url=state["base_url"], model=state["model"],
            query=state["query"], trajectory=trajectory, final_answer=answer,
        )
        correct = None

    state["attempts"].append(
        {
            "attempt": attempt_number,
            "context": {"hkg": bool(state["knowledge_context"]), "far_memory": bool(memory)},
            "tool_plan": plan,
            "trajectory": trajectory,
            "executor": {
                "success": executor_success,
                "reasons": executor_reasons,
                "result": executor_result,
            },
            "answer": answer,
            "evaluation": evaluation,
            "model_calls": total_model_calls,
            "checkpoint_path": react_result["checkpoint_path"],
        }
    )
    state["artifacts"] = sorted(set(state["artifacts"] + artifacts))
    state["candidates"] = {}
    _trace_event(
        state, "evaluate",
        f"Evaluation verdict: {evaluation.get('verdict', 'unknown')} · "
        f"confidence={evaluation.get('confidence', 'n/a')}",
    )

    if state["mode"] == "case":
        if correct:
            state["status"] = "completed_case_correct"
        elif attempt_number >= state["max_attempts"]:
            state["status"] = "max_attempts_reached"
        else:
            state["candidates"] = generate_reflection_candidates(
                api_call=budgeted_api_completion, api_key=api_key,
                base_url=state["base_url"], model=state["model"],
                query=state["query"], knowledge_context=state["knowledge_context"],
                trajectory=_reflection_trace(state["attempts"][-1]),
                ftro_builder=_ftro_candidate_view,
            )
            _trace_event(
                state, "reflect",
                "Reflection candidates: " + ", ".join(
                    f"{name}={info.get('status')}"
                    for name, info in state["candidates"].items()
                ),
            )
            state["status"] = "awaiting_reflection"
            state["attempts"][-1]["model_calls"] = total_model_calls
    else:
        state["status"] = (
            "max_attempts_reached"
            if attempt_number >= state["max_attempts"]
            else "awaiting_user_decision"
        )
    yield "complete", "The attempt is complete."


PHASE_LABELS = {
    "start": "initialization",
    "retrieve": "knowledge retrieval",
    "plan": "ReAct reasoning and tool selection",
    "progress": "ReAct reasoning and tool selection",
    "execute": "MCP action and observation",
    "answer": "answer synthesis",
    "evaluate": "evaluation",
    "reflect": "reflection generation",
    "complete": "completion",
}


def _trace_event(state: dict, phase: str, detail: str, **extra) -> None:
    state.setdefault("debug_trace", []).append(
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "detail": detail,
            **extra,
        }
    )


def _write_debug_trace(state: dict) -> str | None:
    """Dump the full run state (never credentials — none are stored in it)
    to a downloadable JSON so users can report exactly where a run failed.

    The archive copy stays complete for the operator. Visitors receive a
    sanitized file served from the downloads directory, so the URL and the
    JSON both omit the archive layout.
    """
    run_dir = state.get("run_dir")
    if not run_dir or not Path(run_dir).is_dir():
        return None
    payload = dict(state)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    path = Path(run_dir) / "rsmem_debug_trace.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    run_storage.write_trace(state.get("archive_dir"), payload)
    sanitized = run_storage.sanitize_trace_for_download(
        payload,
        archive_dir=state.get("archive_dir"),
        run_dir=run_dir,
    )
    return run_storage.publish_trace(state.get("run_id"), sanitized) or str(path)


def _execute_interactive_attempt(state: dict, api_key: str) -> dict:
    phase = "start"
    try:
        for phase, detail in _execute_interactive_attempt_steps(state, api_key):
            if phase != "progress":
                _trace_event(state, phase, detail)
    except Exception as exc:
        state["status"] = "failed"
        _trace_event(
            state, phase, f"Stage failed: {exc}",
            error=str(exc), traceback=traceback.format_exc(),
        )
        _write_debug_trace(state)
        raise RuntimeError(
            f"error during {PHASE_LABELS.get(phase, phase)}: {exc}"
        ) from exc
    return state


def _processing_timeline(active_phase: str, detail: str, started_at: str | None = None) -> str:
    stages = [
        ("retrieve", "Retrieve", "domain knowledge"),
        ("agent", "ReAct agent loop", "reason ↔ tool call ↔ observation"),
        ("answer", "Answer", "evidence synthesis"),
        ("evaluate", "Evaluate", "judge or failure gate"),
    ]
    order = [item[0] for item in stages]
    display_phase = "agent" if active_phase in {"plan", "execute", "progress"} else active_phase
    active_index = order.index(display_phase) if display_phase in order else len(order)
    cards = []
    for index, (key, name, caption) in enumerate(stages, start=1):
        state_class = "done" if index - 1 < active_index else "current" if key == display_phase else "pending"
        cards.append(
            f'<div class="stage {state_class}"><span>{index}</span><b>{name}</b><small>{caption}</small></div>'
        )
    # No reliable prior exists for how long an attempt takes — real runs have spanned
    # from under a minute to several hours — so this counts up from the attempt's own
    # start rather than guessing at a remaining-time estimate.
    elapsed = (
        f'<span class="run-elapsed" data-started="{html_lib.escape(started_at)}">elapsed 0:00</span>'
        if started_at else ""
    )
    title = (
        html_lib.escape(detail)
        if active_phase == "progress" and detail
        else "RSMeM is working"
    )
    return (
        '<div class="run-progress-panel">'
        '<div class="run-progress-heading"><span class="run-pulse"></span>'
        f'<div><b>{title}</b></div>' + elapsed + '</div>'
        '<div class="pipeline processing-pipeline">' + "".join(cards) + '</div></div>'
    )


def _stream_update(state: dict, phase: str, detail: str):
    status = gr.update(value="", visible=False)
    started_at = state["debug_trace"][0]["time"] if state.get("debug_trace") else None
    return (
        state, status, _processing_timeline(phase, detail, started_at),
        *[gr.skip() for _ in range(18)],
    )


def _stream_error_update(state: dict, phase: str, exc: Exception, trace_path: str | None):
    label = PHASE_LABELS.get(phase, phase)
    status = (
        f"### ❌ Attempt failed during {label}\n\n"
        f"`{exc}`\n\n"
        "Download the debug trace under **Evidence & artifacts** for the full "
        "stage-by-stage log and traceback."
    )
    return (
        state,
        status,
        gr.skip(),
        *[gr.skip() for _ in range(16)],
        gr.update(visible=True),
        gr.update(value=trace_path),
    )


def _initialize_interactive_workflow(
    query: str,
    example_name: str,
    uploaded_files,
    requested_base_url: str,
    requested_model: str,
    user_api_key: str,
) -> tuple[dict, str]:
    query = query.strip()
    if not query:
        raise gr.Error("A remote-sensing task is required.")
    api_key = user_api_key.strip() or os.getenv("RSMEM_API_KEY", "")
    if not api_key:
        raise gr.Error("API key is required; the Studio has no preset planner or judge.")
    try:
        base_url = validate_base_url(requested_base_url or DEFAULT_BASE_URL)
    except ValueError as exc:
        raise gr.Error(f"Invalid Base URL: {exc}") from exc
    model = requested_model.strip() or DEFAULT_MODEL
    if not example_name and not uploaded_files:
        raise gr.Error(
            "Attach GeoTIFF, PNG, or JPEG remote-sensing images for your own task, "
            "or load one of the prepared examples."
        )
    example = EXAMPLES.get(example_name) if example_name else {
        "directory": None,
        "query": query,
        "description": "Visitor-provided task and rasters",
    }
    run_dir = Path(tempfile.mkdtemp(prefix="rsmem_interactive_"))
    raster_paths, source = prepare_rasters(uploaded_files, run_dir, example_name)
    run_id = datetime.now(timezone.utc).strftime("%H%M%S") + "-" + secrets.token_hex(4)
    archive_dir = run_storage.new_run_dir(run_id)
    raster_paths = run_storage.archive_inputs(archive_dir, raster_paths)
    custom = bool(uploaded_files) or example["directory"] is None or example.get("freeform", False)
    hkg_md, hkg_context, retrieved = _retrieve_for_session(query)
    reference = None if custom else CASE_REFERENCES[example["directory"]]
    state = {
        "status": "running",
        "mode": "custom" if custom else "case",
        "max_attempts": 10 if custom else 3,
        "example_name": example_name,
        "query": query,
        "data_source": source,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "archive_dir": str(archive_dir) if archive_dir else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raster_paths": raster_paths,
        "base_url": base_url,
        "model": model,
        "knowledge_markdown": hkg_md,
        "knowledge_context": hkg_context,
        "retrieved": retrieved,
        "reference": reference,
        "case_id": None if custom else example["directory"],
        "agent_thread_id": secrets.token_hex(16),
        "checkpoint_path": str(run_dir / "react_checkpoints.sqlite"),
        "attempts": [],
        "memory": [],
        "candidates": {},
        "artifacts": [],
        "debug_trace": [],
    }
    _trace_event(
        state, "start",
        f"Run initialized · mode={state['mode']} · model={model} · base_url={base_url} · "
        f"{len(raster_paths)} raster(s) from {source}.",
    )
    _trace_event(
        state, "retrieve",
        f"HKG retrieval kept {len(retrieved)} entrie(s) after the low-relevance filter.",
        scores=[record.get("score") for record in retrieved],
    )
    return state, api_key


def start_interactive_workflow(
    query: str,
    example_name: str,
    uploaded_files,
    requested_base_url: str,
    requested_model: str,
    user_api_key: str,
):
    state, api_key = _initialize_interactive_workflow(
        query, example_name, uploaded_files, requested_base_url, requested_model, user_api_key
    )
    try:
        state = _execute_interactive_attempt(state, api_key)
    except Exception as exc:
        raise gr.Error(f"Attempt 1 failed: {exc}") from exc
    return (state, *_render_interactive_state(state))


def start_interactive_workflow_stream(
    query: str,
    example_name: str,
    uploaded_files,
    requested_base_url: str,
    requested_model: str,
    user_api_key: str,
):
    state, api_key = _initialize_interactive_workflow(
        query, example_name, uploaded_files, requested_base_url, requested_model, user_api_key
    )
    phase = "start"
    try:
        for phase, detail in _execute_interactive_attempt_steps(state, api_key):
            if phase != "progress":
                _trace_event(state, phase, detail)
                _write_debug_trace(state)
            if phase == "complete":
                yield (state, *_render_interactive_state(state))
            else:
                yield _stream_update(state, phase, detail)
    except Exception as exc:
        state["status"] = "failed"
        _trace_event(
            state, phase, f"Stage failed: {exc}",
            error=str(exc), traceback=traceback.format_exc(),
        )
        trace_path = _write_debug_trace(state)
        yield _stream_error_update(state, phase, exc, trace_path)
        raise gr.Error(
            f"Attempt 1 failed during {PHASE_LABELS.get(phase, phase)}: {exc}"
        ) from exc


def _esc(value) -> str:
    return html_lib.escape(str(value), quote=True)


def _attempts_interactive_html(state: dict) -> str:
    """Render attempts grouped by ReAct turn: one reasoning block per model
    turn, followed by the tool result(s) that turn's call(s) produced.

    The plan and the trajectory used to be rendered as two separate lists
    (a `Tool calls` list of tool names, then an `Observation` list of step
    cards), which just repeated every tool name twice. Grouping by `turn`
    (see react_harness.parse_agent_messages) also lets a trace older than
    that field -- where every step defaults to turn 0 -- degrade to a
    single group with no reasoning shown, instead of breaking.

    Markdown tables collapse at narrow widths, which is why this renders
    HTML card markup instead.
    """
    sections = []
    for attempt in state.get("attempts", []):
        groups: list[dict] = []
        for step in attempt["trajectory"]:
            turn = step.get("turn") or 0
            if groups and groups[-1]["turn"] == turn:
                groups[-1]["steps"].append(step)
            else:
                groups.append({"turn": turn, "steps": [step]})

        turn_blocks = []
        for group in groups:
            reasoning = str(group["steps"][0].get("reasoning") or "").strip()
            reasoning_html = (
                '<div class="step-reasoning">'
                '<div class="block-label">Reasoning</div>'
                f'<div class="reasoning-text">{_esc(reasoning)}</div>'
                "</div>"
                if reasoning
                else ""
            )
            step_cards = []
            for step in group["steps"]:
                ok = step["status"] == "success"
                badge = (
                    '<span class="badge ok">success</span>'
                    if ok
                    else '<span class="badge err">error</span>'
                )
                step_cards.append(
                    '<details class="step-card">'
                    f'<summary><span class="num">{_esc(step["step"])}</span>'
                    f'<code>{_esc(step["tool"])}</code>{badge}</summary>'
                    '<div class="step-obs"><strong>Inputs</strong>\n'
                    f'{_esc(json.dumps(step.get("parameters", {}), ensure_ascii=False, indent=2))}'
                    '\n\n<strong>Result</strong>\n'
                    f'{_esc(step["observation"])}</div>'
                    "</details>"
                )
            turn_blocks.append(
                '<div class="trace-turn">'
                f"{reasoning_html}"
                '<div class="block-label">Tool results</div>'
                f"{''.join(step_cards)}"
                "</div>"
            )
        sections.append(
            '<div class="trace-attempt">'
            f'<div class="attempt-head">Attempt {_esc(attempt["attempt"])}</div>'
            f"{''.join(turn_blocks)}"
            "</div>"
        )
    joined = '<hr class="trace-divider">'.join(sections)
    return f'<div class="agent-trace">{joined}</div>'


def _interactive_timeline(state: dict) -> str:
    """Render a compact, state-aware RSMeM progress strip."""
    attempt_cards = []
    for attempt in state.get("attempts", []):
        verdict = str(attempt["evaluation"].get("verdict", "unknown"))
        passed = verdict in {"correct", "pass_supported", "pass_recovered"}
        verdict_label = {
            "correct": "CORRECT",
            "incorrect": "RETRY",
            "pass_supported": "SUPPORTED",
            "pass_recovered": "RECOVERED",
            "retry_unsupported": "REVIEW",
            "retry_inconsistent": "REVIEW",
            "uncertain": "UNCERTAIN",
        }.get(verdict, verdict.upper())
        attempt_cards.append(
            f'<div class="attempt {"pass" if passed else "fail"}">'
            f'<b>Attempt {attempt["attempt"]}</b>'
            f'<span>{len(attempt["tool_plan"])} tools</span>'
            f'<strong>{verdict_label}</strong></div>'
        )
    status = state.get("status", "not_started")
    has_attempt = bool(state.get("attempts"))
    has_knowledge = bool(state.get("knowledge_context", "").strip())
    has_memory = bool(state.get("memory"))
    custom_mode = state.get("mode") == "custom"

    def stage_state(name: str) -> str:
        if name == "retrieve":
            return "done" if has_knowledge else "skip"
        if name in {"react", "answer", "evaluate"}:
            return "done" if has_attempt else "pending"
        if name == "human":
            if status == "awaiting_user_decision":
                return "current"
            if status == "completed_user_accepted":
                return "done"
            return "skip" if not custom_mode else "pending"
        if name == "reflect":
            if status == "awaiting_reflection":
                return "current"
            if has_memory:
                return "done"
            if status in {"completed_case_correct", "completed_user_accepted", "max_attempts_reached"}:
                return "skip"
            return "pending"
        return "pending"

    stages = [
        ("1", "Retrieve", "relevant knowledge" if has_knowledge else "skipped: no relevant match", "retrieve"),
        ("2", "ReAct agent loop", "reason ↔ tool call ↔ observation", "react"),
        ("3", "Answer", "evidence synthesis", "answer"),
        ("4", "Evaluate", "case judge or teacher gate", "evaluate"),
        ("5", "Human", "accept or request retry", "human"),
        ("6", "Reflect", "repair and evolve memory", "reflect"),
    ]
    cards = "".join(
        f'<div class="stage {stage_state(key)}"><span>{n}</span><b>{name}</b><small>{detail}</small></div>'
        for n, name, detail, key in stages
    )
    return (
        f'<div class="pipeline" aria-label="RSMeM execution stages">{cards}</div>'
        f'<div class="attempts">{"".join(attempt_cards)}</div>'
    )


REFLECTION_SELECTION_MAP = {
    "Candidate 1": "FTRO",
    "Candidate 2": "Full trajectory",
    "Candidate 3": "LLM summary",
    "Write my own": "User-authored",
}


def _internal_reflection_source(selection: str) -> str:
    return REFLECTION_SELECTION_MAP.get(selection, selection)


def _reflection_candidate_texts(state: dict) -> tuple[str, str, str]:
    values = []
    for source in ("FTRO", "Full trajectory", "LLM summary"):
        candidate = state.get("candidates", {}).get(source, {})
        if candidate.get("status") == "ok":
            values.append(str(candidate.get("reflection", "")).strip())
        else:
            values.append("This candidate is unavailable for the current attempt.")
    return tuple(values)


def _demote_answer_headings(markdown: str, min_level: int = 4) -> str:
    """Shift model-authored headings so none sit above ``min_level``.

    Fenced code blocks are left alone so a Python comment's ``#`` is not
    treated as an ATX heading.
    """
    if not markdown:
        return markdown
    fences = list(
        re.finditer(r"^(`{3,}|~{3,}).*?(?:^\1\s*$|\Z)", markdown, flags=re.MULTILINE | re.DOTALL)
    )

    def in_fence(index: int) -> bool:
        return any(match.start() <= index < match.end() for match in fences)

    levels = [
        len(match.group(1))
        for match in re.finditer(r"^(#{1,6})(?=\s)", markdown, flags=re.MULTILINE)
        if not in_fence(match.start())
    ]
    if not levels:
        return markdown
    shift = max(0, min_level - min(levels))
    if shift == 0:
        return markdown

    def replace(match: re.Match[str]) -> str:
        if in_fence(match.start()):
            return match.group(0)
        return "#" * min(6, len(match.group(1)) + shift) + match.group(2)

    return re.sub(r"^(#{1,6})(\s+)", replace, markdown, flags=re.MULTILINE)


def _evaluation_summary_markdown(evaluation: dict, mode: str) -> str:
    """Turn evaluator internals into a short, user-facing result card."""
    if not evaluation:
        return ""

    verdict = str(evaluation.get("verdict", "unknown")).lower()
    confidence = max(0.0, min(1.0, float(evaluation.get("confidence", 0))))
    confidence_line = f"**Confidence:** {confidence:.0%}"

    if mode == "case":
        labels = {
            "correct": ("✅ Correct", "The result matches the bundled case reference."),
            "incorrect": ("❌ Another attempt is needed", "The result does not match the bundled case reference."),
            "uncertain": ("⚠️ Evaluation is uncertain", "The judge could not verify the result confidently."),
        }
        title, message = labels.get(
            verdict, ("⚠️ Evaluation unavailable", "The case result could not be classified."),
        )
        if verdict != "correct" and evaluation.get("comparison"):
            message += f"\n\n**Reason:** {str(evaluation['comparison']).strip()}"
        return f"#### {title}\n\n{message}\n\n{confidence_line}"

    retry = bool(evaluation.get("retry"))
    if retry:
        title = "⚠️ Please review this result"
        message = "The automated check found a possible problem. You make the final decision."
        detail = evaluation.get("root_cause") or evaluation.get("correction_guidance")
        if detail:
            message += f"\n\n**Possible issue:** {str(detail).strip()}"
    elif verdict in {"pass_supported", "pass_recovered"}:
        title = "✅ No clear problem detected"
        message = "The automated check found sufficient supporting evidence. You make the final decision."
    else:
        title = "⚠️ Automated check is uncertain"
        message = "Please inspect the result and decide whether another attempt is needed."
    return f"#### {title}\n\n{message}\n\n{confidence_line}"


def _render_interactive_state(state: dict):
    status = state.get("status", "not_started")
    if state.get("mode") == "case":
        mode_label = "reference-aware built-in case"
    elif str(state.get("data_source", "")).startswith("bundled"):
        mode_label = "open-ended bundled case"
    else:
        mode_label = "user-controlled upload"
    status_md = (
        f"### Workflow status: `{status}`\n\n"
        f"Mode: **{mode_label}** · Attempt {len(state.get('attempts', []))}/{state.get('max_attempts', '?')} · "
        f"Model: `{state.get('model', '')}`"
    )
    latest = state["attempts"][-1] if state.get("attempts") else {}
    evaluation = latest.get("evaluation", {})
    if status == "awaiting_user_decision":
        decision = (
            "The Teacher Failure Gate has no answer key and only provides an advisory check. "
            "Choose one of the two actions below; your decision is final."
        )
    elif status == "awaiting_reflection":
        decision = (
            "### Reflection required\n\nCompare the three model-generated repairs or write your own. "
            "Nothing is written to FAM until you confirm one option."
        )
    elif status == "completed_case_correct":
        decision = "### Completed\n\nThe reference-aware judge found the result semantically correct."
    elif status == "completed_user_accepted":
        decision = "### Completed\n\nThe user accepted the current result."
    elif status == "max_attempts_reached":
        decision = "### Stopped\n\nThe configured attempt limit has been reached."
    else:
        decision = ""

    show_decision = status == "awaiting_user_decision"
    show_candidates = status == "awaiting_reflection"
    show_knowledge = bool(state.get("knowledge_markdown", "").strip())
    candidate_1, candidate_2, candidate_3 = _reflection_candidate_texts(state)
    final_answer = _demote_answer_headings(latest.get("answer", ""))
    evaluation_md = _evaluation_summary_markdown(evaluation, state.get("mode", "case"))
    final_md = (
        f"### User query\n\n{state.get('query', '')}\n\n"
        f"### Current result\n\n{final_answer}\n\n"
        f"**Data source:** {state.get('data_source', '')}\n\n"
        f"### Evaluator assessment\n\n{evaluation_md}"
    )
    artifacts = list(state.get("artifacts", []))
    if artifacts:
        run_storage.archive_outputs(state.get("archive_dir"), artifacts)
        artifacts = run_storage.publish_downloads(state.get("run_id"), artifacts)
    return (
        status_md,
        _interactive_timeline(state),
        state.get("knowledge_markdown", ""),
        gr.update(visible=show_knowledge),
        _attempts_interactive_html(state),
        _evaluation_summary_markdown(evaluation, state.get("mode", "case")),
        decision,
        gr.update(visible=show_decision),
        gr.update(visible=show_candidates),
        candidate_1,
        candidate_2,
        candidate_3,
        gr.update(value=None),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        final_md,
        artifacts,
        gr.update(visible=bool(state.get("attempts"))),
        gr.update(value=_write_debug_trace(state)),
    )


def accept_custom_result(state: dict):
    if not state or state.get("mode") != "custom" or state.get("status") != "awaiting_user_decision":
        raise gr.Error("There is no custom-upload result awaiting acceptance.")
    updated = copy.deepcopy(state)
    updated["status"] = "completed_user_accepted"
    return (updated, *_render_interactive_state(updated))


def request_custom_reflections(state: dict, user_api_key: str):
    if not state or state.get("mode") != "custom" or state.get("status") != "awaiting_user_decision":
        raise gr.Error("Reflection can only be requested for the current custom-upload result.")
    api_key = user_api_key.strip() or os.getenv("RSMEM_API_KEY", "")
    if not api_key:
        raise gr.Error("API key is required to generate reflection candidates.")
    updated = copy.deepcopy(state)
    latest = updated["attempts"][-1]
    total_model_calls = int(latest.get("model_calls", 0))

    def budgeted_api_completion(*args, **kwargs):
        nonlocal total_model_calls
        if total_model_calls >= MAX_MODEL_CALLS_PER_ATTEMPT:
            raise RuntimeError(
                f"Attempt reached the {MAX_MODEL_CALLS_PER_ATTEMPT}-model-call budget."
            )
        total_model_calls += 1
        return api_completion(*args, **kwargs)

    try:
        updated["candidates"] = generate_reflection_candidates(
            api_call=budgeted_api_completion, api_key=api_key,
            base_url=updated["base_url"], model=updated["model"],
            query=updated["query"], knowledge_context=updated["knowledge_context"],
            trajectory=_reflection_trace(latest), ftro_builder=_ftro_candidate_view,
        )
    except Exception as exc:
        updated["status"] = "failed"
        _trace_event(
            updated, "reflect", f"Reflection generation failed: {exc}",
            error=str(exc), traceback=traceback.format_exc(),
        )
        _write_debug_trace(updated)
        raise gr.Error(f"Reflection generation failed: {exc}") from exc
    latest["model_calls"] = total_model_calls
    _trace_event(
        updated, "reflect",
        "Reflection candidates: " + ", ".join(
            f"{name}={info.get('status')}" for name, info in updated["candidates"].items()
        ),
    )
    updated["status"] = "awaiting_reflection"
    return (updated, *_render_interactive_state(updated))


def select_reflection_candidate(
    selection: str, state: dict, user_diagnosis: str, user_plan: str
) -> str:
    source = _internal_reflection_source(selection)
    if source == "User-authored":
        diagnosis = user_diagnosis.strip()
        plan = user_plan.strip()
        return (
            f"Failure diagnosis: {diagnosis}\nNext-attempt plan: {plan}"
            if diagnosis or plan else ""
        )
    candidate = (state or {}).get("candidates", {}).get(source, {})
    return str(candidate.get("reflection", "")) if candidate.get("status") == "ok" else ""


def confirm_reflection_and_retry(
    state: dict,
    selection: str,
    edited_reflection: str,
    user_diagnosis: str,
    user_plan: str,
    user_api_key: str,
):
    if not state or state.get("status") != "awaiting_reflection":
        raise gr.Error("No reflection decision is currently pending.")
    if len(state["attempts"]) >= state["max_attempts"]:
        raise gr.Error("The attempt limit has already been reached.")
    api_key = user_api_key.strip() or os.getenv("RSMEM_API_KEY", "")
    if not api_key:
        raise gr.Error("API key is required for the next attempt.")
    source = _internal_reflection_source(selection)
    reflection = edited_reflection.strip()
    if source == "User-authored":
        if not user_diagnosis.strip() or not user_plan.strip():
            raise gr.Error("Provide both your error diagnosis and next-attempt plan.")
        reflection = (
            f"Failure diagnosis: {user_diagnosis.strip()}\n"
            f"Next-attempt plan: {user_plan.strip()}"
        )
    elif selection:
        candidate = state["candidates"].get(source, {})
        if candidate.get("status") != "ok":
            reason = str(candidate.get("error", "")).strip()
            raise gr.Error(
                f"{selection} is unavailable for this attempt"
                + (f" ({reason})" if reason else "")
                + " — choose another candidate or write your own."
            )
        if not reflection:
            # The radio-change event fills the editable box asynchronously; if
            # the user confirms before it lands, fall back to the candidate
            # text itself rather than failing.
            reflection = str(candidate.get("reflection", "")).strip()
    if not selection or not reflection:
        raise gr.Error("Select a candidate and confirm a non-empty reflection.")

    updated = copy.deepcopy(state)
    updated["memory"].append(
        {
            "source_attempt": len(state["attempts"]),
            "source": source,
            "reflection": reflection,
        }
    )
    updated["status"] = "running"
    try:
        updated = _execute_interactive_attempt(updated, api_key)
    except Exception as exc:
        raise gr.Error(f"Next attempt failed: {exc}") from exc
    return (updated, *_render_interactive_state(updated))


def _session_id_from_request(request) -> str | None:
    cookies = dict(getattr(request, "cookies", None) or {})
    return cookies.get(msoauth.SESSION_COOKIE) or None


def _effective_session_id(browser_sid: str, request) -> str | None:
    """Prefer the BrowserState id (survives the ModelScope proxy, which
    swallows Set-Cookie on the shell-served routes); fall back to the
    cookie for plain local deployments."""
    return (browser_sid or "").strip() or _session_id_from_request(request)


def _resolve_connection(
    mode: str,
    ms_model: str,
    base_url: str,
    model: str,
    api_key: str,
    session_id: str | None,
) -> tuple[str, str, str]:
    """Map the selected connection mode to (base_url, model, api_key).

    Credentials are resolved strictly by mode: the ModelScope session token is
    never used in OpenAI mode and the pasted key is never used in ModelScope
    mode. Without a valid authorization nothing runs — there is no fallback.
    """
    if mode == MODE_MODELSCOPE:
        if not msoauth.oauth_configured():
            raise gr.Error(
                "ModelScope quota mode is not configured on this deployment yet. "
                "Switch to the OpenAI-compatible API mode, or ask the Studio owner "
                "to enable OAuth under Settings -> adjust deployment settings -> "
                "OAuth settings and redeploy."
            )
        token = msoauth.SESSION_STORE.get_token(session_id)
        if not token:
            raise gr.Error(
                "Sign in with ModelScope first. RSMeM runs only with your own "
                "API-Inference authorization and never uses a preset token."
            )
        return (
            msoauth.INFERENCE_BASE_URL,
            (ms_model or "").strip() or MODELSCOPE_DEFAULT_MODEL,
            token,
        )
    api_key = (api_key or "").strip() or os.getenv("RSMEM_API_KEY", "")
    if not api_key:
        raise gr.Error("API key is required; the Studio has no preset planner or judge.")
    return ((base_url or "").strip() or DEFAULT_BASE_URL, (model or "").strip() or DEFAULT_MODEL, api_key)


def _ms_status_html(session_id: str | None) -> tuple[str, bool]:
    """Build the ModelScope session status box. Returns (html, signed_in)."""
    if not msoauth.oauth_configured():
        return (
            '<div class="auth-box"><div class="st"><span class="dot">●</span> '
            "ModelScope quota mode · not configured on this deployment</div>"
            "<p>The Studio owner has not enabled OAuth under <b>Settings → "
            "adjust deployment settings → OAuth settings</b> yet. Use the "
            "OpenAI-compatible API mode instead.</p></div>",
            False,
        )
    if not msoauth.studio_host() and not os.getenv("OAUTH_REDIRECT_URI", "").strip():
        return (
            '<div class="auth-box"><div class="st"><span class="dot">●</span> '
            "ModelScope quota mode · callback domain unknown</div>"
            "<p>This deployment did not receive the Studio access domain, so "
            "ModelScope would reject the sign-in redirect. The owner needs to "
            "enable OAuth in the Studio settings and redeploy. Use the "
            "OpenAI-compatible API mode meanwhile.</p></div>",
            False,
        )
    if not session_id:
        return (
            '<div class="auth-box"><div class="st"><span class="dot">●</span> '
            "ModelScope session · initializing</div>"
            "<p>Press <b>Refresh sign-in status</b> to start a browser session, "
            "then sign in.</p></div>",
            False,
        )
    session = msoauth.SESSION_STORE.get_session(session_id)
    if session:
        user = _esc(session.get("user", "ModelScope user"))
        return (
            '<div class="auth-box connected"><div class="st ok"><span class="dot">●</span> '
            f'Signed in as <b data-no-translate>{user}</b> · runs use your own API-Inference quota</div>'
            "<p>The token stays in this browser session only and is never stored "
            "by the Studio. Requires an Alibaba Cloud-bound, real-name verified "
            "ModelScope account.</p></div>",
            True,
        )
    nonce = msoauth.SESSION_STORE.mint_login_nonce(session_id)
    return (
        '<div class="auth-box"><div class="st"><span class="dot">●</span> '
        "ModelScope session · not connected</div>"
        "<p>Authorize API-Inference for this browser session — no key field is "
        "needed. The sign-in page opens in a new tab; afterwards return here "
        "and press <b>Refresh sign-in status</b>.</p>"
        f'<a class="ms-signin" href="oauth/login?nonce={nonce}" target="_blank" '
        'rel="noopener">Sign in with ModelScope</a></div>',
        False,
    )


def refresh_ms_session(browser_sid: str, request: gr.Request):
    session_id = _effective_session_id(browser_sid, request)
    if not session_id:
        session_id = secrets.token_urlsafe(24)
    status_html, signed_in = _ms_status_html(session_id)
    return status_html, gr.update(visible=signed_in), session_id


def sign_out_ms(browser_sid: str, request: gr.Request):
    session_id = _effective_session_id(browser_sid, request)
    msoauth.SESSION_STORE.clear(session_id)
    return refresh_ms_session(browser_sid, request)


def switch_connection_mode(mode: str):
    """Toggle the mode panes and drop the other mode's pasted credential."""
    is_ms = mode == MODE_MODELSCOPE
    return (
        gr.update(visible=is_ms),
        gr.update(visible=not is_ms),
        gr.update(value="") if is_ms else gr.skip(),
    )


def start_workflow_ui(
    query: str,
    example_name: str,
    uploaded_files,
    mode: str,
    ms_model: str,
    requested_base_url: str,
    requested_model: str,
    user_api_key: str,
    browser_sid: str,
    request: gr.Request,
):
    base_url, model, api_key = _resolve_connection(
        mode, ms_model, requested_base_url, requested_model, user_api_key,
        _effective_session_id(browser_sid, request),
    )
    yield from start_interactive_workflow_stream(
        query, example_name, uploaded_files, base_url, model, api_key
    )


def request_custom_reflections_ui(
    state: dict,
    mode: str,
    ms_model: str,
    requested_base_url: str,
    requested_model: str,
    user_api_key: str,
    browser_sid: str,
    request: gr.Request,
):
    _, _, api_key = _resolve_connection(
        mode, ms_model, requested_base_url, requested_model, user_api_key,
        _effective_session_id(browser_sid, request),
    )
    return request_custom_reflections(state, api_key)


def confirm_reflection_and_retry_ui(
    state: dict,
    selection: str,
    edited_reflection: str,
    user_diagnosis: str,
    user_plan: str,
    mode: str,
    ms_model: str,
    requested_base_url: str,
    requested_model: str,
    user_api_key: str,
    browser_sid: str,
    request: gr.Request,
):
    _, _, api_key = _resolve_connection(
        mode, ms_model, requested_base_url, requested_model, user_api_key,
        _effective_session_id(browser_sid, request),
    )
    return confirm_reflection_and_retry(
        state, selection, edited_reflection, user_diagnosis, user_plan, api_key
    )



with gr.Blocks(title=PROJECT_TITLE, delete_cache=(3600, 21600)) as demo:
    gr.HTML(NAV, js_on_load=language_script(), apply_default_css=False, elem_classes=["rsmem-localized"])

    with gr.Tab("Run RSMeM"):
        session_state = gr.State({})
        browser_session = gr.BrowserState("", storage_key="rsmem_session")
        gr.HTML(hero_html(len(EXAMPLES) - 1), elem_id="studio-intro", apply_default_css=False, elem_classes=["rsmem-localized"])
        gr.HTML(walkthrough_html(), js_on_load=WALKTHROUGH_JS, elem_id="studio-walkthrough", apply_default_css=False, elem_classes=["rsmem-localized"])
        gr.HTML(WORKSPACE_HEADING, apply_default_css=False, elem_classes=["rsmem-localized"])

        with gr.Group(elem_id="input-card"):
            with gr.Row(elem_id="attach-row"):
                with gr.Column(scale=6, min_width=300, elem_id="question-col"):
                    gr.Markdown(
                        "### 01 / Ask your question\n"
                        "Describe what you want to find out.",
                        elem_classes=["card-heading"],
                    )
                    query_input = gr.Textbox(
                        label="Query",
                        show_label=False,
                        value="",
                        placeholder="Type your remote-sensing question…",
                        lines=5,
                        elem_id="query-input",
                    )
                    with gr.Accordion(
                        f"Need an idea? Explore {len(EXAMPLES) - 1} prepared cases",
                        open=False,
                        elem_id="example-picker",
                    ):
                        example_input = gr.Radio(
                            show_label=False,
                            choices=[name for name, item in EXAMPLES.items() if item["directory"]],
                            value=None,
                            elem_id="example-choices",
                        )
                with gr.Column(scale=6, min_width=300, elem_id="imagery-col"):
                    with gr.Row(elem_id="imagery-head"):
                        gr.Markdown(
                            "### 02 / Add your imagery\n"
                            "GeoTIFF / TIFF / PNG / JPEG · multiple files",
                            elem_classes=["card-heading"],
                            elem_id="imagery-heading",
                        )
                        raster_add = gr.UploadButton(
                            "Upload imagery",
                            file_count="multiple",
                            file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                            type="filepath",
                            size="sm",
                            variant="secondary",
                            elem_id="raster-add",
                        )
                    with gr.Column(elem_id="preview-stack"):
                        raster_preview_output = gr.Image(
                            value=None,
                            type="numpy",
                            interactive=False,
                            show_label=False,
                            elem_id="raster-preview",
                            buttons=["fullscreen"],
                        )
                        raster_drop = gr.File(
                            show_label=False,
                            file_count="multiple",
                            file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                            type="filepath",
                            elem_id="raster-drop",
                        )
                    raster_input = gr.File(
                        show_label=False,
                        file_count="multiple",
                        file_types=[".tif", ".tiff", ".png", ".jpg", ".jpeg"],
                        type="filepath",
                        elem_id="raster-upload",
                        visible=False,
                    )
                    case_files = gr.Radio(
                        show_label=False,
                        choices=[],
                        value=None,
                        visible=False,
                        elem_id="case-files",
                    )
                    case_paths = gr.State([])
        example_input.input(
            load_example_for_ui,
            inputs=[example_input],
            outputs=[
                query_input, raster_preview_output, raster_input, case_files, case_paths,
            ],
        )
        raster_input.upload(
            select_uploaded_source,
            inputs=[raster_input, example_input],
            outputs=[
                raster_preview_output, example_input, raster_input, case_files, case_paths,
            ],
        )
        raster_input.clear(
            select_uploaded_source,
            inputs=[raster_input, example_input],
            outputs=[
                raster_preview_output, example_input, raster_input, case_files, case_paths,
            ],
        )
        raster_input.delete(
            select_uploaded_source,
            inputs=[raster_input, example_input],
            outputs=[
                raster_preview_output, example_input, raster_input, case_files, case_paths,
            ],
        )
        raster_input.select(
            preview_selected_upload,
            inputs=[raster_input],
            outputs=[raster_preview_output],
        )
        raster_add.upload(
            append_uploaded_sources,
            inputs=[raster_add, raster_input, example_input],
            outputs=[
                raster_input, raster_preview_output, example_input, case_files, case_paths,
            ],
        )
        raster_drop.upload(
            drop_uploaded_sources,
            inputs=[raster_drop, raster_input, example_input],
            outputs=[
                raster_drop, raster_input, raster_preview_output,
                example_input, case_files, case_paths,
            ],
        )
        case_files.input(
            preview_selected_case,
            inputs=[case_files, case_paths],
            outputs=[raster_preview_output],
        )
        with gr.Group(elem_id="model-service"):
            gr.Markdown(
                "### 03 / Connect a model\n"
                "Use your ModelScope quota or your own API.",
                elem_classes=["card-heading"],
            )
            connection_mode = gr.Radio(
                choices=[MODE_OPENAI, MODE_MODELSCOPE],
                value=MODE_OPENAI,
                show_label=False,
                elem_id="connection-mode",
            )
            with gr.Column(visible=False, elem_id="ms-pane") as ms_pane:
                ms_status = gr.HTML(elem_classes=["rsmem-localized"])
                with gr.Row():
                    refresh_ms_button = gr.Button(
                        "Refresh sign-in status", size="sm", elem_id="ms-refresh"
                    )
                    signout_button = gr.Button(
                        "Sign out", size="sm", visible=False, elem_id="ms-signout"
                    )
                ms_model_input = gr.Dropdown(
                    label="ModelScope model",
                    choices=MODELSCOPE_MODEL_CHOICES,
                    value=MODELSCOPE_DEFAULT_MODEL,
                    allow_custom_value=True,
                    info="Pick an API-Inference model or type any model identifier.",
                )
            with gr.Column(visible=True, elem_id="oa-pane") as oa_pane:
                with gr.Row():
                    base_url_input = gr.Textbox(
                        label="OpenAI-compatible Base URL",
                        value=DEFAULT_BASE_URL,
                        max_lines=1,
                        placeholder="https://provider.example/v1",
                        info="Only public HTTPS endpoints are accepted.",
                    )
                    model_input = gr.Textbox(
                        label="Model",
                        value=DEFAULT_MODEL,
                        max_lines=1,
                        placeholder="Provider model identifier",
                        info="Use the exact model identifier exposed by your provider.",
                    )
                api_key_input = gr.Textbox(
                    label="API key",
                    type="password",
                    placeholder="Enter your key for this request",
                    info="Used for this run only and never saved by the Studio.",
                )

        with gr.Row(elem_id="run-row"):
            run_button = gr.Button(
                "Analyze with RSMeM   ↗", variant="primary", elem_id="run-rsmem-button"
            )

        api_status = gr.Markdown(
            '<span class="rsmem-localized">Choose data and connect a model to begin. Live progress, your answer, and available downloads will appear below. '
            'Your model authorization is required.</span>',
            elem_id="workflow-status",
        )
        timeline_output = gr.HTML(elem_id="workflow-timeline", js_on_load=RUN_TIMER_JS)

        with gr.Column(visible=False, elem_id="rsmem-workspace") as workspace_group:
            gr.Markdown(
                "## Summary",
                elem_classes=["section-title"],
            )
            final_output = gr.Markdown(
                "Run an example or upload your own rasters to see the result.",
                elem_id="current-result",
                elem_classes=["workspace-content"],
            )

            gr.Markdown(
                "## Agent trace\nOpen a tool call to inspect its inputs, observations, or errors.",
                elem_classes=["section-title"],
            )
            attempts_output = gr.HTML(elem_id="agent-trace-body")

            gr.Markdown(
                "## Evidence & artifacts\nExplore the retrieved knowledge and download files created by this run.",
                elem_classes=["section-title"],
            )
            with gr.Accordion("Retrieved knowledge", open=False) as hkg_accordion:
                hkg_output = gr.Markdown(elem_classes=["workspace-content"])
            evaluation_output = gr.Markdown(visible=False, elem_id="evaluation-card")
            with gr.Accordion("Generated files", open=True):
                artifact_output = gr.File(
                    label="Download generated artifacts",
                    file_count="multiple",
                    elem_id="artifact-files",
                )
            with gr.Accordion("Debug trace", open=False):
                gr.Markdown(
                    "Stage-by-stage log of this run — knowledge scores, planned tools, "
                    "real tool observations, model verdicts, and full tracebacks on "
                    "errors. Attach it when reporting a problem.",
                    elem_classes=["upload-heading"],
                )
                trace_output = gr.File(
                    label="Download run trace (JSON)",
                    file_count="single",
                    elem_id="trace-file",
                )

            with gr.Group(visible=False, elem_id="decision-panel") as decision_group:
                gr.Markdown("### Review and confirm the result")
                decision_output = gr.Markdown()
                with gr.Row():
                    accept_button = gr.Button(
                        "✓ Accept result and finish",
                        variant="primary",
                        elem_id="accept-result-button",
                    )
                    reflect_button = gr.Button(
                        "↻ Incorrect — reflect and retry",
                        elem_id="retry-result-button",
                    )

        with gr.Group(
            visible=False, elem_id="reflection-panel"
        ) as candidate_group:
            gr.Markdown(
                "### Reflection candidates\n\nCompare the three suggestions below, or write your own. "
                "The selected text remains editable before the next attempt."
            )
            candidate_1_output = gr.Textbox(
                label="Candidate 1", lines=5, interactive=False,
            )
            candidate_2_output = gr.Textbox(
                label="Candidate 2", lines=5, interactive=False,
            )
            candidate_3_output = gr.Textbox(
                label="Candidate 3", lines=5, interactive=False,
            )
            candidate_choice = gr.Radio(
                choices=["Candidate 1", "Candidate 2", "Candidate 3", "Write my own"],
                label="Choose one option",
            )
            with gr.Row():
                user_diagnosis = gr.Textbox(
                    label="Your diagnosis (required for Write my own)", lines=3
                )
                user_plan = gr.Textbox(
                    label="Your next-attempt plan (required for Write my own)", lines=3
                )
            edited_reflection = gr.Textbox(
                label="Final editable reflection to write into FAM", lines=6
            )
            retry_button = gr.Button(
                "Confirm this reflection and execute the next attempt", variant="primary"
            )

        workflow_outputs = [
            session_state,
            api_status,
            timeline_output,
            hkg_output,
            hkg_accordion,
            attempts_output,
            evaluation_output,
            decision_output,
            decision_group,
            candidate_group,
            candidate_1_output,
            candidate_2_output,
            candidate_3_output,
            candidate_choice,
            user_diagnosis,
            user_plan,
            edited_reflection,
            final_output,
            artifact_output,
            workspace_group,
            trace_output,
        ]
        # These short session/UI operations must work before a queued model
        # run is available. In the hosted Studio, queue/join can fail at the
        # gateway with an empty response even when the app is serving config.
        # Use Gradio's direct JSON route for these events; keep execution queued.
        connection_mode.change(
            switch_connection_mode,
            inputs=[connection_mode],
            outputs=[ms_pane, oa_pane, api_key_input],
            queue=False,
        )
        refresh_ms_button.click(
            refresh_ms_session,
            inputs=[browser_session],
            outputs=[ms_status, signout_button, browser_session],
            queue=False,
        )
        signout_button.click(
            sign_out_ms,
            inputs=[browser_session],
            outputs=[ms_status, signout_button, browser_session],
            queue=False,
        )
        demo.load(
            refresh_ms_session,
            inputs=[browser_session],
            outputs=[ms_status, signout_button, browser_session],
            queue=False,
        )
        run_button.click(
            start_workflow_ui,
            inputs=[
                query_input,
                example_input,
                raster_input,
                connection_mode,
                ms_model_input,
                base_url_input,
                model_input,
                api_key_input,
                browser_session,
            ],
            outputs=workflow_outputs,
            show_progress="minimal",
        )
        accept_button.click(
            accept_custom_result,
            inputs=[session_state],
            outputs=workflow_outputs,
        )
        reflect_button.click(
            request_custom_reflections_ui,
            inputs=[
                session_state,
                connection_mode,
                ms_model_input,
                base_url_input,
                model_input,
                api_key_input,
                browser_session,
            ],
            outputs=workflow_outputs,
        )
        candidate_choice.change(
            select_reflection_candidate,
            inputs=[candidate_choice, session_state, user_diagnosis, user_plan],
            outputs=[edited_reflection],
        )
        user_diagnosis.change(
            select_reflection_candidate,
            inputs=[candidate_choice, session_state, user_diagnosis, user_plan],
            outputs=[edited_reflection],
        )
        user_plan.change(
            select_reflection_candidate,
            inputs=[candidate_choice, session_state, user_diagnosis, user_plan],
            outputs=[edited_reflection],
        )
        retry_button.click(
            confirm_reflection_and_retry_ui,
            inputs=[
                session_state,
                candidate_choice,
                edited_reflection,
                user_diagnosis,
                user_plan,
                connection_mode,
                ms_model_input,
                base_url_input,
                model_input,
                api_key_input,
                browser_session,
            ],
            outputs=workflow_outputs,
        )
        gr.HTML(FOOTER, apply_default_css=False, elem_classes=["rsmem-localized"])

    with gr.Tab("Open-source resources"):
        learn_resource = (
            f"- [ModelScope Learn article]({LEARN_URL})"
            if LEARN_URL
            else "- ModelScope Learn article: **pending publication by the team leader**"
        )
        gr.Markdown(
            f"""
### Reproduce and extend RSMeM

- [Source code on GitHub]({GITHUB_URL})
{learn_resource}
- [Earth-Bench example-data source](https://huggingface.co/datasets/Sssunset/Earth-Bench) (Apache-2.0)
- Code license: MIT · bundled example-data license: Apache-2.0

### Cite the paper

If RSMeM supports your research, please cite the arXiv paper:

```bibtex
{PAPER_BIBTEX}
```

[Paper abstract](https://arxiv.org/abs/2607.24772) ·
[PDF](https://arxiv.org/pdf/2607.24772) ·
[DOI](https://doi.org/10.48550/arXiv.2607.24772)
            """
        )
        with gr.Accordion("Method overview and paper figures", open=False, elem_id="method-overview"):
            gr.Markdown(
                "RSMeM combines **Hierarchical Knowledge Grounding (HKG)** with "
                "**Failure-Aware Experience Refinement (FAR)**. HKG supplies structured "
                "geoscience guidance before tool use; FAR turns a confirmed failed trajectory "
                "into reusable memory for the next attempt."
            )
            gr.Image("assets/figure1.png", label="Remote-sensing agent paradigms")
            gr.Image("assets/figure2.png", label="RSMeM workflow")

    with gr.Tab("Challenge submission"):
        gr.Markdown(
            f"""
### Project title

**{PROJECT_TITLE}**

### Project Team

**Contributors**

- Bingxian Wu — Lead developer
- [Jin Bai](https://kbaijin.github.io/) — Frontend development

**Advisors**

- [Tang Liu](https://scholar.google.com/citations?user=AtyxShQAAAAJ)
- [Zonghao Guo](https://guozonghao96.github.io/)

### Project abstract ({PROJECT_ABSTRACT_WORDS} words)

{PROJECT_ABSTRACT}
            """
        )


    # Localize interface labels and static copy, never live model answers or evidence.
    # The browser translator also skips input/textarea values and file-name cells.
    _live_markdown_ids = {
        component._id for component in (
            api_status, final_output, hkg_output, evaluation_output, decision_output,
            case_files,
        )
    }
    for _component in demo.blocks.values():
        if isinstance(_component, gr.Accordion):
            _component.elem_classes.append("rsmem-localized-label")
        elif isinstance(_component, (gr.Markdown, gr.Textbox, gr.Button, gr.Radio, gr.Dropdown, gr.File, gr.Image, gr.UploadButton)):
            if _component._id not in _live_markdown_ids:
                _component.elem_classes.append("rsmem-localized")


def _oauth_page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>RSMeM · ModelScope sign-in</title><style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "background:#f7f7f7;color:#242424;display:flex;align-items:center;"
        "justify-content:center;min-height:90vh;margin:0}"
        ".card{background:#fff;border:1px solid #e3e3e3;border-radius:14px;"
        "padding:28px 32px;max-width:460px;line-height:1.6}"
        "h1{font-size:18px;margin:0 0 8px}p{font-size:14px;color:#555;margin:6px 0}"
        "</style></head><body><div class='card'>"
        f"<h1>{title}</h1>{body}</div></body></html>"
    )


def _request_redirect_uri(headers) -> str:
    return msoauth.resolve_redirect_uri(
        headers.get("x-forwarded-host"), headers.get("host")
    )


def _session_cookie_attributes(host: str | None) -> str:
    """Local dev gets a plain Lax cookie; any deployed host gets the
    partitioned cross-site variant so the cookie survives inside the
    modelscope.cn iframe under third-party-cookie blocking (CHIPS)."""
    bare = (host or "").split(",")[0].strip().lower()
    if bare.split(":")[0] in {"127.0.0.1", "localhost", "0.0.0.0", "testserver"}:
        return "; HttpOnly; Path=/; Max-Age=604800; SameSite=Lax"
    return "; HttpOnly; Path=/; Max-Age=604800; SameSite=None; Secure; Partitioned"


def create_app():
    """Wrap the Gradio app in FastAPI so the OAuth routes share the origin."""
    # Serving the Studio is the deployment decision that mcp_runtime's gate
    # defers to; the gate stays off by default so importing this package or
    # running the suite never inherits arbitrary execution. setdefault rather
    # than assignment, so an operator can still pin the flag to 0.
    os.environ.setdefault("RSMEM_ENABLE_CODE_TOOLS", "1")
    try:
        swept = run_storage.sweep_local_runs()
        if swept:
            print(f"[run-storage] swept {swept} stale local run director(ies)", flush=True)
        swept_downloads = run_storage.sweep_downloads()
        if swept_downloads:
            print(f"[run-storage] swept {swept_downloads} stale download(s)", flush=True)
    except Exception as exc:  # noqa: BLE001 - never fail serving because a sweep failed
        print(f"[run-storage] sweep skipped ({exc})", flush=True)
    fastapi_app = FastAPI()

    @fastapi_app.middleware("http")
    async def ensure_session_cookie(request: Request, call_next):
        response = await call_next(request)
        # On modelscope.cn the "/" document is the platform shell and never
        # reaches this app; "/config" is the first same-origin request the
        # shell makes, so the cookie is set there (and only there, to avoid
        # racing session ids across parallel requests).
        if request.url.path in ("/", "/config") and not request.cookies.get(
            msoauth.SESSION_COOKIE
        ):
            session_id = secrets.token_urlsafe(24)
            attributes = _session_cookie_attributes(
                request.headers.get("x-forwarded-host") or request.headers.get("host")
            )
            response.headers.append(
                "set-cookie", f"{msoauth.SESSION_COOKIE}={session_id}{attributes}"
            )
        return response

    @fastapi_app.get("/rsmem/gpu-status")
    async def gpu_status():
        """Read-only bootstrap/worker status; no credentials are exposed."""
        from fastapi.responses import JSONResponse

        manager = get_gpu_service_manager()
        if manager is None:
            return JSONResponse({"startup_running": False, "started": {}, "skipped": {}})
        return JSONResponse(manager.status())

    @fastapi_app.get("/rsmem/gpu-logs/{service}")
    async def gpu_logs(service: str):
        """Read-only worker startup log. Only known service names are served."""
        from fastapi.responses import PlainTextResponse

        try:
            return PlainTextResponse(read_worker_log(service))
        except KeyError:
            return PlainTextResponse(f"unknown service: {service}", status_code=404)

    @fastapi_app.get("/oauth/login")
    async def oauth_login(request: Request):
        if not msoauth.oauth_configured():
            return PlainTextResponse(
                "ModelScope OAuth is not configured on this deployment.", status_code=503
            )
        nonce = request.query_params.get("nonce", "")
        if not msoauth.SESSION_STORE.nonce_valid(nonce):
            return HTMLResponse(
                _oauth_page(
                    "Sign-in link expired",
                    "<p>Go back to the RSMeM tab, press <b>Refresh sign-in status</b>, "
                    "and use the new sign-in button.</p>",
                ),
                status_code=400,
            )
        try:
            redirect_uri = _request_redirect_uri(request.headers)
        except ValueError:
            return HTMLResponse(
                _oauth_page(
                    "Callback domain is not configured",
                    "<p>This deployment could not determine the Studio's own "
                    "access domain, and ModelScope refuses a redirect to any "
                    "other host. The Studio owner needs to enable OAuth in the "
                    "Studio settings and redeploy.</p>",
                ),
                status_code=500,
            )
        authorize_url = msoauth.build_authorize_url(
            msoauth.oauth_client_id(), redirect_uri, nonce
        )
        LOGGER.info(
            "OAuth authorize request: client_id=%s redirect_uri=%s endpoint=%s",
            msoauth.oauth_client_id(),
            redirect_uri,
            authorize_url.split("?", 1)[0],
        )
        return RedirectResponse(authorize_url)

    @fastapi_app.get("/oauth/callback")
    async def oauth_callback(request: Request):
        params = request.query_params
        if params.get("error"):
            detail = html_lib.escape(
                (params.get("error_description") or params.get("error") or "").strip()
            )
            return HTMLResponse(
                _oauth_page(
                    "Sign-in was not completed",
                    "<p>The authorization was cancelled or rejected. You can close "
                    "this tab and try again from the RSMeM page.</p>"
                    + (f"<p>ModelScope reported: {detail}</p>" if detail else ""),
                ),
                status_code=400,
            )
        code = params.get("code", "")
        state = params.get("state", "")
        session_id = msoauth.SESSION_STORE.consume_nonce(state)
        if not code or not session_id:
            return HTMLResponse(
                _oauth_page(
                    "Sign-in link expired",
                    "<p>Go back to the RSMeM tab, press <b>Refresh sign-in status</b>, "
                    "and use the new sign-in button.</p>",
                ),
                status_code=400,
            )
        try:
            redirect_uri = _request_redirect_uri(request.headers)
            payload = msoauth.exchange_code(
                code, msoauth.oauth_client_id(), msoauth.oauth_client_secret(), redirect_uri
            )
        except ValueError as exc:
            return HTMLResponse(
                _oauth_page("Sign-in failed", f"<p>{html_lib.escape(str(exc))}</p>"),
                status_code=502,
            )
        userinfo = msoauth.fetch_userinfo(payload["access_token"])
        msoauth.SESSION_STORE.store_token(session_id, payload, userinfo)
        user = html_lib.escape(msoauth.display_name(userinfo))
        return HTMLResponse(
            _oauth_page(
                "✅ Connected to ModelScope",
                f"<p>Signed in as <b>{user}</b>. RSMeM will now use your own "
                "API-Inference quota for this browser session.</p>"
                "<p>Return to the RSMeM tab, press <b>Refresh sign-in status</b>, "
                "and close this tab.</p>",
            )
        )

    demo.queue()
    downloads = run_storage.downloads_root()
    try:
        downloads.mkdir(parents=True, exist_ok=True)
        allowed = [str(downloads)]
    except OSError as exc:
        print(f"[run-storage] downloads dir unavailable ({exc})", flush=True)
        allowed = []
    gr.mount_gradio_app(
        fastapi_app, demo, path="/",
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"), css=CSS,
        i18n=gr.I18n(**{locale: ENGLISH_CONTROLS for locale in ("en", "zh", "zh-CN", "zh-TW")}),
        allowed_paths=allowed,
    )
    return fastapi_app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        port=int(os.getenv("GRADIO_SERVER_PORT", os.getenv("PORT", "7860"))),
    )
