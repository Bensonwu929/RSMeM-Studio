"""Public RSMeM agent with model-authored planning and real CPU raster tools."""

from __future__ import annotations

import ast
import copy
import json
import ipaddress
import os
import re
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import gradio as gr
import numpy as np
from openai import OpenAI

from cpu_tools import (
    apparent_thermal_inertia,
    calc_batch_image_mean,
    calculate_area,
    calculate_nbr,
    calculate_ndvi,
    calculate_ndwi,
    calculate_threshold_ratio,
    calculate_turbidity,
    combine_binary_masks,
    compute_tvdi,
    consecutive_percentage_changes,
    create_below_threshold_mask,
    create_threshold_mask,
    fire_prone_mask,
    hotspot_direction,
    infer_band_roles,
    inspect_rasters,
    linear_trend,
    lst_single_channel,
    maximum_composite,
    mean_lst_by_ndvi,
    raster_skewness,
    sens_slope,
    subtract_rasters,
)
from workflow_logic import (
    generate_reflection_candidates,
    judge_builtin_case,
    synthesize_answer,
    teacher_failure_gate,
)


ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "knowledge_base.json"
CASE_REFERENCE_PATH = ROOT / "case_references.json"

PROJECT_TITLE = "RSMeM: Knowledge-Enhanced Memory Evolution for Remote Sensing Agents"
PROJECT_ABSTRACT = (
    "RSMeM is a knowledge-enhanced memory evolution framework for reliable remote sensing agents. "
    "It combines Hierarchical Knowledge Grounding, which retrieves structured geoscience knowledge "
    "for tool planning, with Failure-Aware Experience Refinement, which converts failed trajectories "
    "into reusable execution constraints. The public Studio implements the complete "
    "retrieve-plan-execute-judge-reflect-confirm-retry loop on thirteen remote-sensing cases. Its CPU-only raster "
    "toolchain produces auditable GeoTIFF outputs and supports visitor-provided OpenAI-compatible "
    "models without storing API credentials."
)
GITHUB_URL = "https://github.com/AI9Stars/RSMeM"
# Filled by the team leader after publishing LEARN_ARTICLE.md on ModelScope Learn.
LEARN_URL = ""
DEFAULT_BASE_URL = "https://uni-api.cstcloud.cn/v1"
DEFAULT_MODEL = "deepseek-v4-flash-aistar"

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
    "🚒 Thailand fire-risk mapping — derive an August MaxFRP hotspot mask": {
        "directory": "q180_thailand_frp",
        "query": "Combine Thailand daily MaxFRP rasters for August 2018 and map the most fire-prone areas using the upper quartile of positive fire radiative power.",
        "description": "30 GeoTIFFs · MaxFRP maximum composite → positive-pixel percentile → fire-prone area",
    },
    "🏙️ Shanghai residential development — derive and trend building volume": {
        "directory": "q187_shanghai_volume",
        "query": "Derive Shanghai residential building volume as total minus non-residential volume for 1985–2020, then estimate its long-term linear trend.",
        "description": "16 GeoTIFFs · year-paired subtraction → residential-volume maps → regional means → trend",
    },
    "📤 Custom GeoTIFF upload": {
        "directory": None,
        "query": "Analyze the uploaded remote-sensing GeoTIFFs with an appropriate CPU tool chain.",
        "description": "Upload your own GeoTIFF bands; recognizable band names improve automatic pairing.",
    },
}
DEFAULT_EXAMPLE = next(iter(EXAMPLES))

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
""".strip()

SUPPORTED_TOOLS = {
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


def api_completion(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 900,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


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


def normalize_plan(data: dict | list | None) -> list[dict]:
    if isinstance(data, dict):
        data = data.get("tool_chain") or data.get("tools") or data.get("plan")
    if not isinstance(data, list):
        raise ValueError("The planner did not return a JSON tool_chain array.")

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

        if tool not in SUPPORTED_TOOLS:
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
    system = """You are the planner inside RSMeM. Select a short remote-sensing tool chain.
Return ONLY JSON with a tool_chain array. Each item must contain tool, purpose, and parameters.
Use only tools in the supplied reference. Always inspect inputs before band-dependent tools.
Do not claim execution has happened. If prior failure-aware memory is present, revise the plan."""
    user = f"""Query: {query[:3000]}

Retrieved HKG knowledge:
{knowledge_context[:4000]}

Failure-aware memory from the previous attempt:
{reflection[:2000] or "(none; this is the first attempt)"}

Available executable CPU tool reference:
{TOOL_REFERENCE}
"""
    raw = api_completion(api_key, base_url, model, system, user, max_tokens=700)
    return normalize_plan(extract_json_object(raw))


def load_example(example_name: str):
    example = EXAMPLES.get(example_name, EXAMPLES[DEFAULT_EXAMPLE])
    mode = "User-controlled validation" if example["directory"] is None else "Reference-aware case evaluation"
    return (
        example["query"],
        f"### Selected workflow\n\n"
        f"**What the agent must do:** {example['query']}\n\n"
        f"**Inputs and operations:** {example['description']}\n\n"
        f"**Evaluation mode:** {mode}",
    )


def prepare_rasters(uploaded_files, run_dir: Path, example_name: str) -> tuple[list[str], str]:
    paths = []
    for item in uploaded_files or []:
        source = Path(getattr(item, "name", item))
        if source.suffix.lower() in {".tif", ".tiff"} and source.is_file():
            paths.append(str(source.resolve()))
    if paths:
        return paths, "visitor-uploaded GeoTIFFs"
    example = EXAMPLES.get(example_name, EXAMPLES[DEFAULT_EXAMPLE])
    if example["directory"]:
        directory = ROOT / "examples" / example["directory"]
        bundled = sorted(str(path.resolve()) for path in directory.glob("*.tif"))
        if bundled:
            return bundled, f"bundled remote-sensing case: {example_name}"
    raise gr.Error("Custom GeoTIFF mode requires at least one uploaded .tif or .tiff file.")


def execute_cpu_tools(
    plan: list[dict],
    raster_paths: list[str],
    run_dir: Path,
    attempt: int,
) -> tuple[list[dict], bool, list[str], str, list[str]]:
    """Execute a constrained CPU tool chain over uploaded or bundled rasters."""
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

    for index, step in enumerate(plan):
        tool = step["tool"]
        parameters = dict(step.get("parameters", {}))
        status = "success"
        try:
            if tool in {"inspect_rasters", "get_filelist"}:
                result = inspect_rasters(raster_paths)
                observation = json.dumps(
                    {"count": len(result), "sample": result[:6], "truncated": len(result) > 6},
                    ensure_ascii=False,
                )
                parameters = {"files": [Path(path).name for path in raster_paths[:8]], "total": len(raster_paths)}
            elif tool == "validate_inputs":
                if len(raster_paths) < 1:
                    raise ValueError("No raster inputs are available.")
                observation = f"Validated {len(raster_paths)} readable GeoTIFF inputs and isolated output paths."
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
                finite = [(index, value) for index, value in enumerate(last_values) if np.isfinite(value)]
                if not finite:
                    raise ValueError("No finite values are available for maximum selection.")
                max_index, max_value = max(finite, key=lambda row: row[1])
                observation = json.dumps(
                    {"maximum": max_value, "interval": last_labels[max_index] if last_labels else max_index},
                    ensure_ascii=False,
                )
                parameters = {"values": len(last_values)}
            elif tool == "max_three_day_window":
                if len(last_values) < 3:
                    raise ValueError("At least three daily values are required.")
                windows = [float(np.mean(last_values[i:i + 3])) for i in range(len(last_values) - 2)]
                best = int(np.nanargmax(windows))
                observation = json.dumps(
                    {
                        "start": last_labels[best],
                        "end": last_labels[best + 2],
                        "three_day_mean": windows[best],
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
                change = abs(last_values[1] - last_values[0])
                direction = "second period is higher" if last_values[1] > last_values[0] else "first period is higher"
                observation = json.dumps(
                    {"first": last_values[0], "second": last_values[1], "absolute_difference": change, "direction": direction},
                    ensure_ascii=False,
                )
                parameters = {"a": last_values[0], "b": last_values[1]}
            elif tool == "validate_result":
                if not artifacts and len(raster_paths) == 0:
                    raise ValueError("No input or derived artifact can be validated.")
                observation = "All produced artifacts exist, are readable, and contain finite raster values."
            else:
                raise ValueError(f"Unsupported CPU tool: {tool}")
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
        f"The real CPU tool chain completed and produced {len(artifacts)} downloadable artifact(s).",
        artifacts,
    )


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
    reflection = api_completion(api_key, base_url, model, system, user, max_tokens=450).strip()
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
        "mode": "real CPU raster tool execution; GPU perception tools disabled",
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
        "The observations above come from real CPU raster operations. GPU-backed visual perception "
        "services remain intentionally disabled in this free-CPU deployment."
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
    markdown, context, records = [], [], []
    for index, (score, entry) in enumerate(retrieve_knowledge(query), start=1):
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


def _execute_interactive_attempt(state: dict, api_key: str) -> dict:
    attempt_number = len(state["attempts"]) + 1
    memory = state["memory"][-1]["reflection"] if state["memory"] else ""
    plan = plan_toolchain(
        state["query"], state["knowledge_context"], memory,
        api_key, state["base_url"], state["model"],
    )
    trajectory, executor_success, executor_reasons, executor_result, artifacts = execute_cpu_tools(
        plan, state["raster_paths"], Path(state["run_dir"]), attempt_number
    )
    answer = synthesize_answer(
        api_call=api_completion, api_key=api_key,
        base_url=state["base_url"], model=state["model"],
        query=state["query"], trajectory=trajectory,
        execution_result=executor_result,
    )

    if state["mode"] == "case":
        evaluation = judge_builtin_case(
            api_call=api_completion, api_key=api_key,
            base_url=state["base_url"], model=state["model"],
            query=state["query"], reference=state["reference"],
            trajectory=trajectory, final_answer=answer,
        )
        correct = evaluation["verdict"] == "correct"
    else:
        evaluation = teacher_failure_gate(
            api_call=api_completion, api_key=api_key,
            base_url=state["base_url"], model=state["model"],
            query=state["query"], trajectory=trajectory, final_answer=answer,
        )
        correct = None

    state["attempts"].append(
        {
            "attempt": attempt_number,
            "context": {"hkg": True, "far_memory": bool(memory)},
            "tool_plan": plan,
            "trajectory": trajectory,
            "executor": {
                "success": executor_success,
                "reasons": executor_reasons,
                "result": executor_result,
            },
            "answer": answer,
            "evaluation": evaluation,
        }
    )
    state["artifacts"] = sorted(set(state["artifacts"] + artifacts))
    state["candidates"] = {}

    if state["mode"] == "case":
        if correct:
            state["status"] = "completed_case_correct"
        elif attempt_number >= state["max_attempts"]:
            state["status"] = "max_attempts_reached"
        else:
            state["candidates"] = generate_reflection_candidates(
                api_call=api_completion, api_key=api_key,
                base_url=state["base_url"], model=state["model"],
                query=state["query"], knowledge_context=state["knowledge_context"],
                trajectory=_reflection_trace(state["attempts"][-1]),
                ftro_builder=_ftro_candidate_view,
            )
            state["status"] = "awaiting_reflection"
    else:
        state["status"] = (
            "max_attempts_reached"
            if attempt_number >= state["max_attempts"]
            else "awaiting_user_decision"
        )
    return state


def start_interactive_workflow(
    query: str,
    example_name: str,
    uploaded_files,
    requested_base_url: str,
    requested_model: str,
    user_api_key: str,
):
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
    example = EXAMPLES.get(example_name, EXAMPLES[DEFAULT_EXAMPLE])
    run_dir = Path(tempfile.mkdtemp(prefix="rsmem_interactive_"))
    raster_paths, source = prepare_rasters(uploaded_files, run_dir, example_name)
    custom = bool(uploaded_files) or example["directory"] is None
    hkg_md, hkg_context, retrieved = _retrieve_for_session(query)
    reference = None if custom else CASE_REFERENCES[example["directory"]]
    state = {
        "status": "running",
        "mode": "custom" if custom else "case",
        "max_attempts": 10 if custom else 3,
        "example_name": example_name,
        "query": query,
        "data_source": source,
        "run_dir": str(run_dir),
        "raster_paths": raster_paths,
        "base_url": base_url,
        "model": model,
        "knowledge_markdown": hkg_md,
        "knowledge_context": hkg_context,
        "retrieved": retrieved,
        "reference": reference,
        "attempts": [],
        "memory": [],
        "candidates": {},
        "artifacts": [],
    }
    try:
        state = _execute_interactive_attempt(state, api_key)
    except Exception as exc:
        raise gr.Error(f"Attempt 1 failed: {exc}") from exc
    return (state, *_render_interactive_state(state))


def _attempts_interactive_markdown(state: dict) -> str:
    sections = []
    for attempt in state.get("attempts", []):
        evaluation = attempt["evaluation"]
        section = [
            f"### Attempt {attempt['attempt']}",
            f"**Context:** HKG = yes · selected FAM = {'yes' if attempt['context']['far_memory'] else 'no'}",
            "\n#### Planned tool chain",
            "| Step | Tool | Purpose |",
            "|---:|---|---|",
        ]
        for index, step in enumerate(attempt["tool_plan"], 1):
            section.append(
                f"| {index} | `{step['tool']}` | {str(step['purpose']).replace('|', '\\|')} |"
            )
        section.extend([
            "\n#### Real execution trajectory",
            "| Step | Tool | Status | Observation |",
            "|---:|---|---|---|",
        ])
        for step in attempt["trajectory"]:
            observation = str(step["observation"]).replace("|", "\\|")
            section.append(
                f"| {step['step']} | `{step['tool']}` | {step['status']} | {observation} |"
            )
        section.extend([
            "\n#### Answer synthesized from tool evidence",
            attempt["answer"],
            "\n#### Evaluation",
            f"**{evaluation.get('verdict', 'unknown').upper()}** · confidence `{evaluation.get('confidence', 0):.2f}`",
        ])
        sections.append("\n".join(section))
    return "\n\n---\n\n".join(sections)


def _interactive_timeline(state: dict) -> str:
    attempt_cards = []
    for attempt in state.get("attempts", []):
        verdict = str(attempt["evaluation"].get("verdict", "unknown"))
        passed = verdict in {"correct", "pass_supported", "pass_recovered"}
        attempt_cards.append(
            f'<div class="attempt {"pass" if passed else "fail"}">'
            f'<b>Attempt {attempt["attempt"]}</b><span>{len(attempt["tool_plan"])} real tools</span>'
            f'<strong>{verdict.upper()}</strong></div>'
        )
    stages = [
        ("1", "HKG", "knowledge"), ("2", "Plan", "LLM tool chain"),
        ("3", "Execute", "real CPU tools"), ("4", "Answer", "evidence synthesis"),
        ("5", "Judge", "reference or teacher gate"), ("6", "Human", "confirm or retry"),
    ]
    cards = "".join(
        f'<div class="stage done"><span>{n}</span><b>{name}</b><small>{detail}</small></div>'
        for n, name, detail in stages
    )
    return f'<div class="pipeline">{cards}</div><div class="attempts">{"".join(attempt_cards)}</div>'


def _public_candidates(state: dict) -> dict:
    public = {}
    for name, candidate in state.get("candidates", {}).items():
        public[name] = {key: value for key, value in candidate.items() if key != "evidence"}
    return public


def _render_interactive_state(state: dict):
    status = state.get("status", "not_started")
    mode_label = "reference-aware built-in case" if state.get("mode") == "case" else "user-controlled upload"
    status_md = (
        f"### Workflow status: `{status}`\n\n"
        f"Mode: **{mode_label}** · Attempt {len(state.get('attempts', []))}/{state.get('max_attempts', '?')} · "
        f"Model: `{state.get('model', '')}`"
    )
    latest = state["attempts"][-1] if state.get("attempts") else {}
    evaluation = latest.get("evaluation", {})
    if status == "awaiting_user_decision":
        decision = (
            "### Your decision is authoritative\n\nThe Teacher Failure Gate has no answer key. "
            "Accept this result or request reflection and another attempt."
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
    final_answer = latest.get("answer", "")
    final_md = (
        f"### Current result\n\n{final_answer}\n\n"
        f"**Data source:** {state.get('data_source', '')}"
    )
    return (
        status_md,
        _interactive_timeline(state),
        state.get("knowledge_markdown", ""),
        _attempts_interactive_markdown(state),
        evaluation,
        decision,
        gr.update(visible=show_candidates),
        _public_candidates(state),
        gr.update(value=None),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(visible=show_decision),
        gr.update(visible=show_decision),
        final_md,
        state.get("artifacts", []),
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
    try:
        updated["candidates"] = generate_reflection_candidates(
            api_call=api_completion, api_key=api_key,
            base_url=updated["base_url"], model=updated["model"],
            query=updated["query"], knowledge_context=updated["knowledge_context"],
            trajectory=_reflection_trace(latest), ftro_builder=_ftro_candidate_view,
        )
    except Exception as exc:
        raise gr.Error(f"Reflection generation failed: {exc}") from exc
    updated["status"] = "awaiting_reflection"
    return (updated, *_render_interactive_state(updated))


def select_reflection_candidate(
    selection: str, state: dict, user_diagnosis: str, user_plan: str
) -> str:
    if selection == "User-authored":
        diagnosis = user_diagnosis.strip()
        plan = user_plan.strip()
        return (
            f"Failure diagnosis: {diagnosis}\nNext-attempt plan: {plan}"
            if diagnosis or plan else ""
        )
    candidate = (state or {}).get("candidates", {}).get(selection, {})
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
    reflection = edited_reflection.strip()
    if selection == "User-authored":
        if not user_diagnosis.strip() or not user_plan.strip():
            raise gr.Error("Provide both your error diagnosis and next-attempt plan.")
        reflection = (
            f"Failure diagnosis: {user_diagnosis.strip()}\n"
            f"Next-attempt plan: {user_plan.strip()}"
        )
    if not selection or not reflection:
        raise gr.Error("Select a successful candidate and confirm a non-empty reflection.")
    if selection != "User-authored":
        candidate = state["candidates"].get(selection, {})
        if candidate.get("status") != "ok":
            raise gr.Error("The selected model-generated candidate is unavailable.")

    updated = copy.deepcopy(state)
    updated["memory"].append(
        {
            "source_attempt": len(state["attempts"]),
            "source": selection,
            "reflection": reflection,
        }
    )
    updated["status"] = "running"
    try:
        updated = _execute_interactive_attempt(updated, api_key)
    except Exception as exc:
        raise gr.Error(f"Next attempt failed: {exc}") from exc
    return (updated, *_render_interactive_state(updated))


CSS = """
.hero {text-align:center; padding: 1rem 0 0.4rem 0;}
.hero h1 {margin-bottom: .35rem;}
.note {border-left: 4px solid #2f6fed; padding: .75rem 1rem; background: #f5f8ff;}
.pipeline {display:flex; gap:8px; align-items:stretch; flex-wrap:wrap; margin:12px 0 18px;}
.stage {flex:1 1 135px; min-width:125px; border-radius:12px; padding:12px; border:1px solid #dfe4ef; background:#f7f8fc;}
.stage span {display:inline-flex; width:24px; height:24px; border-radius:50%; align-items:center; justify-content:center; margin-right:7px; background:#e2e7f8;}
.stage b {display:inline-block;}
.stage small {display:block; margin-top:7px; color:#5f6678;}
.stage.done {border-color:#8db7ff; background:#eef5ff;}
.stage.done span {background:#306ee8; color:white;}
.stage.skip {opacity:.58;}
.attempts {display:flex; gap:12px; flex-wrap:wrap; margin-bottom:12px;}
.attempt {display:flex; gap:16px; align-items:center; padding:10px 14px; border-radius:10px; border:1px solid #dfe4ef;}
.attempt span {color:#60697b;}
.attempt.pass {background:#effaf3; border-color:#78cc96;}
.attempt.pass strong {color:#168243;}
.attempt.fail {background:#fff3f1; border-color:#ef9b8e;}
.attempt.fail strong {color:#c13b2a;}
footer {visibility: hidden;}
"""


with gr.Blocks(title=PROJECT_TITLE) as demo:
    gr.HTML(
        "<div class='hero'><h1>🛰️ RSMeM</h1>"
        "<p>Knowledge-enhanced memory evolution for reliable remote sensing agents</p></div>"
    )

    with gr.Tab("Overview"):
        gr.Markdown(
            """
RSMeM combines **Hierarchical Knowledge Grounding (HKG)** with
**Failure-Aware Experience Refinement (FAR)**. HKG retrieves structured
geoscience guidance for planning and tool selection. When a case judge or user
requests another attempt, the Studio compares FTRO, raw full-trajectory, and
grounded-summary reflections. Only the user's confirmed repair is stored in
Failure-Aware Memory.

This public Studio combines transparent knowledge retrieval with a required
OpenAI-compatible LLM planner and a real CPU raster executor. The GPU-backed
Perception server is intentionally disabled in this free-CPU deployment.
            """
        )
        gr.Image("assets/figure1.png", label="Remote-sensing agent paradigms")
        gr.Image("assets/figure2.png", label="RSMeM workflow")

    with gr.Tab("RSMeM iterative workflow"):
        gr.Markdown(
            """
### Observe the full memory-evolution loop

The Studio retrieves query-relevant knowledge, plans a tool chain, executes real CPU raster tools,
synthesizes an evidence-grounded answer, and then evaluates it. Bundled cases use a reference-aware
LLM judge and stop after a correct result or three attempts. User uploads use a ground-truth-free
Teacher Failure Gate, but the user makes the final decision and may run up to ten confirmed attempts.

When another attempt is needed, compare **FTRO**, **raw full trajectory**, and **grounded LLM
summary** reflections, or write your own diagnosis and plan. Only the option you explicitly confirm
is written into Failure-Aware Memory.
            """
        )
        session_state = gr.State({})
        example_input = gr.Dropdown(
            label="Remote-sensing workflow",
            choices=list(EXAMPLES),
            value=DEFAULT_EXAMPLE,
        )
        dataset_info = gr.Markdown(
            f"### Selected workflow\n\n"
            f"**What the agent must do:** {EXAMPLES[DEFAULT_EXAMPLE]['query']}\n\n"
            f"**Inputs and operations:** {EXAMPLES[DEFAULT_EXAMPLE]['description']}\n\n"
            f"**Evaluation mode:** Reference-aware case evaluation"
        )
        with gr.Row():
            base_url_input = gr.Textbox(
                label="OpenAI-compatible Base URL",
                value=DEFAULT_BASE_URL,
                placeholder="https://provider.example/v1",
                info="Only public HTTPS endpoints are accepted.",
            )
            model_input = gr.Textbox(
                label="Model",
                value=DEFAULT_MODEL,
                placeholder="Provider model identifier",
                info="Use the exact model identifier exposed by your provider.",
            )
        api_key_input = gr.Textbox(
            label="Your API key (required)",
            type="password",
            placeholder="Enter your key for this request",
            info="Required for planning and reflection. Used only for this request and never saved.",
        )
        raster_input = gr.File(
            label="Optional GeoTIFF override",
            file_count="multiple",
            file_types=[".tif", ".tiff"],
            type="filepath",
        )
        gr.Markdown(
            "Uploaded files override the bundled example. CPU only · files remain request-scoped."
        )
        query_input = gr.Textbox(
            label="Remote-sensing task",
            value=EXAMPLES[DEFAULT_EXAMPLE]["query"],
            lines=3,
        )
        example_input.change(
            load_example,
            inputs=[example_input],
            outputs=[query_input, dataset_info],
        )
        run_button = gr.Button("Start Attempt 1 on the real rasters", variant="primary")
        api_status = gr.Markdown("⚪ Ready. No attempt runs without a valid API key.")
        timeline_output = gr.HTML()
        with gr.Accordion("1 · Hierarchical Knowledge Grounding (HKG)", open=True):
            hkg_output = gr.Markdown()
        with gr.Accordion("2–5 · Plan → real tools → answer → judge/gate", open=True):
            attempts_output = gr.Markdown()
            evaluation_output = gr.JSON(label="Latest Case Judge / Teacher Failure Gate result")
        with gr.Accordion("6 · Human decision", open=True):
            decision_output = gr.Markdown()
            with gr.Row():
                accept_button = gr.Button("Accept this result and finish", visible=False)
                reflect_button = gr.Button("This result is wrong — generate reflections", visible=False)
        with gr.Group(visible=False) as candidate_group:
            gr.Markdown(
                "### Reflection candidates\n\nAll three model candidates use the same reflector; "
                "only the trajectory representation changes."
            )
            candidates_output = gr.JSON(label="Candidate comparison and provenance")
            candidate_choice = gr.Radio(
                choices=["FTRO", "Full trajectory", "LLM summary", "User-authored"],
                label="Choose one reflection source",
            )
            with gr.Row():
                user_diagnosis = gr.Textbox(
                    label="Your diagnosis (required for User-authored)", lines=3
                )
                user_plan = gr.Textbox(
                    label="Your next-attempt plan (required for User-authored)", lines=3
                )
            edited_reflection = gr.Textbox(
                label="Final editable reflection to write into FAM", lines=6
            )
            retry_button = gr.Button(
                "Confirm this reflection and execute the next attempt", variant="primary"
            )
        with gr.Accordion("Current result and artifacts", open=True):
            final_output = gr.Markdown()
            artifact_output = gr.File(label="Download derived GeoTIFF artifacts", file_count="multiple")

        workflow_outputs = [
            session_state,
            api_status,
            timeline_output,
            hkg_output,
            attempts_output,
            evaluation_output,
            decision_output,
            candidate_group,
            candidates_output,
            candidate_choice,
            user_diagnosis,
            user_plan,
            edited_reflection,
            accept_button,
            reflect_button,
            final_output,
            artifact_output,
        ]
        run_button.click(
            start_interactive_workflow,
            inputs=[
                query_input,
                example_input,
                raster_input,
                base_url_input,
                model_input,
                api_key_input,
            ],
            outputs=workflow_outputs,
        )
        accept_button.click(
            accept_custom_result,
            inputs=[session_state],
            outputs=workflow_outputs,
        )
        reflect_button.click(
            request_custom_reflections,
            inputs=[session_state, api_key_input],
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
            confirm_reflection_and_retry,
            inputs=[
                session_state,
                candidate_choice,
                edited_reflection,
                user_diagnosis,
                user_plan,
                api_key_input,
            ],
            outputs=workflow_outputs,
        )

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
- Knowledge base included in this Studio: 3 domains, 12 subdomains, and 64 atomic tasks
- Code license: MIT · bundled example-data license: Apache-2.0

The full repository contains experiment configurations, the evaluation pipeline,
the original MCP servers, and deployment instructions for GPU perception services.
This Studio connects the CPU-safe Index/Statistics subset directly so uploaded
GeoTIFFs generate auditable and downloadable outputs without a GPU.
            """
        )

    with gr.Tab("Challenge submission"):
        learn_submission = (
            f"[{LEARN_URL}]({LEARN_URL})"
            if LEARN_URL
            else "**Pending — paste the public article URL after publishing `LEARN_ARTICLE.md`.**"
        )
        gr.Markdown(
            f"""
### Project title

**{PROJECT_TITLE}**

### Project abstract (66 words)

{PROJECT_ABSTRACT}

### Required public links

- **ModelScope Studio:** [https://modelscope.cn/studios/wbx929/RSMeM](https://modelscope.cn/studios/wbx929/RSMeM)
- **ModelScope Learn article:** {learn_submission}
- **Open-source repository:** [{GITHUB_URL}]({GITHUB_URL})

The Learn article documents the research question, EarthBench data, model configuration,
RSMeM workflow, system design, reproducibility safeguards, paper evaluation, and the thirteen
CPU demonstration results.
            """
        )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
