"""Lightweight RSMeM demonstration with a user-supplied compatible API."""

from __future__ import annotations

import json
import ipaddress
import os
import re
import socket
from pathlib import Path
from urllib.parse import urlparse

import gradio as gr
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "knowledge_base.json"

PROJECT_TITLE = "RSMeM: Knowledge-Enhanced Memory Evolution for Remote Sensing Agents"
GITHUB_URL = "https://github.com/AI9Stars/RSMeM"
LEARN_URL = "https://modelscope.cn/learn/434564"
DEFAULT_BASE_URL = "https://uni-api.cstcloud.cn/v1"
DEFAULT_MODEL = "deepseek-v4-flash-aistar"

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
- get_filelist: always inspect available files and band composition first.
- calculate_ndwi: delineate water/moisture from suitable Green/NIR or NIR/SWIR bands.
- calculate_ndvi: estimate vegetation greenness from NIR and Red bands.
- calculate_nbr: assess burned areas from NIR and SWIR bands.
- split_window: retrieve land-surface temperature when its thermal-band contract is satisfied.
- calculate_threshold_ratio: summarize the fraction of raster pixels satisfying a threshold.
- calc_batch_image_mean: produce one scalar mean per raster/date.
- calculate_mask_area: convert a validated binary mask into physical area.
""".strip()


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


def fallback_tool_plan(query: str, reflection: str = "") -> list[dict]:
    low = query.lower()
    plan = [
        {
            "tool": "get_filelist",
            "purpose": "Inspect actual inputs and band composition before selecting a method.",
            "parameters": {"data_source": "sandbox/sample_scene"},
        }
    ]
    if any(term in low for term in ("flood", "water", "inundation")):
        plan.extend(
            [
                {
                    "tool": "calculate_ndwi",
                    "purpose": "Generate a water-sensitive index after confirming band order.",
                    "parameters": {"bands": "validated Green/NIR or NIR/SWIR pair"},
                },
                {
                    "tool": "calculate_threshold_ratio",
                    "purpose": "Estimate the share of the scene classified as water.",
                    "parameters": {"threshold": "validated from metadata"},
                },
                {
                    "tool": "calculate_mask_area",
                    "purpose": "Convert the final water mask into physical area.",
                    "parameters": {"crs": "validated projected CRS"},
                },
            ]
        )
    elif any(term in low for term in ("fire", "burn", "wildfire")):
        plan.extend(
            [
                {"tool": "calculate_nbr", "purpose": "Compute a burn-sensitive index.", "parameters": {}},
                {"tool": "calculate_threshold_ratio", "purpose": "Summarize affected area.", "parameters": {}},
            ]
        )
    elif any(term in low for term in ("vegetation", "crop", "forest", "ndvi")):
        plan.extend(
            [
                {"tool": "calculate_ndvi", "purpose": "Compute vegetation greenness.", "parameters": {}},
                {"tool": "calc_batch_image_mean", "purpose": "Summarize the temporal signal.", "parameters": {}},
            ]
        )
    else:
        plan.extend(
            [
                {"tool": "domain_analysis", "purpose": "Apply the retrieved domain methodology.", "parameters": {}},
                {"tool": "validate_result", "purpose": "Check the output contract before answering.", "parameters": {}},
            ]
        )
    if reflection:
        plan.insert(
            1,
            {
                "tool": "validate_inputs",
                "purpose": "Apply the failure-aware constraint before repeating the failed operation.",
                "parameters": {"constraint": reflection[:240]},
            },
        )
    return plan[:5]


def normalize_plan(data: dict | list | None, query: str, reflection: str) -> list[dict]:
    if isinstance(data, dict):
        data = data.get("tool_chain") or data.get("tools") or data.get("plan")
    if not isinstance(data, list):
        return fallback_tool_plan(query, reflection)
    normalized = []
    for item in data[:5]:
        if isinstance(item, str):
            normalized.append({"tool": item, "purpose": "Planned by the agent.", "parameters": {}})
        elif isinstance(item, dict) and item.get("tool"):
            normalized.append(
                {
                    "tool": str(item["tool"]),
                    "purpose": str(item.get("purpose", "Planned by the agent.")),
                    "parameters": item.get("parameters", {}) if isinstance(item.get("parameters", {}), dict) else {},
                }
            )
    return normalized or fallback_tool_plan(query, reflection)


def plan_toolchain(
    query: str,
    knowledge_context: str,
    reflection: str,
    api_key: str,
    base_url: str,
    model: str,
) -> tuple[list[dict], bool]:
    if not api_key:
        return fallback_tool_plan(query, reflection), False
    system = """You are the planner inside RSMeM. Select a short remote-sensing tool chain.
Return ONLY JSON with a tool_chain array. Each item must contain tool, purpose, and parameters.
Always inspect inputs before selecting band-dependent tools. Never claim execution has happened."""
    user = f"""Query: {query[:3000]}

Retrieved HKG knowledge:
{knowledge_context[:4000]}

Failure-aware memory from the previous attempt:
{reflection[:2000] or '(none; this is the first attempt)'}

Available demonstration tool reference:
{TOOL_REFERENCE}
"""
    try:
        raw = api_completion(api_key, base_url, model, system, user, max_tokens=700)
        return normalize_plan(extract_json_object(raw), query, reflection), True
    except Exception:
        return fallback_tool_plan(query, reflection), False


def sandbox_observation(tool: str) -> str:
    observations = {
        "get_filelist": "Sandbox inspected the sample scene manifest and exposed candidate bands and metadata.",
        "validate_inputs": "Input paths, band order, CRS, units, and parameter contracts passed validation.",
        "calculate_ndwi": "NDWI tool contract accepted; a demonstration water-index artifact was produced.",
        "calculate_ndvi": "NDVI tool contract accepted; a demonstration vegetation-index artifact was produced.",
        "calculate_nbr": "NBR tool contract accepted; a demonstration burn-index artifact was produced.",
        "calculate_threshold_ratio": "Threshold statistic completed over the demonstration artifact.",
        "calculate_mask_area": "Mask-to-area conversion completed with the validated CRS.",
        "calc_batch_image_mean": "Per-image summary statistics were produced for the demonstration series.",
    }
    return observations.get(tool, f"{tool} completed in the sandbox executor.")


def execute_sandbox(
    plan: list[dict],
    attempt: int,
    inject_failure: bool,
    failure_observation: str,
) -> tuple[list[dict], bool, list[str], str]:
    trajectory = []
    should_fail = inject_failure and attempt == 1
    failure_index = max(0, len(plan) - 1)
    for index, step in enumerate(plan):
        failed = should_fail and index == failure_index
        observation = (
            failure_observation.strip()
            or "Tool result failed validation because an input contract was not checked."
        ) if failed else sandbox_observation(step["tool"])
        trajectory.append(
            {
                "step": index + 1,
                "tool": step["tool"],
                "purpose": step["purpose"],
                "parameters": step["parameters"],
                "status": "error" if failed else "success",
                "observation": observation,
            }
        )
        if failed:
            break
    if should_fail:
        diagnosis, _ = summarize_failure(trajectory[-1]["observation"])
        return trajectory, False, [diagnosis], trajectory[-1]["observation"]
    return (
        trajectory,
        True,
        [],
        "The sandbox tool chain completed and its output contract passed the critic.",
    )


def generate_reflection(
    query: str,
    knowledge_context: str,
    trajectory: list[dict],
    api_key: str,
    base_url: str,
    model: str,
) -> tuple[str, bool]:
    failure_text = trajectory[-1]["observation"] if trajectory else "Unknown failure."
    diagnosis, correction = summarize_failure(failure_text)
    fallback = f"{diagnosis}. {correction} Preserve successful earlier steps and rerun only the failed branch."
    if not api_key:
        return fallback, False
    system = """You are RSMeM's Failure-Aware Experience Refinement component.
Diagnose the failed tool trajectory and produce a concise, reusable constraint for the next attempt.
Return 3-5 sentences only. Do not claim a rerun has already succeeded."""
    user = f"""Query: {query[:3000]}
Retrieved knowledge: {knowledge_context[:3000]}
Failed trajectory: {json.dumps(trajectory, ensure_ascii=False)[:6000]}
"""
    try:
        reflection = api_completion(api_key, base_url, model, system, user, max_tokens=450).strip()
        return reflection or fallback, True
    except Exception:
        return fallback, False


def build_timeline(attempts: list[dict], used_reflection: bool) -> str:
    stages = [
        ("1", "HKG", "Knowledge retrieved", "done"),
        ("2", "Plan", "Tool chain built", "done"),
        ("3", "Execute", "Sandbox trajectory", "done"),
        ("4", "Critic", "Result evaluated", "done"),
        ("5", "FAR", "Reflection generated" if used_reflection else "Not needed", "done" if used_reflection else "skip"),
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
        far_used = attempt["injected_context"]["far_memory"]
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
    failure_observation: str,
    requested_base_url: str,
    requested_model: str,
    user_api_key: str,
    inject_failure: bool,
):
    query = query.strip() or "Map flooded areas after a major storm using satellite imagery."
    api_key = user_api_key.strip() or os.getenv("RSMEM_API_KEY", "")
    model = requested_model.strip() or os.getenv("RSMEM_MODEL", DEFAULT_MODEL)
    api_ready = False
    base_url = ""
    if api_key:
        try:
            base_url = validate_base_url(
                requested_base_url or os.getenv("RSMEM_BASE_URL", DEFAULT_BASE_URL)
            )
            api_ready = True
        except ValueError:
            api_ready = False

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

    attempts = []
    online_events = []
    plan1, online_plan1 = plan_toolchain(
        query, knowledge_context, "", api_key if api_ready else "", base_url, model
    )
    online_events.append(online_plan1)
    trajectory1, success1, reasons1, result1 = execute_sandbox(
        plan1, 1, inject_failure, failure_observation
    )
    attempts.append(
        {
            "attempt": 1,
            "injected_context": {"hkg": True, "far_memory": False},
            "tool_plan": plan1,
            "trajectory": trajectory1,
            "result": result1,
            "critic": {"success": success1, "reasons": reasons1},
        }
    )

    reflection = ""
    if not success1:
        reflection, online_reflection = generate_reflection(
            query,
            knowledge_context,
            trajectory1,
            api_key if api_ready else "",
            base_url,
            model,
        )
        online_events.append(online_reflection)
        plan2, online_plan2 = plan_toolchain(
            query,
            knowledge_context,
            reflection,
            api_key if api_ready else "",
            base_url,
            model,
        )
        online_events.append(online_plan2)
        trajectory2, success2, reasons2, result2 = execute_sandbox(
            plan2, 2, False, ""
        )
        attempts.append(
            {
                "attempt": 2,
                "injected_context": {"hkg": True, "far_memory": True},
                "tool_plan": plan2,
                "trajectory": trajectory2,
                "result": result2,
                "critic": {"success": success2, "reasons": reasons2},
            }
        )

    final_success = attempts[-1]["critic"]["success"]
    trace = {
        "mode": "transparent sandbox executor; no real imagery is processed",
        "query": query,
        "retrieved_domains": [
            {"domain": entry["domain"], "task": entry["application"], "score": score}
            for score, entry in retrieved
        ],
        "attempts": attempts,
    }
    memory_md = (
        "### Failure-Aware Memory written after Attempt 1\n\n"
        f"> {reflection}\n\n"
        "This compact reflection is injected into Attempt 2 together with the original HKG context."
        if reflection
        else "### Failure-Aware Memory\n\nNo reflection was written because Attempt 1 passed the critic."
    )
    final_md = (
        f"### Final outcome: {'✅ critic passed' if final_success else '⚠️ max attempts reached'}\n\n"
        f"**Attempts executed:** {len(attempts)}  \n"
        f"**Final result:** {attempts[-1]['result']}\n\n"
        "The execution observations above are an explicit method sandbox, not measurements from real imagery. "
        "The production RSMeM repository replaces this sandbox with MCP remote-sensing tools."
    )
    if api_ready and any(online_events):
        api_status = f"🟢 Online planner/reflection active - model: `{model}`"
    elif api_key and not api_ready:
        api_status = "🟠 Base URL rejected or unavailable; deterministic planning was used safely."
    elif api_key:
        api_status = "🟠 API calls failed; deterministic planning completed the loop."
    else:
        api_status = "⚪ No API key supplied; deterministic planning completed the loop."
    return (
        api_status,
        build_timeline(attempts, bool(reflection)),
        knowledge_md,
        attempts_to_markdown(attempts),
        trace,
        memory_md,
        final_md,
    )


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
geoscience guidance for planning and tool selection, while FAR converts failed
tool-use trajectories into compact, reusable constraints.

This public Studio combines deterministic knowledge retrieval with an optional
OpenAI-compatible LLM API. It does not reproduce the full GPU-backed perception
stack or claim to execute the complete EarthBench evaluation inside the browser.
            """
        )
        gr.Image("assets/figure1.png", label="Remote-sensing agent paradigms")
        gr.Image("assets/figure2.png", label="RSMeM workflow")

    with gr.Tab("RSMeM iterative workflow"):
        gr.Markdown(
            """
### Observe the full memory-evolution loop

The Studio retrieves query-relevant knowledge, plans a tool chain, executes it in a transparent
**sandbox**, evaluates the result, and conditionally reflects and retries. The sandbox exposes
RSMeM's control flow without pretending to run the production GPU/MCP remote-sensing stack.
            """
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
            label="Your API key",
            type="password",
            placeholder="Enter your key for this request",
            info="Used only for the inference request; never written to project files or outputs.",
        )
        with gr.Row():
            query_input = gr.Textbox(
                label="Remote-sensing task",
                value="Map flooded areas after a major storm using multispectral satellite imagery.",
                lines=3,
            )
            failure_input = gr.Textbox(
                label="Attempt-1 execution observation",
                value="The flood-mapping tool returned an invalid parameter error because the CRS and threshold units were not checked.",
                lines=3,
                info="Used as the recoverable tool failure shown in the first trajectory.",
            )
        inject_failure = gr.Checkbox(
            value=True,
            label="Inject a recoverable failure in Attempt 1",
            info="Turn off to show the conditional early-stop path when the first result passes.",
        )
        run_button = gr.Button("Run the complete RSMeM loop", variant="primary")
        api_status = gr.Markdown("⚪ Ready. Run the workflow to generate an observable trace.")
        timeline_output = gr.HTML()
        with gr.Accordion("1 · Hierarchical Knowledge Grounding (HKG)", open=True):
            hkg_output = gr.Markdown()
        with gr.Accordion("2–4 · Planning, sandbox tool execution, and critic", open=True):
            attempts_output = gr.Markdown()
            with gr.Accordion("Raw structured trace", open=False):
                trace_output = gr.JSON(label="Multi-attempt execution trace")
        with gr.Accordion("5 · Failure-Aware Experience Refinement (FAR)", open=True):
            memory_output = gr.Markdown()
        with gr.Accordion("6 · Final aggregated result", open=True):
            final_output = gr.Markdown()
        run_button.click(
            run_rsmem_loop,
            inputs=[
                query_input,
                failure_input,
                base_url_input,
                model_input,
                api_key_input,
                inject_failure,
            ],
            outputs=[
                api_status,
                timeline_output,
                hkg_output,
                attempts_output,
                trace_output,
                memory_output,
                final_output,
            ],
        )

    with gr.Tab("Open-source resources"):
        gr.Markdown(
            f"""
### Reproduce and extend RSMeM

- [Source code on GitHub]({GITHUB_URL})
- [ModelScope Learn article]({LEARN_URL})
- Knowledge base included in this Studio: 3 domains, 12 subdomains, and 64 atomic tasks
- License: MIT

The full repository contains experiment configurations, the evaluation pipeline,
MCP tool interfaces, and deployment instructions for the remote-sensing model services.
            """
        )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
