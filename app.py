"""Lightweight, API-free RSMeM demonstration for ModelScope Studio."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import gradio as gr
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "knowledge_base.json"

PROJECT_TITLE = "RSMeM: Knowledge-Enhanced Memory Evolution for Remote Sensing Agents"
GITHUB_URL = "https://github.com/AI9Stars/RSMeM"
LEARN_URL = "https://modelscope.cn/learn/434564"
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

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


def call_rsmem_api(
    query: str,
    trajectory: str,
    retrieved: list[tuple[int, dict]],
    diagnosis: str,
    correction: str,
    user_api_key: str,
    requested_model: str,
) -> tuple[str, str]:
    api_key = (
        user_api_key.strip()
        or os.getenv("RSMEM_API_KEY")
        or os.getenv("MODELSCOPE_API_TOKEN")
        or os.getenv("DASHSCOPE_API_KEY")
    )
    base_url = os.getenv("RSMEM_BASE_URL", MODELSCOPE_BASE_URL).strip()
    model = requested_model.strip() or os.getenv("RSMEM_MODEL", DEFAULT_MODEL).strip()

    if not api_key:
        fallback = (
            "### Offline fallback plan\n"
            "1. Validate the imagery path, CRS, spatial extent, units, and required parameters.\n"
            "2. Select the smallest tool sequence supported by the retrieved domain guidance.\n"
            f"3. Apply the reusable constraint: {correction}\n"
            "4. Inspect every intermediate observation before continuing downstream.\n"
            "5. Record failures and corrected strategies for the next attempt."
        )
        return fallback, "⚪ Enter your own ModelScope token to enable online reasoning."

    knowledge_context = []
    for _, entry in retrieved:
        summary = re.sub(r"\s+", " ", entry["summary"]).strip()[:900]
        knowledge_context.append(
            f"Domain: {entry['domain']}\nTask: {entry['application']}\nGuidance: {summary}"
        )

    system_prompt = """You are the planning and reflection component of RSMeM, a remote-sensing agent.
Use the retrieved hierarchical domain knowledge and the failure-aware constraint to produce an auditable plan.
Do not claim that tools or models were actually executed. Do not invent measurements, files, or experimental results.
Return concise Markdown with exactly these sections: Task interpretation, Retrieved guidance, Corrected execution plan,
Validation checks, and Memory to retain."""
    user_prompt = f"""Remote-sensing task:
{query[:4000]}

Observed failed trajectory:
{trajectory[:4000] or 'No prior trajectory supplied.'}

Failure diagnosis:
{diagnosis}

Reusable constraint:
{correction}

Retrieved knowledge:
{chr(10).join(knowledge_context)}
"""

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=900,
        )
        content = response.choices[0].message.content or "The API returned an empty response."
        return content, f"🟢 Online RSMeM reasoning enabled - model: `{model}`"
    except Exception as exc:
        safe_error = type(exc).__name__
        fallback = (
            "### API unavailable - safe fallback\n"
            f"The online request could not be completed (`{safe_error}`). No credential was exposed.\n\n"
            f"**Reusable constraint:** {correction}\n\n"
            "Validate inputs and tool contracts, execute the smallest valid step, inspect its observation, and retry."
        )
        return fallback, "🟠 API request failed - the Studio remains available in fallback mode."


def run_demo(query: str, trajectory: str, user_api_key: str, requested_model: str):
    query = query.strip() or "Map flooded areas after a major storm using satellite imagery."
    retrieved = retrieve_knowledge(query)
    diagnosis, correction = summarize_failure(trajectory)

    knowledge_lines = []
    for index, (score, entry) in enumerate(retrieved, start=1):
        excerpt = re.sub(r"\s+", " ", entry["summary"]).strip()[:360]
        knowledge_lines.append(
            f"**{index}. {entry['domain']} → {entry['application']}**  \n"
            f"Retrieval score: `{score}`  \n{excerpt}..."
        )

    hkg = "\n\n".join(knowledge_lines)
    far = (
        f"**Diagnosis:** {diagnosis}  \n"
        f"**Reusable constraint:** {correction}  \n"
        "**Memory policy:** Preserve the originating trajectory and retrieve this constraint only for similar tasks."
    )
    prompt, api_status = call_rsmem_api(
        query,
        trajectory,
        retrieved,
        diagnosis,
        correction,
        user_api_key,
        requested_model,
    )
    comparison = (
        "| Agent | Planning context | Failure reuse | Expected behavior |\n"
        "|---|---|---|---|\n"
        "| Baseline | General LLM prompt | None | May repeat invalid tool calls |\n"
        "| RSMeM | Hierarchical domain guidance | Compact, traceable constraint | More auditable and robust retry |"
    )
    return api_status, hkg, far, prompt, comparison


CSS = """
.hero {text-align:center; padding: 1rem 0 0.4rem 0;}
.hero h1 {margin-bottom: .35rem;}
.note {border-left: 4px solid #2f6fed; padding: .75rem 1rem; background: #f5f8ff;}
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

    with gr.Tab("Interactive memory demo"):
        gr.Markdown(
            "Enter a remote-sensing task and an example failed trajectory. The demo retrieves domain knowledge "
            "and produces a transparent, reusable execution constraint."
        )
        with gr.Row():
            api_key_input = gr.Textbox(
                label="Your ModelScope API token",
                type="password",
                placeholder="Enter your token for this request",
                info="Used only for the inference request; never saved by this application.",
            )
            model_input = gr.Dropdown(
                label="Model",
                choices=[
                    "deepseek-ai/DeepSeek-V4-Flash-0731",
                    "Qwen/Qwen3-235B-A22B-Instruct-2507",
                    "Qwen/Qwen3-32B",
                ],
                value="deepseek-ai/DeepSeek-V4-Flash-0731",
                allow_custom_value=True,
                info="Availability depends on your ModelScope account and current inference service.",
            )
        with gr.Row():
            query_input = gr.Textbox(
                label="Remote-sensing task",
                value="Map flooded areas after a major storm using multispectral satellite imagery.",
                lines=3,
            )
            failure_input = gr.Textbox(
                label="Example failed trajectory",
                value="The flood-mapping tool returned an invalid parameter error because the CRS and threshold units were not checked.",
                lines=3,
            )
        run_button = gr.Button("Run RSMeM memory demonstration", variant="primary")
        api_status = gr.Markdown("⚪ Ready. Click the button to run the demonstration.")
        with gr.Row():
            hkg_output = gr.Markdown(label="HKG retrieval")
            far_output = gr.Markdown(label="FAR experience")
        prompt_output = gr.Markdown(label="Combined planning context")
        comparison_output = gr.Markdown(label="Baseline comparison")
        run_button.click(
            run_demo,
            inputs=[query_input, failure_input, api_key_input, model_input],
            outputs=[api_status, hkg_output, far_output, prompt_output, comparison_output],
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
