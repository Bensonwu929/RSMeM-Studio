"""Lightweight, API-free RSMeM demonstration for ModelScope Studio."""

from __future__ import annotations

import json
import re
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "knowledge_base.json"

PROJECT_TITLE = "RSMeM: Knowledge-Enhanced Memory Evolution for Remote Sensing Agents"
GITHUB_URL = "https://github.com/AI9Stars/RSMeM"
LEARN_URL = "https://modelscope.cn/learn/434564"

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


def run_demo(query: str, trajectory: str):
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
    prompt = (
        "### RSMeM planning context\n"
        f"**User task:** {query}\n\n"
        f"**Domain grounding:** {retrieved[0][1]['domain']} / {retrieved[0][1]['application']}\n\n"
        f"**Failure-aware constraint:** {correction}\n\n"
        "**Next action:** Validate inputs and tool contracts, execute the smallest valid step, inspect the observation, "
        "then continue the workflow."
    )
    comparison = (
        "| Agent | Planning context | Failure reuse | Expected behavior |\n"
        "|---|---|---|---|\n"
        "| Baseline | General LLM prompt | None | May repeat invalid tool calls |\n"
        "| RSMeM | Hierarchical domain guidance | Compact, traceable constraint | More auditable and robust retry |"
    )
    return hkg, far, prompt, comparison


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

This public Studio is an **API-free explanatory demo** of the memory workflow.
It does not reproduce the full GPU-backed perception stack or claim to execute
the complete EarthBench evaluation inside the browser.
            """
        )
        gr.Image("assets/figure1.png", label="Remote-sensing agent paradigms", show_download_button=False)
        gr.Image("assets/figure2.png", label="RSMeM workflow", show_download_button=False)

    with gr.Tab("Interactive memory demo"):
        gr.Markdown(
            "Enter a remote-sensing task and an example failed trajectory. The demo retrieves domain knowledge "
            "and produces a transparent, reusable execution constraint."
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
        with gr.Row():
            hkg_output = gr.Markdown(label="HKG retrieval")
            far_output = gr.Markdown(label="FAR experience")
        prompt_output = gr.Markdown(label="Combined planning context")
        comparison_output = gr.Markdown(label="Baseline comparison")
        run_button.click(
            run_demo,
            inputs=[query_input, failure_input],
            outputs=[hkg_output, far_output, prompt_output, comparison_output],
        )
        demo.load(
            run_demo,
            inputs=[query_input, failure_input],
            outputs=[hkg_output, far_output, prompt_output, comparison_output],
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
