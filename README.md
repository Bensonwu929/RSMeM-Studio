---
title: RSMeM
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: mit
tags:
  - geoai-challenge
  - remote-sensing
  - geoscience
  - agents
---

# RSMeM: Knowledge-Enhanced Memory Evolution for Remote Sensing Agents

## Challenge submission

**Project abstract (66 words):** RSMeM is a knowledge-enhanced memory evolution
framework for reliable remote sensing agents. It combines Hierarchical Knowledge
Grounding, which retrieves structured geoscience knowledge for tool planning,
with Failure-Aware Experience Refinement, which converts failed trajectories into
reusable execution constraints. The public Studio demonstrates the complete
retrieve-plan-execute-judge-reflect-confirm-retry loop on thirteen remote-sensing cases. Its
CPU-only raster toolchain produces auditable GeoTIFF outputs and supports
visitor-provided OpenAI-compatible models without storing API credentials.

- Studio: https://modelscope.cn/studios/wbx929/RSMeM
- Learn article: [RSMeM on ModelScope Learn](https://modelscope.cn/learn/435887)
- Full article source: [`LEARN_ARTICLE.md`](LEARN_ARTICLE.md)

RSMeM combines Hierarchical Knowledge Grounding (HKG) and Failure-Aware
Experience Refinement (FAR) to improve the reliability and reproducibility of
multi-step remote-sensing agents.

This ModelScope Studio provides a CPU-friendly, executable demonstration of the
core memory workflow. It retrieves structured domain knowledge, plans and runs
real raster operations, synthesizes an evidence-grounded answer, applies either a
reference-aware case judge or a ground-truth-free Teacher Failure Gate, and lets
the user choose which reflection enters memory.
The Studio starts the released Analysis, Index, Inversion, and Statistics
servers as four independent stdio MCP processes. The full experimental
framework and GPU Perception stack remain available in the public repository.

## Observable RSMeM loop

The interactive workflow mirrors the control flow implemented by the released
framework:

1. Retrieve query-relevant hierarchical knowledge (HKG).
2. Plan a short remote-sensing tool chain.
3. Discover the live CPU tool catalog with MCP `list_tools`, execute selected
   operations with MCP `call_tool`, and record parameters, observations, and
   downloadable artifacts.
4. Synthesize a final answer using only real tool evidence.
5. Evaluate bundled cases with a reference-aware LLM judge; evaluate uploads
   with a ground-truth-free Teacher Failure Gate whose advice the user may
   accept or override.
6. When retrying, generate three comparable repairs: **FTRO**, **raw full
   trajectory**, and **grounded LLM summary**. FTRO retains the first
   three and last two elements of long lists, retains two representative
   key-value pairs from long dictionaries, and allocate a larger budget to
   error messages while preserving tool purpose.
7. Let the user select/edit one of those repairs or provide a fourth,
   user-authored diagnosis and plan. Only the confirmed option enters FAM.
8. Retry with HKG + selected FAM. Bundled cases stop after a correct judgment
   or three attempts; uploads stop when the user accepts or after ten attempts.

Reference answers are isolated inside the case judge and never enter the
reflector or next planner. For uploads, the Teacher Failure Gate has no answer
key and provides advisory evidence only; the user's decision is authoritative.

The hosted Studio accepts optional `.tif`/`.tiff` uploads and recognizes common
band names such as `B4_red`, `B5_nir`, `B6_swir1`, and `B7_swir2`. Single-band
files are safest because raster calculations read band 1. JPG, PNG, NetCDF,
and HDF are not accepted by this CPU deployment. The separate GPU perception
services in the full RSMeM repository can process ordinary image inputs, but
they are intentionally not connected here. If no files are uploaded, visitors
can run one of thirteen bundled demonstrations:

1. Sichuan agricultural drought: NDVI + LST → TVDI → dry-area ratio.
2. California wildfire trend: 11 paired dates → NBR means → Sen's slope.
3. Bhutan nighttime lights: two regional means → month-to-month difference.
4. Black Forest: Landsat single-channel LST → NDVI-stratified temperature contrast.
5. Cyprus: daytime/nighttime temperature + albedo → ATI change.
6. Mumbai: 2013–2024 nighttime-light means → long-term trend.
7. Rosario: vegetation means → maximum consecutive growth.
8. Dongting Lake: red reflectance → turbidity → most polluted three-day window.
9. Congo Rainforest: daily precipitation → monthly trend.
10. Lake Urmia: turbidity maps → distribution skewness and anomaly diagnosis.
11. California hotspots: NBR → low-NBR masks → dominant direction.
12. Thailand: MaxFRP composite → fire-prone upper-quartile mask.
13. Shanghai: total minus non-residential volume → 1985–2020 trend.

The CPU runtime exposes the four released EarthAgent/RSMeM MCP services plus
rasterio-only case adapters: **Analysis, Index, Inversion, and Statistics**.
GPU-backed **Perception** is explicitly disabled on the free CPU instance.

## Data attribution

Bundled example GeoTIFFs are subsets of
[Sssunset/Earth-Bench](https://huggingface.co/datasets/Sssunset/Earth-Bench),
released under Apache-2.0. See [`DATA_LICENSE.md`](DATA_LICENSE.md). RSMeM Studio
code is MIT-licensed; the third-party example data retains its original license.

## API use

The public interface lets each visitor enter an OpenAI-compatible Base URL,
model identifier, and their own API key. The API key uses a password field. The
tested defaults are `https://uni-api.cstcloud.cn/v1` and
`deepseek-v4-flash-aistar`; no credential is included in this repository.

API keys are used only to construct the current inference request and are not
written to project files or application output. Availability and quota depend
on the visitor's chosen provider.

For safety, the public Studio accepts only HTTPS endpoints that resolve to
globally routable addresses. Localhost, private networks, link-local addresses,
reserved addresses, and credentials embedded in URLs are rejected.

For a team-managed deployment, environment variables remain supported:
`RSMEM_API_KEY`, `RSMEM_BASE_URL`, and `RSMEM_MODEL`. Never commit credentials.
If no key is supplied or the provider is unavailable, execution stops with an
explicit error; the Studio never replaces model planning with a preset fallback.

## Resources

- Source: https://github.com/AI9Stars/RSMeM
- ModelScope Learn: https://modelscope.cn/learn/435887
- License: MIT

## Local launch

```bash
pip install -r requirements.txt
python app.py
```

## Scope

The Studio is an executable artifact for the Open Source GeoAI Practice
Challenge. It runs four CPU MCP services, but does not claim to run the GPU
Perception service or the complete EarthBench evaluation.
