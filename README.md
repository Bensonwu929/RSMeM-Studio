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
models:
  - wbx929/RSMeM-Perception-Models
---

# RSMeM: Knowledge-Enhanced Memory Evolution for Remote Sensing Agents

## Challenge submission

**Project abstract (66 words):** RSMeM is a knowledge-enhanced memory evolution
framework for reliable remote sensing agents. It combines Hierarchical Knowledge
Grounding, which retrieves structured geoscience knowledge for tool planning,
with Failure-Aware Experience Refinement, which converts failed trajectories into
reusable execution constraints. The public Studio demonstrates the complete
retrieve-plan-execute-judge-reflect-confirm-retry loop on thirteen remote-sensing cases. Its
MCP toolchain combines auditable raster operations with health-gated GPU
perception and supports visitor-provided models without storing API credentials.

- Studio: https://modelscope.cn/studios/wbx929/RSMeM
- Learn article: [RSMeM on ModelScope Learn](https://modelscope.cn/learn/435887)
- Full article source: [`LEARN_ARTICLE.md`](LEARN_ARTICLE.md)

RSMeM combines Hierarchical Knowledge Grounding (HKG) and Failure-Aware
Experience Refinement (FAR) to improve the reliability and reproducibility of
multi-step remote-sensing agents.

This ModelScope xGPU Studio provides an executable demonstration of the
core memory workflow. It retrieves structured domain knowledge, runs a ReAct
reason–tool–observe loop over real raster operations, synthesizes an evidence-grounded answer, applies either a
reference-aware case judge or a ground-truth-free Teacher Failure Gate, and lets
the user choose which reflection enters memory.
The Studio starts Analysis, Index, Inversion, Statistics, EarthBench Perception,
and OEA Perception as independent stdio MCP processes. GPU tools are visible to
the ReAct agent only while their local model worker reports healthy.

## Observable RSMeM loop

The interactive workflow mirrors the control flow implemented by the released
framework:

1. Retrieve query-relevant hierarchical knowledge (HKG).
2. Discover the live tool catalog with MCP `list_tools` and give it to a
   LangChain/LangGraph ReAct agent.
3. Let the model choose an MCP `call_tool`, observe its real result, and adapt
   the next tool call until it has enough evidence; record every parameter,
   observation, and downloadable artifact.
4. Synthesize a final answer using only real tool evidence.
5. Evaluate bundled cases with a conservative deterministic contract first and
   fall back to a reference-aware LLM judge when the contract is inconclusive; evaluate uploads
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
reflector or next ReAct attempt. For uploads, the Teacher Failure Gate has no answer
key and provides advisory evidence only; the user's decision is authoritative.

The hosted Studio accepts `.tif`, `.tiff`, `.png`, `.jpg`, and `.jpeg` uploads.
Raster calculations recognize common
band names such as `B4_red`, `B5_nir`, `B6_swir1`, and `B7_swir2`. Single-band
files are safest because raster calculations read band 1. PNG and JPEG inputs
are routed only to compatible perception tools; NetCDF and HDF are not accepted.
If no files are uploaded, visitors
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

The runtime exposes four raster MCP services—**Analysis, Index, Inversion, and
Statistics**—plus two perception façades: **EarthBench Perception** and **OEA
Perception**. The xGPU resident profile covers RemoteCLIP, SAM2, ChangeOS,
BRIGHT, Grounding DINO, LAE-DINO, Qwen3-VL, and OCR. An unavailable worker is
removed from the ReAct tool catalog rather than replaced with a fake result.

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

The interactive backend uses a bounded LangChain/LangGraph ReAct loop over the
live stdio MCP catalogue. See [docs/REACT_HARNESS.md](docs/REACT_HARNESS.md) for
budgets, SQLite checkpoint semantics, provider compatibility probes, and the
20-case tool-selection benchmark.

```bash
pip install -r requirements.txt
python app.py
```

## xGPU perception deployment

The exact resident models, QY_11 source paths, upstream links, and xGPU target
paths are listed in [`XGPU_MODEL_MANIFEST.md`](XGPU_MODEL_MANIFEST.md).

Large sources and checkpoints are deliberately excluded from Git. Keep them on
the persistent xGPU volume (default `/mnt/workspace`):

```bash
export MODELSCOPE_API_TOKEN=...  # configure as a ModelScope Studio secret
python scripts/prepare_xgpu_workspace.py --download-deployment-bundle
python scripts/prepare_xgpu_workspace.py --bootstrap-envs
python scripts/prepare_xgpu_workspace.py  # validates every required asset
```

`snapshot_download()` is guarded by exact-file validation and an inter-process
lock. It runs only when the persistent `/mnt/workspace/models` bundle is absent
or incomplete; ordinary user requests never download weights. The private
deployment bundle defaults to
`wbx929/RSMeM-Perception-Models` and can be overridden with
`RSMEM_MODELSCOPE_MODEL_REPO`.

The same private repository contains a minimal source archive, EasyOCR weights,
and six relocatable conda-pack archives. Bootstrap preserves the validated QY_11
environment boundaries: separate Python 3.10 prefixes for `sam2`, `remoteclip`,
`changeos`, and `bright`, plus OEA's original `oea-e1` and `oea-e3` split under
`/mnt/workspace/envs`; it never runs Git, conda, or pip and never contacts
GitHub, Hugging Face, PyPI, or Anaconda. Subsequent Studio restarts reuse
checksum-backed completion markers. `RSMEM_GPU_BOOTSTRAP_ENVS=0` disables
automatic environment extraction.

After extracting the packaged sources, the bootstrap applies three small,
version-controlled worker fixes from [`xgpu_workers/`](xgpu_workers/README.md).
These are the same ChangeOS preprocessing, LAE-DINO/BERT compatibility, and
safe-registration versions exercised on QY_11; no model implementation is
replaced by a mock or placeholder.

All checkpoints, sources, OCR assets, and worker environments are downloaded as
one versioned ModelScope bundle. Configure separate Python interpreters only
when overriding the packaged defaults:

```bash
export RSMEM_GPU_AUTOSTART=1
export RSMEM_ENABLE_GPU_PERCEPTION=1
export RSMEM_OEA_E1_PYTHON=/mnt/workspace/envs/oea-e1/bin/python
export RSMEM_OEA_E3_PYTHON=/mnt/workspace/envs/oea-e3/bin/python
export RSMEM_EARTHBENCH_PYTHON=/mnt/workspace/envs/sam2/bin/python
```

Optional `RSMEM_GPU_SERVICES` is a comma-separated allowlist of service names
from `gpu_service_manager.py`. Default concurrency is one request per OEA
worker. Logs go to `/mnt/workspace/rsmem-gpu-logs`; no credential or checkpoint
is committed to this repository.

The recommended 48 GB resident profile is approximately 37.6 GB after warm-up.
Keep steady allocation below 40 GB and peak allocation below 44 GB. Do not run
the legacy `start_all.sh`, because it also starts the redundant/on-demand models
and may stop already healthy services.
