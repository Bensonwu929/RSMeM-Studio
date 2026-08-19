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

RSMeM combines Hierarchical Knowledge Grounding (HKG) and Failure-Aware
Experience Refinement (FAR) to improve the reliability and reproducibility of
multi-step remote-sensing agents.

This ModelScope Studio provides a CPU-friendly, executable demonstration of the
core memory workflow. It retrieves structured domain knowledge, plans and runs
real raster operations, applies a critic, turns failed trajectories into reusable
constraints, and can call an OpenAI-compatible LLM API for planning/reflection.
The full experimental framework and MCP tool stack remain available in the
public source repository.

## Observable RSMeM loop

The interactive workflow mirrors the control flow implemented by the released
framework:

1. Retrieve query-relevant hierarchical knowledge (HKG).
2. Plan a short remote-sensing tool chain.
3. Execute a constrained Index/Statistics tool subset over real GeoTIFF data and
   record tool calls, parameters, observations, and downloadable artifacts.
4. Apply a critic to the result.
5. On failure only, compress the trajectory into a reusable FAR reflection and
   write it to Failure-Aware Memory.
6. Inject both HKG and the latest reflection into the next attempt, rerun the
   failed task, and stop after success or the attempt limit.

The hosted Studio accepts optional `.tif`/`.tiff` uploads and recognizes common
band names such as `B4_red`, `B5_nir`, `B6_swir1`, and `B7_swir2`. If no files
are uploaded, it creates a deterministic synthetic GeoTIFF scene so the whole
pipeline remains testable. The CPU runtime mirrors selected public RSMeM
Index/Statistics tool contracts using Rasterio and NumPy. GPU perception tools
are intentionally disabled on the free CPU instance.

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

For a team-managed deployment, optional environment variables remain supported:
`RSMEM_API_KEY`, `RSMEM_BASE_URL`, and `RSMEM_MODEL`. Never commit credentials.
If no key is supplied or the provider is temporarily
unavailable, the Studio remains usable through a deterministic offline fallback.

## Resources

- Source: https://github.com/AI9Stars/RSMeM
- ModelScope Learn: https://modelscope.cn/learn/434564
- License: MIT

## Local launch

```bash
pip install -r requirements.txt
python app.py
```

## Scope

The Studio is an executable artifact for the Open Source GeoAI Practice
Challenge. It runs a safe CPU subset of the released tool stack, but does not
claim to run the GPU perception services or the complete EarthBench evaluation.
