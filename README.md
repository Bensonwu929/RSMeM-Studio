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

This lightweight ModelScope Studio provides a CPU-friendly demonstration of the
core memory workflow. It retrieves structured domain knowledge, turns an example
failure trajectory into a transparent reusable constraint, and can call an
OpenAI-compatible LLM API to generate an auditable corrected plan. The full
experimental framework and MCP-based remote-sensing tool stack remain available
in the public source repository.

## Observable RSMeM loop

The interactive workflow mirrors the control flow implemented by the released
framework:

1. Retrieve query-relevant hierarchical knowledge (HKG).
2. Plan a short remote-sensing tool chain.
3. Execute the plan and record tool calls, parameters, and observations.
4. Apply a critic to the result.
5. On failure only, compress the trajectory into a reusable FAR reflection and
   write it to Failure-Aware Memory.
6. Inject both HKG and the latest reflection into the next attempt, rerun the
   failed task, and stop after success or the attempt limit.

The hosted Studio uses a clearly labelled sandbox executor because the full
production stack requires external datasets, MCP servers, and GPU perception
services. It demonstrates state transitions and memory evolution; it does not
report sandbox observations as real Earth-observation measurements.

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

The Studio is an explanatory artifact for the Open Source GeoAI Practice
Challenge. It does not claim to run the full GPU perception stack or the
complete EarthBench evaluation in-browser.
