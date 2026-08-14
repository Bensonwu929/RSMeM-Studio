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

## API use

The public interface lets each visitor enter their own ModelScope API token in a
password field and select a model. The default model is
`deepseek-ai/DeepSeek-V4-Flash-0731`, using the ModelScope OpenAI-compatible
inference endpoint `https://api-inference.modelscope.cn/v1`.

Tokens are used only to construct the current inference request and are not
written to files, application output, or logs. Model availability and quota
depend on the visitor's ModelScope account.

For a team-managed deployment, optional environment variables remain supported:
`RSMEM_API_KEY`, `MODELSCOPE_API_TOKEN`, `RSMEM_BASE_URL`, and `RSMEM_MODEL`.
Never commit credentials. If no token is supplied or the provider is temporarily
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
