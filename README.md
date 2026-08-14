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

This lightweight ModelScope Studio provides an API-free, CPU-friendly
demonstration of the core memory workflow. It retrieves structured domain
knowledge and turns an example failure trajectory into a transparent reusable
constraint. The full experimental framework and MCP-based remote-sensing tool
stack remain available in the public source repository.

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
Challenge. It does not require private credentials and does not claim to run the
full GPU perception stack or the complete EarthBench evaluation in-browser.
