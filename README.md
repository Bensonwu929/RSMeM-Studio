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

# RSMeM

RSMeM is a remote-sensing agent Studio for exploring geospatial imagery and
running reproducible analysis with raster and perception tools. Visitors can
upload supported images or try the bundled examples.

- [Open the Studio](https://modelscope.cn/studios/wbx929/RSMeM)
- [Read the published article](https://modelscope.cn/learn/435887)

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

The interface accepts `.tif`, `.tiff`, `.png`, `.jpg`, and `.jpeg` files. Agent
inference requires an OpenAI-compatible model endpoint and an API key supplied
through the interface or environment variables. GPU perception tools also
require the corresponding model assets and a suitable GPU environment.

## Data and licenses

The bundled example GeoTIFFs include subsets of
[Sssunset/Earth-Bench](https://huggingface.co/datasets/Sssunset/Earth-Bench)
under Apache-2.0. See [DATA_LICENSE.md](DATA_LICENSE.md). Studio code is
MIT-licensed; the example data keeps its original license.

## Citation

If you find RSMeM useful in your research, please consider citing our paper:

```bibtex
@inproceedings{wu-etal-2026-rsmem,
  title = "{RSM}e{M}: Knowledge-Enhanced Memory Evolution for Remote {S}ensing Agents with Systematic Evaluation",
  author = "Wu, Bingxian and Zhang, Yu and Guo, Zonghao and Liu, Tang and Qian, Chen and Lu, Yuxiang and Du, Xingbo and Li, Yanghao and Zhang, Yidan and Chen, Chi and Yao, Ling and Zhou, Chenghu and Sun, Maosong",
  booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
  month = jul,
  year = "2026",
  address = "San Diego, California, United States",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2026.acl-long.1519/",
  doi = "10.18653/v1/2026.acl-long.1519",
  pages = "32899--32915"
}
```
