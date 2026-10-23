# RSMeM Studio

**An end-to-end interactive remote sensing agent platform deployed on ModelScope.**

RSMeM Studio brings remote sensing analysis into an interactive workflow, from natural-language requests and image uploads to tool execution and result visualization. It supports both perception tasks, such as object detection, segmentation, and change detection, and quantitative parameter retrieval, including land surface temperature and water turbidity.

🏆 **RSMeM won Second Prize in the Open Source GeoAI Practice Challenge at IEEE APGRSS 2026.**

[**Try the Studio →**](https://modelscope.cn/studios/wbx929/RSMeM) · [Research repository](https://github.com/AI9Stars/RSMeM) · [Published article](https://modelscope.cn/learn/435887)

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
under Apache-2.0. See [DATA_LICENSE.md](../DATA_LICENSE.md). Studio code is
MIT-licensed; the example data keeps its original license.

The optional GPU perception stack uses SAM 2.1 (segmentation), RemoteCLIP
(scene classification), ChangeOS (change detection), BRIGHT/SiamCRNN (damage
assessment), Grounding DINO and LAE-DINO (object localization and detection),
and Qwen3-VL-8B-Instruct (visual description and counting), alongside an OCR
worker. Their checkpoints and supporting assets come from
[RSMeM-Perception-Models](https://modelscope.cn/models/wbx929/RSMeM-Perception-Models).
The Studio's MIT license covers its code; third-party model weights and source
packages are subject to their respective upstream terms.

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
