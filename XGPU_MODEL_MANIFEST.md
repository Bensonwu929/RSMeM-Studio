# xGPU perception model manifest

This is the 48 GB Ada **resident** profile used by the Studio. It deliberately
does not load every historical EarthBench/OEA model at once. Large files stay in
`/mnt/workspace/models` and must not be committed to Git.

| Capability / MCP tool | Backend | Download source on your server | xGPU destination | Upstream reference |
|---|---|---|---|---|
| bbox-guided segmentation / `SAM2` | SAM 2.1 Hiera Small | `QY_11:/slow_share2/ultrafast_share/gzh_data/wubingxian/RSMeM-public/models/sam2.1_hiera_small.pt` | `/mnt/workspace/models/sam2.1_hiera_small.pt` | [Meta SAM 2.1 checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt) |
| scene classification / `RemoteCLIP` | RemoteCLIP ViT-L/14 | `QY_11:/slow_share2/ultrafast_share/gzh_data/wubingxian/RSMeM-public/models/RemoteCLIP-ViT-L-14.pt` | `/mnt/workspace/models/RemoteCLIP-ViT-L-14.pt` | [RemoteCLIP model repository](https://huggingface.co/chendelong/RemoteCLIP/blob/main/RemoteCLIP-ViT-L-14.pt) |
| bi-temporal change mask / `ChangeOS` | ChangeOS R101 TorchScript export | `QY_11:/slow_share2/ultrafast_share/gzh_data/wubingxian/RSMeM-public/models/changeos_r101.pt` | `/mnt/workspace/models/changeos_r101.pt` | [ChangeOS source](https://github.com/Z-Zheng/ChangeOS) (use the tested QY_11 export) |
| building damage / `BRIGHT_DamageAssessment` | BRIGHT SiamCRNN | `QY_11:/slow_share2/ultrafast_share/gzh_data/wubingxian/RSMeM-public/models/SiamCRNN_20250211_033615_1/best_model.pth` | `/mnt/workspace/models/SiamCRNN_20250211_033615_1/best_model.pth` | [BRIGHT source](https://github.com/ChenHongruixuan/BRIGHT) (use the tested QY_11 checkpoint) |
| text-to-box / `TextToBbox` | Grounding DINO Swin-T OGC | `QY_11:/slow_share2/ultrafast_share/gzh_data/baijin/OpenEarthAgent/models/groundingdino_swint_ogc.pth` | `/mnt/workspace/models/groundingdino_swint_ogc.pth` | [Official release checkpoint](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) |
| open-vocabulary detection / `ObjectDetection` | LAE-DINO Swin-T | `QY_11:/slow_share2/ultrafast_share/gzh_data/baijin/OpenEarthAgent/models/lae_dino_swint_lae1m-28ca3a15.pth` | `/mnt/workspace/models/lae_dino_swint_lae1m-28ca3a15.pth` | [LAE-DINO source](https://github.com/jaychempan/LAE-DINO), [official OEA checkpoint link](https://drive.google.com/uc?id=1EiR8KtNRYIeOfvtIe9C82cQk_uOMIQ8U) |
| LAE-DINO language encoder (auxiliary) | BERT base uncased, safetensors-only bundle | `QY_11:/slow_share2/ultrafast_share/gzh_data/baijin/OpenEarthAgent/models/bert-base-uncased/` | `/mnt/workspace/models/bert-base-uncased/` | [Google BERT](https://huggingface.co/google-bert/bert-base-uncased) |
| description, region attributes, counting / three OEA tools | Qwen3-VL-8B-Instruct | `QY_11:/slow_share2/ultrafast_share/gzh_data/baijin/OpenEarthAgent/models/Qwen3-VL-8B-Instruct/` | `/mnt/workspace/models/Qwen3-VL-8B-Instruct/` | [ModelScope model](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct) |
| OCR / `OCR` | EasyOCR detector + Chinese/English recognizers | `QY_11:/slow_share2/ultrafast_share/gzh_data/baijin/easyocr/model/` | `/mnt/workspace/easyocr/model/` (set `EASYOCR_MODULE_PATH=/mnt/workspace/easyocr`) | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |

Verified sizes on QY_11 (2026-09-03): SAM2 184,416,285 bytes;
RemoteCLIP 1,710,613,765; ChangeOS 233,553,904; BRIGHT 132,432,363;
Grounding DINO 693,997,677; LAE-DINO 724,694,893; the minimal BERT inference
bundle 441,147,956; Qwen3-VL directory 17,545,917,594 bytes. EasyOCR weights are
included in the offline source bundle; workers do not download them at startup.

## Offline deployment archives

The private ModelScope model repository also stores the following build outputs
under `offline/`:

- `rsmem-perception-sources-v1.tar.gz`: minimal EarthBench/OEA worker sources,
  GroundingDINO, LAE-DINO, and EasyOCR weights;
- `sam2-py310-cu121-v1.tar.gz`, `remoteclip-py310-cu121-v1.tar.gz`,
  `changeos-py310-cu121-v1.tar.gz`, and `bright-py310-cu121-v1.tar.gz`: the
  four independently validated EarthBench runtimes, kept separate exactly as
  on QY_11;
- `oea-e1-py310-cu124-v1.tar.gz`: controller, OCR, GroundingDINO, and Qwen3-VL;
- `oea-e3-py310-cu124-v1.tar.gz`: repaired and validated LAE-DINO runtime;
- `SHA256SUMS`: mandatory integrity manifest.

Studio startup downloads only from ModelScope, verifies every archive, extracts
it into `/mnt/workspace`, runs `conda-unpack`, and then starts HTTP workers. No
runtime Git clone or dependency installation is permitted.

## Source trees and configs

- EarthBench service source:
  `/slow_share2/ultrafast_share/gzh_data/wubingxian/RSMeM-public/tools/deploy_tools`
- OEA worker source:
  `/slow_share2/ultrafast_share/gzh_data/baijin/OpenEarthAgent/tool_server/tool_workers`
- Grounding DINO config target:
  `/mnt/workspace/models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py`
- LAE-DINO config target:
  `/mnt/workspace/models/LAE-DINO/mmdetection_lae/configs/lae_dino/lae_dino_swin-t_pretrain_LAE-1M.py`

## Not resident on the 48 GB profile

RemoteSAM, Strip-R-CNN, SM3Det, Qwen2.5-VL + InstructSAM, OEA SAM2-large,
and TEOChat remain on-demand. Loading them together with Qwen3-VL would add
redundant capabilities and reduce the safety margin for inference peaks.
