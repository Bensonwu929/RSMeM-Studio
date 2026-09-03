# Tested xGPU worker overrides

These files are small source-code overrides copied from the working QY_11
deployment on 2026-09-03. They are applied after cloning the public source trees:

- `changeos_api.py`: correct six-channel pair normalization and TorchScript call.
- `ObjectDetection_worker.py`: stable LAE-DINO/BERT attention handling and startup retry.
- `base_tool_worker.py`: register a worker only after its model is initialized.

`ObjectDetection_worker.py` and `base_tool_worker.py` derive from
[OpenEarthAgent](https://github.com/mbzuai-oryx/OpenEarthAgent) and retain its
upstream license. `changeos_api.py` derives from the public RSMeM deployment
source. No checkpoint or credential is stored here.
