import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


from deploy_paths import models_dir

MODEL_ROOT = models_dir()
MODEL_PATH = Path(os.environ.get("CHANGEOS_MODEL_PATH", str(MODEL_ROOT / "changeos_r101.pt")))
DEVICE_NAME = os.environ.get("CHANGEOS_DEVICE", "cuda:0")
HOST = os.environ.get("CHANGEOS_HOST", "0.0.0.0")
PORT = int(os.environ.get("CHANGEOS_PORT", "8019"))
OUTPUT_DIR = Path(os.environ.get("CHANGEOS_OUTPUT_DIR", "./changeos_results"))

logger = logging.getLogger("ChangeOSAPI")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

app = FastAPI(title="ChangeOS API", version="1.0.0")

model = None
device = torch.device("cpu")
load_error: Optional[str] = None


class ChangeOSRequest(BaseModel):
    pre_image_path: str
    post_image_path: str
    output_path: str


class ChangeOSResponse(BaseModel):
    success: bool
    output_path: Optional[str] = None
    task_type: Optional[str] = None
    inference_time: Optional[float] = None
    total_time: Optional[float] = None
    error: Optional[str] = None


def _read_image(image_path: str) -> tuple[np.ndarray, tuple[int, int]]:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    original_size = (image.shape[1], image.shape[0])
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    return image_resized, original_size


def _preprocess_pair(pre_image: np.ndarray, post_image: np.ndarray) -> torch.Tensor:
    pair = np.concatenate([pre_image, post_image], axis=2)
    tensor = torch.from_numpy(pair).permute(2, 0, 1).float()
    mean = torch.tensor([123.675, 116.28, 103.53, 123.675, 116.28, 103.53]).view(6, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375, 58.395, 57.12, 57.375]).view(6, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0).to(device)


def _to_mask(prediction: torch.Tensor) -> np.ndarray:
    array = prediction.detach().float().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3:
        if array.shape[0] > 1:
            array = np.argmax(array, axis=0)
        else:
            array = array[0]
    array = np.squeeze(array)
    if array.dtype != np.uint8:
        array = (array > 0.5).astype(np.uint8)
    return array


def _visualize(building_mask: np.ndarray, damage_mask: Optional[np.ndarray]) -> np.ndarray:
    height, width = building_mask.shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[building_mask > 0] = (0, 255, 0)
    if damage_mask is not None:
        canvas[damage_mask > 0] = (0, 0, 255)
        both = (building_mask > 0) & (damage_mask > 0)
        canvas[both] = (0, 255, 255)
    return canvas


@app.on_event("startup")
async def startup_event() -> None:
    global model, device, load_error
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        device = torch.device(DEVICE_NAME if torch.cuda.is_available() else "cpu")
        logger.info("Loading ChangeOS model from %s on %s", MODEL_PATH, device)
        model = torch.jit.load(str(MODEL_PATH), map_location=device)
        model.eval()
        load_error = None
    except Exception as exc:
        model = None
        load_error = str(exc)
        logger.error("Failed to load ChangeOS: %s", exc, exc_info=True)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ready" if model is not None else "error",
        "model_loaded": model is not None,
        "device": str(device),
        "checkpoint": str(MODEL_PATH),
        "error": load_error,
    }


@app.post("/changeos", response_model=ChangeOSResponse)
async def changeos_infer(request: ChangeOSRequest) -> ChangeOSResponse:
    if model is None:
        raise HTTPException(status_code=503, detail=f"ChangeOS model unavailable: {load_error}")

    started = datetime.now()
    pre_path = Path(request.pre_image_path).resolve()
    post_path = Path(request.post_image_path).resolve()
    output_path = Path(request.output_path).resolve()

    if not pre_path.exists() or not post_path.exists():
        raise HTTPException(status_code=404, detail="Input image path not found")

    try:
        pre_image, original_size = _read_image(str(pre_path))
        post_image, _ = _read_image(str(post_path))
        input_tensor = _preprocess_pair(pre_image, post_image)

        inference_started = datetime.now()
        with torch.no_grad():
            outputs = model(input_tensor)
        inference_time = (datetime.now() - inference_started).total_seconds()

        if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            building_mask = _to_mask(outputs[0])
            damage_mask = _to_mask(outputs[1])
            task_type = "building_extraction" if pre_path == post_path else "change_detection"
        else:
            building_mask = _to_mask(outputs)
            damage_mask = None
            task_type = "building_extraction"

        visualization = _visualize(building_mask, None if task_type == "building_extraction" else damage_mask)
        visualization = cv2.resize(visualization, original_size, interpolation=cv2.INTER_NEAREST)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), visualization)

        total_time = (datetime.now() - started).total_seconds()
        return ChangeOSResponse(
            success=True,
            output_path=str(output_path),
            task_type=task_type,
            inference_time=inference_time,
            total_time=total_time,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ChangeOS inference failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
