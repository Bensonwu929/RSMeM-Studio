import io
import logging
import uuid
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

import torch
import numpy as np
import cv2 # 用于图像处理和分割结果的绘制/保存
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
import uvicorn

# 假设以下导入是有效的，它们是您代码中使用的SAM2组件
# 注意：在实际部署环境中，您需要确保这些路径在容器或服务器上是可访问的
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 全局参数 (SAM2 特有) ---
from deploy_paths import models_dir

MODEL_ROOT = models_dir()
SAM2_CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT",
    str(MODEL_ROOT / "sam2.1_hiera_small.pt"),
)
SAM2_MODEL_CFG = os.environ.get(
    "SAM2_MODEL_CFG",
    "configs/sam2.1/sam2.1_hiera_s.yaml",
)
SAM2_DEVICE = os.environ.get("SAM2_DEVICE", "cuda:0")
SAM2_HOST = os.environ.get("SAM2_HOST", "0.0.0.0")
SAM2_PORT = int(os.environ.get("SAM2_PORT", "5554"))
OUTPUT_DIR = Path("./sam2_server_results")
IMAGE_STORAGE_DIR = OUTPUT_DIR / "images"
RESULT_STORAGE_DIR = OUTPUT_DIR / "results"

# --- 全局配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SAM2SegmentationAPI")

app = FastAPI(
    title="SAM2 图像分割 API",
    description="支持 BBox/点提示的图像分割服务接口，基于 SAM2 模型",
    version="1.0.0"
)

# 全局变量
sam2_model: Optional[Any] = None
sam2_predictor: Optional[SAM2ImagePredictor] = None
device: Optional[torch.device] = None

# --- API 数据模型 ---
class SegmentationRequest(BaseModel):
    bbox: Optional[Tuple[int, int, int, int]] = Field(None, description="可选的边界框提示 (x, y, w, h)")
    point_coords: Optional[List[Tuple[int, int]]] = Field(None, description="可选的点坐标提示 [(x1, y1), ...]")
    point_labels: Optional[List[int]] = Field(None, description="点坐标对应的标签 [1, 0, ...]")

class SegmentationResponse(BaseModel):
    request_id: str = Field(..., description="请求的唯一标识符")
    input_image_url: str = Field(..., description="已保存的输入图像URL")
    bounding_boxes: List[Tuple[int, int, int, int]] = Field(..., description="检测到的对象边界框列表 (x, y, w, h)")
    visualization_url: str = Field(..., description="分割结果可视化图URL")
    mask_url: str = Field(..., description="最佳分割 Mask 二值图URL")
    status: str = Field(..., description="请求处理状态")
    device_used: Optional[str] = Field(None, description="处理请求使用的设备")


# --- 辅助函数 (从原代码移植并适应API环境) ---

def get_available_devices() -> str:
    if torch.cuda.is_available():
        return f"GPU (CUDA: {torch.cuda.get_device_name(0)})"
    return "CPU"

def process_and_save_image_bgr(image_file: UploadFile, request_id: str) -> Path:
    """处理上传的图像并保存，返回保存的路径"""
    try:
        # 创建图像存储目录
        IMAGE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 读取内容
        contents = image_file.file.read()
        image_np = np.frombuffer(contents, np.uint8)
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise ValueError("无法解析图像内容，请检查文件格式。")

        # 生成唯一图像文件名
        image_ext = image_file.filename.split(".")[-1].lower() if "." in image_file.filename else "jpg"
        image_filename = f"{request_id}_input.{image_ext}"
        image_path = IMAGE_STORAGE_DIR / image_filename
        
        # 保存图像（OpenCV默认BGR）
        cv2.imwrite(str(image_path), image_bgr)
        logger.info(f"输入图像保存至: {image_path}")
        
        return image_path
        
    except Exception as e:
        logger.error(f"处理图像失败: {e}")
        raise HTTPException(status_code=400, detail=f"无法处理图像: {e}")
    finally:
        image_file.file.close()

# 导入原代码中的辅助函数（需要手动复制到实际文件中或导入）
# from [您的模块] import mask_to_bounding_boxes, visualize_masks_with_boxes, save_mask_image, save_original_with_mask_overlay
# ... (假设这些函数已包含在您的运行环境中)
# 为了使代码可执行，我将这些函数简化/假设存在，但您应使用您原代码中的完整实现。

def _mask_to_bounding_boxes(masks: np.ndarray, min_area: int = 100) -> List[Tuple[int, int, int, int]]:
    # 假设使用原代码中的实现
    # ... [原 mask_to_bounding_boxes 函数代码]
    bounding_boxes = []
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    
    for mask in masks:
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))
    return bounding_boxes

def _save_mask_image(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.dtype != np.uint8 else mask
    cv2.imwrite(str(output_path), mask_uint8)

def _save_visualization(image_rgb: np.ndarray, masks: np.ndarray, bboxes: List[Tuple], output_path: Path) -> None:
    """Tint only mask pixels so the rest of the frame keeps source brightness."""
    vis_image = image_rgb.copy()
    mask_list = [masks] if masks.ndim == 2 else list(masks)
    fill = np.array([0, 200, 255], dtype=np.float32)
    fill_alpha = 0.35
    outline_color = np.array([0, 220, 255], dtype=np.uint8)
    edge_kernel = np.ones((5, 5), dtype=np.uint8)

    for mask in mask_list:
        binary = mask > 0
        vis_image[binary] = (
            vis_image[binary].astype(np.float32) * (1.0 - fill_alpha) + fill * fill_alpha
        ).astype(np.uint8)
        mask_u8 = binary.astype(np.uint8) * 255
        eroded = cv2.erode(mask_u8, edge_kernel, iterations=1)
        vis_image[(mask_u8 > 0) & (eroded == 0)] = outline_color

    for x, y, w, h in bboxes:
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 60, 60), 3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))


# --- 服务启动事件 ---
@app.on_event("startup")
async def startup_event():
    global sam2_model, sam2_predictor, device
    logger.info("SAM2 服务器启动，开始加载模型...")
    
    # 创建必要目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录 '{OUTPUT_DIR.resolve()}' 已准备就绪。")
    
    try:
        device = torch.device(SAM2_DEVICE if torch.cuda.is_available() else "cpu")
        logger.info(f"检测到可用设备: {get_available_devices()}")
        logger.info(f"将使用设备: {device}")
        
        # 加载 SAM2 模型
        logger.info(f"加载 SAM2 模型...")
        sam2_model = build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT)
        sam2_model = sam2_model.eval().to(device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        
        # 预热 (可选，但推荐)
        logger.info("正在进行模型预热...")
        # ... (此处可添加一个 SAM2 模型的 dummy 预热步骤) ...
        logger.info("模型预热完成")
        
        logger.info("SAM2 模型加载完成，服务准备就绪")
        
    except Exception as e:
        logger.error(f"初始化 SAM2 模型时发生错误: {e}", exc_info=True)
        sam2_model = None
        sam2_predictor = None
        device = None


# --- SAM2 核心推理逻辑 ---

def segment_image_sam2(
    image_bgr: np.ndarray, 
    bbox: Optional[Tuple[int, int, int, int]] = None, 
    point_coords: Optional[np.ndarray] = None, 
    point_labels: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
    """执行 SAM2 分割，返回所有 masks, 最佳 mask 和对应的 BBox 列表"""
    
    if sam2_predictor is None:
        raise RuntimeError("SAM2 预测器未加载。")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 变量初始化
    final_point_coords = point_coords
    final_point_labels = point_labels
    
    # 默认中心点逻辑
    if bbox is None and point_coords is None:
        H, W, _ = image_rgb.shape
        cx, cy = W // 2, H // 2
        final_point_coords = np.array([[cx, cy]])
        final_point_labels = np.array([1])
        logger.warning(f"无提示输入，默认使用中心点 ({cx}, {cy})。")

    try:
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            sam2_predictor.set_image(image_rgb)
            # A box is a non-ambiguous prompt; SAM2's dedicated single-mask
            # token (masks[:, 0:1]) is the documented setting. The 3-way
            # whole/part/subpart triplet is only useful for a point click.
            multimask_output = bbox is None
            masks, scores, _ = sam2_predictor.predict(
                point_coords=final_point_coords,
                point_labels=final_point_labels,
                box=np.array(bbox) if bbox is not None else None,
                multimask_output=multimask_output,
            )
            
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        bounding_boxes = _mask_to_bounding_boxes(best_mask)
        
        return masks, best_mask, bounding_boxes
    
    except Exception as e:
        logger.error(f"SAM2 推理失败: {e}", exc_info=True)
        raise RuntimeError(f"SAM2 推理过程中发生错误: {str(e)}")


# --- API 端点 ---

@app.get("/health", summary="健康检查")
async def health_check():
    status = "ready" if sam2_predictor is not None else "model_loading_failed"
    device_info = str(device) if device else None
    
    return {
        "status": status,
        "model_loaded": sam2_predictor is not None,
        "device": device_info,
        "available_devices": get_available_devices(),
        "model_checkpoint": SAM2_CHECKPOINT
    }

@app.get("/results/{filename}", summary="获取结果文件")
async def get_result_file(filename: str):
    file_path = RESULT_STORAGE_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    from fastapi.responses import FileResponse
    return FileResponse(file_path)

@app.get("/images/{image_filename}", summary="获取输入图像")
async def get_input_image(image_filename: str):
    image_path = IMAGE_STORAGE_DIR / image_filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="图像不存在")
    
    from fastapi.responses import FileResponse
    return FileResponse(image_path)


@app.post("/segment", response_model=SegmentationResponse, summary="SAM2 图像分割")
async def segment(
    image: UploadFile = File(..., description="要分割的图像文件"),
    bbox_str: Optional[str] = Form(None, description="可选的边界框提示 (x, y, w, h)，例如 '100,200,300,400'")
):
    """接收图像和可选提示（BBox），使用 SAM2 进行分割"""
    
    # 1. 检查服务状态
    if sam2_predictor is None:
        raise HTTPException(status_code=503, detail="服务不可用：SAM2 预测器未成功加载。")
    if device is None:
        raise HTTPException(status_code=503, detail="服务不可用：未检测到可用计算设备。")

    request_id = str(uuid.uuid4())
    logger.info(f"收到新的分割请求 {request_id}，使用设备: {device}")
    
    # 2. 解析 BBox 参数
    input_bbox: Optional[Tuple[int, int, int, int]] = None
    if bbox_str:
        try:
            input_bbox = tuple(map(int, bbox_str.split(',')))
            if len(input_bbox) != 4:
                 raise ValueError("BBox 必须包含 4 个整数")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"BBox 格式错误，请使用 'x,y,w,h' 格式的整数: {e}")

    # 3. 处理并保存图像
    input_image_path = process_and_save_image_bgr(image, request_id)
    input_image_bgr = cv2.imread(str(input_image_path))
    
    # 4. 执行 SAM2 分割
    try:
        # masks 包含所有预测结果，best_mask 是分数最高的 Mask
        all_masks, best_mask, bboxes = segment_image_sam2(
            image_bgr=input_image_bgr,
            bbox=input_bbox
        )
        
        # 5. 保存结果文件
        vis_filename = f"{request_id}_vis.jpg"
        mask_filename = f"{request_id}_mask.png"

        vis_path = RESULT_STORAGE_DIR / vis_filename
        mask_path = RESULT_STORAGE_DIR / mask_filename
        
        # Visualize the mask that was actually saved, not the rejected candidates.
        image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
        _save_visualization(image_rgb, best_mask, bboxes, vis_path)
        _save_mask_image(best_mask, mask_path)
        
        # 6. 构建响应 URL
        input_image_url = f"/images/{input_image_path.name}"
        visualization_url = f"/results/{vis_filename}"
        mask_url = f"/results/{mask_filename}"
        
        logger.info(f"请求 {request_id} 处理完成，检测到 {len(bboxes)} 个对象。")
        
        return SegmentationResponse(
            request_id=request_id,
            input_image_url=input_image_url,
            bounding_boxes=bboxes,
            visualization_url=visualization_url,
            mask_url=mask_url,
            status="success",
            device_used=str(device)
        )
        
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"SAM2推理失败: {str(e)}")
    except Exception as e:
        logger.error(f"处理请求 {request_id} 时发生未知错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")


# --- 运行服务 ---
if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=SAM2_HOST,
        port=SAM2_PORT,
        timeout_keep_alive=600
    )