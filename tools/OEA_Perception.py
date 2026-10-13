"""MCP facade for the resident OpenEarthAgent perception workers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import requests
from fastmcp import FastMCP


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--controller-url", default=os.getenv("OEA_CONTROLLER_URL", "http://127.0.0.1:21001"))
args, _unknown = parser.parse_known_args()
CONTROLLER_URL = args.controller_url.rstrip("/")
TIMEOUT = float(os.getenv("OEA_TOOL_TIMEOUT_SEC", "300"))

mcp = FastMCP("oea_perception")
_session = requests.Session()
_SUPPORTED = {
    "TextToBbox",
    "ObjectDetection",
    "ImageDescription",
    "RegionAttributeDescription",
    "CountGivenObject",
    "OCR",
}


def _image(path: str) -> str:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise ValueError(f"Input image does not exist: {candidate}")
    return str(candidate)


def _models() -> set[str]:
    try:
        response = _session.post(f"{CONTROLLER_URL}/list_models", timeout=0.75)
        response.raise_for_status()
        return set(response.json().get("models", []))
    except (requests.RequestException, ValueError):
        return set()


@mcp.tool(description="Return only live OpenEarthAgent perception capabilities.")
def available_tools() -> list[str]:
    models = _models()
    capabilities = models & _SUPPORTED
    if "BaseModel" in models:
        capabilities |= {"ImageDescription", "RegionAttributeDescription", "CountGivenObject"}
    return sorted(capabilities)


def _worker(tool: str) -> str:
    worker_model = "BaseModel" if tool in {
        "ImageDescription", "RegionAttributeDescription", "CountGivenObject"
    } else tool
    response = _session.post(
        f"{CONTROLLER_URL}/get_worker_address",
        json={"model": worker_model},
        timeout=5,
    )
    response.raise_for_status()
    address = response.json().get("address", "").rstrip("/")
    if not address:
        raise RuntimeError(f"No live OEA worker is registered for {worker_model}.")
    return address


def _call(tool: str, payload: dict[str, Any]) -> dict[str, Any]:
    address = _worker(tool)
    response = _session.post(f"{address}/worker_generate", json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    result = response.json()
    return {
        "status": "success" if result.get("error_code", 0) == 0 else "worker_error",
        "tool": tool,
        "text": result.get("text", ""),
        "error_code": result.get("error_code", 0),
    }


@mcp.tool(description="Locate objects from an English prompt using Grounding DINO.")
def TextToBbox(image: str, text: str, top1: bool = False, box_threshold: float = 0.25, text_threshold: float = 0.25):
    return _call("TextToBbox", {"image": _image(image), "text": text, "top1": top1, "box_threshold": box_threshold, "text_threshold": text_threshold})


@mcp.tool(description="Detect remote-sensing objects with LAE-DINO.")
def ObjectDetection(image: str, text: str = "", top1: bool = False, box_threshold: float = 0.25):
    return _call("ObjectDetection", {"image": _image(image), "text": text, "top1": top1, "box_threshold": box_threshold})


@mcp.tool(description="Describe a remote-sensing image with the resident vision-language model.")
def ImageDescription(image: str):
    return _call("ImageDescription", {"image": _image(image), "query": "Describe this remote-sensing image using observable evidence."})


@mcp.tool(description="Describe a requested attribute inside an optional pixel bounding box.")
def RegionAttributeDescription(image: str, attribute: str, bbox: str = ""):
    query = f"Describe the {attribute}"
    if bbox:
        query += f" inside pixel bounding box {bbox}"
    return _call("RegionAttributeDescription", {"image": _image(image), "query": query})


@mcp.tool(description="Count instances of a requested object in a remote-sensing image.")
def CountGivenObject(image: str, text: str, bbox: str = ""):
    query = f"Count the visible instances of {text}."
    if bbox:
        query += f" Restrict the count to pixel bounding box {bbox}."
    return _call("CountGivenObject", {"image": _image(image), "query": query})


@mcp.tool(description="Recognize text and return text with pixel bounding boxes.")
def OCR(image: str):
    return _call("OCR", {"image": _image(image)})


if __name__ == "__main__":
    mcp.run()
