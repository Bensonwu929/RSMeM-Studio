"""Lean EarthBench perception MCP facade for the xGPU Studio.

The GPU models live in local HTTP workers.  This process intentionally contains
no model code, so MCP discovery remains fast and a failed worker does not take
down the agent runtime.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from fastmcp import FastMCP


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--temp_dir", required=True)
args, _unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir).resolve()
TEMP_DIR.mkdir(parents=True, exist_ok=True)
TIMEOUT = float(os.getenv("RSMEM_PERCEPTION_TIMEOUT_SEC", "300"))

SAM2_BASE = os.getenv("SAM2_API_URL", "http://127.0.0.1:5554").rstrip("/")
REMOTECLIP_BASE = os.getenv("REMOTECLIP_API_BASE", "http://127.0.0.1:5555").rstrip("/")
CHANGEOS_BASE = os.getenv("CHANGEOS_API_BASE", "http://127.0.0.1:8019").rstrip("/")
BRIGHT_BASE = os.getenv("BRIGHT_API_BASE", "http://127.0.0.1:8120").rstrip("/")

mcp = FastMCP("earthbench_perception")
_session = requests.Session()

_SERVICES = {
    "RemoteCLIP": (f"{REMOTECLIP_BASE}/health", f"{REMOTECLIP_BASE}/classify"),
    "SAM2": (f"{SAM2_BASE}/health", f"{SAM2_BASE}/segment"),
    "ChangeOS": (f"{CHANGEOS_BASE}/health", f"{CHANGEOS_BASE}/changeos"),
    "BRIGHT_DamageAssessment": (
        f"{BRIGHT_BASE}/health",
        f"{BRIGHT_BASE}/bright_damage_assessment",
    ),
}


def _input(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise ValueError(f"Input image does not exist: {candidate}")
    return candidate


def _output(path: str, default_name: str) -> Path:
    requested = Path(path or default_name)
    candidate = requested.resolve() if requested.is_absolute() else (TEMP_DIR / requested).resolve()
    if not candidate.is_relative_to(TEMP_DIR):
        raise ValueError("Perception outputs must stay inside the request output directory.")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _ready(name: str) -> bool:
    try:
        response = _session.get(_SERVICES[name][0], timeout=0.5)
        if not response.ok:
            return False
        payload = response.json()
        return payload.get("status") in {"ready", "ok", "healthy"} or payload.get("model_loaded") is True
    except (requests.RequestException, ValueError):
        return False


def _download_worker_artifact(relative_url: str, destination: Path) -> None:
    """Copy a worker result into the request-scoped MCP output directory."""
    parsed = urlparse(relative_url)
    if parsed.scheme or parsed.netloc or not parsed.path.startswith("/results/"):
        raise RuntimeError("SAM2 returned an invalid result URL.")
    response = _session.get(urljoin(f"{SAM2_BASE}/", relative_url.lstrip("/")), timeout=TIMEOUT)
    response.raise_for_status()
    destination.write_bytes(response.content)


@mcp.tool(description="Return only EarthBench perception workers whose health endpoint is ready.")
def available_tools() -> list[str]:
    return [name for name in _SERVICES if _ready(name)]


@mcp.tool(description="Classify a remote-sensing scene with RemoteCLIP and return top predictions.")
def RemoteCLIP(input_image_path: str) -> dict[str, Any]:
    image = _input(input_image_path)
    if not _ready("RemoteCLIP"):
        raise RuntimeError("RemoteCLIP worker is not ready.")
    with image.open("rb") as handle:
        response = _session.post(
            _SERVICES["RemoteCLIP"][1],
            files={"file": (image.name, handle, "application/octet-stream")},
            timeout=TIMEOUT,
        )
    response.raise_for_status()
    return response.json()


@mcp.tool(description="Segment an object with SAM2 from a pixel bbox [x, y, width, height].")
def SAM2(input_image_path: str, bbox: list[float], output_path: str = "sam2_mask.png") -> dict[str, Any]:
    image = _input(input_image_path)
    output = _output(output_path, "sam2_mask.png")
    if len(bbox) != 4:
        raise ValueError("bbox must contain [x, y, width, height].")
    if not _ready("SAM2"):
        raise RuntimeError("SAM2 worker is not ready.")
    bbox_str = ",".join(str(int(value)) for value in bbox)
    with image.open("rb") as handle:
        response = _session.post(
            _SERVICES["SAM2"][1],
            files={"image": (image.name, handle, "application/octet-stream")},
            data={"bbox_str": bbox_str, "output_path": str(output)},
            timeout=TIMEOUT,
        )
    response.raise_for_status()
    payload = response.json()
    mask_url = payload.get("mask_url", "")
    if not isinstance(mask_url, str) or not mask_url:
        raise RuntimeError("SAM2 response did not contain a mask artifact.")
    _download_worker_artifact(mask_url, output)
    payload["output_path"] = str(output)
    return payload


def _paired_call(name: str, pre_image_path: str, post_image_path: str, output_path: str) -> dict[str, Any]:
    pre = _input(pre_image_path)
    post = _input(post_image_path)
    output = _output(output_path, f"{name.lower()}_result.png")
    if not _ready(name):
        raise RuntimeError(f"{name} worker is not ready.")
    response = _session.post(
        _SERVICES[name][1],
        json={
            "pre_image_path": str(pre),
            "post_image_path": str(post),
            "output_path": str(output),
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


@mcp.tool(description="Detect semantic change between aligned pre-event and post-event images with ChangeOS.")
def ChangeOS(pre_image_path: str, post_image_path: str, output_path: str = "changeos_result.png") -> dict[str, Any]:
    return _paired_call("ChangeOS", pre_image_path, post_image_path, output_path)


@mcp.tool(description="Assess building damage from aligned pre-event and post-event images with BRIGHT.")
def BRIGHT_DamageAssessment(
    pre_image_path: str,
    post_image_path: str,
    output_path: str = "bright_damage_result.png",
) -> dict[str, Any]:
    return _paired_call("BRIGHT_DamageAssessment", pre_image_path, post_image_path, output_path)


if __name__ == "__main__":
    mcp.run()
