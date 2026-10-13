#!/usr/bin/env python3
"""Drive N concurrent ``/start_workflow_ui`` calls against the live Studio.

LLM credentials are read from the environment and never printed:

    RSMEM_PROBE_API_KEY
    RSMEM_PROBE_BASE_URL
    RSMEM_PROBE_MODEL

The ModelScope api-inference token comes from ``git credential fill`` for
``www.modelscope.cn`` and is also never written to disk.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import argparse
import html
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx
import requests
from gradio_client import Client, handle_file

STUDIO_DEFAULT = "https://studio-wbx929-rsmem.api-inference.modelscope.net"
MODE_OPENAI = "API"
DEFAULT_QUERY = "检测图中的船，然后输出给我带目标检测框的图像"

# Gradio's client omits State / Accordion / Group from the streamed tuple, so
# this is the 16-slot payload ``/start_workflow_ui`` actually returns, not the
# 21-entry ``workflow_outputs`` list in app.py.
OUTPUT_NAMES = [
    "status",
    "timeline",
    "hkg",
    "attempts",
    "evaluation",
    "decision",
    "candidate_1",
    "candidate_2",
    "candidate_3",
    "candidate_choice",
    "user_diagnosis",
    "user_plan",
    "edited_reflection",
    "final_answer",
    "artifacts",
    "trace",
]

ERROR_MARKERS = (
    "worker is not ready",
    "model is busy",
    "cuda out of memory",
    "outofmemoryerror",
    "out of memory",
    "503",
    "queue is full",
    "queue_full",
    "readtimeout",
    "connecttimeout",
    "remotedisconnected",
    "server disconnected",
    "connectionreset",
    "gateway time-out",
    "504",
    "429",
)


def modelscope_token() -> str:
    raw = subprocess.run(
        ["git", "credential", "fill"],
        input=b"protocol=https\nhost=www.modelscope.cn\n\n",
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.decode()
    token = dict(
        line.split("=", 1) for line in raw.splitlines() if "=" in line
    ).get("password", "")
    if not token:
        raise SystemExit("No ModelScope token from git credential helper")
    return token


def _redact(text: str, secrets: list[str]) -> str:
    out = str(text)
    for secret in secrets:
        if secret:
            out = out.replace(secret, "<redacted>")
    return out


def strip_html(value) -> str:
    text = re.sub(r"<br\s*/?>", "\n", str(value or ""))
    text = re.sub(r"</(p|div|li|tr|h\d)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def tools_used(attempts_html: str) -> list[str]:
    seen: list[str] = []
    for name in re.findall(r"<code>([A-Za-z_][A-Za-z_0-9]*)</code>", str(attempts_html or "")):
        if name not in seen:
            seen.append(name)
    return seen


def classify_errors(*blobs: str) -> list[str]:
    joined = "\n".join(str(blob or "") for blob in blobs).lower()
    return [marker for marker in ERROR_MARKERS if marker in joined]


def _load_trace(trace_path) -> dict | None:
    path = Path(str(trace_path or ""))
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _trace_events(payload: dict) -> list[dict]:
    events = []
    attempts = payload.get("attempts") or []
    if not attempts:
        attempts = [payload]
    for attempt in attempts:
        for event in attempt.get("trajectory") or attempt.get("events") or []:
            if isinstance(event, dict):
                events.append(event)
    return events


def tools_from_trace(payload: dict | None) -> list[str]:
    if not payload:
        return []
    seen: list[str] = []
    for event in _trace_events(payload):
        name = event.get("tool")
        if name and name not in seen:
            seen.append(name)
    return seen


def _trace_tool_errors(payload: dict | None) -> list[dict]:
    if not payload:
        return []
    errors = []
    for event in _trace_events(payload):
        status = str(event.get("status") or "").lower()
        observation = str(event.get("observation") or "")
        if status in {"error", "failed"} or "worker is not ready" in observation.lower():
            errors.append(
                {
                    "tool": event.get("tool"),
                    "status": event.get("status"),
                    "observation": observation[:400],
                }
            )
    return errors


def _summarize_gpu(payload: dict) -> dict:
    started = payload.get("started") or {}
    skipped = payload.get("skipped") or {}
    storage = payload.get("storage") or {}
    identity = payload.get("identity") or {}
    running = sorted(name for name, info in started.items() if (info or {}).get("running"))
    free = {
        name: (info or {}).get("free_gib")
        for name, info in storage.items()
        if isinstance(info, dict)
    }
    return {
        "http": payload.get("http"),
        "error": payload.get("error"),
        "startup_running": payload.get("startup_running"),
        "running_count": len(running),
        "running": running,
        "skipped": sorted(skipped),
        "storage_free_gib": free,
        "identity": {
            "hostname": identity.get("hostname"),
            "pid": identity.get("pid"),
            "instance_id": identity.get("instance_id"),
        },
    }


def poll_gpu_status(
    studio: str,
    token: str,
    stop: threading.Event,
    samples: list[dict],
    interval: float,
    log_path: Path,
) -> None:
    session = requests.Session()
    session.trust_env = False
    headers = {"Authorization": f"Bearer {token}"}
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    while not stop.is_set():
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        elapsed = round(time.time() - started, 1)
        try:
            response = session.get(
                f"{studio.rstrip('/')}/rsmem/gpu-status",
                headers=headers,
                timeout=20,
            )
            if response.ok:
                payload = response.json()
            else:
                payload = {"http": response.status_code, "body": response.text[:300]}
        except Exception as exc:  # noqa: BLE001
            payload = {"error": f"{type(exc).__name__}: {exc}"}
        summary = {"t": stamp, "elapsed_s": elapsed, **_summarize_gpu(payload)}
        samples.append(summary)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        free = summary.get("storage_free_gib") or {}
        free_txt = ",".join(f"{k}={v}" for k, v in free.items() if v is not None) or "n/a"
        print(
            f"[gpu {stamp}] running={summary['running_count']} "
            f"skipped={len(summary['skipped'])} free_gib={free_txt} "
            f"boot={summary.get('startup_running')}",
            flush=True,
        )
        stop.wait(interval)


def _watch_job(job, timeout: float) -> tuple[object, dict, float | None, list[dict]]:
    started = time.time()
    first_seen: dict[str, float] = {}
    first_output: float | None = None
    queue_samples: list[dict] = []
    last_code = None
    deadline = started + timeout
    while True:
        status = job.status()
        code = status.code.name if status.code else None
        now = round(time.time() - started, 2)
        if code and code not in first_seen:
            first_seen[code] = now
            print(f"        status {code} at +{now}s rank={status.rank} q={status.queue_size}", flush=True)
        if code != last_code or (status.rank is not None and (not queue_samples or queue_samples[-1].get("rank") != status.rank)):
            queue_samples.append(
                {
                    "t": now,
                    "code": code,
                    "rank": status.rank,
                    "queue_size": status.queue_size,
                    "eta": status.eta,
                }
            )
            last_code = code
        outputs = job.outputs()
        if outputs and first_output is None:
            first_output = now
            print(f"        first yield at +{now}s", flush=True)
        if job.done():
            break
        if time.time() >= deadline:
            try:
                job.cancel()
            except Exception:
                pass
            raise TimeoutError(
                f"job still {code or 'unknown'} after {timeout:.0f}s "
                f"(first_status={first_seen}, first_yield={first_output})"
            )
        time.sleep(0.5)
    return job.result(timeout=30), first_seen, first_output, queue_samples


def run_lane(
    lane: int,
    studio: str,
    token: str,
    image: Path,
    query: str,
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    download_dir: Path,
) -> dict:
    secrets = [api_key, token]
    record: dict = {
        "lane": lane,
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    started = time.time()
    print(f"[lane {lane}] submit", flush=True)
    try:
        client = Client(
            studio,
            headers={"Authorization": f"Bearer {token}"},
            verbose=False,
            download_files=str(download_dir),
            httpx_kwargs={
                "timeout": httpx.Timeout(None),
                "trust_env": False,
            },
        )
        job = client.submit(
            query,
            None,
            [handle_file(str(image))],
            MODE_OPENAI,
            "",
            base_url,
            model,
            api_key,
            "",
            api_name="/start_workflow_ui",
        )
        result, first_seen, first_output, queue_samples = _watch_job(job, timeout)
        slots = {}
        if isinstance(result, (list, tuple)):
            if len(result) == len(OUTPUT_NAMES):
                slots = dict(zip(OUTPUT_NAMES, result))
            else:
                record["output_arity"] = len(result)
                for index, name in enumerate(OUTPUT_NAMES):
                    if index < len(result):
                        slots[name] = result[index]
        record["seconds"] = round(time.time() - started, 1)
        record["first_status_s"] = first_seen
        record["first_yield_s"] = first_output
        record["queue_samples"] = queue_samples[-40:]
        record["api_status"] = strip_html(slots.get("status", ""))[:400]
        record["final_answer"] = strip_html(slots.get("final_answer", ""))[:2000]
        record["evaluation"] = strip_html(slots.get("evaluation", ""))[:800]
        trace_payload = _load_trace(slots.get("trace"))
        record["tools"] = tools_from_trace(trace_payload) or tools_used(slots.get("attempts", ""))
        record["attempts_text"] = strip_html(slots.get("attempts", ""))[:4000]
        record["trace_tool_errors"] = _trace_tool_errors(trace_payload)
        record["error_markers"] = classify_errors(
            record.get("api_status"),
            record.get("attempts_text"),
            record.get("final_answer"),
            json.dumps(record.get("trace_tool_errors") or [], ensure_ascii=False),
        )
        record["ok"] = not record["error_markers"] and not record["trace_tool_errors"]
        print(
            f"[lane {lane}] done in {record['seconds']}s "
            f"yield={first_output} tools={record['tools']} "
            f"errors={record['error_markers'] or []}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        record["seconds"] = round(time.time() - started, 1)
        record["ok"] = False
        record["error"] = _redact(f"{type(exc).__name__}: {exc}", secrets)[:800]
        record["error_markers"] = classify_errors(record["error"])
        print(f"[lane {lane}] ERROR after {record['seconds']}s: {record['error']}", flush=True)
    return record


def print_table(records: list[dict], wall_s: float) -> None:
    print("\n=== concurrency probe ===", flush=True)
    print(
        f"{'lane':<6}{'total_s':>9}{'yield_s':>10}{'queue_s':>10}"
        f"{'ok':>6}  {'errors':<32} tools",
        flush=True,
    )
    for record in sorted(records, key=lambda item: item["lane"]):
        first_seen = record.get("first_status_s") or {}
        queue_s = first_seen.get("IN_QUEUE")
        processing_s = first_seen.get("PROCESSING") or first_seen.get("ITERATING")
        yield_s = record.get("first_yield_s")
        errors = ",".join(record.get("error_markers") or []) or (record.get("error") or "")[:30]
        print(
            f"{record['lane']:<6}{record.get('seconds', 0):>9.1f}"
            f"{'' if yield_s is None else f'{yield_s:.1f}':>10}"
            f"{'' if queue_s is None else f'{queue_s:.1f}':>10}"
            f"{'yes' if record.get('ok') else 'no':>6}  "
            f"{errors:<32} {record.get('tools') or []}",
            flush=True,
        )
        if processing_s is not None:
            print(f"       first PROCESSING/ITERATING at +{processing_s:.1f}s", flush=True)
    print(f"wall_clock_s={wall_s:.1f}  n={len(records)}", flush=True)


def preflight(studio: str, token: str) -> dict:
    response = requests.get(
        f"{studio.rstrip('/')}/rsmem/gpu-status",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    summary = _summarize_gpu(payload)
    print(
        f"[preflight] running={summary['running_count']} "
        f"skipped={summary['skipped'] or '(none)'} "
        f"boot={summary['startup_running']} "
        f"storage={summary['storage_free_gib'] or 'n/a'}",
        flush=True,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio", default=os.getenv("RSMEM_PROBE_STUDIO", STUDIO_DEFAULT))
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--image", required=True)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--timeout", type=int, default=2700, help="per-lane seconds")
    parser.add_argument("--poll", type=float, default=10.0)
    parser.add_argument("--label", default="probe")
    parser.add_argument("--out", default="")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("RSMEM_PROBE_API_KEY", "").strip()
    base_url = os.getenv("RSMEM_PROBE_BASE_URL", "").strip()
    model = os.getenv("RSMEM_PROBE_MODEL", "").strip()
    if not args.preflight_only and not all((api_key, base_url, model)):
        parser.error("Set RSMEM_PROBE_API_KEY, RSMEM_PROBE_BASE_URL, and RSMEM_PROBE_MODEL")

    image = Path(args.image).expanduser().resolve()
    if not image.is_file():
        raise SystemExit(f"image not found: {image}")

    token = modelscope_token()
    payload = preflight(args.studio, token)
    if args.preflight_only:
        print(json.dumps(_summarize_gpu(payload), ensure_ascii=False, indent=2))
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) if args.out else Path(f"/tmp/rsmem_concurrency_probe/{args.label}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[probe] n={args.n} image={image.name} out={out_dir}", flush=True)

    gpu_samples: list[dict] = []
    stop = threading.Event()
    poller = threading.Thread(
        target=poll_gpu_status,
        args=(args.studio, token, stop, gpu_samples, args.poll, out_dir / "gpu_status.jsonl"),
        name="gpu-status-poller",
        daemon=True,
    )
    poller.start()

    wall_started = time.time()
    records: list[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=args.n) as pool:
            futures = [
                pool.submit(
                    run_lane,
                    lane,
                    args.studio,
                    token,
                    image,
                    args.query,
                    base_url,
                    model,
                    api_key,
                    float(args.timeout),
                    out_dir / f"lane_{lane}",
                )
                for lane in range(1, args.n + 1)
            ]
            for future in as_completed(futures):
                records.append(future.result())
    finally:
        stop.set()
        poller.join(timeout=15)

    wall_s = round(time.time() - wall_started, 1)
    records.sort(key=lambda item: item["lane"])
    print_table(records, wall_s)

    summary = {
        "label": args.label,
        "n": args.n,
        "query": args.query,
        "image": str(image),
        "model": model,
        "base_url": base_url,
        "studio": args.studio,
        "wall_clock_s": wall_s,
        "lanes": records,
        "gpu_samples": gpu_samples,
        "error_counts": {},
    }
    counts: dict[str, int] = {}
    for record in records:
        for marker in record.get("error_markers") or []:
            counts[marker] = counts.get(marker, 0) + 1
        if record.get("error") and not record.get("error_markers"):
            counts["other"] = counts.get("other", 0) + 1
    summary["error_counts"] = counts
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[probe] wrote {out_dir / 'summary.json'}", flush=True)
    print(f"[probe] error_counts={counts or '{}'}", flush=True)
    return 0 if all(record.get("ok") for record in records) else 1


if __name__ == "__main__":
    sys.exit(main())
