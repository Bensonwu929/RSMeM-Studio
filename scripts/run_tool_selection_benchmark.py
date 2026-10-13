#!/usr/bin/env python3
"""Dry-run RSMeM tool selection on 20 curated EarthBench questions.

This script never executes a geospatial tool.  It asks the configured model to
emit OpenAI-style tool calls, then reports protocol support, expected-tool
recall, and tools missing from the current MCP deployment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcp_runtime import MCPRuntime  # noqa: E402
from react_harness import build_mcp_tools  # noqa: E402

EQUIVALENTS = {
    "calculate_nbr": {"calculate_nbr", "calculate_batch_nbr", "studio_calculate_nbr"},
    "calculate_water_turbidity_ntu": {"calculate_water_turbidity_ntu", "studio_calculate_turbidity"},
    "compute_linear_trend": {"compute_linear_trend", "studio_linear_trend"},
    "percentage_change": {"percentage_change", "studio_consecutive_percentage_changes"},
    "analyze_hotspot_direction": {"analyze_hotspot_direction", "studio_hotspot_direction"},
    "ATI": {"ATI", "studio_apparent_thermal_inertia"},
    "calculate_area": {"calculate_area", "studio_calculate_area"},
    "difference": {"difference", "studio_difference"},
}


def variants(name: str) -> set[str]:
    return EQUIVALENTS.get(name, {name})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("RSMEM_BASE_URL", ""))
    parser.add_argument("--model", default=os.getenv("RSMEM_MODEL", ""))
    parser.add_argument("--api-key", default=os.getenv("RSMEM_API_KEY", ""))
    parser.add_argument("--partition", choices=["all", "cpu", "perception_last_40"], default="all")
    args = parser.parse_args()
    if not all((args.base_url, args.model, args.api_key)):
        parser.error("Set RSMEM_BASE_URL, RSMEM_MODEL, and RSMEM_API_KEY (or pass flags).")

    cases = json.loads(
        (ROOT / "tests" / "fixtures" / "tool_selection_cases.json").read_text(encoding="utf-8")
    )
    if args.partition != "all":
        cases = [case for case in cases if case["partition"] == args.partition]

    with tempfile.TemporaryDirectory(prefix="rsmem_selection_") as directory:
        runtime = MCPRuntime(Path(directory), timeout=60).start()
        try:
            tools = build_mcp_tools(runtime, [], Path(directory))
            available = {tool.name for tool in tools}
            model = ChatOpenAI(
                api_key=args.api_key, base_url=args.base_url, model=args.model,
                temperature=0, timeout=180, max_retries=1, max_completion_tokens=4096,
            ).bind_tools(tools, tool_choice="auto")
            results = []
            for case in cases:
                message = model.invoke(
                    "You are doing tool-selection evaluation only. Select all tools needed for the "
                    "task as tool calls; do not solve the task and do not invent tool names.\n\n"
                    + case["query"]
                )
                selected = [call.get("name") for call in (message.tool_calls or [])]
                expected = case["expected_tools"]
                matched = sorted(
                    name for name in expected if variants(name) & set(selected)
                )
                missing_from_deployment = sorted(
                    name for name in expected if not (variants(name) & available)
                )
                results.append({
                    "question_id": case["question_id"],
                    "partition": case["partition"],
                    "protocol_tool_call": bool(selected),
                    "selected": selected,
                    "expected": expected,
                    "matched": matched,
                    "expected_recall": len(matched) / len(expected),
                    "missing_from_deployment": missing_from_deployment,
                })
                print(json.dumps(results[-1], ensure_ascii=False))
        finally:
            runtime.close()

    summary = {
        "cases": len(results),
        "protocol_success_rate": sum(item["protocol_tool_call"] for item in results) / len(results),
        "case_expected_tool_hit_rate": sum(bool(item["matched"]) for item in results) / len(results),
        "mean_expected_tool_recall": sum(item["expected_recall"] for item in results) / len(results),
        "cases_with_deployment_gaps": sum(bool(item["missing_from_deployment"]) for item in results),
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
