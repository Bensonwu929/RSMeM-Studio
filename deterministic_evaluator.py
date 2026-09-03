"""Conservative deterministic checks for bundled RSMeM cases.

These checks only claim ``correct`` when auditable tool evidence satisfies a
case-specific contract.  Ambiguous outputs return ``None`` so the existing LLM
judge can perform the semantic comparison.  A real tool error is decisive and
returns ``incorrect`` without spending another judge call.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NumericExpectation:
    value: float
    tolerance: float


@dataclass(frozen=True)
class CaseContract:
    tool_groups: tuple[tuple[str, ...], ...]
    numbers: tuple[NumericExpectation, ...] = ()
    text_groups: tuple[tuple[str, ...], ...] = ()


CONTRACTS: dict[str, CaseContract] = {
    "tvdi_sichuan": CaseContract(
        (("compute_tvdi",), ("calculate_threshold_ratio",)),
        (NumericExpectation(28.07, 0.35),),
    ),
    "nbr_california": CaseContract(
        (("calculate_nbr", "calculate_batch_nbr"), ("sens_slope",)),
        (NumericExpectation(0.00336, 0.00035),),
        (("increasing", "increase", "positive"),),
    ),
    "nightlights_bhutan": CaseContract(
        (("calc_batch_image_mean",), ("difference",)),
        (
            NumericExpectation(0.32277, 0.003),
            NumericExpectation(0.41059, 0.003),
            NumericExpectation(0.08782, 0.003),
        ),
        (("june",), ("lower",)),
    ),
    "q10_black_forest": CaseContract(
        (("lst_single_channel",), ("calculate_mean_lst_by_ndvi", "studio_mean_lst_by_ndvi")),
        (
            NumericExpectation(290.94, 0.8),
            NumericExpectation(281.97, 0.8),
            NumericExpectation(8.97, 0.8),
        ),
        (("warmer", "no cooling", "does not show"),),
    ),
    "q41_cyprus_ati": CaseContract(
        (("ati", "studio_apparent_thermal_inertia"), ("difference",)),
        (
            NumericExpectation(3.656, 0.08),
            NumericExpectation(4.295, 0.08),
            NumericExpectation(0.6391, 0.08),
        ),
        (("higher", "increase", "increasing"),),
    ),
    "q102_mumbai_lights": CaseContract(
        (("calc_batch_image_mean",), ("compute_linear_trend", "studio_linear_trend")),
        (NumericExpectation(0.2884, 0.035), NumericExpectation(12.388, 0.25)),
        (("increasing", "positive"),),
    ),
    "q108_rosario_ndvi": CaseContract(
        (("percentage_change", "studio_consecutive_percentage_changes"), ("max_value_and_index",)),
        (NumericExpectation(329.51, 3.5),),
        (("30 january", "january 30"), ("10 february", "february 10")),
    ),
    "q124_dongting_turbidity": CaseContract(
        (("calculate_water_turbidity_ntu", "studio_calculate_turbidity"),),
        (NumericExpectation(6573.55, 80.0),),
        (("21",), ("23",), ("september",)),
    ),
    "q145_congo_rainfall": CaseContract(
        (("calc_batch_image_mean",), ("compute_linear_trend", "studio_linear_trend")),
        (NumericExpectation(6.5466, 0.15), NumericExpectation(0.0252, 0.008)),
        (("increasing", "positive"),),
    ),
    "q156_urmia_turbidity": CaseContract(
        (("calculate_water_turbidity_ntu", "studio_calculate_turbidity"), ("calc_batch_image_skewness",)),
        (NumericExpectation(0.5448, 0.08), NumericExpectation(1.6166, 0.15)),
        (("right-skew", "right skew", "positive skew"),),
    ),
    "q161_california_hotspots": CaseContract(
        (("calculate_nbr", "calculate_batch_nbr"), ("analyze_hotspot_direction", "studio_hotspot_direction")),
        (NumericExpectation(53198, 3),),
        (("west",),),
    ),
    "q180_thailand_frp": CaseContract(
        (("identify_fire_prone_areas", "studio_fire_prone_mask"),),
        (NumericExpectation(115, 2),),
        (("upper quartile", "75th", "75 percent"),),
    ),
    "q187_shanghai_volume": CaseContract(
        (("subtract", "studio_subtract_rasters"), ("compute_linear_trend", "studio_linear_trend")),
        (NumericExpectation(149.76, 4.0),),
        (("increasing", "positive"),),
    ),
}


def _canonical_tool(name: str) -> str:
    value = name.lower().strip()
    for prefix in ("analysis_", "index_", "inversion_", "statistics_"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return value


def _numbers(text: str) -> list[float]:
    values = []
    for token in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text):
        try:
            value = float(token)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _contains_expected_number(values: list[float], expected: NumericExpectation) -> bool:
    # Accept percentages written either as 28.07 or 0.2807.
    targets = [expected.value]
    if abs(expected.value) > 1:
        targets.append(expected.value / 100.0)
    return any(
        abs(observed - target) <= expected.tolerance * (0.01 if target != expected.value else 1)
        for observed in values
        for target in targets
    )


def evaluate_case_deterministically(
    case_id: str,
    trajectory: list[dict[str, Any]],
    final_answer: str,
) -> dict[str, Any] | None:
    """Return a decisive verdict, or ``None`` when an LLM judge is needed."""
    contract = CONTRACTS.get(case_id)
    if contract is None:
        return None

    if not trajectory:
        return {
            "verdict": "incorrect",
            "confidence": 1.0,
            "comparison": "The agent returned an answer without executing an MCP tool.",
            "reason_codes": ["deterministic_no_tool_evidence"],
            "evidence_step_ids": [],
            "source": "deterministic",
        }

    errors = [step for step in trajectory if str(step.get("status", "")).lower() == "error"]
    if errors:
        return {
            "verdict": "incorrect",
            "confidence": 1.0,
            "comparison": f"A required MCP execution failed at step {errors[0].get('step', '?')}.",
            "reason_codes": ["deterministic_tool_error"],
            "evidence_step_ids": [errors[0].get("step")],
            "source": "deterministic",
        }

    tool_names = {_canonical_tool(str(step.get("tool", ""))) for step in trajectory}
    tools_ok = all(
        any(_canonical_tool(candidate) in tool_names for candidate in group)
        for group in contract.tool_groups
    )
    if not tools_ok:
        return None

    evidence_text = final_answer + "\n" + "\n".join(
        json.dumps(step.get("observation", ""), ensure_ascii=False, default=str)
        for step in trajectory
    )
    lowered = evidence_text.lower()
    values = _numbers(evidence_text)
    numbers_ok = all(_contains_expected_number(values, expected) for expected in contract.numbers)
    text_ok = all(any(option in lowered for option in group) for group in contract.text_groups)
    if not (numbers_ok and text_ok):
        return None

    return {
        "verdict": "correct",
        "confidence": 1.0,
        "comparison": "The executed tools and reported evidence satisfy the deterministic case contract.",
        "reason_codes": ["deterministic_contract_satisfied"],
        "evidence_step_ids": [step.get("step") for step in trajectory],
        "source": "deterministic",
    }
