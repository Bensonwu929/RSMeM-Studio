"""Conservative deterministic checks for bundled RSMeM cases.

A case is judged on what its answer reports, not on which tools produced it.
An answer is ``correct`` when it states the expected values within tolerance
and reaches the expected conclusion, and when a successful execution in the
same attempt produced those values as well.  Requiring both sides keeps a
recited reference number from passing on its own while leaving the agent free
to reach the answer by whatever route works, including code it wrote itself.

Ambiguous outputs return ``None`` so the existing LLM judge can perform the
semantic comparison.  An attempt in which nothing executed successfully is
decisively ``incorrect`` without spending another judge call.
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
    """What a bundled case's answer has to say to be judged correct.

    Contracts describe the destination, not the route. An answer is correct
    when it reports the expected values and conclusion, whichever tools or
    code produced them -- there are several valid ways to reach the same
    number, and pinning the verdict to one of them sent correct answers to the
    LLM judge merely for arriving differently.
    """

    numbers: tuple[NumericExpectation, ...] = ()
    text_groups: tuple[tuple[str, ...], ...] = ()


CONTRACTS: dict[str, CaseContract] = {
    "tvdi_sichuan": CaseContract(
        numbers=(NumericExpectation(28.07, 0.35),),
    ),
    "nbr_california": CaseContract(
        numbers=(NumericExpectation(0.00336, 0.00035),),
        text_groups=(("increasing", "increase", "positive"),),
    ),
    "nightlights_bhutan": CaseContract(
        numbers=(
            NumericExpectation(0.32277, 0.003),
            NumericExpectation(0.41059, 0.003),
            NumericExpectation(0.08782, 0.003),
        ),
        text_groups=(("june",), ("lower",)),
    ),
    "q10_black_forest": CaseContract(
        numbers=(
            NumericExpectation(290.94, 0.8),
            NumericExpectation(281.97, 0.8),
            NumericExpectation(8.97, 0.8),
        ),
        text_groups=(("warmer", "no cooling", "does not show"),),
    ),
    "q41_cyprus_ati": CaseContract(
        numbers=(
            NumericExpectation(3.656, 0.08),
            NumericExpectation(4.295, 0.08),
            NumericExpectation(0.6391, 0.08),
        ),
        text_groups=(("higher", "increase", "increasing"),),
    ),
    "q102_mumbai_lights": CaseContract(
        numbers=(NumericExpectation(0.2884, 0.035), NumericExpectation(12.388, 0.25)),
        text_groups=(("increasing", "positive"),),
    ),
    "q108_rosario_ndvi": CaseContract(
        numbers=(NumericExpectation(329.51, 3.5),),
        text_groups=(("30 january", "january 30"), ("10 february", "february 10")),
    ),
    "q124_dongting_turbidity": CaseContract(
        numbers=(NumericExpectation(6573.55, 80.0),),
        text_groups=(("21",), ("23",), ("september",)),
    ),
    "q145_congo_rainfall": CaseContract(
        numbers=(NumericExpectation(6.5466, 0.15), NumericExpectation(0.0252, 0.008)),
        text_groups=(("increasing", "positive"),),
    ),
    "q156_urmia_turbidity": CaseContract(
        numbers=(NumericExpectation(0.5448, 0.08), NumericExpectation(1.6166, 0.15)),
        text_groups=(("right-skew", "right skew", "positive skew"),),
    ),
    "q161_california_hotspots": CaseContract(
        numbers=(NumericExpectation(53198, 3),),
        text_groups=(("west",),),
    ),
    "q187_shanghai_volume": CaseContract(
        numbers=(NumericExpectation(149.76, 4.0),),
        text_groups=(("increasing", "positive"),),
    ),
}


#: Tools that run agent-authored code. They report a failing script through a
#: non-zero ``exit_code`` rather than by failing the call, because the tool did
#: its job. That distinction keeps ordinary debugging iterations out of the
#: hard tool-error verdict below, but it also means the output of a run that
#: crashed must never be mistaken for evidence.
CODE_TOOL_NAMES = frozenset({"run_python"})


def _canonical_tool(name: str) -> str:
    value = name.lower().strip()
    for prefix in ("analysis_", "index_", "inversion_", "statistics_", "codeexec_"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return value


def _is_discredited_code_step(step: dict[str, Any]) -> bool:
    """Whether a code step's output must be excluded from the evidence pool.

    A run that crashed, timed out, or reported an unreadable result proves
    nothing, even when it printed numbers before dying. Anything unparseable is
    treated as discredited so the case falls through to the LLM judge rather
    than being scored on output nobody can vouch for.
    """
    if _canonical_tool(str(step.get("tool", ""))) not in CODE_TOOL_NAMES:
        return False
    observation = step.get("observation", "")
    if isinstance(observation, str):
        try:
            observation = json.loads(observation)
        except (TypeError, ValueError):
            return True
    if not isinstance(observation, dict):
        return True
    if observation.get("timed_out"):
        return True
    return observation.get("exit_code") != 0


def step_succeeded(step: dict[str, Any]) -> bool:
    """Whether a step actually produced something worth believing.

    A code step reports a crashed script through its exit code rather than by
    failing the call, so ``status`` alone overstates success once the agent can
    write code.
    """
    if str(step.get("status", "")).lower() != "success":
        return False
    return not _is_discredited_code_step(step)


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

    credible = [step for step in trajectory if step_succeeded(step)]
    if not credible:
        return {
            "verdict": "incorrect",
            "confidence": 1.0,
            "comparison": "Every execution in this attempt failed, so nothing supports the answer.",
            "reason_codes": ["deterministic_no_credible_evidence"],
            "evidence_step_ids": [step.get("step") for step in trajectory],
            "source": "deterministic",
        }

    observation_text = "\n".join(
        json.dumps(step.get("observation", ""), ensure_ascii=False, default=str)
        for step in credible
    )
    answer_values = _numbers(final_answer)
    observed_values = _numbers(observation_text)

    # Both sides are required. The answer has to state the value, or the case
    # was not actually answered; and an execution has to have produced it, or a
    # recited reference number would pass on its own.
    numbers_ok = all(
        _contains_expected_number(answer_values, expected)
        and _contains_expected_number(observed_values, expected)
        for expected in contract.numbers
    )
    lowered_answer = final_answer.lower()
    text_ok = all(
        any(option in lowered_answer for option in group)
        for group in contract.text_groups
    )
    if not (numbers_ok and text_ok):
        return None

    return {
        "verdict": "correct",
        "confidence": 1.0,
        "comparison": "The answer reports the expected values and conclusion, corroborated by successful executions.",
        "reason_codes": ["deterministic_contract_satisfied"],
        "evidence_step_ids": [step.get("step") for step in credible],
        "source": "deterministic",
    }
