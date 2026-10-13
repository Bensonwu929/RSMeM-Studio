"""End-to-end checks for the code tools over a live CodeExec MCP server.

These exercise the real stdio boundary rather than a fake runtime, because the
failure this design most needs to rule out -- agent output corrupting the
JSONRPC framing -- only exists across that boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import deterministic_evaluator as evaluator
import mcp_runtime
import react_harness as harness
from tools import code_executor as executor


CODE_TOOLS = {"run_python", "read_text_file", "list_workspace", "write_script"}


@pytest.fixture
def code_runtime(tmp_path):
    """A live CodeExec server rooted at a throwaway attempt directory."""
    runtime = mcp_runtime.MCPRuntime(tmp_path, timeout=60, servers=("CodeExec",)).start()
    try:
        yield runtime
    finally:
        runtime.close()


# --- gating ------------------------------------------------------------------


def test_code_tools_are_off_by_default():
    assert mcp_runtime.code_tools_enabled({}) is False
    assert mcp_runtime.configured_servers({}) == mcp_runtime.CPU_SERVERS


def test_code_tools_switch_on_explicitly():
    env = {"RSMEM_ENABLE_CODE_TOOLS": "1"}
    assert mcp_runtime.code_tools_enabled(env) is True
    assert "CodeExec" in mcp_runtime.configured_servers(env)


def test_gates_are_independent():
    both = {"RSMEM_ENABLE_CODE_TOOLS": "true", "RSMEM_ENABLE_GPU_PERCEPTION": "on"}
    servers = mcp_runtime.configured_servers(both)
    assert "CodeExec" in servers
    assert "Perception" in servers

    code_only = mcp_runtime.configured_servers({"RSMEM_ENABLE_CODE_TOOLS": "yes"})
    assert "CodeExec" in code_only
    assert "Perception" not in code_only


# --- discovery ---------------------------------------------------------------


def test_all_four_tools_are_discovered(code_runtime):
    discovered = {item["name"] for item in code_runtime.catalog.values()}
    assert CODE_TOOLS <= discovered


def test_tools_reach_the_agent_as_structured_tools(code_runtime, tmp_path):
    tools = harness.build_mcp_tools(code_runtime, [], tmp_path)
    exposed = {tool.name for tool in tools}
    assert CODE_TOOLS <= exposed

    run_python = next(tool for tool in tools if tool.name == "run_python")
    assert "FRESH interpreter" in run_python.description
    assert "perception" in run_python.description.lower()


# --- execution across the stdio boundary -------------------------------------


def test_printing_does_not_corrupt_the_jsonrpc_channel(code_runtime):
    """A print inside agent code must not reach the server's own stdout.

    Running the script in a grandchild with redirected streams is what makes
    this safe; the server keeps answering afterwards, which is the real check.
    """
    noisy = code_runtime.call(
        "CodeExec",
        "run_python",
        {"code": "print('x' * 5000)\nprint('second line')"},
    )
    assert noisy["exit_code"] == 0
    assert "second line" in noisy["stdout"]

    still_alive = code_runtime.call("CodeExec", "run_python", {"code": "print(2 + 2)"})
    assert still_alive["stdout"].strip() == "4"


def test_failing_code_comes_back_as_a_normal_result(code_runtime):
    result = code_runtime.call(
        "CodeExec", "run_python", {"code": "raise ValueError('bad band index')"}
    )

    assert result["exit_code"] != 0
    assert result["timed_out"] is False
    assert "ValueError" in result["stderr"]
    assert "bad band index" in result["stderr"]


def test_write_run_list_read_round_trip(code_runtime, tmp_path):
    """The four tools have to compose, or the extra three earn nothing."""
    code_runtime.call(
        "CodeExec",
        "write_script",
        {"path": "helper.py", "content": "def total(values):\n    return sum(values)\n"},
    )

    run = code_runtime.call(
        "CodeExec",
        "run_python",
        {
            "code": (
                "import json, helper\n"
                "value = helper.total([1, 2, 3, 4])\n"
                "json.dump({'total': value}, open('result.json', 'w'))\n"
                "print('wrote', value)\n"
            )
        },
    )
    assert run["exit_code"] == 0
    assert "wrote 10" in run["stdout"]

    listing = code_runtime.call("CodeExec", "list_workspace", {})
    names = {entry["path"] for entry in listing["files"]}
    assert {"helper.py", "result.json"} <= names

    read = code_runtime.call("CodeExec", "read_text_file", {"path": "result.json"})
    assert json.loads(read["content"]) == {"total": 10}


def test_generated_scripts_stay_out_of_the_listing(code_runtime):
    code_runtime.call("CodeExec", "run_python", {"code": "print('hi')"})

    listing = code_runtime.call("CodeExec", "list_workspace", {})
    assert all(
        executor.CODE_SUBDIR not in entry["path"] for entry in listing["files"]
    )


def test_truncated_output_is_recoverable_through_read_text_file(code_runtime):
    result = code_runtime.call(
        "CodeExec",
        "run_python",
        {"code": f"print('Z' * {executor.MAX_STDOUT_BYTES * 3})"},
    )
    assert result["truncated"] is True

    recovered = code_runtime.call(
        "CodeExec", "read_text_file", {"path": result["stdout_path"]}
    )
    assert recovered["bytes_total"] > executor.MAX_STDOUT_BYTES
    assert recovered["content"].startswith("ZZZ")


def test_reading_outside_the_workspace_is_refused(code_runtime):
    """Refusals are reported, not raised.

    LangGraph re-raises anything that is not a ToolException, so raising here
    would end the attempt over a mistyped path. The refusal still has to be a
    refusal: no content comes back.
    """
    read = code_runtime.call("CodeExec", "read_text_file", {"path": "/etc/passwd"})
    assert "outside this request" in read["error"]
    assert "content" not in read

    write = code_runtime.call(
        "CodeExec", "write_script", {"path": "../escaped.py", "content": "x = 1"}
    )
    assert "outside this request" in write["error"]
    assert not (Path(code_runtime.output_dir).parent / "escaped.py").exists()


def test_recoverable_mistakes_do_not_end_the_attempt(code_runtime):
    """A bad path should cost one turn, and the agent should be able to continue."""
    missing = code_runtime.call(
        "CodeExec", "read_text_file", {"path": "never_written.json"}
    )
    assert "does not exist" in missing["error"]

    empty = code_runtime.call("CodeExec", "run_python", {"code": "   "})
    assert "error" in empty

    recovered = code_runtime.call("CodeExec", "run_python", {"code": "print('back')"})
    assert recovered["stdout"].strip() == "back"


def test_timeout_is_reported_without_losing_earlier_output(code_runtime):
    result = code_runtime.call(
        "CodeExec",
        "run_python",
        {"code": "print('partial result')\nimport time\ntime.sleep(30)", "timeout_s": 2},
    )

    assert result["timed_out"] is True
    assert "partial result" in result["stdout"]


# --- harness and evaluator integration ---------------------------------------


def test_artifact_scan_hides_generated_scripts_but_keeps_results(tmp_path):
    executor.run_python("open('answer.csv', 'w').write('a,b\\n1,2\\n')", tmp_path)

    parsed = harness.parse_agent_messages([], tmp_path)
    names = {Path(item).name for item in parsed["artifacts"]}

    assert "answer.csv" in names
    assert not any(name.startswith("step_") for name in names)


def _code_step(step: int, exit_code: int, observation_extra: dict | None = None) -> dict:
    payload = {"exit_code": exit_code, "stdout": "", "stderr": "", "timed_out": False}
    payload.update(observation_extra or {})
    return {
        "step": step,
        "tool": "run_python",
        "parameters": {},
        "status": "success",
        "observation": json.dumps(payload),
    }


def test_output_of_a_crashed_run_is_not_treated_as_evidence():
    """A run that died proves nothing, even if it printed the right number.

    Without this the deterministic contract could be satisfied by numbers a
    half-finished script happened to emit before crashing.
    """
    good = {
        "step": 1,
        "tool": "calc_batch_image_mean",
        "parameters": {},
        "status": "success",
        "observation": "[0.32277, 0.41059]",
    }
    crashed = _code_step(2, exit_code=1, observation_extra={"stdout": "0.08782"})

    verdict = evaluator.evaluate_case_deterministically(
        "nightlights_bhutan", [good, crashed], "June was lower."
    )

    assert verdict is None, "a crashed run must not satisfy the numeric contract"


def test_output_of_a_successful_run_counts_as_evidence():
    """Code may supply the evidence the named tools would otherwise have.

    The case is judged on the answer's values, so a run that produced them
    counts regardless of which tool the number came out of.
    """
    good = {
        "step": 1,
        "tool": "calc_batch_image_mean",
        "parameters": {},
        "status": "success",
        "observation": "[0.32277, 0.41059]",
    }
    succeeded = _code_step(2, exit_code=0, observation_extra={"stdout": "0.08782"})

    verdict = evaluator.evaluate_case_deterministically(
        "nightlights_bhutan",
        [good, succeeded],
        "June is lower at 0.32277 against 0.41059; the difference is 0.08782.",
    )

    assert verdict is not None
    assert verdict["verdict"] == "correct"


def test_an_answer_reciting_the_reference_still_needs_a_run_behind_it():
    """The red line: reference-like text is not evidence.

    Dropping the tool-name requirement must not let a recited number pass, so
    every expected value has to appear in a successful execution too.
    """
    unrelated = {
        "step": 1,
        "tool": "studio_inspect_rasters",
        "parameters": {},
        "status": "success",
        "observation": "[{'width': 3441, 'height': 1731}]",
    }

    verdict = evaluator.evaluate_case_deterministically(
        "nightlights_bhutan",
        [unrelated],
        "June is lower at 0.32277 against 0.41059; the difference is 0.08782.",
    )

    assert verdict is None, "unexecuted numbers must fall through to the judge"


def test_the_route_no_longer_decides_the_verdict():
    """The same answer passes whether code or the named tools produced it.

    tvdi_sichuan previously failed its contract for computing the exceedance
    share in code instead of calling calculate_threshold_ratio, even though the
    value matched the reference.
    """
    by_code = [
        {
            "step": 1,
            "tool": "compute_tvdi",
            "parameters": {},
            "status": "success",
            "observation": "wrote tvdi.tif",
        },
        _code_step(2, exit_code=0, observation_extra={"stdout": "28.0701%"}),
    ]

    verdict = evaluator.evaluate_case_deterministically(
        "tvdi_sichuan", by_code, "28.07% of valid pixels exceed a TVDI of 0.75."
    )

    assert verdict is not None
    assert verdict["verdict"] == "correct"


def test_timed_out_run_is_discredited_even_with_a_zero_exit_code():
    step = _code_step(1, exit_code=0, observation_extra={"timed_out": True})
    assert evaluator._is_discredited_code_step(step) is True


def test_unparseable_code_observation_is_discredited():
    step = {
        "step": 1,
        "tool": "run_python",
        "status": "success",
        "observation": "not json at all",
    }
    assert evaluator._is_discredited_code_step(step) is True


def test_geospatial_steps_are_never_discredited_by_the_code_rule():
    step = {
        "step": 1,
        "tool": "studio_calculate_ndvi",
        "status": "success",
        "observation": "/tmp/ndvi.tif",
    }
    assert evaluator._is_discredited_code_step(step) is False
