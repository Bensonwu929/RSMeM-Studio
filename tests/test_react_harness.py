from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

import react_harness as harness


class ToolCallingFake(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class FakeRuntime:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.calls = []
        self.catalog = {
            "Statistics.inspect_fixture": {
                "service": "Statistics",
                "name": "inspect_fixture",
                "description": "Inspect one fixture path.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"input_path": {"type": "string"}},
                    "required": ["input_path"],
                },
            }
        }

    def start(self):
        return self

    def call(self, service, tool, arguments):
        self.calls.append((service, tool, arguments))
        return {"readable": True, "name": Path(arguments["input_path"]).name}

    def health(self):
        return {"transport": "stdio", "discovered_tools": 1}

    def is_tool_available(self, service, tool):
        return True

    def close(self):
        pass


def test_budget_is_explicit_and_not_low():
    assert harness.MAX_MODEL_CALLS_PER_ATTEMPT == 30
    assert harness.GRAPH_RECURSION_LIMIT >= 2 * harness.MAX_MODEL_CALLS_PER_ATTEMPT
    assert harness.MAX_COMPLETION_TOKENS == 16_384


def test_path_policy_rebases_outputs_and_rejects_unrelated_inputs(tmp_path):
    raster = tmp_path / "inputs" / "sample.tif"
    raster.parent.mkdir()
    raster.write_bytes(b"fixture")
    output = tmp_path / "run"
    output.mkdir()
    policy = harness.ToolPathPolicy((raster.resolve(),), output.resolve())

    safe = policy.sanitize({"input_path": "sample.tif", "output_path": "/tmp/answer.tif"})
    assert safe["input_path"] == str(raster.resolve())
    assert safe["output_path"] == str((output / "answer.tif").resolve())
    assert policy.sanitize({"direction": "west", "file_type": "tif"}) == {
        "direction": "west", "file_type": "tif"
    }
    with pytest.raises(harness.ReActHarnessError, match="outside this request"):
        policy.sanitize({"input_path": "/etc/passwd"})
    assert policy.sanitize({"image": "sample.tif"})["image"] == str(raster.resolve())
    with pytest.raises(harness.ReActHarnessError, match="outside this request"):
        policy.sanitize({"pre_image": "/etc/passwd"})


def test_unhealthy_gpu_tools_are_not_exposed_to_react(tmp_path):
    runtime = FakeRuntime(tmp_path)
    runtime.catalog["Perception.RemoteCLIP"] = {
        "service": "Perception",
        "name": "RemoteCLIP",
        "description": "GPU classifier",
        "inputSchema": {
            "type": "object",
            "properties": {"input_image_path": {"type": "string"}},
            "required": ["input_image_path"],
        },
    }
    runtime.is_tool_available = lambda service, tool: service != "Perception"
    tools = harness.build_mcp_tools(runtime, [], tmp_path)
    assert [tool.name for tool in tools] == ["inspect_fixture"]


def test_parse_agent_messages_preserves_react_order(tmp_path):
    messages = [
        AIMessage(content="", tool_calls=[{
            "name": "inspect_fixture", "args": {"input_path": "a.tif"}, "id": "c1"
        }]),
        ToolMessage(content=json.dumps({"readable": True}), tool_call_id="c1", name="inspect_fixture"),
        AIMessage(content="The raster is readable."),
    ]
    parsed = harness.parse_agent_messages(messages, tmp_path)
    assert parsed["model_calls"] == 2
    assert [step["tool"] for step in parsed["trajectory"]] == ["inspect_fixture"]
    assert parsed["answer"] == "The raster is readable."


def test_react_agent_calls_mcp_then_answers_and_writes_sqlite(tmp_path):
    raster = tmp_path / "input.tif"
    raster.write_bytes(b"fixture")
    fake_model = ToolCallingFake(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "inspect_fixture", "args": {"input_path": "input.tif"}, "id": "c1"
        }]),
        AIMessage(content="The MCP observation confirms the input is readable."),
    ])
    runtimes = []

    def runtime_factory(output_dir):
        runtime = FakeRuntime(output_dir)
        runtimes.append(runtime)
        return runtime

    result = harness.run_react_attempt(
        query="Inspect this raster.", raster_paths=[str(raster)],
        output_dir=tmp_path / "attempt", api_key="test-key",
        base_url="https://example.com/v1", model="fake-model",
        checkpoint_path=tmp_path / "checkpoints.sqlite", thread_id="thread-1",
        runtime_factory=runtime_factory,
        model_factory=lambda **kwargs: fake_model,
    )

    assert result["model_calls"] == 2
    assert result["trajectory"][0]["status"] == "success"
    assert result["answer"].startswith("The MCP observation")
    assert runtimes[0].calls[0][0:2] == ("Statistics", "inspect_fixture")
    assert (tmp_path / "checkpoints.sqlite").is_file()


def test_twenty_case_selection_fixture_has_required_perception_tail_split():
    path = Path(__file__).parent / "fixtures" / "tool_selection_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))
    assert len(cases) == 20
    assert sum(case["partition"] == "cpu" for case in cases) == 10
    tail = [case for case in cases if case["partition"] == "perception_last_40"]
    assert len(tail) == 10
    assert all(209 <= case["question_id"] <= 248 for case in tail)
    assert all(case["expected_tools"] for case in cases)


def test_selection_expectations_match_question_json_when_source_is_available():
    source = Path("/Users/wu/Downloads/RSMeM/benchmark/question.json")
    if not source.is_file():
        pytest.skip("The external EarthBench question catalog is not mounted.")
    catalog = json.loads(source.read_text(encoding="utf-8"))
    cases = json.loads(
        (Path(__file__).parent / "fixtures" / "tool_selection_cases.json").read_text(encoding="utf-8")
    )
    for case in cases:
        expected = []
        for dialog in catalog[str(case["question_id"])]["dialogs"]:
            for call in dialog.get("tool_calls", []):
                name = call.get("function", {}).get("name")
                if name and name not in expected:
                    expected.append(name)
        assert case["expected_tools"] == expected


def _mock_openai_client(message):
    def handler(request: httpx.Request):
        payload = json.loads(request.content)
        assert payload["tools"][0]["type"] == "function"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 1,
                "model": "fake", "choices": [{"index": 0, "finish_reason": "tool_calls",
                "message": message}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_compatible_probe_accepts_standard_tool_calls():
    client = _mock_openai_client({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "compute_tvdi", "arguments": "{}"
        }}],
    })
    result = harness.tool_call_protocol_probe(
        api_key="test", base_url="https://provider.example/v1", model="fake",
        query="Compute TVDI.", tool_names=["compute_tvdi"], http_client=client,
    )
    assert result["supported"] is True
    assert result["tool_calls"][0]["name"] == "compute_tvdi"


def test_openai_compatible_probe_flags_text_only_provider_response():
    client = _mock_openai_client({"role": "assistant", "content": "I would use compute_tvdi."})
    result = harness.tool_call_protocol_probe(
        api_key="test", base_url="https://provider.example/v1", model="fake",
        query="Compute TVDI.", tool_names=["compute_tvdi"], http_client=client,
    )
    assert result["supported"] is False
    assert "compute_tvdi" in result["raw_text"]
