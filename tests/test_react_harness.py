from __future__ import annotations

import json
import os
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
    assert harness.MAX_MODEL_CALLS_PER_ATTEMPT == 100
    assert harness.GRAPH_RECURSION_LIMIT >= 2 * harness.MAX_MODEL_CALLS_PER_ATTEMPT
    assert harness.DEFAULT_COMPLETION_TOKENS == 65_536
    assert harness.FALLBACK_COMPLETION_TOKENS == 16_384
    assert harness.completion_token_budget({}) == 65_536
    assert harness.MAX_COMPLETION_TOKENS == harness.DEFAULT_COMPLETION_TOKENS
    assert harness.MODEL_RETRY_ATTEMPTS == 2


def test_recursion_limit_matches_the_compiled_graphs_steps_per_round():
    """ModelCallLimitMiddleware inserts before_model/after_model nodes.

    That makes one round that ends in a tool call four graph steps
    (before_model -> model -> after_model -> tools), not two. A limit sized
    for two silently cut the configured 30-call budget to 24: with the old
    value of 96, GraphRecursionError fired after only 24 model calls, so the
    30-call limit the middleware was configured with never got to run.
    """
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelCallLimitMiddleware
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import StructuredTool
    from langgraph.errors import GraphRecursionError

    calls = {"n": 0}

    class NeverStopsFake(ToolCallingFake):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            calls["n"] += 1
            from langchain_core.outputs import ChatGeneration, ChatResult

            message = AIMessage(
                content="", tool_calls=[{"name": "noop", "args": {}, "id": f"c{calls['n']}"}]
            )
            return ChatResult(generations=[ChatGeneration(message=message)])

    tool = StructuredTool.from_function(func=lambda: "ok", name="noop", description="d")
    agent = create_agent(
        model=NeverStopsFake(responses=[]),
        tools=[tool],
        middleware=[
            ModelCallLimitMiddleware(
                run_limit=harness.MAX_MODEL_CALLS_PER_ATTEMPT, exit_behavior="error"
            )
        ],
        checkpointer=None,
    )

    try:
        agent.invoke(
            {"messages": [HumanMessage(content="go")]},
            config={"recursion_limit": harness.GRAPH_RECURSION_LIMIT},
        )
        pytest.fail("expected the model call limit to stop the agent")
    except GraphRecursionError:
        pytest.fail(
            f"hit the recursion limit after only {calls['n']} model call(s); "
            f"the configured budget of {harness.MAX_MODEL_CALLS_PER_ATTEMPT} was "
            "never reached"
        )
    except Exception as exc:  # the middleware's own error is the expected outcome
        assert "run limit" in str(exc)
        assert calls["n"] == harness.MAX_MODEL_CALLS_PER_ATTEMPT


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


def test_a_failing_tool_is_reported_to_the_model_not_raised(tmp_path):
    """One unusable tool must not discard the whole attempt.

    A missing GDAL binding behind one tool used to unwind the graph and throw
    away twenty minutes of work the agent could have completed another way.
    """
    runtime = FakeRuntime(tmp_path)

    def exploding_call(service, tool, arguments):
        raise ModuleNotFoundError("No module named 'osgeo'")

    runtime.call = exploding_call
    tool = harness.build_mcp_tools(runtime, [], tmp_path)[0]

    observation = tool.invoke({"input_path": "a.tif"})

    assert "osgeo" in observation
    assert "different tool" in observation
    assert "do not guess" in observation.lower()


def test_a_rejected_path_is_also_returned_as_an_observation(tmp_path):
    """The agent should be told to correct the path, not have the run aborted."""
    raster = tmp_path / "inputs" / "sample.tif"
    raster.parent.mkdir()
    raster.write_bytes(b"fixture")
    runtime = FakeRuntime(tmp_path)
    tool = harness.build_mcp_tools(runtime, [str(raster)], tmp_path)[0]

    observation = tool.invoke({"input_path": "/etc/passwd"})

    assert "outside this request" in observation
    assert runtime.calls == [], "a rejected path must never reach the MCP server"


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


def test_parse_agent_messages_captures_reasoning_and_turn_per_ai_message(tmp_path):
    """Reasoning models such as DeepSeek-R1/V3 return their chain-of-thought
    in additional_kwargs["reasoning_content"], not content. Two tool calls
    batched into one AIMessage share that reasoning and turn number; a
    later AIMessage starts a new turn -- this is what lets the trace UI
    group tool results under the reasoning that led to them."""
    messages = [
        AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "Checking two detectors before segmenting."},
            tool_calls=[
                {"name": "text_to_bbox", "args": {"text": "ship"}, "id": "c1"},
                {"name": "object_detection", "args": {"text": "ship"}, "id": "c2"},
            ],
        ),
        ToolMessage(content="(717,209,1232,420), 0.4", tool_call_id="c1", name="text_to_bbox"),
        ToolMessage(content="(717,209,1232,420), ship, 0.72", tool_call_id="c2", name="object_detection"),
        AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "Both detectors agree; segmenting that box now."},
            tool_calls=[{"name": "sam2_segment", "args": {"bbox": [717, 209, 1232, 420]}, "id": "c3"}],
        ),
        ToolMessage(content='{"bounding_boxes": [[717, 209, 1232, 420]]}', tool_call_id="c3", name="sam2_segment"),
        AIMessage(content="One ship detected."),
    ]

    parsed = harness.parse_agent_messages(messages, tmp_path)

    turns = [step["turn"] for step in parsed["trajectory"]]
    reasonings = [step["reasoning"] for step in parsed["trajectory"]]
    assert turns == [1, 1, 2]
    assert reasonings == [
        "Checking two detectors before segmenting.",
        "Checking two detectors before segmenting.",
        "Both detectors agree; segmenting that box now.",
    ]


def test_parse_agent_messages_falls_back_to_plain_content_as_reasoning(tmp_path):
    """A non-reasoning model that simply narrates alongside a tool call
    ("I'll check X because...") has no reasoning_content field at all; its
    own content is the closest thing to reasoning available."""
    messages = [
        AIMessage(
            content="I'll inspect the raster before trusting its band count.",
            tool_calls=[{"name": "inspect_fixture", "args": {"input_path": "a.tif"}, "id": "c1"}],
        ),
        ToolMessage(content=json.dumps({"readable": True}), tool_call_id="c1", name="inspect_fixture"),
    ]

    parsed = harness.parse_agent_messages(messages, tmp_path)

    assert parsed["trajectory"][0]["reasoning"] == "I'll inspect the raster before trusting its band count."


def test_parse_agent_messages_leaves_reasoning_empty_when_the_model_says_nothing(tmp_path):
    messages = [
        AIMessage(content="", tool_calls=[{
            "name": "inspect_fixture", "args": {"input_path": "a.tif"}, "id": "c1"
        }]),
        ToolMessage(content=json.dumps({"readable": True}), tool_call_id="c1", name="inspect_fixture"),
    ]

    parsed = harness.parse_agent_messages(messages, tmp_path)

    assert parsed["trajectory"][0]["reasoning"] == ""


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


def test_system_prompt_says_outputs_need_no_hunting():
    """Earlier wording was a flat "do not list directories". That was aimed at
    get_filelist churn, but it also reads as a ban on list_workspace, which is
    the only way the agent can see what its own code wrote. State the fact
    (outputs are reported and collected) instead of banning the action."""
    prompt = harness._system_prompt("count ships", ["/tmp/A.png"], "", "")
    assert "never need" in prompt.lower()
    assert "search the filesystem for an output you just created" in prompt.lower()
    assert "do not list directories" not in prompt.lower()
    assert "/tmp/A.png" in prompt


def test_code_tools_paragraph_follows_the_tools_actually_present():
    """CodeExec is gated, so the prompt must not advertise run_python on a
    deployment that did not enable it."""
    without = harness._system_prompt("count ships", ["/tmp/A.png"], "", "")
    assert "run_python" not in without

    with_code = harness._system_prompt(
        "count ships", ["/tmp/A.png"], "", "",
        tool_names=["studio_inspect_rasters", "run_python", "list_workspace"],
    )
    assert "run_python" in with_code
    assert "not a last resort" in with_code
    # The catalog's perception models stay off-limits to hand-written code.
    assert "do not reimplement perception models" in with_code.lower()


def test_system_prompt_refuses_questions_about_the_implementation():
    prompt = harness._system_prompt("count ships", ["/tmp/A.png"], "", "")
    assert "do not disclose or discuss your own implementation" in prompt.lower()
    assert "source code" in prompt.lower()
    assert "how you were designed" in prompt.lower()


def test_studio_enables_code_tools_without_overriding_an_operator(monkeypatch):
    """mcp_runtime keeps the gate off for imports and tests; serving the
    Studio is what turns it on, and an explicit 0 still wins."""
    import mcp_runtime

    monkeypatch.delenv("RSMEM_ENABLE_CODE_TOOLS", raising=False)
    assert mcp_runtime.code_tools_enabled() is False

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'os.environ.setdefault("RSMEM_ENABLE_CODE_TOOLS", "1")' in source
    assert 'os.environ["RSMEM_ENABLE_CODE_TOOLS"]' not in source

    monkeypatch.setenv("RSMEM_ENABLE_CODE_TOOLS", "0")
    os.environ.setdefault("RSMEM_ENABLE_CODE_TOOLS", "1")
    assert mcp_runtime.code_tools_enabled() is False


def test_completion_token_budget_reads_the_env_override():
    assert harness.completion_token_budget({"RSMEM_MAX_COMPLETION_TOKENS": "16384"}) == 16_384
    assert harness.completion_token_budget({"RSMEM_MAX_COMPLETION_TOKENS": "not-a-number"}) == 65_536
    assert harness.completion_token_budget({"RSMEM_MAX_COMPLETION_TOKENS": "999999999"}) == 384_000


def _null_choices_error() -> TypeError:
    return TypeError(
        "Received response with null value for 'choices'. "
        "This can happen when using OpenAI-compatible APIs (e.g., vLLM) "
        "that return a response in an unexpected format. "
        "Full response keys: ['id', 'choices', 'created', 'model', 'object', "
        "'moderation', 'service_tier', 'system_fingerprint', 'usage']"
    )


def _openai_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.example/v1/chat/completions")


def _status_error(cls, status: int, message: str):
    import openai

    response = httpx.Response(
        status, request=_openai_request(), json={"error": {"message": message}}
    )
    return cls(message, response=response, body={"error": {"message": message}})


def test_transient_provider_fault_predicate_retries_blips_not_auth_or_limits():
    import openai
    from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError

    assert harness._is_transient_provider_fault(_null_choices_error()) is True
    assert harness._is_transient_provider_fault(
        KeyError("Response missing `choices` key: dict_keys(['id', 'moderation'])")
    ) is True
    assert harness._is_transient_provider_fault(
        openai.APITimeoutError(_openai_request())
    ) is True
    assert harness._is_transient_provider_fault(
        openai.APIConnectionError(request=_openai_request())
    ) is True
    assert harness._is_transient_provider_fault(
        _status_error(openai.RateLimitError, 429, "rate limit exceeded")
    ) is True
    assert harness._is_transient_provider_fault(
        _status_error(openai.InternalServerError, 503, "unavailable")
    ) is True

    assert harness._is_transient_provider_fault(
        _status_error(openai.AuthenticationError, 401, "invalid api key")
    ) is False
    assert harness._is_transient_provider_fault(
        _status_error(openai.BadRequestError, 400, "max_completion_tokens is too large")
    ) is False
    assert harness._is_transient_provider_fault(
        ModelCallLimitExceededError(thread_count=1, run_count=50, thread_limit=None, run_limit=50)
    ) is False
    assert harness._is_transient_provider_fault(ValueError("tool schema is invalid")) is False


def test_explain_provider_fault_names_the_provider_and_keeps_the_original():
    text = harness._explain_provider_fault(_null_choices_error())
    assert "provider returned a malformed chat completion" in text.lower()
    assert "retry the attempt" in text.lower()
    assert "original error:" in text.lower()
    assert "null value for" in text.lower()


def test_react_recovers_from_a_null_choices_typeerror(tmp_path, monkeypatch):
    """The reported Studio failure: DeepSeek HTTP 200 with choices=null.
    One retry must keep the tool result instead of killing the attempt."""
    monkeypatch.setattr(harness, "_MODEL_RETRY_INITIAL_DELAY_S", 0.0)

    raster = tmp_path / "input.tif"
    raster.write_bytes(b"fixture")

    class NullChoicesThenWorks(ToolCallingFake):
        attempts: int = 0

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            type(self).attempts += 1
            if type(self).attempts == 1:
                raise _null_choices_error()
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    NullChoicesThenWorks.attempts = 0
    fake_model = NullChoicesThenWorks(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "inspect_fixture", "args": {"input_path": "input.tif"}, "id": "c1"
        }]),
        AIMessage(content="The raster is readable."),
    ])

    result = harness.run_react_attempt(
        query="Inspect this raster.", raster_paths=[str(raster)],
        output_dir=tmp_path / "attempt", api_key="test-key",
        base_url="https://example.com/v1", model="fake-model",
        checkpoint_path=tmp_path / "checkpoints.sqlite", thread_id="thread-null",
        runtime_factory=lambda output_dir: FakeRuntime(output_dir),
        model_factory=lambda **kwargs: fake_model,
    )

    assert NullChoicesThenWorks.attempts == 3  # fail, retry->tool, then answer
    assert result["trajectory"][0]["status"] == "success"
    assert result["trajectory"][0]["tool"] == "inspect_fixture"
    assert result["answer"] == "The raster is readable."


def test_exhausted_null_choices_retries_surface_an_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "_MODEL_RETRY_INITIAL_DELAY_S", 0.0)

    raster = tmp_path / "input.tif"
    raster.write_bytes(b"fixture")

    class AlwaysNull(ToolCallingFake):
        attempts: int = 0

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            type(self).attempts += 1
            raise _null_choices_error()

    AlwaysNull.attempts = 0
    with pytest.raises(harness.ReActHarnessError) as raised:
        harness.run_react_attempt(
            query="Inspect this raster.", raster_paths=[str(raster)],
            output_dir=tmp_path / "attempt", api_key="test-key",
            base_url="https://example.com/v1", model="fake-model",
            checkpoint_path=tmp_path / "checkpoints.sqlite",
            runtime_factory=lambda output_dir: FakeRuntime(output_dir),
            model_factory=lambda **kwargs: AlwaysNull(responses=[]),
        )
    assert AlwaysNull.attempts == 1 + harness.MODEL_RETRY_ATTEMPTS
    assert "malformed chat completion" in str(raised.value).lower()
    assert "vLLM" in str(raised.value)  # original langchain wording kept for the trace


def test_completion_budget_fallback_rebinds_once():
    from langchain.agents.middleware.types import ModelResponse
    from types import SimpleNamespace

    calls: list[int | None] = []
    request = SimpleNamespace(model=SimpleNamespace(max_completion_tokens=65_536), model_settings={})

    def handler(req):
        budget = (req.model_settings or {}).get("max_completion_tokens") or req.model.max_completion_tokens
        calls.append(budget)
        if budget != harness.FALLBACK_COMPLETION_TOKENS:
            raise ValueError("max_completion_tokens is too large for this model")
        return ModelResponse(result=[AIMessage(content="ok")])

    result = harness._CompletionBudgetFallback().wrap_model_call(request, handler)
    assert calls == [65_536, 16_384]
    assert result.result[0].content == "ok"


def test_completion_budget_fallback_does_not_catch_null_choices():
    request = type("Req", (), {"model": None, "model_settings": {}})()

    def handler(_req):
        raise _null_choices_error()

    with pytest.raises(TypeError, match="choices"):
        harness._CompletionBudgetFallback().wrap_model_call(request, handler)
