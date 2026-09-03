"""LangChain 1.x ReAct harness over RSMeM's persistent stdio MCP runtime.

The harness deliberately keeps MCP process ownership in :mod:`mcp_runtime`.
LangChain receives one StructuredTool for every discovered MCP tool, while all
actual execution still crosses the MCP stdio boundary.  This avoids a second
set of short-lived MCP subprocesses for every agent step.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from mcp_runtime import MCPRuntime


MAX_MODEL_CALLS_PER_ATTEMPT = 30
# A model call followed by a tool round normally consumes two graph supersteps.
# Ninety-six leaves headroom for middleware/finalization without lowering the
# explicit 30-model-call budget requested for one attempt.
GRAPH_RECURSION_LIMIT = 96
MAX_COMPLETION_TOKENS = 16_384


class ReActHarnessError(RuntimeError):
    """Raised when the ReAct graph cannot produce a usable attempt."""


def _safe_tool_name(service: str, native_name: str, duplicates: set[str]) -> str:
    candidate = native_name if native_name not in duplicates else f"{service}_{native_name}"
    candidate = re.sub(r"[^A-Za-z0-9_-]", "_", candidate)
    return candidate[:64]


def _within(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


@dataclass(frozen=True)
class ToolPathPolicy:
    """Keep model-authored paths inside the current request's data sandbox."""

    raster_paths: tuple[Path, ...]
    output_dir: Path

    @property
    def input_roots(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(path.parent for path in self.raster_paths))

    @property
    def allowed_roots(self) -> tuple[Path, ...]:
        return self.input_roots + (self.output_dir,)

    def _input_candidate(self, value: str) -> str:
        requested = Path(value).expanduser()
        if not requested.is_absolute():
            exact = next((path for path in self.raster_paths if path.name == requested.name), None)
            if exact:
                return str(exact)
            for root in self.input_roots:
                candidate = (root / requested).resolve()
                if candidate.exists() and _within(candidate, self.allowed_roots):
                    return str(candidate)
            candidate = (self.output_dir / requested).resolve()
        else:
            candidate = requested.resolve()
        if not _within(candidate, self.allowed_roots):
            raise ReActHarnessError(f"MCP input path is outside this request: {requested.name}")
        return str(candidate)

    def _output_candidate(self, value: str) -> str:
        # A model may suggest an absolute path from a prompt/example.  Outputs
        # are always rebased to this attempt directory and never escape it.
        requested = Path(value or "derived_output.tif")
        relative = Path(requested.name) if requested.is_absolute() else requested
        candidate = (self.output_dir / relative).resolve()
        if not _within(candidate, (self.output_dir,)):
            raise ReActHarnessError("MCP output path attempted to leave the request directory.")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return str(candidate)

    def sanitize(self, arguments: dict[str, Any]) -> dict[str, Any]:
        def visit(key: str, value: Any) -> Any:
            lowered = key.lower()
            path_like = lowered in {
                "image", "pre_image", "post_image", "path", "paths", "file", "files",
                "folder", "directory", "dir",
            } or lowered.endswith(
                (
                    "_path", "_paths", "_file", "_files", "_folder", "_folders",
                    "_directory", "_directories", "_dir", "_dirs",
                )
            )
            if isinstance(value, list):
                return [visit(key, item) for item in value]
            if not path_like or not isinstance(value, str) or not value.strip():
                return value
            if "output" in lowered or lowered.startswith("save"):
                return self._output_candidate(value)
            return self._input_candidate(value)

        return {key: visit(key, value) for key, value in arguments.items()}


def build_mcp_tools(runtime: MCPRuntime, raster_paths: list[str], output_dir: Path) -> list[StructuredTool]:
    """Expose every discovered tool from the selected MCP servers to ReAct."""
    policy = ToolPathPolicy(
        tuple(Path(path).expanduser().resolve() for path in raster_paths),
        Path(output_dir).resolve(),
    )
    counts: dict[str, int] = {}
    available_items = [
        item for item in runtime.catalog.values()
        if runtime.is_tool_available(item["service"], item["name"])
    ]
    for item in available_items:
        counts[item["name"]] = counts.get(item["name"], 0) + 1
    duplicates = {name for name, count in counts.items() if count > 1}

    tools: list[StructuredTool] = []
    for item in available_items:
        service, native_name = item["service"], item["name"]
        public_name = _safe_tool_name(service, native_name, duplicates)

        def invoke_tool(
            _service: str = service,
            _native_name: str = native_name,
            **arguments: Any,
        ) -> str:
            safe_arguments = policy.sanitize(arguments)
            result = runtime.call(_service, _native_name, safe_arguments)
            return json.dumps(result, ensure_ascii=False, default=str)

        description = (item.get("description") or "Execute an RSMeM geospatial operation.").strip()
        description += f" [MCP server: {service}; native tool: {native_name}]"
        tools.append(
            StructuredTool.from_function(
                func=invoke_tool,
                name=public_name,
                description=description,
                args_schema=item["inputSchema"],
                infer_schema=False,
            )
        )
    return tools


@contextmanager
def _checkpointer(path: Path):
    """Use SQLite when installed, with an in-memory fallback for dev imports."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:  # pragma: no cover - deployment installs requirements.txt
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                texts.append(str(item.get("text", "")))
        return "\n".join(texts)
    return str(content)


def parse_agent_messages(messages: list[BaseMessage], output_dir: Path) -> dict[str, Any]:
    """Convert LangChain messages into RSMeM's UI-facing plan and trajectory."""
    calls: dict[str, dict[str, Any]] = {}
    plan: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    answers: list[str] = []
    model_calls = 0

    for message in messages:
        if isinstance(message, AIMessage):
            model_calls += 1
            tool_calls = list(message.tool_calls or [])
            if not tool_calls and _message_text(message).strip():
                answers.append(_message_text(message).strip())
            for call in tool_calls:
                call_id = str(call.get("id") or f"call_{len(calls) + 1}")
                record = {
                    "tool": str(call.get("name", "unknown_tool")),
                    "purpose": "Selected by the ReAct agent from the current observation.",
                    "parameters": dict(call.get("args") or {}),
                }
                calls[call_id] = record
                plan.append(record)
        elif isinstance(message, ToolMessage):
            call = calls.get(str(message.tool_call_id), {})
            text = _message_text(message)
            status = "error" if getattr(message, "status", None) == "error" else "success"
            trajectory.append(
                {
                    "step": len(trajectory) + 1,
                    "tool": call.get("tool") or message.name or "unknown_tool",
                    "purpose": call.get("purpose", "Executed by the ReAct agent."),
                    "parameters": call.get("parameters", {}),
                    "status": status,
                    "observation": text,
                }
            )

    answer = answers[-1] if answers else ""
    artifacts = []
    for path in Path(output_dir).rglob("*"):
        if path.is_file() and path.name != "checkpoints.sqlite":
            artifacts.append(str(path.resolve()))
    return {
        "tool_plan": plan,
        "trajectory": trajectory,
        "answer": answer,
        "model_calls": model_calls,
        "artifacts": sorted(set(artifacts)),
    }


def _system_prompt(
    query: str,
    raster_paths: list[str],
    knowledge_context: str,
    failure_memory: str,
) -> str:
    files = "\n".join(f"- {Path(path).resolve()}" for path in raster_paths)
    return f"""You are RSMeM, a ReAct remote-sensing agent. Solve the user's task by alternating
between reasoning, one or more real MCP tool calls, observing their returned evidence, and adapting
the next action. You are not following a frozen one-shot plan. Use only tool observations for numeric
or visual claims. Inspect available inputs before band- or date-dependent operations. Do not invent
files, values, successful calls, or perception results. Stop and answer as soon as the requested
conclusion is supported. If a tool fails, diagnose the observation and choose a corrected action.

Current task:
{query}

Files available to this request (all outputs must be written through tool output_path arguments):
{files}

Retrieved hierarchical knowledge (omitted when retrieval score <= 1):
{knowledge_context or "(no sufficiently relevant knowledge retrieved)"}

Confirmed failure-aware memory from the previous attempt:
{failure_memory or "(none; this is the first attempt)"}
"""


def run_react_attempt(
    *,
    query: str,
    raster_paths: list[str],
    output_dir: Path,
    api_key: str,
    base_url: str,
    model: str,
    knowledge_context: str = "",
    failure_memory: str = "",
    thread_id: str | None = None,
    checkpoint_path: Path | None = None,
    runtime_factory: Callable[..., MCPRuntime] = MCPRuntime,
    model_factory: Callable[..., Any] = ChatOpenAI,
    event_callback: Callable[[str], None] | None = None,
    model_call_limit: int = MAX_MODEL_CALLS_PER_ATTEMPT,
) -> dict[str, Any]:
    """Run one bounded ReAct attempt over all live MCP tools."""
    if not api_key:
        raise ReActHarnessError("A valid model credential is required for a ReAct attempt.")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_path or output_dir / "checkpoints.sqlite"
    thread_id = thread_id or uuid.uuid4().hex

    runtime = runtime_factory(output_dir)
    runtime.start()
    try:
        tools = build_mcp_tools(runtime, raster_paths, output_dir)
        if not tools:
            raise ReActHarnessError("No MCP tools were discovered for the ReAct agent.")
        chat_model = model_factory(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0,
            timeout=180.0,
            max_retries=2,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        middleware = [
            ModelCallLimitMiddleware(
                run_limit=model_call_limit,
                exit_behavior="error",
            )
        ]
        with _checkpointer(Path(checkpoint_path)) as saver:
            agent = create_agent(
                model=chat_model,
                tools=tools,
                system_prompt=_system_prompt(
                    query, raster_paths, knowledge_context, failure_memory
                ),
                middleware=middleware,
                checkpointer=saver,
                name="rsmem_react",
            )
            if event_callback:
                event_callback(f"ReAct started with {len(tools)} live MCP tools.")
            result = agent.invoke(
                {"messages": [HumanMessage(content=query)]},
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": GRAPH_RECURSION_LIMIT,
                },
            )
        parsed = parse_agent_messages(list(result.get("messages", [])), output_dir)
        if not parsed["answer"]:
            raise ReActHarnessError("The ReAct agent stopped without a final answer.")
        if event_callback:
            event_callback(
                f"ReAct completed after {parsed['model_calls']} model call(s) and "
                f"{len(parsed['trajectory'])} MCP tool call(s)."
            )
        parsed.update(
            {
                "thread_id": thread_id,
                "checkpoint_path": str(Path(checkpoint_path).resolve()),
                "mcp_health": runtime.health(),
            }
        )
        return parsed
    except Exception as exc:
        if isinstance(exc, ReActHarnessError):
            raise
        raise ReActHarnessError(str(exc)) from exc
    finally:
        runtime.close()


def tool_call_protocol_probe(
    *, api_key: str, base_url: str, model: str, query: str, tool_names: list[str],
    http_client: Any | None = None,
) -> dict[str, Any]:
    """Probe whether an OpenAI-compatible endpoint returns standard tool calls.

    No geospatial tool is executed.  This separates protocol compatibility from
    tool-selection quality and is safe to run against a new provider.
    """
    schemas = [
        StructuredTool.from_function(
            func=lambda **kwargs: kwargs,
            name=name,
            description=f"Candidate geospatial tool {name}.",
            args_schema={"type": "object", "properties": {}},
            infer_schema=False,
        )
        for name in tool_names
    ]
    chat_model = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0,
        timeout=120,
        max_retries=0,
        max_completion_tokens=2048,
        http_client=http_client,
    ).bind_tools(schemas, tool_choice="auto")
    message = chat_model.invoke(
        "Select the appropriate geospatial tool or tools for this task. Do not solve it directly.\n"
        + query
    )
    calls = list(getattr(message, "tool_calls", []) or [])
    return {
        "supported": bool(calls),
        "tool_calls": [
            {"name": call.get("name"), "args": call.get("args", {})} for call in calls
        ],
        "raw_text": _message_text(message),
    }
