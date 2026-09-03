# RSMeM ReAct harness

## Runtime architecture

One attempt now uses LangChain 1.x `create_agent`, which is backed by LangGraph. The model receives every tool discovered from the four CPU MCP servers (`Analysis`, `Index`, `Inversion`, and `Statistics`). A tool selected by the model is still executed by the persistent stdio `MCPRuntime`; LangChain does not call the Python implementation directly.

```text
query + HKG + confirmed FAM + input paths
  -> model reasoning / tool call
  -> MCP execution
  -> tool observation
  -> model chooses the next tool or finishes
  -> evidence-grounded answer
```

This is ReAct rather than plan-then-execute: later tool choices can depend on real observations and errors from earlier calls.

## Budgets

- Hard limit: **30 model calls per attempt** (`ModelCallLimitMiddleware`).
- LangGraph recursion limit: **96 supersteps**. This is not a second model budget; it leaves room for up to 30 model turns plus tool and middleware nodes.
- Maximum completion tokens per model call: **16,384**.
- Provider retries: **2** for transient API failures.
- No independent hard tool-call, total-token, or wall-clock limit was added. MCP calls retain their existing per-call timeout.
- Built-in cases remain limited to 3 attempts; visitor-uploaded tasks remain limited to 10 user-confirmed attempts.

## What SQLite solves

SQLite is the local LangGraph checkpoint store. It records graph state after supersteps, including messages, tool calls, and tool observations. It supports resuming after a human confirmation or recoverable interruption, inspecting the exact state behind a tool choice, and avoiding full replay of completed nodes when a graph resumes later.

It does not improve model quality and it is not Failure-Aware Memory. API credentials live in the model client/configuration and are never inserted into checkpoint state.

Claude Code and Codex use the same broad pattern with product-specific persistence: append-only conversation or rollout records reconstruct a task. LangGraph exposes persistence as an application checkpointer, so RSMeM can use SQLite locally and later replace it with PostgreSQL without changing graph semantics.

## Why a resumed human node can appear to run twice

LangGraph `interrupt()` pauses with a resumable graph interrupt. On resume, the interrupted node starts again from its first line; the `interrupt()` call then returns the supplied human value. Therefore this is unsafe:

```python
charge_credit_card()       # may run again
decision = interrupt(...)  # pause
```

RSMeM keeps user confirmation at a separate boundary: expensive model/tool work and reflection generation finish before waiting; state-changing work happens after the confirmed value is returned. Remaining side effects should be idempotent, keyed by thread and attempt ID.

## Deterministic evaluation first

For bundled cases the evaluator order is:

```text
case-specific deterministic contract
  -> decisive pass or real tool error: use it
  -> otherwise: reference-aware LLM judge
```

The deterministic evaluator is conservative. Missing or ambiguous evidence falls back to semantic judging instead of being labeled wrong.

## OpenAI-compatible provider tests

Unit protocol tests use a mock OpenAI `/chat/completions` endpoint and verify that standard `tool_calls` are parsed while text-only responses are flagged as unsupported.

`scripts/run_tool_selection_benchmark.py` performs a credentialed dry run without executing geospatial tools. It covers 20 curated questions: 10 CPU cases and 10 perception-heavy cases from questions 209–248. It reports protocol success, expected-tool recall, and tools absent from the deployment.

```bash
RSMEM_BASE_URL=https://provider.example/v1 \
RSMEM_MODEL=provider-model-name \
RSMEM_API_KEY='...' \
python scripts/run_tool_selection_benchmark.py
```

The perception partition intentionally reveals deployment gaps. GPU-backed tools are not advertised to the live agent unless their workers pass health checks.

