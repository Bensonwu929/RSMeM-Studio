# RSMeM ModelScope Studio — Next Agent Handoff

Last updated: 2026-09-03 (Asia/Shanghai)  
Repository: `/Users/wu/Documents/APGRSS/rsmem-studio-deploy`  
Stable commit: `80ec984055583872b08c9d7d4f581ca305e8275a`

## 1. Read this first

This repository is the public ModelScope Studio for RSMeM, not the complete
research repository. The user is iteratively improving the public competition
artifact and will continue giving UI, workflow, deployment, and evaluation
requirements.

Do not assume that a visually plausible demo is sufficient. The Studio must
execute real MCP tools, use a real LLM-backed ReAct loop, expose auditable
observations, and never fabricate a successful run when no API authorization is
available.

Never place API keys, ModelScope tokens, OAuth secrets, private server paths, or
credentials in source code, tests, documentation, Git history, SQLite files, or
debug traces. The user has previously pasted credentials in chat. Treat those
as secrets, never repeat them, and recommend rotation if credential hygiene is
being discussed.

## 2. Git state and required workflow

At handoff time the following refs all point to the same commit:

```text
local master                    80ec984
github/master                   80ec984
modelscope/master               80ec984
feat/langgraph-react-harness    80ec984
```

The local `master` was fast-forwarded with:

```bash
git switch master
git merge --ff-only feat/langgraph-react-harness
```

The worktree was clean at the time of the fast-forward.

For every new feature or non-trivial fix, follow this process:

```bash
git switch master
git pull --ff-only github master
git switch -c feat/<short-feature-name>   # or fix/<short-name>

# implement and test
python -m pytest -q
git diff --check

# only after verification
git switch master
git merge --ff-only feat/<short-feature-name>
git push github master
git push modelscope master
```

Do not develop directly on `master`. Do not push a feature branch directly to a
remote `master` before locally integrating and verifying it. Never force-push a
deployment branch unless the user explicitly authorizes it and understands the
risk.

## 3. Remotes and public pages

```text
github     https://github.com/Bensonwu929/RSMeM-Studio.git
modelscope https://www.modelscope.cn/studios/wbx929/RSMeM.git
```

Public resources:

- Studio: https://modelscope.cn/studios/wbx929/RSMeM
- Main open-source repository: https://github.com/AI9Stars/RSMeM
- ModelScope Learn article: https://modelscope.cn/learn/435887
- arXiv paper: https://arxiv.org/abs/2607.24772
- Competition: https://modelscope.cn/events/263

## 4. Product objective

RSMeM is a knowledge-enhanced memory-evolution framework for remote-sensing
agents. The Studio should communicate and execute this loop:

```text
Retrieve relevant geoscience knowledge
                ↓
ReAct agent loop
Reason ↔ MCP tool call ↔ Observation
                ↓
Evidence-grounded answer
                ↓
Evaluation
  ├─ bundled case: deterministic evaluator, then LLM Judge if ambiguous
  └─ user upload: Teacher Failure Gate advice + authoritative user decision
                ↓
If incorrect: generate reflection candidates
  ├─ FTRO-compressed trajectory
  ├─ full trajectory
  ├─ grounded LLM summary
  └─ user-authored diagnosis and next plan
                ↓
Only the user-confirmed reflection is written to Failure-Aware Memory
                ↓
Next attempt
```

Attempt limits:

- Bundled case: maximum 3 attempts.
- User-provided task: maximum 10 attempts.
- Each attempt: maximum 30 total model calls, including agent, judge/gate, and
  reflection calls.

There must be no keyless fake execution or preset “correct-looking” result.
Correct is correct; incorrect is incorrect.

## 5. Current architecture

### ReAct harness

Primary implementation: `react_harness.py`

- Uses current `langchain.agents.create_agent`.
- Uses LangGraph middleware for the 30-model-call budget.
- Uses an arbitrary OpenAI-compatible `base_url` through `ChatOpenAI`.
- Dynamically converts the live MCP tool catalogue into LangChain
  `StructuredTool` objects.
- Tool execution goes through the persistent `MCPRuntime`; it is not a hidden
  direct-Python substitute.
- Uses a SQLite LangGraph checkpointer for durable attempt state and audit
  groundwork.
- Each attempt gets a separate checkpoint namespace. Cross-attempt learning is
  carried only through confirmed FAM, not by silently replaying the entire
  previous transcript.

Key budgets:

```text
MAX_MODEL_CALLS_PER_ATTEMPT = 30
GRAPH_RECURSION_LIMIT       = 96
MAX_COMPLETION_TOKENS       = 16384
```

### Outer workflow

Primary implementation: `app.py`

The current Judge, Teacher Failure Gate, three reflection modes, user
confirmation, and retry behavior remain outside the inner ReAct agent. This is
intentional: tool autonomy belongs inside the agent; correctness policy and
human authority belong in the outer workflow.

### Evaluation

Primary implementation: `deterministic_evaluator.py`

For bundled cases:

```text
deterministic evaluator
        ↓ unable to decide
LLM Judge
```

The deterministic evaluator must not pass an answer merely because it contains
reference-like text. Missing tool evidence or failed tool calls cannot be
treated as a verified success.

For uploaded data, the Teacher Failure Gate is advisory. The user makes the
final decision.

### Knowledge retrieval

Retrieved HKG items are shown and injected into agent context only when at least
one retrieval score is greater than 1. If all scores are `<= 1`, knowledge is
neither displayed nor appended to the prompt.

### MCP servers

Configuration: `configs/agent_mcp.json`

CPU server families:

- Analysis
- Index
- Inversion
- Statistics

Perception/GPU server families are health-gated and must not be advertised when
their workers are unavailable. Relevant files include:

- `tools/Perception.py`
- `tools/OEA_Perception.py`
- `gpu_service_manager.py`
- `xgpu_runtime.py`
- `xgpu_workers/`
- `requirements-xgpu.txt`
- `XGPU_MODEL_MANIFEST.md`

Local CPU discovery previously found approximately 117 unique tools. The full
serialized schema is large (roughly 29k tokens), which is an important latency
and tool-selection risk. The user previously chose to expose the full catalogue
first and evaluate tool selection before introducing aggressive routing.

## 6. Current UI decisions

The public UI is intentionally restrained and uses a Codex-like neutral black,
white, and gray palette rather than copying another Studio's color scheme.

Current top-level tabs:

- Run RSMeM
- Open-source resources
- Challenge submission

The old Overview tab was removed because it duplicated information and delayed
the main task.

Input behavior:

- The user can write a custom remote-sensing query.
- The user can upload GeoTIFF/TIFF/PNG/JPEG imagery.
- The user can load one of 13 bundled cases; loading a case fills an editable
  query and bundled input files.
- A raster/image preview is shown on the right side.
- Two model connection modes are intended:
  - ModelScope quota through OAuth/browser session.
  - User-provided OpenAI-compatible Base URL, model, and API key.
- No run should begin without valid model authorization.

Current progress visualization:

```text
Retrieve → ReAct agent loop → Answer → Evaluate
```

The ReAct card explicitly represents:

```text
reason ↔ tool call ↔ observation
```

Do not split Reason and Act/Observe into misleading one-pass linear stages.

Current result layout:

1. Summary
   - User query
   - Current result
   - Evaluator assessment
2. Agent trace
   - Attempt number
   - Tool calls
   - Observation (each tool observation is collapsed by default)
3. Evidence & artifacts
   - Retrieved knowledge
   - Generated files
   - Debug trace

Deliberately removed from visible Agent trace:

- `Context: HKG = ... · selected FAM = ... · model calls = ...`
- “Selected by the ReAct agent from the current observation.”
- duplicate final-answer rendering
- the Agent trace subtitle

The internal information may remain in the downloadable debug trace.

Also removed:

- “Starting a bounded ReAct loop over the live MCP tool catalogue.”
- the hero subtitle beginning “From remote-sensing imagery to verified
  answers...”.

Reflection candidates are presented as Candidate 1, Candidate 2, Candidate 3,
or user-authored text. Do not expose a large raw JSON comparison block in the
normal UI.

## 7. Tests and verified behavior

The most recent full automated test run reported:

```text
95 passed in 20.38s
```

A later focused run of core workflow modules reported:

```text
71 passed in 3.20s
```

Test areas include:

- ReAct message/tool-call parsing.
- MCP discovery and runtime behavior.
- deterministic evaluation and judge fallback.
- ModelScope/OpenAI-compatible connection modes.
- interactive state transitions.
- reflection confirmation and retry.
- UI structure and visibility rules.
- GPU perception health gating.
- 20 tool-selection fixtures selected from the research benchmark, including
  10 perception-heavy cases from its last 40 questions.

Earlier real backend smoke test (before the most recent UI-only changes):

- Case: Bhutan nighttime-light comparison.
- Model calls: 3.
- Selected tools in order:
  1. `studio_inspect_rasters`
  2. `calc_batch_image_mean`
  3. `difference`
- All calls succeeded.
- June mean: approximately `0.3228`.
- August mean: approximately `0.4106`.
- Absolute difference: approximately `0.0878`.
- Correct conclusion: June was lower.

This proves sequential observation-driven tool selection in the backend, but it
does **not** replace a fresh browser end-to-end test of the currently deployed
UI.

## 8. Known gaps and risks

### Browser end-to-end test is not complete

The latest UI structure has strong automated coverage, but a complete browser
run of the current version—entering credentials, choosing a case, clicking Run,
waiting for the result, and visually checking Summary/Agent trace/evaluator—has
not yet been completed.

The local `python app.py` process launched from a short-lived tool session has
repeatedly stopped after that session ended, causing `ERR_CONNECTION_REFUSED`.
This is a local process-lifecycle/testing issue, not evidence that the deployed
ModelScope service crashes. Use a durable foreground terminal or controlled
background process for browser testing and clean it up afterward.

### Latency

Potential contributors:

- Several LLM calls per ReAct attempt.
- Very large MCP tool schemas sent to the model.
- Initial MCP stdio server startup.
- External OpenAI-compatible provider latency.
- Judge/gate/reflection calls after the agent finishes.

Do not claim “the UI is slow” without measuring these components separately.
Add timestamps for model calls, MCP discovery/startup, individual tools,
evaluation, and total attempt duration before optimizing.

### ModelScope OAuth

The UI implements two connection modes, but a local run may show that the
ModelScope OAuth client is not configured on that deployment. Verify deployed
environment variables and callback URLs rather than removing the quota mode.

### GPU perception

GPU services may be intentionally disabled locally. CPU mode must remain
functional. Perception tools must only appear when workers pass health checks.
Do not silently substitute fabricated vision outputs.

### OpenAI-compatible tool calling

Some providers do not reliably implement standard tool calls. Preserve protocol
tests and fail clearly when a provider returns only text instead of callable
tool structures. Do not fall back to pretending a textual tool plan was
executed.

## 9. Important local research references

Use these only as read-only implementation references unless the user asks to
modify them:

- Public RSMeM code:
  `/Users/wu/Desktop/RSMeM-public`
- RSMeM paper source:
  `/Users/wu/Desktop/260106ACL/ACL_2026_RSMeM_Arixv`
- Vision/MCP guidance:
  `/Users/wu/Desktop/RSMeM-public/docs/VISION_MCP_CODEX_CLAUDE_CODE.md`
- Benchmark questions:
  `/Users/wu/Downloads/RSMeM/benchmark/question.json`
- Server research run directory:
  `/ultrafast_share/gzh_data/wubingxian/RSMeM-public/runs/deepseek_v4flash_repro_repair/dsml_parser_20260729_v2`
- Teacher Failure Gate prompts/source on server:
  `/ultrafast_share/gzh_data/wubingxian/RSMeM-public/private_research/teacher_failure_gate/`
- FTRO/full-trajectory/summary reflector prompts/source on server:
  `/ultrafast_share/gzh_data/wubingxian/RSMeM-public/private_research/ftro_fulltraj_summary_reflector/`

The SSH host alias previously used by the user is `QY_11`. Treat remote access
as read-only unless the user explicitly asks for changes. Never copy server
secrets into this public repository.

## 10. Definition of done for future changes

Before merging and deploying:

- Work is on a new `feat/*` or `fix/*` branch.
- No unrelated user changes are overwritten.
- `python -m pytest -q` passes.
- `git diff --check` passes.
- No credentials or private data are present in the diff or generated files.
- Relevant browser flow is manually smoke-tested.
- At least one real API + MCP case succeeds for changes affecting execution.
- GPU-disabled behavior is tested when relevant.
- README/deployment documentation is updated when behavior changes.
- The user reviews material UI or workflow changes before deployment when they
  requested prior confirmation.
- Only verified code is fast-forwarded/merged into `master`.
- GitHub `master` is pushed first, then ModelScope `master`.
- Both deployment remotes are verified to point to the same commit.

## 11. Suggested prompt for the next agent

Copy the prompt below into the next task together with this file path:

```text
Please continue collaborating with me on the RSMeM ModelScope Studio project.

Repository:
/Users/wu/Documents/APGRSS/rsmem-studio-deploy

Before doing any work, read this handoff completely:
/Users/wu/Documents/APGRSS/rsmem-studio-deploy/docs/NEXT_AGENT_HANDOFF.md

Then inspect the current repository and confirm the current branch, clean/dirty
state, HEAD commit, GitHub remote commit, and ModelScope remote commit. Treat the
handoff as context, but verify repository facts directly before modifying code.

For every new feature or non-trivial fix, create a new feat/* or fix/* branch
from current master. Do not develop directly on master. Preserve unrelated
changes. Use apply_patch for edits. Never write or upload API keys, OAuth
secrets, ModelScope tokens, or private credentials.

The Studio must remain a real MCP-backed ReAct agent rather than a simulated
demo. It must not execute without valid model authorization, fabricate tool
results, hard-code correctness, or bypass the Judge/Teacher Gate/user-decision
workflow.

When I give you the next requirement:
1. Inspect the relevant implementation and tests first.
2. Explain any important architecture or UX trade-off briefly.
3. Implement the requested change on a feature/fix branch.
4. Add or update high-quality tests.
5. Run focused tests, then the complete suite and git diff --check.
6. Show me the local result before merging or deploying when the change is
   visual or behaviorally significant.
7. Merge to master and push GitHub/ModelScope only after I explicitly approve
   that release step.

Start by summarizing your understanding of the current system and reporting any
discrepancy between the handoff and the actual repository. Do not make changes
until I provide the next requirement.
```

