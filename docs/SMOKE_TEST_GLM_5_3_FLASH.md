# GLM-5.3-Flash smoke test

Tested against the OpenAI-compatible endpoint at `uni-api.cstcloud.cn/v1`. The credential was provided only through a temporary shell environment variable and is not present in the repository, checkpoint, test fixture, or this report.

## Tool-calling protocol and dry selection

- Model: `glm-5.3-flash`
- CPU questions: 10
- Standard OpenAI `tool_calls` returned: 10/10
- Cases selecting at least one reference-appropriate tool: 10/10
- Mean full-chain recall in a single response: 0.342
- Expected CPU tools absent from deployment: 0

The low one-response full-chain recall is expected for a ReAct agent: on many questions the model correctly selected only the first executable action (`get_filelist`, inspection, or the main inversion/index tool) and waited for an observation instead of pretending it already knew the inputs needed by later tools.

## Real ReAct + MCP execution

Task: compare Bhutan nighttime-light rasters from June and August 2024, identify the lower month, and report the absolute difference.

- Model calls: 3
- Live MCP catalogue: 117 tools from Analysis, Index, Inversion, and Statistics
- Selected execution sequence:
  1. `studio_inspect_rasters`
  2. `calc_batch_image_mean`
  3. `difference`
- MCP status: all three calls succeeded
- Reported June mean: 0.3228
- Reported August mean: 0.4106
- Reported absolute difference: 0.0878
- Conclusion: June is lower

The result matches the bundled deterministic case contract. This validates the provider's tool-call wire format, sequential observation-driven tool selection, MCP execution, and evidence-grounded final answer.
