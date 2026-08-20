# RSMeM: Knowledge-Enhanced Memory Evolution for Remote Sensing Agents

**Paper:** [arXiv abstract](https://arxiv.org/abs/2607.24772) ·
[PDF](https://arxiv.org/pdf/2607.24772) ·
**Code:** [GitHub](https://github.com/AI9Stars/RSMeM) ·
**Demo:** [ModelScope Studio](https://modelscope.cn/studios/wbx929/RSMeM)

## Abstract

RSMeM is a knowledge-enhanced memory evolution framework for reliable remote
sensing agents. It combines Hierarchical Knowledge Grounding, which retrieves
structured geoscience knowledge for tool planning, with Failure-Aware Experience
Refinement, which converts failed trajectories into reusable execution
constraints. The public Studio demonstrates the complete
retrieve-plan-execute-judge-reflect-confirm-retry loop on thirteen remote-sensing cases. Its
CPU-only raster toolchain produces auditable GeoTIFF outputs and supports
visitor-provided OpenAI-compatible models without storing API credentials.

## 1. Research question

Remote-sensing analysis rarely consists of a single model prediction. A useful
agent must identify the correct data products, retrieve domain-specific
knowledge, select tools in the right order, respect units and band semantics,
inspect intermediate outputs, and recover when a step fails. General-purpose
LLMs often produce plausible plans but remain brittle in these multi-step
workflows: one incorrect band pairing, threshold, path, or variable definition
can invalidate every downstream result.

RSMeM asks:

> Can a remote-sensing agent combine structured geoscience knowledge with
> reusable experience distilled from failed executions, so that it becomes more
> reliable without accumulating long raw trajectories?

The framework addresses this question with two complementary memories:

- **Hierarchical Knowledge Grounding (HKG)** retrieves task-level guidance from
  an expert-curated domain hierarchy before tool planning.
- **GeoTraceCompress** performs type-aware compression of failed execution
  evidence before any LLM reflection: long lists retain their first three and
  last two elements, dictionaries retain representative key-value pairs, and
  errors receive a larger text budget.
- **Failure-Aware Experience Refinement (FAR)** supplies the compressed
  trajectory, symbolic critic diagnosis, query, and HKG context to the
  Reflection Generator, producing a short actionable constraint for the next
  attempt.

## 2. Dataset and demonstration cases

The research evaluation uses **EarthBench**, a benchmark introduced with
Earth-Agent for tool-augmented Earth-observation workflows. The public Studio
bundles thirteen Apache-2.0-licensed EarthBench subsets, selected because they
exercise different CPU-compatible geoscience workflows without requiring a GPU
perception model.

| Case | Inputs | Scientific workflow |
|---|---:|---|
| Sichuan agricultural drought | 2 GeoTIFFs | NDVI + LST → TVDI → dryness ratio and area |
| California wildfire trend | 22 GeoTIFFs, 11 dates | paired reflectance bands → NBR series → regional means → Sen's slope |
| Bhutan nighttime-light comparison | 2 GeoTIFFs | spatial means → temporal comparison → absolute difference |
| Black Forest temperature contrast | 3 GeoTIFFs | NDVI emissivity → single-channel LST → land-cover contrast |
| Cyprus thermal inertia | 6 GeoTIFFs | paired day/night BT + albedo → ATI change |
| Mumbai night-light trend | 12 GeoTIFFs | annual means → 2013–2024 linear trend |
| Rosario vegetation growth | 17 GeoTIFFs | temporal means → consecutive percentage change |
| Dongting Lake turbidity | 29 GeoTIFFs | red reflectance → turbidity → three-day episode |
| Congo rainfall | 30 GeoTIFFs | daily rainfall means → monthly trend |
| Lake Urmia turbidity | 24 GeoTIFFs | turbidity maps → means and skewness |
| California fire hotspots | 22 GeoTIFFs | NBR → low-NBR masks → dominant direction |
| Thailand fire-prone zones | 30 GeoTIFFs | MaxFRP composite → upper-quartile mask |
| Shanghai residential volume | 20 GeoTIFFs | paired subtraction → 1985–2020 trend |

Visitors may instead upload their own GeoTIFFs. Uploaded files override the
bundled case and are used only for the current request.

**Data source:** [EarthBench](https://huggingface.co/datasets/Sssunset/Earth-Bench)  
**Original project:** [Earth-Agent](https://github.com/opendatalab/Earth-Agent)  
**Dataset license:** Apache-2.0

## 3. Models and knowledge resources

RSMeM is model-agnostic. The paper evaluates four representative backbones:
DeepSeek-V3.2, Kimi-K2, Qwen3-32B, and Qwen3-8B. The hosted Studio accepts any
public HTTPS OpenAI-compatible endpoint. Its default fields demonstrate the
tested `deepseek-v4-flash-aistar` endpoint identifier, while every visitor
provides their own Base URL, model name, and API key.

No credential is stored in project files or application outputs. The key is
held only long enough to construct the current planning or reflection request.
A valid API key is required: the Studio does not substitute a preset or
deterministic tool plan when the model is unavailable.

The public knowledge base contains three top-level domains, 12 subdomains, and
64 atomic task cards. The full RSMeM implementation uses hybrid semantic and
lexical retrieval. The CPU Studio uses transparent lexical retrieval to supply
evidence to the model-authored planner.

## 4. Workflow

For every task, the Studio exposes the complete RSMeM state transition rather
than returning only a final answer:

1. **Retrieve knowledge.** HKG finds query-relevant geoscience task cards.
2. **Plan tools.** The agent constructs a short tool chain and validates required
   dependencies such as raster inspection before band-dependent operations.
3. **Execute tools.** CPU raster tools run on real GeoTIFF inputs and record their
   parameters, observations, and derived artifacts.
4. **Synthesize an answer.** A separate LLM summarizes only claims supported by
   the real tool trajectory.
5. **Evaluate the attempt.** Bundled cases use a reference-aware LLM judge;
   uploads use a no-answer Teacher Failure Gate and wait for the user's decision.
6. **Generate comparable repairs.** FTRO, raw full trajectory, and a grounded
   LLM summary are passed separately through the same reflector prompt.
7. **Confirm memory.** The user selects and may edit one model repair, or writes
   a fourth diagnosis and plan. Only that confirmed text enters FAM.
8. **Retry conditionally.** The next planner receives HKG plus the selected FAM.
   Bundled cases allow three attempts and uploads allow ten user-confirmed attempts.

The interface does not inject a demonstration failure. Reference information is
confined to the bundled-case judge and never enters any reflection or later
planner prompt. For uploaded data, the Teacher Failure Gate remains
ground-truth-free and advisory; the user makes the final accept/retry decision.

## 5. System design

The hosted artifact is organized into six auditable layers:

- **Gradio interface:** case selection, custom GeoTIFF uploads, user-supplied API
  configuration, execution trace, and artifact download.
- **Knowledge retriever:** lightweight retrieval over the released hierarchical
  geoscience knowledge base.
- **Planner:** required LLM-based tool planning constrained to the published CPU
  tool registry; malformed, empty, or unsupported plans are reported as errors.
- **CPU tool runtime:** Rasterio and NumPy implementations of selected public
  RSMeM Index, Statistics, Inversion, and Analysis semantics.
- **Judge/Gate and FAR memory:** isolated case judging, an advisory no-answer
  upload gate, three trajectory representations through one reflector, and
  explicit user confirmation before memory injection.
- **Safety boundary:** public HTTPS endpoint validation, blocked private or
  loopback Base URLs, request-scoped uploads, isolated temporary outputs, and no
  embedded API credentials.

GPU-based visual perception services are intentionally excluded from the free
CPU deployment. The complete open-source repository retains the original MCP
tool interfaces and deployment documentation for extending the Studio.

## 6. Demonstration results

All values below are produced by the public CPU workflow from the bundled
GeoTIFFs.

### 6.1 Sichuan agricultural drought

The Studio validates the NDVI and LST scale factors, fits dry and wet edges in
NDVI-LST feature space, creates a TVDI GeoTIFF, and thresholds it at 0.75.

- Valid pixels above TVDI 0.75: **28.07%**
- Dry-pixel count: **367,090**
- Area from the 30 m projected grid: **330,381,000 m²**

### 6.2 California wildfire trend

The executor pairs B02 and B07 rasters by acquisition date, generates 11 NBR
GeoTIFFs, computes one regional mean per date, and estimates a robust trend.

- Paired observations: **11**
- Sen's slope: **+0.00336 per observation**
- Direction: **increasing NBR**

### 6.3 Bhutan nighttime lights

The executor calculates the spatial mean of each dated raster and preserves
chronological ordering before comparison.

- June 2024 mean: **0.322772**
- August 2024 mean: **0.410588**
- Absolute difference: **0.087816**
- Interpretation: August has the higher regional nighttime-light intensity.

### 6.4 Ten additional domain cases

The expanded Studio adds ten cases selected after reviewing all 248 EarthBench
questions, seven experiment arms, and their attempt-level transitions. Selection
favored scientific clarity, diverse Earth-observation applications, real failure-to-
recovery behavior, and CPU reproducibility. Together they cover thermal remote
sensing, apparent thermal inertia, urban nighttime lights, vegetation dynamics,
inland-water quality, precipitation, wildfire indicators, fire radiative power,
and long-term built-volume change. Each case exposes its real input inspection,
derived GeoTIFFs, regional statistics, critic verdict, FAR constraint, and retry.

## 7. Paper evaluation

RSMeM is evaluated on EarthBench across DeepSeek-V3.2, Kimi-K2, Qwen3-32B, and
Qwen3-8B using tool-use correctness, outcome accuracy, and efficiency metrics.
The full framework consistently improves tool-use and end-to-end performance.
On DeepSeek-V3.2, RSMeM reaches **57.89% outcome accuracy**, a **6.07 percentage
point absolute improvement**, while reflection accounts for only **0.35%** of
the total inference-token budget. This supports the central hypothesis that
compact failure-derived constraints can improve reliability without brute-force
context expansion.

## 8. Reproducibility and open-source resources

- **Try the Studio:** https://modelscope.cn/studios/wbx929/RSMeM
- **Source code:** https://github.com/AI9Stars/RSMeM
- **EarthBench:** https://huggingface.co/datasets/Sssunset/Earth-Bench
- **Code license:** MIT
- **Bundled example-data license:** Apache-2.0

No demo video is required to reproduce the workflow: the Studio exposes every
retrieved task card, planned tool, parameter, observation, critic verdict,
reflection, retry, and downloadable GeoTIFF directly in the browser.
