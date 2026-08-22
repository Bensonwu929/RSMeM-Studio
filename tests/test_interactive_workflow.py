from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app


class StudioApi:
    def __init__(self, case_verdicts=None, gate_verdict="pass_supported"):
        self.case_verdicts = list(case_verdicts or ["correct"])
        self.gate_verdict = gate_verdict
        self.calls = []

    def __call__(self, api_key, base_url, model, system_prompt, user_prompt, max_tokens=900):
        self.calls.append((system_prompt, user_prompt))
        if "planner inside RSMeM" in system_prompt:
            return json.dumps({"tool_chain": [{
                "tool": "inspect_rasters", "purpose": "inspect the real inputs", "parameters": {}
            }]})
        if "answer synthesizer" in system_prompt:
            return "The supplied raster was inspected successfully; no unsupported numeric claim is made."
        if "reference-aware evaluator" in system_prompt:
            verdict = self.case_verdicts.pop(0) if self.case_verdicts else "correct"
            return json.dumps({
                "verdict": verdict, "confidence": .9,
                "candidate_interpretation": "candidate", "reference_interpretation": "reference",
                "comparison": "comparison", "reason_codes": ["test"], "evidence_step_ids": [1],
            })
        if "conservative failure verifier" in system_prompt:
            retry = self.gate_verdict.startswith("retry_")
            return json.dumps({
                "verdict": self.gate_verdict, "retry": retry, "confidence": .8,
                "task_requirement": "inspect data", "verification": {
                    "trajectory_complete": True, "tool_execution_valid": True,
                    "evidence_sufficient": not retry, "calculation_consistent": True,
                    "answer_supported": not retry},
                "reason_codes": ["test"], "first_decisive_error_step": 1 if retry else None,
                "root_cause": "test" if retry else None,
                "evidence": [{"step": 1, "observation": "inspection"}],
                "correction_guidance": "retry" if retry else None, "missing_information": [],
            })
        if "trajectory state compressor" in system_prompt:
            return json.dumps({
                "task_summary": {"text": "Inspect the data.", "source_step_ids": [1]},
                "attempted_strategy": [{"text": "Inspection was attempted.", "source_step_ids": [1]}],
                "verified_state": [{"text": "A raster was read.", "source_step_ids": [1]}],
                "observed_failures": [], "unfinished_subgoals": [], "budget_signals": [],
            })
        if "repair reflector" in system_prompt:
            label = user_prompt.split("Evidence representation:", 1)[1].splitlines()[0].strip()
            return json.dumps({
                "reflective_text": f"Use the {label} evidence to preserve valid inspection. Correct only the unsupported step, validate its output, and stop when the requested result is supported."
            })
        raise AssertionError(system_prompt)


class InteractiveWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.case_name = next(iter(app.EXAMPLES))
        self.custom_name = next(name for name, case in app.EXAMPLES.items() if case["directory"] is None)
        self.one_tif = next((app.ROOT / "examples").glob("*/*.tif"))
        self.url_patch = patch.object(app, "validate_base_url", lambda value: value)
        self.url_patch.start()
        self.addCleanup(self.url_patch.stop)

    def _start_case(self, api):
        with patch.object(app, "api_completion", api):
            return app.start_interactive_workflow(
                app.EXAMPLES[self.case_name]["query"], self.case_name, [],
                "https://example.com/v1", "test-model", "private-test-key",
            )

    def _start_custom(self, api):
        with patch.object(app, "api_completion", api):
            return app.start_interactive_workflow(
                "Inspect my uploaded raster and report only supported findings.", self.custom_name,
                [str(self.one_tif)], "https://example.com/v1", "test-model", "private-test-key",
            )

    def test_all_case_labels_hide_benchmark_question_numbers_and_are_descriptive(self):
        for name, case in app.EXAMPLES.items():
            if case["directory"] is None:
                continue
            self.assertNotIn("EarthBench", name)
            self.assertNotRegex(name, r"\bQ\d+\b")
            self.assertIn("—", name)
            self.assertGreater(len(name), 55)

    def test_reference_catalog_covers_every_builtin_case(self):
        directories = {case["directory"] for case in app.EXAMPLES.values() if case["directory"]}
        self.assertEqual(directories, set(app.CASE_REFERENCES))

    def test_raster_preview_is_display_ready_and_keeps_source_unchanged(self):
        before = self.one_tif.stat().st_mtime_ns
        preview = app.raster_preview([str(self.one_tif)])
        self.assertEqual(preview.dtype.name, "uint8")
        self.assertEqual(preview.ndim, 3)
        self.assertEqual(preview.shape[2], 3)
        self.assertLessEqual(max(preview.shape[:2]), 640)
        self.assertEqual(before, self.one_tif.stat().st_mtime_ns)

    def test_loading_case_returns_query_and_preview(self):
        query, preview = app.load_example(self.case_name)
        self.assertEqual(query, app.EXAMPLES[self.case_name]["query"])
        self.assertEqual(preview.shape[2], 3)

    def test_streaming_run_exposes_each_major_phase_before_final_result(self):
        api = StudioApi(["correct"])
        with patch.object(app, "api_completion", api):
            updates = list(app.start_interactive_workflow_stream(
                app.EXAMPLES[self.case_name]["query"], self.case_name, [],
                "https://example.com/v1", "test-model", "private-test-key",
            ))
        self.assertEqual(len(updates), 5)
        for update in updates:
            self.assertEqual(len(update), 21)
        progress_html = "\n".join(str(update[2]) for update in updates[:-1])
        for label in ("Plan", "Execute", "Answer", "Evaluate"):
            self.assertIn(label, progress_html)
        self.assertEqual(updates[-1][0]["status"], "completed_case_correct")

    def test_custom_query_requires_uploaded_rasters_when_no_case_is_loaded(self):
        with self.assertRaisesRegex(Exception, "Attach GeoTIFF"):
            app._initialize_interactive_workflow(
                "Inspect my own data.", None, [], "https://example.com/v1",
                "test-model", "private-test-key",
            )

    def test_correct_case_evaluation_is_concise_and_hides_judge_json(self):
        summary = app._evaluation_summary_markdown({
            "verdict": "correct",
            "confidence": 0.95,
            "candidate_interpretation": "internal candidate details",
            "reference_interpretation": "internal reference details",
            "comparison": "A long internal comparison.",
            "reason_codes": ["correct_band_usage"],
        }, "case")
        self.assertIn("Correct", summary)
        self.assertIn("95%", summary)
        self.assertNotIn("internal", summary)
        self.assertNotIn("reason_codes", summary)

    def test_incorrect_case_summary_keeps_only_actionable_reason(self):
        summary = app._evaluation_summary_markdown({
            "verdict": "incorrect", "confidence": 0.8,
            "comparison": "The reported trend direction conflicts with the reference.",
        }, "case")
        self.assertIn("Another attempt is needed", summary)
        self.assertIn("trend direction", summary)

    def test_teacher_gate_summary_remains_advisory(self):
        summary = app._evaluation_summary_markdown({
            "verdict": "retry_unsupported", "retry": True, "confidence": 0.7,
            "root_cause": "The final answer lacks evidence from the raster output.",
        }, "custom")
        self.assertIn("You make the final decision", summary)
        self.assertIn("lacks evidence", summary)

    def test_execution_trace_renders_step_cards_without_evaluation(self):
        state = {
            "attempts": [{
                "attempt": 1,
                "context": {"far_memory": ""},
                "tool_plan": [{"tool": "inspect_rasters", "purpose": "Inspect input"}],
                "trajectory": [{
                    "step": 1, "tool": "inspect_rasters", "status": "success",
                    "observation": "one readable raster & <script>alert(1)</script>",
                }],
                "answer": "The raster is readable.",
                "evaluation": {"verdict": "pass_supported", "confidence": 0.95},
            }],
        }
        rendered = app._attempts_interactive_html(state)
        self.assertIn("Planned tool chain", rendered)
        self.assertIn("Answer synthesized from tool evidence", rendered)
        self.assertIn('class="badge ok"', rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("<script>alert(1)</script>", rendered)
        self.assertNotIn("#### Evaluation", rendered)
        self.assertNotIn("PASS_SUPPORTED", rendered)

    def test_custom_result_reveals_the_whole_decision_panel(self):
        state = {
            "status": "awaiting_user_decision",
            "mode": "custom",
            "max_attempts": 10,
            "model": "test-model",
            "data_source": "visitor-uploaded GeoTIFFs",
            "knowledge_markdown": "",
            "knowledge_context": "",
            "candidates": {},
            "artifacts": [],
            "attempts": [{
                "attempt": 1,
                "context": {"far_memory": ""},
                "tool_plan": [],
                "trajectory": [],
                "answer": "Supported answer.",
                "evaluation": {
                    "verdict": "pass_supported", "retry": False, "confidence": 0.95,
                },
            }],
        }
        rendered = app._render_interactive_state(state)
        self.assertFalse(rendered[3]["visible"])
        self.assertTrue(rendered[7]["visible"])
        self.assertFalse(rendered[8]["visible"])
        self.assertIn("your decision is final", rendered[6])
        self.assertIn("skipped: no relevant match", rendered[1])
        self.assertIn('class="stage current"><span>6</span><b>Human</b>', rendered[1])
        self.assertIn("### Task", rendered[16])
        self.assertIn("Supported answer.", rendered[16])

    def test_reflection_stage_is_current_only_while_reflection_is_pending(self):
        state = {
            "status": "awaiting_reflection",
            "mode": "case",
            "knowledge_context": "relevant guidance",
            "memory": [],
            "attempts": [{
                "attempt": 1,
                "tool_plan": [{"tool": "inspect_rasters"}],
                "evaluation": {"verdict": "incorrect"},
            }],
        }
        rendered = app._interactive_timeline(state)
        self.assertIn('class="stage current"><span>7</span><b>Reflect</b>', rendered)
        self.assertIn('class="stage skip"><span>6</span><b>Human</b>', rendered)

    def test_successful_builtin_case_marks_reflection_as_skipped(self):
        state = {
            "status": "completed_case_correct",
            "mode": "case",
            "knowledge_context": "relevant guidance",
            "memory": [],
            "attempts": [{
                "attempt": 1,
                "tool_plan": [{"tool": "inspect_rasters"}],
                "evaluation": {"verdict": "correct"},
            }],
        }
        rendered = app._interactive_timeline(state)
        self.assertIn('class="stage skip"><span>7</span><b>Reflect</b>', rendered)
        self.assertIn("CORRECT", rendered)

    def test_all_low_score_knowledge_is_removed_from_ui_and_context(self):
        entry = {
            "domain": "Environmental Monitoring",
            "application": "Atmospheric Parameter Estimation",
            "summary": "Low-relevance guidance.",
        }
        with patch.object(app, "retrieve_knowledge", return_value=[(1, entry), (1, entry)]):
            markdown, context, records = app._retrieve_for_session("unrelated task")
        self.assertEqual(markdown, "")
        self.assertEqual(context, "")
        self.assertEqual(records, [])

    def test_relevant_knowledge_is_kept_when_top_score_exceeds_one(self):
        entry = {
            "domain": "Disaster Management",
            "application": "Drought Monitoring",
            "summary": "Use TVDI for drought analysis.",
        }
        with patch.object(app, "retrieve_knowledge", return_value=[(4, entry)]):
            markdown, context, records = app._retrieve_for_session("calculate TVDI")
        self.assertIn("Drought Monitoring", markdown)
        self.assertIn("Use TVDI", context)
        self.assertEqual(records[0]["score"], 4)

    def test_candidate_ui_hides_internal_provenance_names(self):
        state = {"candidates": {
            "FTRO": {"status": "ok", "reflection": "First repair."},
            "Full trajectory": {"status": "ok", "reflection": "Second repair."},
            "LLM summary": {"status": "ok", "reflection": "Third repair."},
        }}
        self.assertEqual(
            app._reflection_candidate_texts(state),
            ("First repair.", "Second repair.", "Third repair."),
        )
        self.assertEqual(
            app.select_reflection_candidate("Candidate 1", state, "", ""),
            "First repair.",
        )

    def test_correct_case_stops_after_one_attempt_without_reflector(self):
        api = StudioApi(["correct"])
        output = self._start_case(api)
        state = output[0]
        self.assertEqual(state["status"], "completed_case_correct")
        self.assertEqual(len(state["attempts"]), 1)
        self.assertFalse(state["candidates"])
        self.assertFalse(any("repair reflector" in system for system, _ in api.calls))
        self.assertNotIn("private-test-key", json.dumps(state))

    def test_incorrect_case_generates_three_candidates_and_waits(self):
        api = StudioApi(["incorrect"])
        state = self._start_case(api)[0]
        self.assertEqual(state["status"], "awaiting_reflection")
        self.assertEqual(state["max_attempts"], 3)
        self.assertEqual(
            {name for name, value in state["candidates"].items() if value["status"] == "ok"},
            {"FTRO", "Full trajectory", "LLM summary"},
        )
        reflector_prompts = [user for system, user in api.calls if "repair reflector" in system]
        self.assertTrue(reflector_prompts)
        self.assertTrue(all("Reference result:" not in prompt for prompt in reflector_prompts))
        self.assertGreaterEqual(sum("answer_synthesizer" in prompt for prompt in reflector_prompts), 2)
        summary_prompts = [user for system, user in api.calls if "trajectory state compressor" in system]
        self.assertEqual(len(summary_prompts), 1)
        self.assertIn("answer_synthesizer", summary_prompts[0])

    def test_only_selected_reflection_enters_next_planner(self):
        api = StudioApi(["incorrect", "correct"])
        state = self._start_case(api)[0]
        chosen = state["candidates"]["FTRO"]["reflection"]
        with patch.object(app, "api_completion", api):
            state2 = app.confirm_reflection_and_retry(
                state, "FTRO", chosen, "", "", "private-test-key"
            )[0]
        self.assertEqual(state2["status"], "completed_case_correct")
        self.assertEqual(state2["memory"][0]["source"], "FTRO")
        planner_prompts = [user for system, user in api.calls if "planner inside RSMeM" in system]
        self.assertIn(chosen, planner_prompts[-1])
        self.assertNotIn(state["candidates"].get("Full trajectory", {}).get("reflection", "__missing__"), planner_prompts[-1])

    def test_case_stops_at_three_incorrect_attempts(self):
        api = StudioApi(["incorrect", "incorrect", "incorrect"])
        state = self._start_case(api)[0]
        for _ in range(2):
            chosen = state["candidates"]["FTRO"]["reflection"]
            with patch.object(app, "api_completion", api):
                state = app.confirm_reflection_and_retry(
                    state, "FTRO", chosen, "", "", "private-test-key"
                )[0]
        self.assertEqual(len(state["attempts"]), 3)
        self.assertEqual(state["status"], "max_attempts_reached")
        self.assertFalse(state["candidates"])

    def test_custom_gate_is_advisory_and_user_can_accept_retry_recommendation(self):
        api = StudioApi(gate_verdict="retry_unsupported")
        state = self._start_custom(api)[0]
        self.assertEqual(state["status"], "awaiting_user_decision")
        self.assertEqual(state["max_attempts"], 10)
        accepted = app.accept_custom_result(state)[0]
        self.assertEqual(accepted["status"], "completed_user_accepted")

    def test_custom_gate_pass_can_be_overridden_to_request_reflection(self):
        api = StudioApi(gate_verdict="pass_supported")
        state = self._start_custom(api)[0]
        with patch.object(app, "api_completion", api):
            reflected = app.request_custom_reflections(state, "private-test-key")[0]
        self.assertEqual(reflected["status"], "awaiting_reflection")
        self.assertEqual(reflected["candidates"]["LLM summary"]["status"], "ok")

    def test_user_authored_option_requires_both_fields(self):
        api = StudioApi(gate_verdict="pass_supported")
        state = self._start_custom(api)[0]
        with patch.object(app, "api_completion", api):
            state = app.request_custom_reflections(state, "private-test-key")[0]
            with self.assertRaises(Exception):
                app.confirm_reflection_and_retry(
                    state, "User-authored", "text", "diagnosis only", "", "private-test-key"
                )

    def test_custom_upload_stops_at_ten_user_confirmed_attempts(self):
        api = StudioApi(gate_verdict="retry_unsupported")
        state = self._start_custom(api)[0]
        for attempt in range(1, 10):
            with patch.object(app, "api_completion", api):
                state = app.request_custom_reflections(state, "private-test-key")[0]
                state = app.confirm_reflection_and_retry(
                    state, "User-authored", "",
                    f"attempt {attempt} diagnosis", f"attempt {attempt} corrective plan",
                    "private-test-key",
                )[0]
        self.assertEqual(len(state["attempts"]), 10)
        self.assertEqual(len(state["memory"]), 9)
        self.assertEqual(state["status"], "max_attempts_reached")

    def test_failed_retry_does_not_mutate_original_memory(self):
        api = StudioApi(["incorrect"])
        state = self._start_case(api)[0]
        chosen = state["candidates"]["FTRO"]["reflection"]
        original = json.dumps(state, sort_keys=True)
        def broken(*args, **kwargs):
            raise RuntimeError("temporary API outage")
        with patch.object(app, "api_completion", broken):
            with self.assertRaises(Exception):
                app.confirm_reflection_and_retry(
                    state, "FTRO", chosen, "", "", "private-test-key"
                )
        self.assertEqual(json.dumps(state, sort_keys=True), original)


def test_run_tab_is_primary_and_overview_content_is_progressively_disclosed():
    config = app.demo.get_config_file()
    tab_labels = [
        component.get("props", {}).get("label")
        for component in config["components"]
        if component.get("type") == "tabitem"
    ]
    assert tab_labels == ["Run RSMeM", "Open-source resources", "Challenge submission"]
    assert "Overview" not in tab_labels

    html_values = [
        str(component.get("props", {}).get("value", ""))
        for component in config["components"]
        if component.get("type") == "html"
    ]
    assert any(
        "Retrieve" in value and "Evaluate" in value and "Reflect" in value
        for value in html_values
    )
    assert any(
        component.get("props", {}).get("label") == "Method overview and paper figures"
        for component in config["components"]
    )


def test_results_flow_is_single_column_with_agent_trace_before_evidence():
    config = app.demo.get_config_file()
    markdown_values = [
        str(component.get("props", {}).get("value", ""))
        for component in config["components"]
        if component.get("type") == "markdown"
    ]
    trace_index = next(
        i for i, value in enumerate(markdown_values) if value.startswith("## Agent trace")
    )
    evidence_index = next(
        i for i, value in enumerate(markdown_values) if value.startswith("## Evidence")
    )
    assert trace_index < evidence_index
    assert not any("Task & decision" in value for value in markdown_values)
    assert not any("Memory-evolution trace" in value for value in markdown_values)


if __name__ == "__main__":
    unittest.main()
