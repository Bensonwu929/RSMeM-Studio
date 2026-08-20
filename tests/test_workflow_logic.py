from __future__ import annotations

import json
import unittest

import workflow_logic as logic


class ScriptedApi:
    def __init__(self):
        self.calls = []

    def __call__(self, api_key, base_url, model, system, user, max_tokens):
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        if "reference-aware evaluator" in system:
            return json.dumps({
                "verdict": "correct", "confidence": 0.98,
                "candidate_interpretation": "12.3 percent",
                "reference_interpretation": "0.123 as a fraction",
                "comparison": "Equivalent after unit conversion.",
                "reason_codes": ["semantic_match"], "evidence_step_ids": [1],
            })
        if "conservative failure verifier" in system:
            return json.dumps({
                "verdict": "uncertain", "retry": False, "confidence": 0.55,
                "task_requirement": "derive a supported result",
                "verification": {"trajectory_complete": True, "tool_execution_valid": True,
                    "evidence_sufficient": False, "calculation_consistent": True,
                    "answer_supported": False},
                "reason_codes": ["insufficient_trace"],
                "first_decisive_error_step": None, "root_cause": None,
                "evidence": [{"step": 1, "observation": "tool succeeded"}],
                "correction_guidance": None, "missing_information": ["independent check"],
            })
        if "trajectory state compressor" in system:
            return json.dumps({
                "task_summary": {"text": "Compute the requested index.", "source_step_ids": [1]},
                "attempted_strategy": [{"text": "Called one tool.", "source_step_ids": [1]}],
                "verified_state": [{"text": "The tool returned output.", "source_step_ids": [1]}],
                "observed_failures": [], "unfinished_subgoals": [], "budget_signals": [],
            })
        if "repair reflector" in system:
            return json.dumps({
                "reflective_text": "Keep the verified input inspection. Correct the earliest unsupported operation, rerun only the affected computation, validate its output, and stop once the requested quantity is supported."
            })
        raise AssertionError(system)


class WorkflowLogicTests(unittest.TestCase):
    def setUp(self):
        self.trajectory = [{
            "step": 1, "tool": "inspect_rasters", "purpose": "inspect",
            "parameters": {"files": ["a.tif"]}, "status": "success",
            "observation": "one valid raster",
        }]

    def test_extracts_complete_object_not_nested_block(self):
        text = 'prefix {"verification":{"answer_supported":true},"verdict":"correct"} suffix'
        self.assertEqual(logic.extract_json_object(text, "verdict")["verdict"], "correct")

    def test_case_judge_handles_semantic_unit_equivalence(self):
        api = ScriptedApi()
        result = logic.judge_builtin_case(
            api_call=api, api_key="secret", base_url="https://example.com/v1", model="m",
            query="report affected fraction", reference={"reference": "0.123", "rubric": "accept 12.3%"},
            trajectory=self.trajectory, final_answer="12.3 percent",
        )
        self.assertEqual(result["verdict"], "correct")
        self.assertIn("Reference result:", api.calls[0]["user"])

    def test_teacher_gate_never_receives_ground_truth(self):
        api = ScriptedApi()
        result = logic.teacher_failure_gate(
            api_call=api, api_key="secret", base_url="https://example.com/v1", model="m",
            query="analyze upload", trajectory=self.trajectory, final_answer="result",
        )
        self.assertEqual(result["verdict"], "uncertain")
        self.assertNotIn("Reference result", api.calls[0]["user"])
        self.assertNotIn("ground_truth", api.calls[0]["user"])

    def test_gate_rejects_inconsistent_pass_retry_pair(self):
        def bad(*args):
            return json.dumps({"verdict": "pass_supported", "retry": True, "confidence": .8})
        with self.assertRaisesRegex(ValueError, "Pass verdict"):
            logic.teacher_failure_gate(
                api_call=bad, api_key="k", base_url="https://e", model="m",
                query="q", trajectory=self.trajectory, final_answer="a",
            )

    def test_full_trajectory_is_uncompressed_and_numbered(self):
        view = logic.full_trajectory_view(self.trajectory)
        self.assertEqual(view["trajectory"][0]["step_id"], 1)
        self.assertEqual(view["trajectory"][0]["observation"], "one valid raster")

    def test_full_trajectory_fails_explicitly_above_limit(self):
        oversized = [dict(self.trajectory[0], observation="x" * 1000)]
        with self.assertRaisesRegex(ValueError, "safety limit"):
            logic.full_trajectory_view(oversized, max_chars=100)

    def test_grounded_summary_requires_valid_step_ids(self):
        def invalid(*args):
            return json.dumps({
                "task_summary": {"text": "summary", "source_step_ids": [99]},
                "attempted_strategy": [], "verified_state": [], "observed_failures": [],
                "unfinished_subgoals": [], "budget_signals": [],
            })
        with self.assertRaisesRegex(ValueError, "valid source step"):
            logic.grounded_summary(
                api_call=invalid, api_key="k", base_url="https://e", model="m", query="q",
                full_view=logic.full_trajectory_view(self.trajectory),
            )

    def test_three_reflectors_share_identical_system_prompt(self):
        api = ScriptedApi()
        candidates = logic.generate_reflection_candidates(
            api_call=api, api_key="secret", base_url="https://example.com/v1", model="m",
            query="q", knowledge_context="hkg", trajectory=self.trajectory,
            ftro_builder=lambda trajectory: {"strategy": "ftro", "trajectory": trajectory},
        )
        self.assertEqual({candidates[name]["status"] for name in ("FTRO", "Full trajectory", "LLM summary")}, {"ok"})
        reflector_systems = [call["system"] for call in api.calls if "repair reflector" in call["system"]]
        self.assertEqual(len(reflector_systems), 3)
        self.assertEqual(len(set(reflector_systems)), 1)
        for call in api.calls:
            if "repair reflector" in call["system"]:
                self.assertNotIn("Reference result:", call["user"])

    def test_format_repair_is_attempted_once(self):
        calls = []
        valid = json.dumps({
            "verdict": "correct", "confidence": .9,
            "candidate_interpretation": "x", "reference_interpretation": "x",
            "comparison": "same", "reason_codes": [], "evidence_step_ids": [1],
        })
        def flaky(*args):
            calls.append(args[4])
            return "not json" if len(calls) == 1 else valid
        result = logic.judge_builtin_case(
            api_call=flaky, api_key="k", base_url="https://e", model="m",
            query="q", reference={"reference": "x", "rubric": "same"},
            trajectory=self.trajectory, final_answer="x",
        )
        self.assertEqual(result["verdict"], "correct")
        self.assertEqual(len(calls), 2)
        self.assertIn("previous response was invalid", calls[1])

    def test_one_reflector_failure_keeps_other_candidates_available(self):
        class Partial(ScriptedApi):
            def __call__(self, api_key, base_url, model, system, user, max_tokens):
                if "repair reflector" in system and "Raw full trajectory" in user:
                    raise RuntimeError("provider rejected long prompt")
                return super().__call__(api_key, base_url, model, system, user, max_tokens)
        candidates = logic.generate_reflection_candidates(
            api_call=Partial(), api_key="k", base_url="https://e", model="m",
            query="q", knowledge_context="h", trajectory=self.trajectory,
            ftro_builder=lambda trajectory: {"trajectory": trajectory},
        )
        self.assertEqual(candidates["Full trajectory"]["status"], "error")
        self.assertEqual(candidates["FTRO"]["status"], "ok")
        self.assertEqual(candidates["LLM summary"]["status"], "ok")
        self.assertEqual(candidates["User-authored"]["status"], "awaiting_user")


if __name__ == "__main__":
    unittest.main()
