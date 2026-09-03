from deterministic_evaluator import evaluate_case_deterministically


def test_deterministic_case_passes_without_llm_when_contract_is_complete():
    trajectory = [
        {"step": 1, "tool": "calc_batch_image_mean", "status": "success", "observation": "[0.32277, 0.41059]"},
        {"step": 2, "tool": "difference", "status": "success", "observation": "0.08782"},
    ]
    result = evaluate_case_deterministically(
        "nightlights_bhutan", trajectory,
        "June is lower at 0.32277; August is 0.41059; the difference is 0.08782.",
    )
    assert result["verdict"] == "correct"
    assert result["source"] == "deterministic"


def test_ambiguous_case_falls_back_to_llm_judge():
    trajectory = [
        {"step": 1, "tool": "calc_batch_image_mean", "status": "success", "observation": "means computed"},
        {"step": 2, "tool": "difference", "status": "success", "observation": "difference computed"},
    ]
    assert evaluate_case_deterministically("nightlights_bhutan", trajectory, "Done.") is None


def test_tool_error_is_deterministically_incorrect():
    result = evaluate_case_deterministically(
        "nightlights_bhutan",
        [{"step": 1, "tool": "calc_batch_image_mean", "status": "error", "observation": "bad input"}],
        "Unable to finish.",
    )
    assert result["verdict"] == "incorrect"
    assert result["confidence"] == 1.0


def test_answer_without_any_tool_evidence_cannot_pass_by_matching_reference_text():
    result = evaluate_case_deterministically(
        "nightlights_bhutan", [],
        "June is 0.32277, August is 0.41059, and the difference is 0.08782.",
    )
    assert result["verdict"] == "incorrect"
    assert result["reason_codes"] == ["deterministic_no_tool_evidence"]
