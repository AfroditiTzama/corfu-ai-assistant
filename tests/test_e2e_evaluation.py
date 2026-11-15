from evaluation.run_e2e_evaluation import run_deterministic_cases


def test_deterministic_end_to_end_evaluation_covers_release_behaviors() -> None:
    report = run_deterministic_cases()
    summary = report["summary"]

    assert summary["case_count"] == 20
    assert summary["task_success_rate"] == 1.0
    assert summary["citation_validation_attempts"] == 9
    assert summary["citation_validation_failures"] == 1
    assert "not a production rate" in summary["citation_validity_metric_definition"]
    cases = {case["id"]: case for case in report["cases"]}
    assert cases["family_eligibility"]["citations_valid"] is True
    assert cases["citation_validation_fallback"]["citations_valid"] is False
    assert cases["instruction_attack_grounding"]["task_success"] is True
    assert cases["feasible_nature_half_day"]["task_success"] is True
    assert cases["multi_turn_preference_change"]["task_success"] is True
    assert summary["recommendation_count_adherence"] == 1.0
    assert summary["answer_card_consistency_rate"] == 1.0
    assert summary["citation_validity_rate"] > 0
    assert summary["constraint_violation_count"] == 0
    assert summary["failed_cases"] == []
    assert "provider_error" in summary["fallback_reasons"]
    assert "parsing_error" in summary["fallback_reasons"]
    assert "structured_output_error" in summary["fallback_reasons"]
    assert "evidence_insufficient" in summary["fallback_reasons"]
    assert "citation_validation_error" in summary["fallback_reasons"]
    assert "recommendation_selection_mismatch" in summary["fallback_reasons"]


def test_fallback_answers_do_not_claim_live_price_or_access_facts() -> None:
    report = run_deterministic_cases()
    row = next(case for case in report["cases"] if case["id"] == "unsupported_price_question")

    assert row["task_success"] is True
    assert row["unsupported_claims"] == 0
    assert row["answer_mode"] == "deterministic"
    assert row["fallback_reason"] == "generator_unavailable"
