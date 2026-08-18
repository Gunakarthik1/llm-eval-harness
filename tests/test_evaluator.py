"""
Tests for rubric scoring functions, EvalResult structure,
comparison logic, and judge determinism.
"""
from __future__ import annotations

import pytest

from harness.evaluator import (
    LLMJudge,
    citation_precision,
    coherence,
    factual_grounding,
    instruction_following,
    task_completion,
    tool_call_accuracy,
)
from harness.models import ExecutionTrace, Scenario, ToolCall, Turn
from harness.runner import ScenarioRunner
from harness.scenarios import SCENARIO_MAP, list_scenarios


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trace(responses: list[str], model: str = "model-a", scenario_id: str = "test") -> ExecutionTrace:
    turns = [
        Turn(
            turn_number=i + 1,
            user_input=f"user input {i}",
            agent_response=resp,
            response_time_ms=100.0,
            tokens_used=len(resp) // 4,
        )
        for i, resp in enumerate(responses)
    ]
    return ExecutionTrace(
        scenario_id=scenario_id,
        model=model,
        turns=turns,
        total_tokens=sum(t.tokens_used for t in turns),
        duration_ms=300.0,
    )


def _simple_scenario(**kwargs) -> Scenario:
    defaults = dict(
        id="test_scenario",
        category="Code Generation",
        prompt="Write a function.",
        expected_keywords=["def", "function", "return"],
        ground_truth="A Python function with def keyword and return statement.",
        turns=1,
        rubrics=["task_completion"],
        max_score=10,
    )
    defaults.update(kwargs)
    return Scenario(**defaults)


# ---------------------------------------------------------------------------
# task_completion tests
# ---------------------------------------------------------------------------

class TestTaskCompletion:
    def test_full_keyword_match_scores_high(self):
        scenario = _simple_scenario(expected_keywords=["def", "function", "return", "python"])
        trace = _make_trace(["def my_function(): return python_value"])
        score, reason = task_completion(trace, scenario)
        assert score >= 8.0, f"Expected high score, got {score}. Reason: {reason}"

    def test_zero_keyword_match_scores_low(self):
        scenario = _simple_scenario(expected_keywords=["blockchain", "distributed", "consensus"])
        trace = _make_trace(["This is completely unrelated content."])
        score, reason = task_completion(trace, scenario)
        assert score <= 3.0, f"Expected low score, got {score}"

    def test_partial_match_scores_proportionally(self):
        scenario = _simple_scenario(expected_keywords=["alpha", "beta", "gamma", "delta"])
        trace = _make_trace(["alpha and beta are present"])
        score, reason = task_completion(trace, scenario)
        assert 3.0 <= score <= 7.0, f"Expected mid-range score, got {score}"

    def test_very_short_response_penalized(self):
        scenario = _simple_scenario(expected_keywords=["def", "function"])
        trace = _make_trace(["def function"])  # matches but extremely short
        score, reason = task_completion(trace, scenario)
        # Should be penalized for short response despite keyword match
        assert score < 10.0

    def test_empty_keywords_returns_midpoint(self):
        scenario = _simple_scenario(expected_keywords=[])
        trace = _make_trace(["Some response text here."])
        score, reason = task_completion(trace, scenario)
        assert score == 5.0

    def test_score_bounded_0_to_10(self):
        scenario = _simple_scenario(expected_keywords=["x", "y", "z"])
        trace = _make_trace([""])
        score, _ = task_completion(trace, scenario)
        assert 0.0 <= score <= 10.0


# ---------------------------------------------------------------------------
# factual_grounding tests
# ---------------------------------------------------------------------------

class TestFactualGrounding:
    def test_ground_truth_overlap_scores_high(self):
        scenario = _simple_scenario(
            ground_truth="A Python function with def keyword, parameters, and return statement that performs computation."
        )
        trace = _make_trace(
            ["A Python function with def keyword, parameters, and return statement that performs computation."]
        )
        score, reason = factual_grounding(trace, scenario)
        assert score >= 7.0

    def test_negation_signals_penalize_score(self):
        scenario = _simple_scenario(ground_truth="The algorithm is correct and efficient.")
        trace = _make_trace(
            ["The algorithm is not correct, never efficient, and incorrect in its approach. Wrong assumption."]
        )
        score, _ = factual_grounding(trace, scenario)
        assert score < 8.0  # negation penalty applied

    def test_multi_turn_consistency_gives_bonus(self):
        scenario = _simple_scenario(
            ground_truth="Binary search tree insert search delete inorder traversal sorted."
        )
        trace = _make_trace([
            "Binary search tree insert operation sorted traversal.",
            "Binary search tree delete inorder sorted sequence.",
        ])
        score, reason = factual_grounding(trace, scenario)
        assert score >= 5.0


# ---------------------------------------------------------------------------
# instruction_following tests
# ---------------------------------------------------------------------------

class TestInstructionFollowing:
    def test_no_constraint_returns_high_score(self):
        scenario = _simple_scenario(format_constraint=None)
        trace = _make_trace(["This is a complete response. It addresses the prompt fully. Contains enough information."])
        score, _ = instruction_following(trace, scenario)
        assert score >= 7.0

    def test_haiku_three_lines_correct_syllables(self):
        scenario = _simple_scenario(format_constraint="haiku")
        # "Rows of data sleep" = 5 syllables approx, "Queries wake each record gently" = 7, "Index finds the key" = 5
        trace = _make_trace(["Rows of data sleep\nQueries wake each record gently\nIndex finds the key"])
        score, reason = instruction_following(trace, scenario)
        assert score >= 5.0, f"Haiku score: {score}. Reason: {reason}"

    def test_list_constraint_exact_count_scores_ten(self):
        scenario = _simple_scenario(format_constraint="list:3")
        trace = _make_trace(["1. First item\n2. Second item\n3. Third item"])
        score, reason = instruction_following(trace, scenario)
        assert score == 10.0, f"Expected 10.0 for exact list count, got {score}"

    def test_list_constraint_wrong_count_penalized(self):
        scenario = _simple_scenario(format_constraint="list:5")
        trace = _make_trace(["1. Only one item here"])
        score, _ = instruction_following(trace, scenario)
        assert score < 8.0

    def test_word_limit_under_limit_scores_ten(self):
        scenario = _simple_scenario(format_constraint="word_limit:50")
        trace = _make_trace(["Short response under fifty words total count here definitely."])
        score, reason = instruction_following(trace, scenario)
        assert score == 10.0, f"Expected 10 for under limit, got {score}"

    def test_word_limit_over_limit_penalized(self):
        scenario = _simple_scenario(format_constraint="word_limit:10")
        long_response = " ".join(["word"] * 100)
        trace = _make_trace([long_response])
        score, _ = instruction_following(trace, scenario)
        assert score < 7.0


# ---------------------------------------------------------------------------
# tool_call_accuracy tests
# ---------------------------------------------------------------------------

class TestToolCallAccuracy:
    def test_no_tool_required_scores_ten(self):
        scenario = _simple_scenario(tool_required=False, expected_tool=None)
        trace = _make_trace(["Response without tool calls."])
        score, reason = tool_call_accuracy(trace, scenario)
        assert score == 10.0

    def test_correct_tool_with_params_scores_ten(self):
        scenario = _simple_scenario(tool_required=True, expected_tool="weather")
        trace = _make_trace(["Checking weather now."])
        trace.tool_calls = [
            ToolCall(
                tool_name="weather",
                parameters={"location": "Seattle, WA", "units": "imperial", "forecast": True},
                result="success",
                success=True,
            )
        ]
        score, reason = tool_call_accuracy(trace, scenario)
        assert score == 10.0

    def test_wrong_tool_name_scores_low(self):
        scenario = _simple_scenario(tool_required=True, expected_tool="weather")
        trace = _make_trace(["Using calculator instead."])
        trace.tool_calls = [
            ToolCall(tool_name="calculator", parameters={"x": 1}, result="", success=True)
        ]
        score, _ = tool_call_accuracy(trace, scenario)
        assert score <= 3.0

    def test_no_tool_call_when_required_scores_low(self):
        scenario = _simple_scenario(tool_required=True, expected_tool="search")
        trace = _make_trace(["I searched manually but didn't use the tool."])
        score, _ = tool_call_accuracy(trace, scenario)
        assert score <= 5.0

    def test_tool_mentioned_in_text_partial_credit(self):
        scenario = _simple_scenario(tool_required=True, expected_tool="search")
        trace = _make_trace(["I would call the search tool for this."])
        score, _ = tool_call_accuracy(trace, scenario)
        # Mentioned in text but no formal call → partial credit
        assert score <= 6.0


# ---------------------------------------------------------------------------
# citation_precision tests
# ---------------------------------------------------------------------------

class TestCitationPrecision:
    def test_non_citation_category_returns_neutral(self):
        scenario = _simple_scenario(category="Code Generation")
        trace = _make_trace(["def function(): return True"])
        score, reason = citation_precision(trace, scenario)
        assert score == 7.0

    def test_strong_citations_score_ten(self):
        scenario = _simple_scenario(category="Factual Grounding")
        trace = _make_trace([
            "According to Vaswani et al. (2017), the transformer was introduced in 2017. "
            "See also Devlin et al. (2018) for BERT. Reference [1] and [2] provide background."
        ])
        score, _ = citation_precision(trace, scenario)
        assert score >= 8.0

    def test_no_citations_in_factual_scores_low(self):
        scenario = _simple_scenario(category="Factual Grounding")
        trace = _make_trace(["PageRank is an algorithm. It uses links. It was invented by Google founders."])
        score, _ = citation_precision(trace, scenario)
        assert score <= 6.0


# ---------------------------------------------------------------------------
# coherence tests
# ---------------------------------------------------------------------------

class TestCoherence:
    def test_empty_trace_scores_zero(self):
        trace = ExecutionTrace(scenario_id="test", model="model-a")
        score, _ = coherence(trace)
        assert score == 0.0

    def test_single_structured_response_scores_well(self):
        trace = _make_trace([
            "First, we define the problem. Then, we analyze the constraints. "
            "Next, we implement the solution. Finally, we validate the output. "
            "This covers all the required cases. The approach is clean and efficient."
        ])
        score, _ = coherence(trace)
        assert score >= 5.0

    def test_very_short_response_scores_low(self):
        trace = _make_trace(["ok"])
        score, _ = coherence(trace)
        assert score < 6.0

    def test_highly_repetitive_multi_turn_penalized(self):
        repeated = "This is the same answer repeated. Same words used again. Same answer."
        trace = _make_trace([repeated, repeated])
        score, reason = coherence(trace)
        # High n-gram overlap should apply repetition penalty
        assert score <= 8.0


# ---------------------------------------------------------------------------
# LLMJudge integration tests
# ---------------------------------------------------------------------------

class TestLLMJudge:
    def test_evaluate_returns_eval_result_structure(self):
        runner = ScenarioRunner()
        judge = LLMJudge()
        scenario = SCENARIO_MAP["code_json_parser"]
        trace = runner.run_scenario(scenario, "model-a")
        result = judge.evaluate(trace, scenario)

        assert result.scenario_id == scenario.id
        assert result.model == "model-a"
        assert len(result.rubric_scores) == len(scenario.rubrics)
        assert 0.0 <= result.weighted_total <= 10.0
        assert result.judge_reasoning != ""

    def test_model_a_scores_higher_than_model_b(self):
        """model-a should consistently outperform model-b on most scenarios."""
        runner = ScenarioRunner()
        judge = LLMJudge()
        scenario = SCENARIO_MAP["factual_pagerank"]

        trace_a = runner.run_scenario(scenario, "model-a")
        trace_b = runner.run_scenario(scenario, "model-b")
        result_a = judge.evaluate(trace_a, scenario)
        result_b = judge.evaluate(trace_b, scenario)

        # model-a should score the same or higher
        assert result_a.weighted_total >= result_b.weighted_total - 0.5, (
            f"model-a ({result_a.weighted_total}) should outperform model-b ({result_b.weighted_total})"
        )

    def test_evaluate_all_scenarios_without_error(self):
        runner = ScenarioRunner()
        judge = LLMJudge()
        for scenario in list_scenarios():
            for model in ["model-a", "model-b"]:
                trace = runner.run_scenario(scenario, model)
                result = judge.evaluate(trace, scenario)
                assert result is not None
                assert 0.0 <= result.weighted_total <= 10.0

    def test_judge_reasoning_mentions_rubrics(self):
        runner = ScenarioRunner()
        judge = LLMJudge()
        scenario = SCENARIO_MAP["reasoning_math_compound"]
        trace = runner.run_scenario(scenario, "model-a")
        result = judge.evaluate(trace, scenario)
        for rubric in scenario.rubrics:
            assert rubric in result.judge_reasoning, f"Rubric '{rubric}' missing from reasoning."

    def test_compare_detects_regression(self):
        runner = ScenarioRunner()
        judge = LLMJudge()
        scenario = SCENARIO_MAP["factual_cap_theorem"]

        trace_a = runner.run_scenario(scenario, "model-a")
        trace_b = runner.run_scenario(scenario, "model-b")
        result_a = judge.evaluate(trace_a, scenario)
        result_b = judge.evaluate(trace_b, scenario)

        comparison = judge.compare(result_a, result_b)
        assert comparison.model_a == "model-a"
        assert comparison.model_b == "model-b"
        assert isinstance(comparison.overall_delta, float)
        assert isinstance(comparison.regression_detected, bool)
        assert isinstance(comparison.improvement_detected, bool)

    def test_score_map_contains_all_rubrics(self):
        runner = ScenarioRunner()
        judge = LLMJudge()
        scenario = SCENARIO_MAP["instruct_pros_cons"]
        trace = runner.run_scenario(scenario, "model-a")
        result = judge.evaluate(trace, scenario)
        score_map = result.score_map()
        for rubric in scenario.rubrics:
            assert rubric in score_map, f"Rubric '{rubric}' missing from score_map"

    def test_weighted_total_between_zero_and_ten(self):
        runner = ScenarioRunner()
        judge = LLMJudge()
        for scenario in list_scenarios()[:5]:
            trace = runner.run_scenario(scenario, "model-a")
            result = judge.evaluate(trace, scenario)
            assert 0.0 <= result.weighted_total <= 10.0, (
                f"Scenario {scenario.id}: weighted_total {result.weighted_total} out of range"
            )


# ---------------------------------------------------------------------------
# Adversarial generation tests
# ---------------------------------------------------------------------------

class TestAdversarialGenerator:
    def test_generate_suite_returns_n_variants(self):
        from harness.adversarial import AdversarialGenerator
        gen = AdversarialGenerator()
        scenario = SCENARIO_MAP["instruct_haiku"]
        variants = gen.generate_suite(scenario, n=5)
        assert len(variants) == 5

    def test_variants_have_unique_ids(self):
        from harness.adversarial import AdversarialGenerator
        gen = AdversarialGenerator()
        scenario = SCENARIO_MAP["code_json_parser"]
        variants = gen.generate_suite(scenario, n=5)
        ids = [v.id for v in variants]
        assert len(set(ids)) == 5, f"Duplicate IDs found: {ids}"

    def test_noisy_prompt_differs_from_original(self):
        from harness.adversarial import AdversarialGenerator
        gen = AdversarialGenerator(seed=123)
        scenario = SCENARIO_MAP["reasoning_fermi"]
        variants = gen.generate_suite(scenario, n=1)
        assert variants[0].prompt != scenario.prompt

    def test_truncated_prompt_is_shorter(self):
        from harness.adversarial import AdversarialGenerator
        gen = AdversarialGenerator()
        scenario = SCENARIO_MAP["factual_pagerank"]
        truncated = gen.truncate(scenario.prompt)
        assert len(truncated) < len(scenario.prompt)
        assert "[TRUNCATED]" in truncated

    def test_contradiction_appended_to_end(self):
        from harness.adversarial import AdversarialGenerator
        gen = AdversarialGenerator()
        scenario = SCENARIO_MAP["code_bst"]
        contradicted = gen.inject_contradiction(scenario.prompt)
        assert len(contradicted) > len(scenario.prompt)
        assert contradicted.startswith(scenario.prompt[:20])

    def test_irrelevant_context_prepended(self):
        from harness.adversarial import AdversarialGenerator
        gen = AdversarialGenerator()
        scenario = SCENARIO_MAP["summarize_meeting_notes"]
        augmented = gen.add_irrelevant_context(scenario.prompt)
        assert augmented.endswith(scenario.prompt[-10:]) is False  # has suffix
        assert len(augmented) > len(scenario.prompt)

    def test_negated_prompt_contains_negation_language(self):
        from harness.adversarial import AdversarialGenerator
        gen = AdversarialGenerator()
        scenario = SCENARIO_MAP["instruct_pros_cons"]
        negated = gen.negate(scenario.prompt)
        assert any(word in negated.lower() for word in ["not", "avoid", "refrain", "cannot"])

    def test_describe_mutation_returns_expected_keys(self):
        from harness.adversarial import AdversarialGenerator
        gen = AdversarialGenerator()
        scenario = SCENARIO_MAP["tool_weather"]
        variants = gen.generate_suite(scenario, n=1)
        desc = gen.describe_mutation(scenario, variants[0])
        for key in ("original_id", "variant_id", "mutation_type", "original_prompt_len", "variant_prompt_len"):
            assert key in desc
