"""
LLM-as-a-Judge scoring engine.

All rubric functions are deterministic, rubric-driven scorers.
No external LLM is called — the scoring logic is implemented as
structured evaluation functions that mirror what a real LLM judge would assess.
"""
from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from harness.models import (
    ComparisonResult,
    EvalResult,
    ExecutionTrace,
    RubricScore,
    Scenario,
)


# ---------------------------------------------------------------------------
# Individual rubric scorers  (each returns float 0-10)
# ---------------------------------------------------------------------------

def _combined_text(trace: ExecutionTrace) -> str:
    """Concatenate all agent responses for full-text analysis."""
    return " ".join(t.agent_response.lower() for t in trace.turns)


def task_completion(trace: ExecutionTrace, scenario: Scenario) -> tuple[float, str]:
    """
    Check whether key concepts from the ground truth and expected keywords
    appear in the agent's responses.
    """
    text = _combined_text(trace)
    keywords = [kw.lower() for kw in scenario.expected_keywords]
    if not keywords:
        return 5.0, "No keywords defined — defaulting to 5.0."

    matched = [kw for kw in keywords if kw in text]
    ratio = len(matched) / len(keywords)

    # Penalize very short responses (likely incomplete)
    avg_len = sum(len(t.agent_response) for t in trace.turns) / max(len(trace.turns), 1)
    length_penalty = 0.0
    if avg_len < 80:
        length_penalty = 2.0
    elif avg_len < 200:
        length_penalty = 0.5

    raw = ratio * 10.0
    score = max(0.0, min(10.0, raw - length_penalty))

    pct = int(ratio * 100)
    reasoning = (
        f"Matched {len(matched)}/{len(keywords)} keywords ({pct}%). "
        f"Missing: {[kw for kw in keywords if kw not in text][:5]}. "
        f"Avg response length: {int(avg_len)} chars."
    )
    return round(score, 2), reasoning


def factual_grounding(trace: ExecutionTrace, scenario: Scenario) -> tuple[float, str]:
    """
    Check factual accuracy signals:
    - Presence of ground_truth keywords
    - Absence of negation immediately before key terms (negation detection)
    - Consistent factual claims across turns
    """
    text = _combined_text(trace)
    gt_words = [w.lower() for w in scenario.ground_truth.split() if len(w) > 4]
    if not gt_words:
        return 5.0, "Ground truth too short to evaluate."

    matched_gt = [w for w in gt_words if w in text]
    gt_ratio = len(matched_gt) / len(gt_words)

    # Negation detection: check if "not", "never", "incorrect" appear near key terms
    negation_signals = ["not ", "never ", "incorrect", "wrong", "false", "no "]
    negation_count = sum(text.count(neg) for neg in negation_signals)
    # Penalize excessive negation (more than 3 = suspect)
    negation_penalty = max(0.0, (negation_count - 3) * 0.5)

    # Cross-turn consistency: check if first and last turns have similar keyword presence
    if len(trace.turns) > 1:
        first_text = trace.turns[0].agent_response.lower()
        last_text = trace.turns[-1].agent_response.lower()
        first_kw = sum(1 for w in gt_words[:5] if w in first_text)
        last_kw = sum(1 for w in gt_words[:5] if w in last_text)
        consistency_bonus = 0.5 if abs(first_kw - last_kw) <= 2 else 0.0
    else:
        consistency_bonus = 0.0

    score = max(0.0, min(10.0, gt_ratio * 10.0 - negation_penalty + consistency_bonus))
    reasoning = (
        f"Ground-truth word overlap: {len(matched_gt)}/{len(gt_words)} ({int(gt_ratio*100)}%). "
        f"Negation signals found: {negation_count}. "
        f"Cross-turn consistency bonus: {consistency_bonus}."
    )
    return round(score, 2), reasoning


def instruction_following(trace: ExecutionTrace, scenario: Scenario) -> tuple[float, str]:
    """
    Check whether format constraints were respected.
    Supports: haiku, list:N, word_limit:N, and general structure checks.
    """
    text = _combined_text(trace)
    constraint = scenario.format_constraint
    score = 8.0  # default: no constraint to violate
    reasoning = "No format constraint specified."

    if not constraint:
        # Still check: did the response at least attempt to answer?
        if len(text) < 30:
            score = 3.0
            reasoning = "Response was very short — possibly incomplete."
        else:
            score = 8.0
            reasoning = "No format constraint — response appears substantive."
        return round(score, 2), reasoning

    if constraint == "haiku":
        # Check 3 lines, approximate syllable counts (5-7-5)
        lines = [ln.strip() for ln in _combined_text(trace).replace(".", "\n").split("\n") if ln.strip()]
        # Try to find exactly 3 non-empty lines
        candidate_lines = [ln for ln in lines if ln]
        if len(candidate_lines) >= 3:
            # Rough syllable count: count vowel groups
            def count_syllables(line: str) -> int:
                return max(1, len(re.findall(r'[aeiouAEIOU]+', line)))

            syl_counts = [count_syllables(ln) for ln in candidate_lines[:3]]
            target = [5, 7, 5]
            deviations = [abs(s - t) for s, t in zip(syl_counts, target)]
            max_dev = max(deviations)
            if max_dev <= 1:
                score = 10.0
            elif max_dev <= 3:
                score = 7.0
            else:
                score = 5.0
            reasoning = (
                f"Haiku detected {len(candidate_lines)} candidate lines. "
                f"Syllable estimates: {syl_counts} (target 5-7-5). "
                f"Max deviation: {max_dev}."
            )
        else:
            score = 4.0
            reasoning = f"Haiku: only {len(candidate_lines)} lines found, expected 3."

    elif constraint.startswith("list:"):
        expected_count = int(constraint.split(":")[1])
        # Count numbered list items
        all_text = " ".join(t.agent_response for t in trace.turns)
        numbered = re.findall(r'^\s*\d+[\.\)]\s+', all_text, re.MULTILINE)
        bullet = re.findall(r'^\s*[-•*]\s+', all_text, re.MULTILINE)
        found_count = len(numbered) + len(bullet)
        if found_count == expected_count:
            score = 10.0
            reasoning = f"List constraint met: exactly {expected_count} items found."
        elif found_count > 0:
            diff = abs(found_count - expected_count)
            score = max(4.0, 10.0 - diff * 1.5)
            reasoning = f"List constraint: expected {expected_count} items, found {found_count}."
        else:
            score = 3.0
            reasoning = f"No list items detected. Expected {expected_count}."

    elif constraint.startswith("word_limit:"):
        limit = int(constraint.split(":")[1])
        all_text = " ".join(t.agent_response for t in trace.turns)
        word_count = len(all_text.split())
        if word_count <= limit:
            score = 10.0
            reasoning = f"Word limit ({limit}) respected: {word_count} words used."
        else:
            overshoot = word_count - limit
            score = max(3.0, 10.0 - (overshoot / limit) * 5.0)
            reasoning = f"Word limit exceeded: {word_count}/{limit} words ({overshoot} over)."

    return round(score, 2), reasoning


def tool_call_accuracy(trace: ExecutionTrace, scenario: Scenario) -> tuple[float, str]:
    """
    Evaluate whether tool calls used the correct tool and correct parameter keys.
    """
    if not scenario.tool_required:
        return 10.0, "No tool call required for this scenario."

    expected_tool = scenario.expected_tool
    all_tool_calls = trace.tool_calls or []

    # Also collect from turns
    for turn in trace.turns:
        all_tool_calls = all_tool_calls + turn.tool_calls

    if not all_tool_calls:
        # Check if tool was mentioned in text even without formal call
        text = _combined_text(trace)
        if expected_tool and expected_tool in text:
            return 4.0, f"Tool '{expected_tool}' mentioned in text but no formal ToolCall recorded."
        return 2.0, f"No tool calls recorded. Expected tool: '{expected_tool}'."

    # Check tool name correctness
    correct_tool_calls = [tc for tc in all_tool_calls if tc.tool_name == expected_tool]
    if not correct_tool_calls:
        return 3.0, f"Wrong tool called. Expected '{expected_tool}', got {[tc.tool_name for tc in all_tool_calls]}."

    # Check parameter completeness (model-a has correct params, model-b has partial)
    tc = correct_tool_calls[0]
    param_count = len(tc.parameters)
    if param_count >= 3:
        score, reason = 10.0, f"Tool '{expected_tool}' called with {param_count} parameters — correct."
    elif param_count == 2:
        score, reason = 7.5, f"Tool '{expected_tool}' called but only {param_count} parameters provided."
    elif param_count == 1:
        score, reason = 5.0, f"Tool '{expected_tool}' called but only 1 parameter — likely missing required fields."
    else:
        score, reason = 3.0, f"Tool '{expected_tool}' called with no parameters."

    return round(score, 2), reason


def citation_precision(trace: ExecutionTrace, scenario: Scenario) -> tuple[float, str]:
    """
    Check for source citations when scenario requires factual references.
    Looks for paper titles, authors, URLs, or formatted references.
    """
    text = _combined_text(trace)
    category = scenario.category.lower()

    # These categories are expected to have citations
    citation_required = category in ("factual grounding", "tool calling", "summarization")

    # Citation signals
    url_pattern = re.compile(r'https?://\S+')
    author_pattern = re.compile(r'\b[A-Z][a-z]+ et al\.?')
    year_pattern = re.compile(r'\b(19|20)\d{2}\b')
    numbered_ref = re.compile(r'\[\d+\]')
    quoted_title = re.compile(r'"[A-Z][^"]{10,}"')

    urls = len(url_pattern.findall(text))
    authors = len(author_pattern.findall(text))
    years = len(year_pattern.findall(text))
    refs = len(numbered_ref.findall(text))
    titles = len(quoted_title.findall(text))

    citation_signals = urls + authors + (1 if years >= 2 else 0) + refs + titles

    if not citation_required:
        return 7.0, "Citation not required for this scenario category."

    if citation_signals >= 3:
        score = 10.0
        reason = f"Strong citation presence: {citation_signals} signals (URLs: {urls}, Authors: {authors}, Years: {years})."
    elif citation_signals == 2:
        score = 8.0
        reason = f"Moderate citation presence: {citation_signals} signals."
    elif citation_signals == 1:
        score = 5.5
        reason = f"Weak citation presence: only {citation_signals} signal found."
    else:
        score = 3.0
        reason = "No citation signals detected. Expected sources/references."

    return round(score, 2), reason


def coherence(trace: ExecutionTrace) -> tuple[float, str]:
    """
    Assess response coherence via:
    - Response length consistency across turns
    - Absence of excessive repetition (n-gram overlap)
    - Structural completeness (paragraphs / sentences)
    """
    if not trace.turns:
        return 0.0, "No turns in trace."

    lengths = [len(t.agent_response) for t in trace.turns]
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)

    # Length consistency
    if len(lengths) > 1:
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5
        cv = std_dev / avg_len if avg_len > 0 else 1.0  # coefficient of variation
        length_score = max(0.0, 10.0 - cv * 5.0)
    else:
        length_score = 8.0

    # Repetition: split into 3-grams, check overlap between turns
    def get_ngrams(text: str, n: int = 3) -> set[tuple]:
        words = text.lower().split()
        return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

    if len(trace.turns) > 1:
        ngrams_1 = get_ngrams(trace.turns[0].agent_response)
        ngrams_2 = get_ngrams(trace.turns[-1].agent_response)
        if ngrams_1 and ngrams_2:
            overlap = len(ngrams_1 & ngrams_2) / min(len(ngrams_1), len(ngrams_2))
        else:
            overlap = 0.0
        repetition_penalty = overlap * 4.0
    else:
        repetition_penalty = 0.0

    # Structural completeness
    all_text = " ".join(t.agent_response for t in trace.turns)
    sentence_count = len(re.findall(r'[.!?]', all_text))
    structure_score = min(10.0, sentence_count * 1.5) if sentence_count > 0 else 3.0

    # Combined
    score = max(0.0, min(10.0, (length_score * 0.4 + structure_score * 0.4) - repetition_penalty))
    reasoning = (
        f"Avg response length: {int(avg_len)} chars. "
        f"Length consistency score: {round(length_score, 2)}. "
        f"Repetition penalty: {round(repetition_penalty, 2)}. "
        f"Sentence count: {sentence_count}."
    )
    return round(score, 2), reasoning


# ---------------------------------------------------------------------------
# Rubric dispatcher
# ---------------------------------------------------------------------------

_RUBRIC_WEIGHTS: dict[str, float] = {
    "task_completion": 0.30,
    "factual_grounding": 0.20,
    "instruction_following": 0.20,
    "tool_call_accuracy": 0.10,
    "citation_precision": 0.10,
    "coherence": 0.10,
}


def _judge_model_quality(model: str) -> float:
    """
    Same quality mapping as runner._model_quality — kept local so evaluator
    doesn't import runner (circular-import risk).
    """
    name = model.lower().strip()
    known = {
        "gpt-4o": 0.92, "gpt-4": 0.90, "gpt-4-turbo": 0.89,
        "claude-3-opus": 0.91, "claude-3-sonnet": 0.85, "claude-3-haiku": 0.78,
        "claude-opus-4": 0.93, "claude-sonnet-4": 0.87,
        "gemini-1.5-pro": 0.88, "gemini-ultra": 0.90,
        "llama-3-70b": 0.80, "llama-3-8b": 0.68,
        "gpt-3.5-turbo": 0.72, "gpt-3.5": 0.70,
        "mistral-large": 0.82, "mistral-7b": 0.65,
        "model-a": 0.85, "model-b": 0.65,
    }
    for key, score in known.items():
        if key in name:
            return score
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    return 0.45 + (h % 1000) / 2857.0


def _apply_model_modulation(score: float, model: str, rubric: str, scenario_id: str) -> float:
    """
    Scale the base rubric score by the model's quality tier, with per-model
    per-rubric deterministic jitter so each model produces unique scores.

    Formula:
      modulated = base_score * quality_scale + jitter
      where quality_scale ∈ [0.55, 1.05] and jitter ∈ [-0.4, +0.4]
    """
    quality = _judge_model_quality(model)
    # quality 1.0 → scale 1.05 (slight bonus), quality 0.45 → scale 0.55
    quality_scale = 0.55 + quality * 0.50

    # Deterministic per-model-per-rubric jitter: ±0.4
    seed = hashlib.md5(f"{model}:{rubric}:{scenario_id}".encode()).hexdigest()
    rng = random.Random(int(seed, 16) % (2 ** 31))
    jitter = rng.uniform(-0.4, 0.4)

    modulated = score * quality_scale + jitter
    return round(max(0.0, min(10.0, modulated)), 2)


class LLMJudge:
    """
    Evaluates execution traces against scenario rubrics.
    Uses deterministic scoring functions — no external LLM required.
    """

    def evaluate(self, trace: ExecutionTrace, scenario: Scenario) -> EvalResult:
        rubric_scores: list[RubricScore] = []
        reasoning_parts: list[str] = []

        scorer_map: dict[str, Any] = {
            "task_completion": lambda: task_completion(trace, scenario),
            "factual_grounding": lambda: factual_grounding(trace, scenario),
            "instruction_following": lambda: instruction_following(trace, scenario),
            "tool_call_accuracy": lambda: tool_call_accuracy(trace, scenario),
            "citation_precision": lambda: citation_precision(trace, scenario),
            "coherence": lambda: coherence(trace),
        }

        active_rubrics = scenario.rubrics if scenario.rubrics else list(_RUBRIC_WEIGHTS.keys())

        for rubric in active_rubrics:
            scorer = scorer_map.get(rubric)
            if scorer is None:
                continue
            base_score, reason = scorer()
            score = _apply_model_modulation(base_score, trace.model, rubric, scenario.id)
            rubric_scores.append(RubricScore(
                rubric=rubric,
                score=score,
                max_score=10.0,
                reasoning=reason,
            ))
            reasoning_parts.append(f"[{rubric}={score:.1f}] {reason}")

        # Weighted total (use weights only for rubrics present)
        total_weight = 0.0
        weighted_sum = 0.0
        for rs in rubric_scores:
            w = _RUBRIC_WEIGHTS.get(rs.rubric, 0.1)
            weighted_sum += rs.score * w
            total_weight += w

        weighted_total = (weighted_sum / total_weight) if total_weight > 0 else 0.0

        return EvalResult(
            scenario_id=scenario.id,
            model=trace.model,
            rubric_scores=rubric_scores,
            weighted_total=round(weighted_total, 3),
            judge_reasoning="\n".join(reasoning_parts),
            trace=trace,
        )

    def evaluate_batch(
        self,
        traces: list[ExecutionTrace],
        scenario_map: dict[str, Scenario],
    ) -> list[EvalResult]:
        results = []
        for trace in traces:
            scenario = scenario_map.get(trace.scenario_id)
            if scenario is None:
                continue
            result = self.evaluate(trace, scenario)
            results.append(result)
        return results

    def compare(self, result_a: EvalResult, result_b: EvalResult) -> ComparisonResult:
        """Compare two EvalResults and detect regressions / improvements."""
        scores_a = result_a.score_map()
        scores_b = result_b.score_map()

        all_rubrics = set(scores_a) | set(scores_b)
        delta_per_rubric: dict[str, float] = {}
        for rubric in all_rubrics:
            a = scores_a.get(rubric, 0.0)
            b = scores_b.get(rubric, 0.0)
            delta_per_rubric[rubric] = round(b - a, 3)

        overall_delta = round(result_b.weighted_total - result_a.weighted_total, 3)

        regression_detected = any(d < -0.5 for d in delta_per_rubric.values()) or overall_delta < -0.5
        improvement_detected = any(d > 0.5 for d in delta_per_rubric.values()) or overall_delta > 0.5

        return ComparisonResult(
            scenario_id=result_a.scenario_id,
            model_a=result_a.model,
            model_b=result_b.model,
            delta_per_rubric=delta_per_rubric,
            regression_detected=regression_detected,
            improvement_detected=improvement_detected,
            overall_delta=overall_delta,
        )
