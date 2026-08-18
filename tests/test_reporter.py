"""
Tests for EvalReporter: aggregation, confidence intervals,
regression detection, and HTML report generation.
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from harness.evaluator import LLMJudge
from harness.models import EvalResult, RubricScore
from harness.reporter import EvalReporter, _t_critical
from harness.runner import ScenarioRunner
from harness.scenarios import SCENARIO_MAP, list_scenarios


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eval_result(
    scenario_id: str = "test",
    model: str = "model-a",
    scores: dict[str, float] | None = None,
    weighted_total: float = 7.5,
) -> EvalResult:
    if scores is None:
        scores = {"task_completion": 8.0, "coherence": 7.0}
    rubric_scores = [RubricScore(rubric=r, score=s) for r, s in scores.items()]
    return EvalResult(
        run_id="test-run",
        scenario_id=scenario_id,
        model=model,
        rubric_scores=rubric_scores,
        weighted_total=weighted_total,
    )


def _build_results(model: str, n: int = 5, weighted_totals: list[float] | None = None) -> list[EvalResult]:
    wts = weighted_totals or [7.0 + i * 0.2 for i in range(n)]
    return [
        _make_eval_result(
            scenario_id=f"sc_{i}",
            model=model,
            scores={"task_completion": wt, "coherence": max(0, wt - 0.5)},
            weighted_total=wt,
        )
        for i, wt in enumerate(wts)
    ]


# ---------------------------------------------------------------------------
# t-critical lookup tests
# ---------------------------------------------------------------------------

class TestTCritical:
    def test_df1_returns_high_value(self):
        # With df=1 and alpha/2=0.025, t ≈ 12.706
        val = _t_critical(1, 0.025)
        assert abs(val - 12.706) < 0.01

    def test_df30_is_near_2_042(self):
        val = _t_critical(30, 0.025)
        assert abs(val - 2.042) < 0.01

    def test_large_df_approaches_1_96(self):
        val = _t_critical(500, 0.025)
        assert abs(val - 1.96) < 0.05

    def test_df0_returns_worst_case(self):
        val = _t_critical(0, 0.025)
        assert val == 12.706

    def test_interpolation_between_table_keys(self):
        # df=7 is exactly in table at 2.365; df=8 at 2.306
        # df=7 should match exactly
        v7 = _t_critical(7, 0.025)
        assert abs(v7 - 2.365) < 0.01
        # df=8 should match exactly
        v8 = _t_critical(8, 0.025)
        assert abs(v8 - 2.306) < 0.01


# ---------------------------------------------------------------------------
# aggregate_batch tests
# ---------------------------------------------------------------------------

class TestAggregateBatch:
    def test_returns_dataframe(self):
        reporter = EvalReporter()
        results = _build_results("model-a", n=3)
        df = reporter.aggregate_batch(results)
        assert isinstance(df, pd.DataFrame)

    def test_row_count_equals_result_count(self):
        reporter = EvalReporter()
        results = _build_results("model-a", n=7)
        df = reporter.aggregate_batch(results)
        assert len(df) == 7

    def test_contains_score_columns(self):
        reporter = EvalReporter()
        results = _build_results("model-a", n=2)
        df = reporter.aggregate_batch(results)
        assert "score_task_completion" in df.columns
        assert "score_coherence" in df.columns
        assert "weighted_total" in df.columns

    def test_empty_results_returns_empty_df(self):
        reporter = EvalReporter()
        df = reporter.aggregate_batch([])
        assert df.empty

    def test_model_column_populated(self):
        reporter = EvalReporter()
        results = _build_results("model-x", n=3)
        df = reporter.aggregate_batch(results)
        assert all(df["model"] == "model-x")


# ---------------------------------------------------------------------------
# compute_confidence_intervals tests
# ---------------------------------------------------------------------------

class TestComputeConfidenceIntervals:
    def test_single_value_returns_same_for_all(self):
        reporter = EvalReporter()
        results = [_make_eval_result(scores={"task_completion": 8.0}, weighted_total=8.0)]
        df = reporter.aggregate_batch(results)
        mean, lower, upper = reporter.compute_confidence_intervals(df, "task_completion")
        assert mean == lower == upper == 8.0

    def test_mean_is_arithmetic_mean(self):
        reporter = EvalReporter()
        scores = [6.0, 7.0, 8.0, 9.0, 10.0]
        results = [
            _make_eval_result(scenario_id=f"s{i}", scores={"task_completion": s}, weighted_total=s)
            for i, s in enumerate(scores)
        ]
        df = reporter.aggregate_batch(results)
        mean, _, _ = reporter.compute_confidence_intervals(df, "task_completion")
        expected_mean = sum(scores) / len(scores)
        assert abs(mean - expected_mean) < 0.001

    def test_ci_lower_lt_mean_lt_upper(self):
        reporter = EvalReporter()
        scores = [5.0, 6.0, 7.0, 8.0, 9.0]
        results = [
            _make_eval_result(scenario_id=f"s{i}", scores={"task_completion": s}, weighted_total=s)
            for i, s in enumerate(scores)
        ]
        df = reporter.aggregate_batch(results)
        mean, lower, upper = reporter.compute_confidence_intervals(df, "task_completion", confidence=0.95)
        assert lower < mean < upper

    def test_wider_ci_for_90pct_vs_95pct(self):
        reporter = EvalReporter()
        scores = [4.0, 6.0, 5.0, 7.0, 8.0, 3.0]
        results = [
            _make_eval_result(scenario_id=f"s{i}", scores={"task_completion": s}, weighted_total=s)
            for i, s in enumerate(scores)
        ]
        df = reporter.aggregate_batch(results)
        _, l95, u95 = reporter.compute_confidence_intervals(df, "task_completion", confidence=0.95)
        _, l90, u90 = reporter.compute_confidence_intervals(df, "task_completion", confidence=0.90)
        # 95% CI should be wider than 90%
        assert (u95 - l95) >= (u90 - l90)

    def test_nonexistent_rubric_returns_zeros(self):
        reporter = EvalReporter()
        results = _build_results("model-a", n=3)
        df = reporter.aggregate_batch(results)
        mean, lower, upper = reporter.compute_confidence_intervals(df, "nonexistent_rubric")
        assert mean == lower == upper == 0.0

    def test_ci_symmetry_around_mean(self):
        """For symmetric data the CI should be symmetric around the mean."""
        reporter = EvalReporter()
        # Symmetric: [5, 7] → mean=6, symmetric CI
        scores = [5.0, 7.0, 5.0, 7.0, 5.0, 7.0]
        results = [
            _make_eval_result(scenario_id=f"s{i}", scores={"coherence": s}, weighted_total=s)
            for i, s in enumerate(scores)
        ]
        df = reporter.aggregate_batch(results)
        mean, lower, upper = reporter.compute_confidence_intervals(df, "coherence")
        assert abs((mean - lower) - (upper - mean)) < 0.001


# ---------------------------------------------------------------------------
# detect_regressions tests
# ---------------------------------------------------------------------------

class TestDetectRegressions:
    def _make_df(self, model: str, tc_scores: list[float], coh_scores: list[float]) -> pd.DataFrame:
        results = [
            _make_eval_result(
                scenario_id=f"s{i}",
                model=model,
                scores={"task_completion": tc, "coherence": coh},
                weighted_total=(tc + coh) / 2,
            )
            for i, (tc, coh) in enumerate(zip(tc_scores, coh_scores))
        ]
        return EvalReporter().aggregate_batch(results)

    def test_no_regression_returns_empty_list(self):
        reporter = EvalReporter()
        baseline_df = self._make_df("model-a", [8.0, 8.0, 8.0], [7.0, 7.0, 7.0])
        current_df  = self._make_df("model-b", [9.0, 9.0, 9.0], [8.0, 8.0, 8.0])
        flags = reporter.detect_regressions(baseline_df, current_df)
        assert flags == []

    def test_regression_below_threshold_detected(self):
        reporter = EvalReporter()
        # Drop of 2.0 > threshold 0.5
        baseline_df = self._make_df("model-a", [8.0, 8.0, 8.0], [8.0, 8.0, 8.0])
        current_df  = self._make_df("model-b", [6.0, 6.0, 6.0], [6.0, 6.0, 6.0])
        flags = reporter.detect_regressions(baseline_df, current_df, threshold=0.5)
        assert len(flags) >= 1

    def test_regression_severity_minor_major_critical(self):
        reporter = EvalReporter()
        # minor: delta ~= -0.7, major: ~-2.0, critical: ~-4.0
        baseline_df = self._make_df("a", [8.0, 8.0, 8.0], [8.0, 8.0, 8.0])

        # critical: current drops to 4.0 → delta = -4.0
        current_critical = self._make_df("b", [4.0, 4.0, 4.0], [4.0, 4.0, 4.0])
        flags = reporter.detect_regressions(baseline_df, current_critical)
        severities = {f.rubric: f.severity for f in flags}
        assert any(s == "critical" for s in severities.values()), f"Expected critical in {severities}"

        # major: drops to 6.0 → delta = -2.0
        current_major = self._make_df("b", [6.0, 6.0, 6.0], [6.0, 6.0, 6.0])
        flags = reporter.detect_regressions(baseline_df, current_major)
        severities = {f.rubric: f.severity for f in flags}
        assert any(s == "major" for s in severities.values())

        # minor: drops to 7.3 → delta = -0.7
        current_minor = self._make_df("b", [7.3, 7.3, 7.3], [7.3, 7.3, 7.3])
        flags = reporter.detect_regressions(baseline_df, current_minor)
        severities = {f.rubric: f.severity for f in flags}
        assert any(s == "minor" for s in severities.values())

    def test_flags_sorted_worst_first(self):
        reporter = EvalReporter()
        # task_completion drops by 4, coherence drops by 1
        baseline_df = self._make_df("a", [9.0, 9.0, 9.0], [8.0, 8.0, 8.0])
        current_df  = self._make_df("b", [5.0, 5.0, 5.0], [7.0, 7.0, 7.0])
        flags = reporter.detect_regressions(baseline_df, current_df, threshold=0.5)
        assert len(flags) >= 2
        assert flags[0].delta <= flags[1].delta  # worst first

    def test_custom_threshold_respected(self):
        reporter = EvalReporter()
        # Drop of 0.3 — below default 0.5 but above 0.2
        baseline_df = self._make_df("a", [8.0, 8.0], [8.0, 8.0])
        current_df  = self._make_df("b", [7.7, 7.7], [7.7, 7.7])

        flags_strict = reporter.detect_regressions(baseline_df, current_df, threshold=0.2)
        flags_loose  = reporter.detect_regressions(baseline_df, current_df, threshold=0.5)
        assert len(flags_strict) >= len(flags_loose)


# ---------------------------------------------------------------------------
# generate_summary_json tests
# ---------------------------------------------------------------------------

class TestGenerateSummaryJson:
    def test_returns_dict_with_models_key(self):
        reporter = EvalReporter()
        results = _build_results("model-a", n=4)
        summary = reporter.generate_summary_json(results)
        assert "models" in summary
        assert "model-a" in summary["models"]

    def test_avg_weighted_total_matches_manual_calc(self):
        reporter = EvalReporter()
        wts = [6.0, 7.0, 8.0, 9.0]
        results = _build_results("model-x", n=4, weighted_totals=wts)
        summary = reporter.generate_summary_json(results)
        expected = sum(wts) / len(wts)
        actual = summary["models"]["model-x"]["avg_weighted_total"]
        assert abs(actual - expected) < 0.01

    def test_empty_results_returns_error_key(self):
        reporter = EvalReporter()
        summary = reporter.generate_summary_json([])
        assert "error" in summary

    def test_rubric_ci_present_for_each_rubric(self):
        reporter = EvalReporter()
        results = _build_results("model-a", n=5)
        summary = reporter.generate_summary_json(results)
        rubrics = summary["models"]["model-a"]["rubrics"]
        for rubric_data in rubrics.values():
            assert "mean" in rubric_data
            assert "ci_lower" in rubric_data
            assert "ci_upper" in rubric_data


# ---------------------------------------------------------------------------
# generate_html_report tests
# ---------------------------------------------------------------------------

class TestGenerateHtmlReport:
    def _run_and_get_results(self) -> tuple[list[EvalResult], list[EvalResult]]:
        runner = ScenarioRunner()
        judge = LLMJudge()
        scenarios = list_scenarios()[:4]  # use 4 scenarios for speed

        results_a, results_b = [], []
        for scenario in scenarios:
            trace_a = runner.run_scenario(scenario, "model-a")
            results_a.append(judge.evaluate(trace_a, scenario))
            trace_b = runner.run_scenario(scenario, "model-b")
            results_b.append(judge.evaluate(trace_b, scenario))
        return results_a, results_b

    def test_generates_html_file(self, tmp_path):
        reporter = EvalReporter()
        results_a, results_b = self._run_and_get_results()
        out = str(tmp_path / "test_report.html")
        reporter.generate_html_report(results_a, results_b, out)
        assert Path(out).exists()

    def test_html_file_not_empty(self, tmp_path):
        reporter = EvalReporter()
        results_a, results_b = self._run_and_get_results()
        out = str(tmp_path / "report.html")
        reporter.generate_html_report(results_a, results_b, out)
        content = Path(out).read_text(encoding="utf-8")
        assert len(content) > 1000  # must be substantial

    def test_html_contains_model_names(self, tmp_path):
        reporter = EvalReporter()
        results_a, results_b = self._run_and_get_results()
        out = str(tmp_path / "report.html")
        reporter.generate_html_report(results_a, results_b, out)
        content = Path(out).read_text(encoding="utf-8")
        assert "model-a" in content
        assert "model-b" in content

    def test_html_contains_rubric_table(self, tmp_path):
        reporter = EvalReporter()
        results_a, results_b = self._run_and_get_results()
        out = str(tmp_path / "report.html")
        reporter.generate_html_report(results_a, results_b, out)
        content = Path(out).read_text(encoding="utf-8")
        assert "<table" in content
        assert "Rubric" in content or "rubric" in content

    def test_html_contains_regression_section(self, tmp_path):
        reporter = EvalReporter()
        results_a, results_b = self._run_and_get_results()
        out = str(tmp_path / "report.html")
        reporter.generate_html_report(results_a, results_b, out)
        content = Path(out).read_text(encoding="utf-8")
        assert "Regression" in content or "regression" in content

    def test_html_contains_ci_chart_element(self, tmp_path):
        reporter = EvalReporter()
        results_a, results_b = self._run_and_get_results()
        out = str(tmp_path / "report.html")
        reporter.generate_html_report(results_a, results_b, out)
        content = Path(out).read_text(encoding="utf-8")
        assert "ci-chart" in content

    def test_html_is_valid_document(self, tmp_path):
        reporter = EvalReporter()
        results_a, results_b = self._run_and_get_results()
        out = str(tmp_path / "report.html")
        reporter.generate_html_report(results_a, results_b, out)
        content = Path(out).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "</html>" in content

    def test_returns_output_path(self, tmp_path):
        reporter = EvalReporter()
        results_a, results_b = self._run_and_get_results()
        out = str(tmp_path / "path_check.html")
        returned = reporter.generate_html_report(results_a, results_b, out)
        assert returned == out

    def test_creates_parent_dirs_if_missing(self, tmp_path):
        reporter = EvalReporter()
        results_a, results_b = self._run_and_get_results()
        nested_out = str(tmp_path / "nested" / "deep" / "report.html")
        reporter.generate_html_report(results_a, results_b, nested_out)
        assert Path(nested_out).exists()

    def test_html_contains_scenario_ids(self, tmp_path):
        reporter = EvalReporter()
        results_a, results_b = self._run_and_get_results()
        out = str(tmp_path / "report.html")
        reporter.generate_html_report(results_a, results_b, out)
        content = Path(out).read_text(encoding="utf-8")
        # At least one of the evaluated scenario IDs should appear in the report
        scenario_ids = {r.scenario_id for r in results_a}
        assert any(sid in content for sid in scenario_ids)


# ---------------------------------------------------------------------------
# Full pipeline integration test
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_end_to_end_evaluation_pipeline(self, tmp_path):
        """
        Integration test: run scenarios → evaluate → aggregate → detect regressions
        → generate report. Verifies the whole pipeline produces correct types.
        """
        runner = ScenarioRunner()
        judge = LLMJudge()
        reporter = EvalReporter()

        scenarios = list_scenarios()[:3]
        results_a, results_b = [], []

        for scenario in scenarios:
            for _ in range(2):
                ta = runner.run_scenario(scenario, "model-a")
                results_a.append(judge.evaluate(ta, scenario))
                tb = runner.run_scenario(scenario, "model-b")
                results_b.append(judge.evaluate(tb, scenario))

        # Aggregate
        df_a = reporter.aggregate_batch(results_a)
        df_b = reporter.aggregate_batch(results_b)
        assert not df_a.empty and not df_b.empty

        # CI for task_completion
        rubric_cols = [c.replace("score_", "") for c in df_a.columns if c.startswith("score_")]
        if rubric_cols:
            mean, lower, upper = reporter.compute_confidence_intervals(df_a, rubric_cols[0])
            assert lower <= mean <= upper

        # Regressions
        flags = reporter.detect_regressions(df_a, df_b, threshold=0.3)
        assert isinstance(flags, list)

        # HTML report
        out = str(tmp_path / "integration_report.html")
        returned = reporter.generate_html_report(results_a, results_b, out)
        assert Path(returned).exists()
        content = Path(returned).read_text(encoding="utf-8")
        assert len(content) > 500
