"""
Pandas-based aggregation, confidence interval computation,
regression detection, and HTML diff report generation.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from harness.models import (
    EvalResult,
    RegressionFlag,
    SEVERITY_CRITICAL,
    SEVERITY_MAJOR,
    SEVERITY_MINOR,
)


# ---------------------------------------------------------------------------
# t-distribution critical values (two-tailed) — hand-computed lookup table
# Covers df 1-120 + inf at common alpha levels; interpolation used for gaps.
# ---------------------------------------------------------------------------

_T_TABLE: dict[int, dict[float, float]] = {
    # df: {alpha/2: t-value}
    1:   {0.025: 12.706, 0.05: 6.314},
    2:   {0.025: 4.303,  0.05: 2.920},
    3:   {0.025: 3.182,  0.05: 2.353},
    4:   {0.025: 2.776,  0.05: 2.132},
    5:   {0.025: 2.571,  0.05: 2.015},
    6:   {0.025: 2.447,  0.05: 1.943},
    7:   {0.025: 2.365,  0.05: 1.895},
    8:   {0.025: 2.306,  0.05: 1.860},
    9:   {0.025: 2.262,  0.05: 1.833},
    10:  {0.025: 2.228,  0.05: 1.812},
    11:  {0.025: 2.201,  0.05: 1.796},
    12:  {0.025: 2.179,  0.05: 1.782},
    13:  {0.025: 2.160,  0.05: 1.771},
    14:  {0.025: 2.145,  0.05: 1.761},
    15:  {0.025: 2.131,  0.05: 1.753},
    16:  {0.025: 2.120,  0.05: 1.746},
    17:  {0.025: 2.110,  0.05: 1.740},
    18:  {0.025: 2.101,  0.05: 1.734},
    19:  {0.025: 2.093,  0.05: 1.729},
    20:  {0.025: 2.086,  0.05: 1.725},
    25:  {0.025: 2.060,  0.05: 1.708},
    30:  {0.025: 2.042,  0.05: 1.697},
    40:  {0.025: 2.021,  0.05: 1.684},
    60:  {0.025: 2.000,  0.05: 1.671},
    80:  {0.025: 1.990,  0.05: 1.664},
    120: {0.025: 1.980,  0.05: 1.658},
    999: {0.025: 1.960,  0.05: 1.645},  # infinity approximation
}


def _t_critical(df: int, alpha_half: float = 0.025) -> float:
    """Return the t-critical value for the given degrees of freedom."""
    if df <= 0:
        return 12.706  # worst case
    sorted_keys = sorted(_T_TABLE.keys())
    # Find nearest key
    lower_key = max((k for k in sorted_keys if k <= df), default=sorted_keys[0])
    upper_key = min((k for k in sorted_keys if k >= df), default=sorted_keys[-1])
    if lower_key == upper_key:
        return _T_TABLE[lower_key].get(alpha_half, 1.96)
    # Linear interpolation
    t_low = _T_TABLE[lower_key].get(alpha_half, 1.96)
    t_high = _T_TABLE[upper_key].get(alpha_half, 1.96)
    frac = (df - lower_key) / (upper_key - lower_key)
    return t_low + frac * (t_high - t_low)


# ---------------------------------------------------------------------------
# EvalReporter
# ---------------------------------------------------------------------------

class EvalReporter:
    """Aggregates evaluation results, computes statistics, generates reports."""

    def aggregate_batch(self, results: list[EvalResult]) -> pd.DataFrame:
        """Build a flat DataFrame from a list of EvalResults."""
        rows: list[dict] = []
        for r in results:
            base = {
                "run_id": r.run_id,
                "scenario_id": r.scenario_id,
                "model": r.model,
                "weighted_total": r.weighted_total,
                "timestamp": r.timestamp,
            }
            for rs in r.rubric_scores:
                base[f"score_{rs.rubric}"] = rs.score
            rows.append(base)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df

    def compute_confidence_intervals(
        self,
        df: pd.DataFrame,
        rubric: str,
        confidence: float = 0.95,
    ) -> tuple[float, float, float]:
        """
        Compute (mean, lower_ci, upper_ci) for the given rubric column.
        Uses the Student t-distribution manually — no scipy required.

        confidence: e.g. 0.95 for 95% CI
        """
        col = f"score_{rubric}" if f"score_{rubric}" in df.columns else rubric
        if col not in df.columns:
            return (0.0, 0.0, 0.0)

        values = df[col].dropna().tolist()
        n = len(values)
        if n == 0:
            return (0.0, 0.0, 0.0)
        if n == 1:
            return (values[0], values[0], values[0])

        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std_err = math.sqrt(variance / n)

        alpha = 1.0 - confidence
        alpha_half = alpha / 2.0
        t_val = _t_critical(df=n - 1, alpha_half=alpha_half)
        margin = t_val * std_err

        return (round(mean, 4), round(mean - margin, 4), round(mean + margin, 4))

    def detect_regressions(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> list[RegressionFlag]:
        """
        Compare baseline and current DataFrames per rubric.
        Returns a list of RegressionFlags where current mean dropped > threshold.
        """
        rubric_cols = [c for c in baseline_df.columns if c.startswith("score_")]
        flags: list[RegressionFlag] = []

        for col in rubric_cols:
            rubric = col[len("score_"):]
            if col not in current_df.columns:
                continue

            b_vals = baseline_df[col].dropna()
            c_vals = current_df[col].dropna()

            if b_vals.empty or c_vals.empty:
                continue

            b_mean = float(b_vals.mean())
            c_mean = float(c_vals.mean())
            delta = c_mean - b_mean

            if delta < -threshold:
                abs_delta = abs(delta)
                if abs_delta >= 3.0:
                    severity = SEVERITY_CRITICAL
                elif abs_delta >= 1.5:
                    severity = SEVERITY_MAJOR
                else:
                    severity = SEVERITY_MINOR

                flags.append(RegressionFlag(
                    rubric=rubric,
                    baseline_mean=round(b_mean, 3),
                    current_mean=round(c_mean, 3),
                    delta=round(delta, 3),
                    severity=severity,
                ))

        # Sort by delta ascending (worst regression first)
        flags.sort(key=lambda f: f.delta)
        return flags

    def generate_summary_json(self, results: list[EvalResult]) -> dict:
        """Produce a structured JSON-serializable summary."""
        if not results:
            return {"error": "No results"}

        df = self.aggregate_batch(results)
        models = df["model"].unique().tolist()
        summary: dict = {"models": {}, "scenarios": {}}

        for model in models:
            mdf = df[df["model"] == model]
            model_entry: dict = {
                "avg_weighted_total": round(float(mdf["weighted_total"].mean()), 3),
                "n_evaluations": int(len(mdf)),
                "rubrics": {},
            }
            rubric_cols = [c for c in mdf.columns if c.startswith("score_")]
            for col in rubric_cols:
                rubric = col[len("score_"):]
                vals = mdf[col].dropna()
                if vals.empty:
                    continue
                mean, lower, upper = self.compute_confidence_intervals(mdf, rubric)
                model_entry["rubrics"][rubric] = {
                    "mean": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n": int(len(vals)),
                }
            summary["models"][model] = model_entry

        # Per-scenario breakdown
        for scenario_id in df["scenario_id"].unique():
            sdf = df[df["scenario_id"] == scenario_id]
            summary["scenarios"][scenario_id] = {
                m: round(float(sdf[sdf["model"] == m]["weighted_total"].mean()), 3)
                for m in models
                if m in sdf["model"].values
            }

        return summary

    def generate_html_report(
        self,
        baseline_results: list[EvalResult],
        current_results: list[EvalResult],
        output_path: str,
    ) -> str:
        """
        Generate a fully self-contained HTML diff report comparing
        baseline and current evaluation results.
        Returns the output path.
        """
        baseline_df = self.aggregate_batch(baseline_results)
        current_df = self.aggregate_batch(current_results)
        regressions = self.detect_regressions(baseline_df, current_df)

        model_a = baseline_results[0].model if baseline_results else "baseline"
        model_b = current_results[0].model if current_results else "current"
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build rubric comparison table data
        rubric_cols = [c for c in baseline_df.columns if c.startswith("score_")]
        rubric_rows_html = ""
        for col in rubric_cols:
            rubric = col[len("score_"):]
            b_mean, b_low, b_high = self.compute_confidence_intervals(baseline_df, rubric)
            c_mean, c_low, c_high = self.compute_confidence_intervals(current_df, rubric)
            delta = round(c_mean - b_mean, 3)
            delta_class = "positive" if delta > 0 else ("negative" if delta < 0 else "neutral")
            delta_sign = "+" if delta > 0 else ""
            severity_badge = ""
            for flag in regressions:
                if flag.rubric == rubric:
                    severity_badge = f'<span class="badge badge-{flag.severity}">{flag.severity.upper()}</span>'
                    break
            rubric_rows_html += f"""
            <tr>
                <td class="rubric-name">{rubric.replace('_', ' ').title()}</td>
                <td>{b_mean:.2f} <span class="ci">[{b_low:.2f}, {b_high:.2f}]</span></td>
                <td>{c_mean:.2f} <span class="ci">[{c_low:.2f}, {c_high:.2f}]</span></td>
                <td class="delta {delta_class}">{delta_sign}{delta:.2f}</td>
                <td>{severity_badge}</td>
            </tr>"""

        # Build regression flags section
        regression_rows_html = ""
        for flag in regressions:
            severity_class = f"badge-{flag.severity}"
            regression_rows_html += f"""
            <tr>
                <td>{flag.rubric.replace('_', ' ').title()}</td>
                <td>{flag.baseline_mean:.2f}</td>
                <td>{flag.current_mean:.2f}</td>
                <td class="delta negative">{flag.delta:.2f}</td>
                <td><span class="badge {severity_class}">{flag.severity.upper()}</span></td>
            </tr>"""

        if not regression_rows_html:
            regression_rows_html = '<tr><td colspan="5" class="no-issues">No regressions detected.</td></tr>'

        # Per-scenario comparison
        baseline_map = {r.scenario_id: r for r in baseline_results}
        current_map = {r.scenario_id: r for r in current_results}
        all_scenario_ids = sorted(set(baseline_map) | set(current_map))

        scenario_rows_html = ""
        for sid in all_scenario_ids:
            br = baseline_map.get(sid)
            cr = current_map.get(sid)
            b_score = f"{br.weighted_total:.2f}" if br else "—"
            c_score = f"{cr.weighted_total:.2f}" if cr else "—"
            if br and cr:
                d = cr.weighted_total - br.weighted_total
                d_sign = "+" if d > 0 else ""
                d_class = "positive" if d > 0 else ("negative" if d < 0 else "neutral")
                delta_html = f'<span class="delta {d_class}">{d_sign}{d:.2f}</span>'
            else:
                delta_html = "—"
            scenario_rows_html += f"""
            <tr>
                <td class="scenario-id">{sid}</td>
                <td>{b_score}</td>
                <td>{c_score}</td>
                <td>{delta_html}</td>
            </tr>"""

        # Build CI chart data (JSON for inline JS)
        ci_data: list[dict] = []
        for col in rubric_cols:
            rubric = col[len("score_"):]
            b_mean, b_low, b_high = self.compute_confidence_intervals(baseline_df, rubric)
            c_mean, c_low, c_high = self.compute_confidence_intervals(current_df, rubric)
            ci_data.append({
                "rubric": rubric.replace("_", " ").title(),
                "b_mean": b_mean, "b_low": b_low, "b_high": b_high,
                "c_mean": c_mean, "c_low": c_low, "c_high": c_high,
            })

        # Example traces for the largest regression
        trace_html = ""
        if regressions and baseline_results and current_results:
            worst_rubric = regressions[0].rubric
            worst_sid = None
            worst_delta = 0.0
            for sid in all_scenario_ids:
                br = baseline_map.get(sid)
                cr = current_map.get(sid)
                if br and cr:
                    bsc = {rs.rubric: rs.score for rs in br.rubric_scores}
                    csc = {rs.rubric: rs.score for rs in cr.rubric_scores}
                    d = csc.get(worst_rubric, 0) - bsc.get(worst_rubric, 0)
                    if d < worst_delta:
                        worst_delta = d
                        worst_sid = sid

            if worst_sid:
                br = baseline_map[worst_sid]
                cr = current_map[worst_sid]
                # Render baseline trace
                if br.trace:
                    b_trace_turns = "".join(
                        f'<div class="turn"><div class="turn-user">User: {t.user_input}</div>'
                        f'<div class="turn-agent"><strong>{model_a}:</strong> {t.agent_response[:400]}{"..." if len(t.agent_response) > 400 else ""}</div></div>'
                        for t in br.trace.turns
                    )
                else:
                    b_trace_turns = "<p>No trace available.</p>"

                if cr.trace:
                    c_trace_turns = "".join(
                        f'<div class="turn"><div class="turn-user">User: {t.user_input}</div>'
                        f'<div class="turn-agent"><strong>{model_b}:</strong> {t.agent_response[:400]}{"..." if len(t.agent_response) > 400 else ""}</div></div>'
                        for t in cr.trace.turns
                    )
                else:
                    c_trace_turns = "<p>No trace available.</p>"

                trace_html = f"""
                <div class="trace-comparison">
                    <h3>Largest Regression Trace: <code>{worst_sid}</code> — rubric: <em>{worst_rubric.replace('_',' ')}</em></h3>
                    <div class="trace-grid">
                        <div class="trace-col">
                            <h4>{model_a} (Baseline)</h4>
                            <div class="trace-score">{br.weighted_total:.2f}/10</div>
                            {b_trace_turns}
                        </div>
                        <div class="trace-col">
                            <h4>{model_b} (Current)</h4>
                            <div class="trace-score regression">{cr.weighted_total:.2f}/10</div>
                            {c_trace_turns}
                        </div>
                    </div>
                </div>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Eval Diff Report — {model_a} vs {model_b}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Inter', sans-serif;
    background: #FAFAF8;
    color: #1a1a1a;
    line-height: 1.6;
    padding: 2rem;
  }}

  h1, h2, h3 {{ font-family: 'Merriweather', serif; }}

  .report-header {{
    border-bottom: 3px solid #5B21B6;
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
  }}

  .report-header h1 {{
    font-size: 2rem;
    color: #5B21B6;
    margin-bottom: 0.5rem;
  }}

  .report-meta {{
    font-size: 0.875rem;
    color: #6b7280;
    font-family: 'JetBrains Mono', monospace;
  }}

  .model-pills {{
    display: flex;
    gap: 1rem;
    margin-top: 0.75rem;
  }}

  .model-pill {{
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
  }}

  .pill-a {{ background: #EDE9FE; color: #5B21B6; border: 1px solid #5B21B6; }}
  .pill-b {{ background: #FEF2F2; color: #DC2626; border: 1px solid #DC2626; }}

  .section {{
    margin-bottom: 3rem;
  }}

  .section h2 {{
    font-size: 1.4rem;
    color: #111;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e5e7eb;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }}

  th {{
    background: #f3f4f6;
    color: #374151;
    font-weight: 600;
    text-align: left;
    padding: 0.6rem 0.8rem;
    border: 1px solid #e5e7eb;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  td {{
    padding: 0.55rem 0.8rem;
    border: 1px solid #e5e7eb;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    vertical-align: middle;
  }}

  tr:nth-child(even) {{ background: #f9fafb; }}

  .rubric-name {{
    font-family: 'Inter', sans-serif;
    font-weight: 500;
  }}

  .ci {{
    font-size: 0.75rem;
    color: #6b7280;
  }}

  .delta {{ font-weight: 700; }}
  .positive {{ color: #4D7C6E; }}
  .negative {{ color: #DC2626; }}
  .neutral  {{ color: #6b7280; }}

  .badge {{
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.05em;
  }}

  .badge-minor    {{ background: #FEF3C7; color: #92400E; }}
  .badge-major    {{ background: #FEE2E2; color: #991B1B; }}
  .badge-critical {{ background: #7F1D1D; color: #ffffff; }}

  .no-issues {{ color: #4D7C6E; font-style: italic; text-align: center; padding: 1rem; }}

  .scenario-id {{ font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }}

  .stats-bar {{
    display: flex;
    gap: 1.5rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
  }}

  .stat-card {{
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    min-width: 180px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }}

  .stat-card .label {{
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
  }}

  .stat-card .value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a1a;
    line-height: 1.2;
  }}

  .stat-card .value.good {{ color: #4D7C6E; }}
  .stat-card .value.bad  {{ color: #DC2626; }}

  /* CI chart */
  .ci-chart {{ margin: 1.5rem 0; }}
  .ci-row {{
    display: flex;
    align-items: center;
    margin-bottom: 0.6rem;
    gap: 0.75rem;
  }}
  .ci-label {{
    width: 180px;
    font-size: 0.8rem;
    font-family: 'Inter', sans-serif;
    flex-shrink: 0;
    color: #374151;
  }}
  .ci-track {{
    flex: 1;
    height: 20px;
    background: #f3f4f6;
    border-radius: 4px;
    position: relative;
  }}
  .ci-bar-a, .ci-bar-b {{
    position: absolute;
    height: 6px;
    border-radius: 3px;
    top: 3px;
  }}
  .ci-bar-a {{ background: #5B21B6; opacity: 0.7; top: 3px; }}
  .ci-bar-b {{ background: #DC2626; opacity: 0.7; top: 11px; }}

  /* Traces */
  .trace-comparison {{
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1.5rem;
    background: white;
    margin-top: 1rem;
  }}

  .trace-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-top: 1rem;
  }}

  .trace-col h4 {{
    font-size: 1rem;
    margin-bottom: 0.5rem;
    font-family: 'Merriweather', serif;
  }}

  .trace-score {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #4D7C6E;
    margin-bottom: 0.75rem;
  }}

  .trace-score.regression {{ color: #DC2626; }}

  .turn {{ margin-bottom: 1rem; }}
  .turn-user {{
    background: #f3f4f6;
    border-radius: 8px 8px 8px 0;
    padding: 0.5rem 0.75rem;
    font-size: 0.85rem;
    color: #374151;
    margin-bottom: 0.35rem;
  }}
  .turn-agent {{
    background: #EDE9FE;
    border-radius: 8px 8px 8px 0;
    padding: 0.5rem 0.75rem;
    font-size: 0.82rem;
    color: #1a1a1a;
    font-family: 'JetBrains Mono', monospace;
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.5;
  }}

  .footer {{
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #e5e7eb;
    font-size: 0.75rem;
    color: #9ca3af;
    font-family: 'JetBrains Mono', monospace;
  }}
</style>
</head>
<body>

<div class="report-header">
  <h1>LLM Evaluation Diff Report</h1>
  <div class="report-meta">Generated: {generated_at}</div>
  <div class="model-pills">
    <span class="model-pill pill-a">Baseline: {model_a}</span>
    <span class="model-pill pill-b">Current:  {model_b}</span>
  </div>
</div>

<!-- Summary Stats -->
<div class="section">
  <div class="stats-bar">
    <div class="stat-card">
      <div class="label">Scenarios Evaluated</div>
      <div class="value">{len(all_scenario_ids)}</div>
    </div>
    <div class="stat-card">
      <div class="label">{model_a} Avg Score</div>
      <div class="value good">{float(baseline_df["weighted_total"].mean()):.2f}</div>
    </div>
    <div class="stat-card">
      <div class="label">{model_b} Avg Score</div>
      <div class="value {"bad" if float(current_df["weighted_total"].mean()) < float(baseline_df["weighted_total"].mean()) else "good"}">{float(current_df["weighted_total"].mean()):.2f}</div>
    </div>
    <div class="stat-card">
      <div class="label">Regressions Found</div>
      <div class="value {"bad" if regressions else "good"}">{len(regressions)}</div>
    </div>
  </div>
</div>

<!-- Rubric Comparison Table -->
<div class="section">
  <h2>Rubric-by-Rubric Comparison</h2>
  <table>
    <thead>
      <tr>
        <th>Rubric</th>
        <th>{model_a} Mean [95% CI]</th>
        <th>{model_b} Mean [95% CI]</th>
        <th>Delta (B − A)</th>
        <th>Status</th>
      </tr>
    </thead>
    <tbody>
      {rubric_rows_html}
    </tbody>
  </table>
</div>

<!-- CI Visualization -->
<div class="section">
  <h2>Confidence Interval Visualization</h2>
  <p style="font-size:0.8rem;color:#6b7280;margin-bottom:1rem;">
    Purple bars = {model_a} (baseline) &nbsp;|&nbsp; Red bars = {model_b} (current). Width represents 95% CI.
  </p>
  <div class="ci-chart" id="ci-chart"></div>
</div>

<!-- Regression Flags -->
<div class="section">
  <h2>Regression Flags</h2>
  <table>
    <thead>
      <tr>
        <th>Rubric</th>
        <th>{model_a} Mean</th>
        <th>{model_b} Mean</th>
        <th>Delta</th>
        <th>Severity</th>
      </tr>
    </thead>
    <tbody>
      {regression_rows_html}
    </tbody>
  </table>
  <p style="margin-top:0.75rem;font-size:0.78rem;color:#6b7280;">
    * Regressions detected using threshold δ &lt; −0.5. Severity: minor (|δ| ≥ 0.5), major (|δ| ≥ 1.5), critical (|δ| ≥ 3.0).
  </p>
</div>

<!-- Per-Scenario Scores -->
<div class="section">
  <h2>Per-Scenario Score Comparison</h2>
  <table>
    <thead>
      <tr>
        <th>Scenario</th>
        <th>{model_a}</th>
        <th>{model_b}</th>
        <th>Delta</th>
      </tr>
    </thead>
    <tbody>
      {scenario_rows_html}
    </tbody>
  </table>
</div>

<!-- Trace Comparison -->
<div class="section">
  <h2>Side-by-Side Trace (Largest Regression)</h2>
  {trace_html if trace_html else '<p style="color:#6b7280;font-style:italic;">No regression traces to display.</p>'}
</div>

<div class="footer">
  LLM-as-a-Judge Benchmark &amp; Agent Evaluation Harness &nbsp;|&nbsp;
  Confidence intervals computed using Student t-distribution (df = n−1) &nbsp;|&nbsp;
  Regression threshold: −0.5 points
</div>

<script>
const ciData = {json.dumps(ci_data)};
const chartEl = document.getElementById('ci-chart');
const MAX_SCORE = 10;

ciData.forEach(d => {{
  const row = document.createElement('div');
  row.className = 'ci-row';

  const label = document.createElement('div');
  label.className = 'ci-label';
  label.textContent = d.rubric;

  const track = document.createElement('div');
  track.className = 'ci-track';

  const barA = document.createElement('div');
  barA.className = 'ci-bar-a';
  barA.style.left   = (d.b_low  / MAX_SCORE * 100) + '%';
  barA.style.width  = ((d.b_high - d.b_low) / MAX_SCORE * 100) + '%';
  barA.title = `{model_a}: mean=${{d.b_mean.toFixed(2)}} CI[${{d.b_low.toFixed(2)}}, ${{d.b_high.toFixed(2)}}]`;

  const barB = document.createElement('div');
  barB.className = 'ci-bar-b';
  barB.style.left   = (d.c_low  / MAX_SCORE * 100) + '%';
  barB.style.width  = ((d.c_high - d.c_low) / MAX_SCORE * 100) + '%';
  barB.title = `{model_b}: mean=${{d.c_mean.toFixed(2)}} CI[${{d.c_low.toFixed(2)}}, ${{d.c_high.toFixed(2)}}]`;

  track.appendChild(barA);
  track.appendChild(barB);
  row.appendChild(label);
  row.appendChild(track);
  chartEl.appendChild(row);
}});
</script>
</body>
</html>"""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return output_path
