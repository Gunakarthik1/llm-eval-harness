"""
Pydantic schemas for the evaluation harness.
"""
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field
import time


# ---------------------------------------------------------------------------
# Scenario primitives
# ---------------------------------------------------------------------------

class Scenario(BaseModel):
    id: str
    category: str
    prompt: str
    expected_keywords: list[str]
    ground_truth: str
    turns: int = 1
    rubrics: list[str]
    max_score: int = 10
    tool_required: bool = False
    expected_tool: Optional[str] = None
    format_constraint: Optional[str] = None  # e.g. "haiku", "list:5", "word_limit:100"


# ---------------------------------------------------------------------------
# Execution trace primitives
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    success: bool = True


class Turn(BaseModel):
    turn_number: int
    user_input: str
    agent_response: str
    response_time_ms: float
    tokens_used: int
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ExecutionTrace(BaseModel):
    run_id: str = ""
    scenario_id: str
    model: str
    turns: list[Turn] = Field(default_factory=list)
    total_tokens: int = 0
    duration_ms: float = 0.0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------

class RubricScore(BaseModel):
    rubric: str
    score: float          # 0-10
    max_score: float = 10.0
    reasoning: str = ""


class EvalResult(BaseModel):
    run_id: str = ""
    scenario_id: str
    model: str
    rubric_scores: list[RubricScore] = Field(default_factory=list)
    weighted_total: float = 0.0
    judge_reasoning: str = ""
    trace: Optional[ExecutionTrace] = None
    timestamp: float = Field(default_factory=time.time)

    def score_map(self) -> dict[str, float]:
        return {rs.rubric: rs.score for rs in self.rubric_scores}


class ComparisonResult(BaseModel):
    scenario_id: str
    model_a: str
    model_b: str
    delta_per_rubric: dict[str, float] = Field(default_factory=dict)
    regression_detected: bool = False
    improvement_detected: bool = False
    overall_delta: float = 0.0


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

SEVERITY_MINOR = "minor"
SEVERITY_MAJOR = "major"
SEVERITY_CRITICAL = "critical"


class RegressionFlag(BaseModel):
    rubric: str
    baseline_mean: float
    current_mean: float
    delta: float
    severity: str  # "minor", "major", "critical"


# ---------------------------------------------------------------------------
# API request / response shapes
# ---------------------------------------------------------------------------

class EvalRunRequest(BaseModel):
    model_a: str = "model-a"
    model_b: str = "model-b"
    scenario_ids: list[str] = Field(default_factory=list)
    n_runs: int = Field(default=3, ge=1, le=10)


class EvalRunStatus(BaseModel):
    run_id: str
    status: str   # "pending" | "running" | "complete" | "error"
    progress: int = 0          # 0-100
    total_scenarios: int = 0
    completed_scenarios: int = 0
    message: str = ""


class ReportRequest(BaseModel):
    run_id: str
    output_filename: Optional[str] = None


class LeaderboardEntry(BaseModel):
    model: str
    avg_weighted_total: float
    runs_evaluated: int
    top_rubric: str
    worst_rubric: str
