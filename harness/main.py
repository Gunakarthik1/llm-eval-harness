"""
FastAPI control plane for the LLM-as-a-Judge Evaluation Harness.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from harness.adversarial import AdversarialGenerator
from harness.evaluator import LLMJudge
from harness.models import (
    EvalResult,
    EvalRunRequest,
    EvalRunStatus,
    LeaderboardEntry,
    ReportRequest,
    Scenario,
)
from harness.reporter import EvalReporter
from harness.runner import ScenarioRunner
from harness.scenarios import SCENARIO_MAP, list_scenarios

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LLM-as-a-Judge Evaluation Harness",
    description="Automated multi-turn agent evaluation with LLM-as-a-Judge scoring.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Serve frontend and reports
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Static frontend
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# Reports directory
app.mount("/reports-static", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to the frontend dashboard."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<h1>LLM Eval Harness</h1><p>Visit <a href='/docs'>/docs</a> for API.</p>"
    )


# ---------------------------------------------------------------------------
# In-memory run store (would be a DB in production)
# ---------------------------------------------------------------------------

# run_id -> {"status": EvalRunStatus, "results_a": list, "results_b": list}
_RUN_STORE: dict[str, dict[str, Any]] = {}

# Leaderboard accumulation: model -> list of weighted totals
_LEADERBOARD: dict[str, list[float]] = {}


# ---------------------------------------------------------------------------
# Background task: run evaluation
# ---------------------------------------------------------------------------

def _run_evaluation_sync(
    run_id: str,
    request: EvalRunRequest,
) -> None:
    """
    Synchronous evaluation worker executed in a background task.
    Runs both model-a and model-b through all requested scenarios.
    """
    store = _RUN_STORE[run_id]
    status: EvalRunStatus = store["status"]

    try:
        # Resolve scenarios
        if request.scenario_ids:
            scenarios = [SCENARIO_MAP[sid] for sid in request.scenario_ids if sid in SCENARIO_MAP]
        else:
            scenarios = list_scenarios()

        if not scenarios:
            status.status = "error"
            status.message = "No valid scenario IDs found."
            return

        status.total_scenarios = len(scenarios)
        status.status = "running"

        runner = ScenarioRunner()
        judge = LLMJudge()

        results_a: list[EvalResult] = []
        results_b: list[EvalResult] = []

        for i, scenario in enumerate(scenarios):
            status.message = f"Running scenario {i + 1}/{len(scenarios)}: {scenario.id}"
            status.progress = int((i / len(scenarios)) * 100)

            # Run model-a
            for _ in range(request.n_runs):
                trace_a = runner.run_scenario(scenario, request.model_a, run_id=run_id)
                eval_a = judge.evaluate(trace_a, scenario)
                eval_a.run_id = run_id
                results_a.append(eval_a)

            # Run model-b
            for _ in range(request.n_runs):
                trace_b = runner.run_scenario(scenario, request.model_b, run_id=run_id)
                eval_b = judge.evaluate(trace_b, scenario)
                eval_b.run_id = run_id
                results_b.append(eval_b)

            status.completed_scenarios = i + 1

        store["results_a"] = results_a
        store["results_b"] = results_b

        # Update leaderboard
        for result in results_a:
            _LEADERBOARD.setdefault(request.model_a, []).append(result.weighted_total)
        for result in results_b:
            _LEADERBOARD.setdefault(request.model_b, []).append(result.weighted_total)

        status.status = "complete"
        status.progress = 100
        status.message = f"Evaluation complete. {len(results_a)} results per model."

    except Exception as exc:
        status.status = "error"
        status.message = str(exc)
        raise


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0", "timestamp": time.time()}


@app.get("/api/scenarios")
async def get_scenarios():
    """List all available evaluation scenarios."""
    return {
        "count": len(list_scenarios()),
        "scenarios": [s.model_dump() for s in list_scenarios()],
    }


@app.post("/api/eval/run", status_code=202)
async def start_evaluation(request: EvalRunRequest, background_tasks: BackgroundTasks):
    """
    Start a background evaluation run.
    Returns a run_id for polling.
    """
    run_id = str(uuid.uuid4())[:12]
    status = EvalRunStatus(
        run_id=run_id,
        status="pending",
        progress=0,
        message="Queued for execution.",
    )
    _RUN_STORE[run_id] = {
        "status": status,
        "request": request,
        "results_a": [],
        "results_b": [],
    }
    background_tasks.add_task(_run_evaluation_sync, run_id, request)
    return {"run_id": run_id, "message": "Evaluation started."}


@app.get("/api/eval/{run_id}/status")
async def get_run_status(run_id: str):
    """Poll the status of an evaluation run."""
    if run_id not in _RUN_STORE:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return _RUN_STORE[run_id]["status"].model_dump()


@app.get("/api/eval/{run_id}/results")
async def get_run_results(run_id: str):
    """Retrieve full evaluation results for a completed run."""
    if run_id not in _RUN_STORE:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    store = _RUN_STORE[run_id]
    status = store["status"]
    if status.status not in ("complete", "error"):
        return JSONResponse(
            status_code=202,
            content={"message": "Run not yet complete.", "status": status.status},
        )

    reporter = EvalReporter()
    results_a: list[EvalResult] = store["results_a"]
    results_b: list[EvalResult] = store["results_b"]

    summary = reporter.generate_summary_json(results_a + results_b)

    baseline_df = reporter.aggregate_batch(results_a)
    current_df = reporter.aggregate_batch(results_b)
    regressions = reporter.detect_regressions(baseline_df, current_df)

    request: EvalRunRequest = store["request"]

    return {
        "run_id": run_id,
        "model_a": request.model_a,
        "model_b": request.model_b,
        "summary": summary,
        "regressions": [r.model_dump() for r in regressions],
        "results_a": [r.model_dump(exclude={"trace"}) for r in results_a],
        "results_b": [r.model_dump(exclude={"trace"}) for r in results_b],
    }


@app.get("/api/eval/{run_id}/results/full")
async def get_run_results_full(run_id: str):
    """Return results with full execution traces (large payload)."""
    if run_id not in _RUN_STORE:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    store = _RUN_STORE[run_id]
    results_a: list[EvalResult] = store["results_a"]
    results_b: list[EvalResult] = store["results_b"]

    return {
        "run_id": run_id,
        "results_a": [r.model_dump() for r in results_a],
        "results_b": [r.model_dump() for r in results_b],
    }


@app.get("/api/eval/{run_id}/regressions")
async def get_regressions(run_id: str, threshold: float = 0.5):
    """Return only the regression flags for a completed run."""
    if run_id not in _RUN_STORE:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    store = _RUN_STORE[run_id]
    reporter = EvalReporter()
    baseline_df = reporter.aggregate_batch(store["results_a"])
    current_df = reporter.aggregate_batch(store["results_b"])
    flags = reporter.detect_regressions(baseline_df, current_df, threshold=threshold)
    return {"run_id": run_id, "regressions": [f.model_dump() for f in flags]}


@app.post("/api/report/generate")
async def generate_report(req: ReportRequest):
    """Generate an HTML diff report for a completed run."""
    if req.run_id not in _RUN_STORE:
        raise HTTPException(status_code=404, detail=f"Run '{req.run_id}' not found.")

    store = _RUN_STORE[req.run_id]
    results_a: list[EvalResult] = store["results_a"]
    results_b: list[EvalResult] = store["results_b"]

    if not results_a or not results_b:
        raise HTTPException(status_code=400, detail="Run has no results yet.")

    filename = req.output_filename or f"report_{req.run_id}.html"
    if not filename.endswith(".html"):
        filename += ".html"

    output_path = str(REPORTS_DIR / filename)
    reporter = EvalReporter()
    reporter.generate_html_report(results_a, results_b, output_path)

    return {
        "filename": filename,
        "download_url": f"/api/reports/{filename}",
        "view_url": f"/reports-static/{filename}",
    }


@app.get("/api/reports/{filename}")
async def serve_report(filename: str):
    """Download a previously generated HTML report."""
    path = REPORTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Report '{filename}' not found.")
    return FileResponse(str(path), media_type="text/html", filename=filename)


@app.get("/api/leaderboard")
async def get_leaderboard():
    """Return aggregated scores across all evaluation runs."""
    entries: list[LeaderboardEntry] = []

    for model, scores in _LEADERBOARD.items():
        if not scores:
            continue
        avg = sum(scores) / len(scores)

        # Find top/worst rubric from stored results
        all_results = [
            r
            for store in _RUN_STORE.values()
            for result_list in [store.get("results_a", []), store.get("results_b", [])]
            for r in result_list
            if r.model == model
        ]

        rubric_totals: dict[str, list[float]] = {}
        for r in all_results:
            for rs in r.rubric_scores:
                rubric_totals.setdefault(rs.rubric, []).append(rs.score)

        rubric_means = {rb: sum(vs) / len(vs) for rb, vs in rubric_totals.items() if vs}
        if rubric_means:
            top_rubric = max(rubric_means, key=lambda k: rubric_means[k])
            worst_rubric = min(rubric_means, key=lambda k: rubric_means[k])
        else:
            top_rubric = worst_rubric = "—"

        entries.append(LeaderboardEntry(
            model=model,
            avg_weighted_total=round(avg, 3),
            runs_evaluated=len(scores),
            top_rubric=top_rubric,
            worst_rubric=worst_rubric,
        ))

    entries.sort(key=lambda e: e.avg_weighted_total, reverse=True)
    return {"leaderboard": [e.model_dump() for e in entries]}


@app.post("/api/adversarial/generate")
async def generate_adversarial(scenario_id: str, n: int = 5, model: str = "model-a"):
    """
    Generate adversarial variants of a scenario and evaluate them.
    Returns original score + scores for each adversarial variant.
    """
    if scenario_id not in SCENARIO_MAP:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found.")

    scenario = SCENARIO_MAP[scenario_id]
    gen = AdversarialGenerator()
    variants = gen.generate_suite(scenario, n=n)

    runner = ScenarioRunner()
    judge = LLMJudge()

    # Evaluate original
    original_trace = runner.run_scenario(scenario, model)
    original_result = judge.evaluate(original_trace, scenario)

    # Evaluate each variant
    variant_results = []
    for variant in variants:
        trace = runner.run_scenario(variant, model)
        result = judge.evaluate(trace, variant)
        description = gen.describe_mutation(scenario, variant)
        variant_results.append({
            "description": description,
            "result": result.model_dump(exclude={"trace"}),
            "score_delta": round(result.weighted_total - original_result.weighted_total, 3),
        })

    return {
        "scenario_id": scenario_id,
        "model": model,
        "original_score": original_result.weighted_total,
        "variants": variant_results,
    }
