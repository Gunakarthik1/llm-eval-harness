# LLM Eval Harness

Rubric-based LLM-as-a-Judge evaluation framework for head-to-head model benchmarking. Compare any two LLMs across 18 scenarios in 6 categories — scored on 6 dimensions with real-time SSE streaming and an interactive arena UI.

**Live:** [llm-eval-harness.onrender.com](https://llm-eval-harness.onrender.com)

---

## What it does

Pick two models (e.g. `gpt-4o` vs `claude-3-5-sonnet`), select a scenario set, and run a head-to-head benchmark. Each model's output is scored by a judge LLM using structured rubrics across 6 evaluation dimensions. Results stream in live and are visualized on a radar chart with per-scenario breakdown.

---

## Evaluation Suite

**18 scenarios across 6 categories:**

| Category | Count | Example tasks |
|----------|-------|---------------|
| Code Generation | 4 | JSON parser, BST implementation, async HTTP client, LRU cache |
| Reasoning | 3 | Compound math, logical fallacy detection, Fermi estimation |
| Tool Calling | 3 | Weather lookup, compound interest calculator, search API |
| Summarization | 3 | Technical article, meeting notes, code diff explanation |
| Instruction Following | 2 | Haiku generation, pros/cons list |
| Factual Grounding | 3 | PageRank explanation, CAP theorem, Transformer architecture |

**6 scoring dimensions (0–10 each):**

`Accuracy` · `Coherence` · `Factuality` · `Safety` · `Tool Use` · `Reasoning`

---

## Arena UI

Four-tab dashboard built for interactive benchmarking:

**⚔ Compare** — Select Model A vs Model B, choose All 18 / Quick 6 / Custom scenarios, watch live SSE progress, then see:
- Chart.js radar pentagon comparing both models across 6 dimensions
- Animated score bars with winner callout and point delta
- Per-scenario table with category tags and expandable judge reasoning

**📊 Models** — Spec cards for 12 major LLMs: MMLU, HumanEval, MATH benchmarks, pricing, context window, strengths. Filter by provider (OpenAI, Anthropic, Google, DeepSeek, Meta, Mistral).

**🏆 Leaderboard** — Historical rankings from all eval runs with gold/silver/bronze badges.

**📡 AI News** — Live AI/LLM news from Hacker News + a model release timeline.

---

## Architecture

```
POST /api/run  (SSE stream)
       │
       ▼
ScenarioRunner          ← generates model responses
       │
  LLMJudge              ← scores each response against rubrics
       │
  EvalReporter          ← aggregates, detects regressions
       │
  SSE events ──▶ frontend radar chart + table
```

Evaluation runs stream progress via Server-Sent Events. Each scenario result fires a `progress` event with the model name, scenario ID, and score. A final `complete` event delivers the full results object with radar data and per-scenario breakdown.

---

## Tech Stack

- **Backend:** Python · FastAPI · asyncio · Pydantic · SSE streaming
- **Evaluation:** Custom LLM-as-a-Judge rubric framework (`harness/evaluator.py`)
- **Adversarial:** Auto-generated test variants via `harness/adversarial.py`
- **Reporting:** HTML/JSON report generation (`harness/reporter.py`)
- **Frontend:** Vanilla JS · Chart.js radar chart · CSS Grid
- **Infra:** Docker · Render

---

## Running locally

```bash
git clone https://github.com/Gunakarthik1/llm-eval-harness
cd llm-eval-harness
pip install -r requirements.txt
uvicorn harness.main:app --reload --port 8000
```

Open `http://localhost:8000`

---

## API

```
POST /api/run
Body: { "models": ["gpt-4o", "claude-3-5-sonnet"], "scenario_ids": [] }
→ SSE stream of progress + complete events

GET  /api/scenarios       ← list all 18 scenarios
GET  /api/leaderboard     ← historical model rankings
GET  /api/models          ← specs for 12 major LLMs
GET  /api/news            ← live AI news from Hacker News
GET  /api/health
```

---

## Adversarial Testing

Generate robustness variants of any scenario:

```
POST /api/adversarial/generate?scenario_id=code_json_parser&n=5&model=gpt-4o
```

Returns the original score plus score deltas for 5 auto-mutated variants — negation, distraction, constraint changes, etc. — to surface model brittleness.
