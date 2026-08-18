"""
Built-in task scenario library — 18 scenarios across 6 categories.
"""
from harness.models import Scenario

SCENARIOS: list[Scenario] = [

    # -----------------------------------------------------------------------
    # CODE GENERATION  (4 scenarios)
    # -----------------------------------------------------------------------
    Scenario(
        id="code_json_parser",
        category="Code Generation",
        prompt=(
            "Write a Python function called `parse_nested_json` that accepts a JSON string "
            "and returns a flattened dictionary where nested keys are joined with dots. "
            "Include type hints and a brief docstring. Show a usage example."
        ),
        expected_keywords=[
            "def parse_nested_json", "import json", "dict", "str", "docstring",
            "flatten", "recursive", "example",
        ],
        ground_truth=(
            "A correct implementation uses recursion or a stack to flatten nested dicts. "
            "Keys at each level are joined with '.' separators. "
            "The function signature is parse_nested_json(json_str: str) -> dict."
        ),
        turns=2,
        rubrics=["task_completion", "instruction_following", "coherence"],
        max_score=10,
        format_constraint=None,
    ),

    Scenario(
        id="code_bst",
        category="Code Generation",
        prompt=(
            "Implement a binary search tree in Python with the following methods: "
            "`insert(value)`, `search(value) -> bool`, `delete(value)`, and "
            "`inorder() -> list`. Include a `Node` dataclass."
        ),
        expected_keywords=[
            "class Node", "class BinarySearchTree", "def insert", "def search",
            "def delete", "def inorder", "left", "right", "dataclass",
        ],
        ground_truth=(
            "A BST implementation uses a Node with left/right pointers. "
            "Insert follows BST property. Delete handles three cases: leaf, one child, two children "
            "(replace with inorder successor). Inorder traversal yields sorted output."
        ),
        turns=2,
        rubrics=["task_completion", "instruction_following", "coherence"],
        max_score=10,
    ),

    Scenario(
        id="code_async_http",
        category="Code Generation",
        prompt=(
            "Write an async Python function `fetch_all(urls: list[str]) -> list[dict]` "
            "that fetches all URLs concurrently using httpx and asyncio.gather, "
            "handles HTTP errors gracefully, and returns a list of response dicts "
            "with keys `url`, `status`, `body`."
        ),
        expected_keywords=[
            "async def fetch_all", "import httpx", "asyncio", "gather",
            "status", "body", "url", "exception", "error",
        ],
        ground_truth=(
            "Uses httpx.AsyncClient and asyncio.gather for concurrency. "
            "Each response dict contains url, status code, and body text. "
            "Errors are caught per-request and stored without crashing the batch."
        ),
        turns=1,
        rubrics=["task_completion", "instruction_following", "coherence"],
        max_score=10,
    ),

    Scenario(
        id="code_lru_cache",
        category="Code Generation",
        prompt=(
            "Implement an LRU cache class in Python with O(1) get and put operations. "
            "Use a doubly-linked list and a hash map. Include capacity parameter in __init__, "
            "and get(key)/put(key, value) methods matching the LeetCode 146 interface."
        ),
        expected_keywords=[
            "class LRUCache", "def get", "def put", "capacity",
            "OrderedDict", "doubly", "O(1)", "hashmap",
        ],
        ground_truth=(
            "LRU cache uses OrderedDict or explicit doubly-linked list + dict for O(1) ops. "
            "get() moves key to end (most recent). put() evicts least-recently-used when at capacity."
        ),
        turns=2,
        rubrics=["task_completion", "instruction_following", "coherence"],
        max_score=10,
    ),

    # -----------------------------------------------------------------------
    # REASONING  (3 scenarios)
    # -----------------------------------------------------------------------
    Scenario(
        id="reasoning_math_compound",
        category="Reasoning",
        prompt=(
            "Solve step by step: Alice invests $5,000 at 7% annual interest compounded monthly. "
            "After 10 years, she withdraws half the balance. The remaining amount continues "
            "compounding at 5% for another 5 years. What is the final balance? "
            "Show every intermediate calculation."
        ),
        expected_keywords=[
            "compound interest", "monthly", "formula", "balance", "withdraw",
            "step", "final", "5000", "7%", "5%",
        ],
        ground_truth=(
            "Step 1: A1 = 5000*(1+0.07/12)^120 ≈ $10,048.31. "
            "Step 2: After withdrawal: $5,024.15. "
            "Step 3: A2 = 5024.15*(1+0.05/12)^60 ≈ $6,440.18."
        ),
        turns=1,
        rubrics=["task_completion", "factual_grounding", "coherence"],
        max_score=10,
        format_constraint=None,
    ),

    Scenario(
        id="reasoning_logical_fallacy",
        category="Reasoning",
        prompt=(
            "Analyze the following argument for logical fallacies and explain each one found:\n"
            "'Everyone is switching to cryptocurrency, so it must be a good investment. "
            "My friend who invested made money, and experts who disagree are just paid shills "
            "who are afraid of change.'"
        ),
        expected_keywords=[
            "bandwagon", "ad hominem", "anecdotal", "fallacy", "appeal",
            "evidence", "argument", "logical",
        ],
        ground_truth=(
            "Fallacies present: (1) Bandwagon fallacy — popularity doesn't imply value. "
            "(2) Anecdotal evidence — one friend's success is not representative. "
            "(3) Ad hominem / poisoning the well — dismissing experts by impugning motives."
        ),
        turns=1,
        rubrics=["task_completion", "factual_grounding", "coherence"],
        max_score=10,
    ),

    Scenario(
        id="reasoning_fermi",
        category="Reasoning",
        prompt=(
            "Estimate how many piano tuners there are in Chicago. "
            "Use Fermi estimation: state each assumption explicitly, show the multiplication chain, "
            "and give a final order-of-magnitude answer with uncertainty bounds."
        ),
        expected_keywords=[
            "population", "pianos", "household", "tuning", "hours", "tuner",
            "assumption", "estimate", "order of magnitude",
        ],
        ground_truth=(
            "Chicago population ~2.7M, ~2 people/household → ~1.35M households. "
            "~20% own pianos → 270,000 pianos. Each tuned ~1/year. "
            "Tuner does ~4/day, ~250 days/yr → 1,000 tunings/yr each. "
            "270,000 / 1,000 ≈ 270 tuners. Typical range: 100–500."
        ),
        turns=1,
        rubrics=["task_completion", "factual_grounding", "coherence"],
        max_score=10,
    ),

    # -----------------------------------------------------------------------
    # TOOL CALLING  (3 scenarios)
    # -----------------------------------------------------------------------
    Scenario(
        id="tool_weather",
        category="Tool Calling",
        prompt=(
            "Use the weather tool to get the current conditions in Seattle, WA. "
            "Then summarize: temperature, humidity, wind speed, and a plain-English forecast "
            "for the next 24 hours."
        ),
        expected_keywords=[
            "temperature", "humidity", "wind", "forecast", "Seattle", "weather",
        ],
        ground_truth=(
            "Calls weather(location='Seattle, WA'). Returns structured data with temp, humidity, "
            "wind speed. Provides a concise human-readable summary of conditions."
        ),
        turns=2,
        rubrics=["task_completion", "tool_call_accuracy", "coherence"],
        max_score=10,
        tool_required=True,
        expected_tool="weather",
    ),

    Scenario(
        id="tool_compound_interest",
        category="Tool Calling",
        prompt=(
            "Use the calculator tool to compute compound interest: principal=$12,000, "
            "annual rate=6.5%, compounded quarterly, over 15 years. "
            "Show the tool call parameters and the final accumulated value."
        ),
        expected_keywords=[
            "calculator", "principal", "rate", "quarterly", "years",
            "compound", "12000", "6.5", "15",
        ],
        ground_truth=(
            "Calls calculator(formula='compound_interest', principal=12000, rate=0.065, "
            "n=4, t=15). Result ≈ $31,371. Shows accumulated value clearly."
        ),
        turns=1,
        rubrics=["task_completion", "tool_call_accuracy", "instruction_following"],
        max_score=10,
        tool_required=True,
        expected_tool="calculator",
    ),

    Scenario(
        id="tool_search_api",
        category="Tool Calling",
        prompt=(
            "Use the search tool to find the three most recent peer-reviewed papers on "
            "'transformer attention mechanisms' from 2023-2024. "
            "For each result, extract: title, authors, year, and a one-sentence summary."
        ),
        expected_keywords=[
            "search", "transformer", "attention", "paper", "title", "authors",
            "year", "summary", "2023", "2024",
        ],
        ground_truth=(
            "Calls search(query='transformer attention mechanisms peer-reviewed 2023 2024'). "
            "Returns 3 results with structured fields: title, authors, year, abstract summary."
        ),
        turns=2,
        rubrics=["task_completion", "tool_call_accuracy", "citation_precision"],
        max_score=10,
        tool_required=True,
        expected_tool="search",
    ),

    # -----------------------------------------------------------------------
    # SUMMARIZATION  (3 scenarios)
    # -----------------------------------------------------------------------
    Scenario(
        id="summarize_technical_article",
        category="Summarization",
        prompt=(
            "Summarize the following technical article in exactly 3 bullet points, "
            "each under 25 words:\n\n"
            "Kubernetes introduced Horizontal Pod Autoscaling (HPA) to dynamically adjust "
            "the number of pod replicas based on observed CPU utilization or custom metrics. "
            "HPA queries the metrics server every 15 seconds by default, computes a desired "
            "replica count using the formula: desiredReplicas = ceil(currentReplicas * "
            "(currentMetricValue / desiredMetricValue)), and applies changes if the ratio "
            "exceeds a configurable tolerance band (default ±10%). Recent versions support "
            "scaling on memory, latency, and external metrics via the custom metrics API, "
            "enabling fine-grained autoscaling policies tied to business KPIs."
        ),
        expected_keywords=[
            "HPA", "autoscal", "replicas", "metrics", "CPU", "formula", "kubernetes",
        ],
        ground_truth=(
            "3 bullets: (1) HPA dynamically adjusts pod replicas based on metrics. "
            "(2) Uses formula ceil(currentReplicas * ratio) with 10% tolerance. "
            "(3) Supports custom metrics including memory, latency, external sources."
        ),
        turns=1,
        rubrics=["task_completion", "instruction_following", "coherence"],
        max_score=10,
        format_constraint="list:3",
    ),

    Scenario(
        id="summarize_meeting_notes",
        category="Summarization",
        prompt=(
            "Extract and list the action items from these meeting notes:\n\n"
            "Meeting: Q3 Planning — June 12th. Attendees: Sarah, Marcus, Devon.\n"
            "Sarah will finalize the API design document by June 20th. Marcus needs to "
            "schedule load testing for the week of June 24th. Devon agreed to write the "
            "migration runbook and share it with the team before the sprint ends July 1st. "
            "All three will attend the architecture review on June 18th. Sarah also offered "
            "to prepare slides for the stakeholder demo on June 25th."
        ),
        expected_keywords=[
            "Sarah", "Marcus", "Devon", "API", "load testing", "migration",
            "runbook", "demo", "slides", "action",
        ],
        ground_truth=(
            "Action items: (1) Sarah: API design doc by June 20. "
            "(2) Marcus: schedule load testing for June 24 week. "
            "(3) Devon: migration runbook before July 1. "
            "(4) All: architecture review June 18. "
            "(5) Sarah: stakeholder demo slides by June 25."
        ),
        turns=1,
        rubrics=["task_completion", "instruction_following", "coherence"],
        max_score=10,
        format_constraint="list:5",
    ),

    Scenario(
        id="summarize_explain_diff",
        category="Summarization",
        prompt=(
            "You are given a code diff. Summarize what changed in plain English for a "
            "non-technical stakeholder in 2-3 sentences:\n\n"
            "- `user_auth.py`: Added rate limiting (max 5 login attempts per minute per IP)\n"
            "- `session.py`: Session tokens now expire after 30 minutes of inactivity\n"
            "- `audit_log.py`: All login/logout events now written to a database table"
        ),
        expected_keywords=[
            "rate limit", "login", "session", "expire", "audit", "log",
            "security", "attempts",
        ],
        ground_truth=(
            "Plain-English: We added security improvements — login attempts are now rate-limited, "
            "sessions expire after inactivity, and all login events are recorded for auditing."
        ),
        turns=1,
        rubrics=["task_completion", "coherence", "instruction_following"],
        max_score=10,
        format_constraint="word_limit:60",
    ),

    # -----------------------------------------------------------------------
    # INSTRUCTION FOLLOWING  (2 scenarios)
    # -----------------------------------------------------------------------
    Scenario(
        id="instruct_haiku",
        category="Instruction Following",
        prompt=(
            "Write a haiku about databases. "
            "A haiku has exactly 3 lines: 5 syllables, 7 syllables, 5 syllables. "
            "Do not add any explanation before or after the haiku."
        ),
        expected_keywords=[
            "data", "row", "query", "table", "index", "store", "byte", "key", "record",
        ],
        ground_truth=(
            "A valid haiku about databases with 5-7-5 syllable structure, "
            "no extra explanation, database-themed imagery."
        ),
        turns=1,
        rubrics=["instruction_following", "coherence"],
        max_score=10,
        format_constraint="haiku",
    ),

    Scenario(
        id="instruct_pros_cons",
        category="Instruction Following",
        prompt=(
            "List exactly 5 pros and exactly 5 cons of microservices architecture. "
            "Use this format:\n"
            "**Pros:**\n1. ...\n2. ...\n3. ...\n4. ...\n5. ...\n"
            "**Cons:**\n1. ...\n2. ...\n3. ...\n4. ...\n5. ..."
        ),
        expected_keywords=[
            "scalab", "deploy", "fault", "complex", "latency", "team",
            "service", "microservice", "independent", "overhead",
        ],
        ground_truth=(
            "Exactly 5 pros and 5 cons in the specified numbered format with **Pros:** and **Cons:** headers."
        ),
        turns=1,
        rubrics=["instruction_following", "task_completion", "coherence"],
        max_score=10,
        format_constraint="list:5",
    ),

    # -----------------------------------------------------------------------
    # FACTUAL GROUNDING  (3 scenarios)
    # -----------------------------------------------------------------------
    Scenario(
        id="factual_pagerank",
        category="Factual Grounding",
        prompt=(
            "Explain the PageRank algorithm. Cover: the intuition behind it, "
            "the mathematical formula, what the damping factor represents, "
            "and one known limitation of the original algorithm."
        ),
        expected_keywords=[
            "PageRank", "link", "damping factor", "probability", "teleport",
            "formula", "eigenvalue", "matrix", "random walk", "limitation",
        ],
        ground_truth=(
            "PageRank models a random web surfer. PR(u) = (1-d)/N + d*sum(PR(v)/L(v)). "
            "Damping factor d≈0.85 is the probability of following a link vs teleporting. "
            "Limitation: susceptible to link spam / dangling nodes."
        ),
        turns=1,
        rubrics=["task_completion", "factual_grounding", "citation_precision"],
        max_score=10,
    ),

    Scenario(
        id="factual_cap_theorem",
        category="Factual Grounding",
        prompt=(
            "Explain the CAP theorem. Define each of the three properties "
            "(Consistency, Availability, Partition Tolerance), state what the theorem claims, "
            "and give a real-world database example for each of CA, CP, and AP systems."
        ),
        expected_keywords=[
            "Consistency", "Availability", "Partition", "CAP", "theorem",
            "network partition", "trade-off", "CP", "AP", "CA",
        ],
        ground_truth=(
            "CAP: a distributed system can guarantee at most 2 of 3. "
            "CA: PostgreSQL (single-node). CP: HBase, Zookeeper. AP: Cassandra, DynamoDB. "
            "In practice partition tolerance is mandatory, so the real choice is CP vs AP."
        ),
        turns=1,
        rubrics=["task_completion", "factual_grounding", "coherence"],
        max_score=10,
    ),

    Scenario(
        id="factual_transformer",
        category="Factual Grounding",
        prompt=(
            "Describe the original Transformer architecture from 'Attention is All You Need'. "
            "Cover: encoder-decoder structure, multi-head self-attention, positional encoding, "
            "feed-forward sublayers, and the purpose of layer normalization."
        ),
        expected_keywords=[
            "encoder", "decoder", "multi-head", "self-attention", "positional encoding",
            "feed-forward", "layer norm", "residual", "softmax", "query", "key", "value",
        ],
        ground_truth=(
            "Transformer encoder: N=6 layers of multi-head self-attention + FFN with residual+LayerNorm. "
            "Decoder adds cross-attention to encoder outputs. Positional encoding uses sin/cos. "
            "LayerNorm stabilizes training by normalizing per-token across features."
        ),
        turns=1,
        rubrics=["task_completion", "factual_grounding", "coherence"],
        max_score=10,
    ),
]

# Index for fast lookup
SCENARIO_MAP: dict[str, Scenario] = {s.id: s for s in SCENARIOS}


def get_scenario(scenario_id: str) -> Scenario:
    if scenario_id not in SCENARIO_MAP:
        raise KeyError(f"Unknown scenario: {scenario_id!r}")
    return SCENARIO_MAP[scenario_id]


def list_scenarios() -> list[Scenario]:
    return list(SCENARIOS)
