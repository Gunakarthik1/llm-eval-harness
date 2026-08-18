"""
Multi-turn scenario runner.

model-a produces high-quality, keyword-rich responses.
model-b produces slightly degraded responses — fewer keywords, shorter answers —
so the regression detection machinery has something real to flag.
"""
from __future__ import annotations

import hashlib
import random
import time
import uuid
from typing import Callable

from harness.models import (
    ExecutionTrace,
    Scenario,
    ToolCall,
    Turn,
)


# ---------------------------------------------------------------------------
# Deterministic response corpora
# ---------------------------------------------------------------------------

# High-quality responses keyed by scenario_id, turn index
_MODEL_A_RESPONSES: dict[str, list[str]] = {
    "code_json_parser": [
        (
            "Here is a complete implementation with type hints and docstring:\n\n"
            "```python\nimport json\nfrom typing import Any\n\n"
            "def parse_nested_json(json_str: str) -> dict:\n"
            "    \"\"\"Flatten a nested JSON string into a dot-separated key dict.\"\"\"\n"
            "    data = json.loads(json_str)\n\n"
            "    def flatten(obj: Any, prefix: str = '') -> dict:\n"
            "        result = {}\n"
            "        if isinstance(obj, dict):\n"
            "            for k, v in obj.items():\n"
            "                new_key = f'{prefix}.{k}' if prefix else k\n"
            "                result.update(flatten(v, new_key))\n"
            "        elif isinstance(obj, list):\n"
            "            for i, v in enumerate(obj):\n"
            "                new_key = f'{prefix}[{i}]'\n"
            "                result.update(flatten(v, new_key))\n"
            "        else:\n"
            "            result[prefix] = obj\n"
            "        return result\n\n"
            "    return flatten(data)\n```\n\n"
            "Usage example:\n```python\njson_str = '{\"a\": {\"b\": {\"c\": 1}}}'\n"
            "result = parse_nested_json(json_str)\n# {'a.b.c': 1}\n```\n\n"
            "The recursive flatten helper handles both dicts and lists."
        ),
        (
            "Great question on edge cases! The function also handles mixed nested structures. "
            "For deeply nested JSON the recursive approach works well, and the docstring "
            "clearly documents the flatten behavior. I'd recommend adding unit tests for "
            "empty dicts and null values as well."
        ),
    ],
    "code_bst": [
        (
            "Here is a complete Binary Search Tree implementation:\n\n"
            "```python\nfrom dataclasses import dataclass\nfrom typing import Optional\n\n"
            "@dataclass\nclass Node:\n    value: int\n    left: Optional['Node'] = None\n"
            "    right: Optional['Node'] = None\n\n"
            "class BinarySearchTree:\n"
            "    def __init__(self):\n        self.root: Optional[Node] = None\n\n"
            "    def insert(self, value: int) -> None:\n"
            "        if not self.root:\n            self.root = Node(value)\n"
            "        else:\n            self._insert_rec(self.root, value)\n\n"
            "    def _insert_rec(self, node: Node, value: int) -> Node:\n"
            "        if value < node.value:\n"
            "            if node.left is None:\n                node.left = Node(value)\n"
            "            else:\n                self._insert_rec(node.left, value)\n"
            "        elif value > node.value:\n"
            "            if node.right is None:\n                node.right = Node(value)\n"
            "            else:\n                self._insert_rec(node.right, value)\n\n"
            "    def search(self, value: int) -> bool:\n"
            "        return self._search_rec(self.root, value)\n\n"
            "    def _search_rec(self, node: Optional[Node], value: int) -> bool:\n"
            "        if node is None:\n            return False\n"
            "        if node.value == value:\n            return True\n"
            "        if value < node.value:\n            return self._search_rec(node.left, value)\n"
            "        return self._search_rec(node.right, value)\n\n"
            "    def inorder(self) -> list:\n"
            "        result = []\n        self._inorder_rec(self.root, result)\n"
            "        return result\n\n"
            "    def _inorder_rec(self, node: Optional[Node], result: list) -> None:\n"
            "        if node:\n            self._inorder_rec(node.left, result)\n"
            "            result.append(node.value)\n"
            "            self._inorder_rec(node.right, result)\n\n"
            "    def delete(self, value: int) -> None:\n"
            "        self.root = self._delete_rec(self.root, value)\n\n"
            "    def _delete_rec(self, node: Optional[Node], value: int) -> Optional[Node]:\n"
            "        if node is None:\n            return None\n"
            "        if value < node.value:\n"
            "            node.left = self._delete_rec(node.left, value)\n"
            "        elif value > node.value:\n"
            "            node.right = self._delete_rec(node.right, value)\n"
            "        else:\n"
            "            if node.left is None:\n                return node.right\n"
            "            if node.right is None:\n                return node.left\n"
            "            # Two children: replace with inorder successor\n"
            "            successor = node.right\n"
            "            while successor.left:\n                successor = successor.left\n"
            "            node.value = successor.value\n"
            "            node.right = self._delete_rec(node.right, successor.value)\n"
            "        return node\n```"
        ),
        (
            "The delete method handles all three cases: leaf node removal, single child, "
            "and two children (replaced by inorder successor). The dataclass for Node keeps "
            "the code clean. Inorder traversal yields a sorted list as expected from a BST."
        ),
    ],
    "code_async_http": [
        (
            "```python\nimport httpx\nimport asyncio\nfrom typing import Any\n\n"
            "async def fetch_all(urls: list[str]) -> list[dict]:\n"
            "    \"\"\"Fetch all URLs concurrently using httpx and asyncio.gather.\"\"\"\n"
            "    async with httpx.AsyncClient(timeout=10.0) as client:\n"
            "        tasks = [_fetch_one(client, url) for url in urls]\n"
            "        return await asyncio.gather(*tasks)\n\n"
            "async def _fetch_one(client: httpx.AsyncClient, url: str) -> dict[str, Any]:\n"
            "    try:\n        response = await client.get(url)\n"
            "        response.raise_for_status()\n"
            "        return {'url': url, 'status': response.status_code, 'body': response.text}\n"
            "    except httpx.HTTPStatusError as e:\n"
            "        return {'url': url, 'status': e.response.status_code, 'body': str(e)}\n"
            "    except Exception as e:\n"
            "        return {'url': url, 'status': 0, 'body': f'Error: {e}'}\n```\n\n"
            "This uses `asyncio.gather` for concurrency and handles both HTTP errors "
            "and network-level exceptions per request without crashing the batch."
        )
    ],
    "code_lru_cache": [
        (
            "Here's an O(1) LRU Cache using Python's OrderedDict:\n\n"
            "```python\nfrom collections import OrderedDict\n\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity: int):\n"
            "        self.capacity = capacity\n"
            "        self.cache: OrderedDict = OrderedDict()\n\n"
            "    def get(self, key: int) -> int:\n"
            "        if key not in self.cache:\n            return -1\n"
            "        self.cache.move_to_end(key)  # most recently used\n"
            "        return self.cache[key]\n\n"
            "    def put(self, key: int, value: int) -> None:\n"
            "        if key in self.cache:\n            self.cache.move_to_end(key)\n"
            "        self.cache[key] = value\n"
            "        if len(self.cache) > self.capacity:\n"
            "            self.cache.popitem(last=False)  # evict LRU\n```\n\n"
            "OrderedDict maintains insertion order and supports O(1) move_to_end "
            "and popitem operations. This satisfies the LeetCode 146 interface exactly. "
            "For a pure doubly-linked list + hashmap approach without OrderedDict, "
            "you'd maintain head/tail sentinel nodes."
        ),
        (
            "The `move_to_end` operation is O(1) thanks to the doubly-linked list "
            "inside OrderedDict. `popitem(last=False)` removes from the front (LRU end). "
            "Capacity is enforced after every put that adds a new key."
        ),
    ],
    "reasoning_math_compound": [
        (
            "Let me solve this step by step.\n\n"
            "**Step 1: Compound interest for 10 years (monthly compounding)**\n"
            "Formula: A = P(1 + r/n)^(nt)\n"
            "P = $5,000, r = 0.07, n = 12, t = 10\n"
            "A1 = 5000 × (1 + 0.07/12)^(12×10)\n"
            "A1 = 5000 × (1.005833...)^120\n"
            "A1 = 5000 × 2.00966 ≈ **$10,048.31**\n\n"
            "**Step 2: Withdrawal of half the balance**\n"
            "Remaining = 10,048.31 / 2 = **$5,024.16**\n\n"
            "**Step 3: Compound interest for 5 more years at 5% monthly**\n"
            "P = $5,024.16, r = 0.05, n = 12, t = 5\n"
            "A2 = 5024.16 × (1 + 0.05/12)^60\n"
            "A2 = 5024.16 × 1.28336 ≈ **$6,447.53**\n\n"
            "**Final balance: approximately $6,447.53**\n\n"
            "Using the compound interest formula at each stage, "
            "with monthly compounding applied consistently."
        )
    ],
    "reasoning_logical_fallacy": [
        (
            "This argument contains three logical fallacies:\n\n"
            "**1. Bandwagon Fallacy (Ad Populum)**\n"
            "'Everyone is switching to cryptocurrency' — popularity does not establish value "
            "or safety. Many people can be wrong simultaneously.\n\n"
            "**2. Anecdotal Evidence**\n"
            "'My friend who invested made money' — a single data point cannot represent "
            "the general outcome. Survivorship bias likely applies here.\n\n"
            "**3. Ad Hominem / Poisoning the Well**\n"
            "'Experts who disagree are paid shills afraid of change' — this dismisses "
            "counter-arguments by attacking the arguer's motives rather than addressing "
            "the substance of their evidence. This is also a form of the genetic fallacy.\n\n"
            "A sound argument would require objective evidence about risk-adjusted returns, "
            "volatility, and regulatory factors — independent of who endorses it."
        )
    ],
    "reasoning_fermi": [
        (
            "**Fermi Estimation: Piano Tuners in Chicago**\n\n"
            "**Assumptions:**\n"
            "1. Chicago metro population: ~2.7 million people\n"
            "2. Average household size: ~2.5 people → ~1.08 million households\n"
            "3. Fraction of households with a piano: ~15% → ~162,000 pianos\n"
            "4. Average tunings per piano per year: 1\n"
            "5. Tuner works 250 days/year, tunes ~4 pianos per day → 1,000 tunings/yr per tuner\n\n"
            "**Calculation chain:**\n"
            "Pianos needing tuning per year = 162,000\n"
            "Tuners needed = 162,000 / 1,000 = **~162 tuners**\n\n"
            "**Order-of-magnitude answer: ~100–200 piano tuners**\n"
            "Uncertainty bounds: 50 (if only 8% own pianos) to 400 (if 25% do and tuning is biannual)\n\n"
            "This is consistent with real estimates for similarly sized cities."
        )
    ],
    "tool_weather": [
        (
            "I'll call the weather tool now.\n\n"
            "[Tool call: weather(location='Seattle, WA')]\n\n"
            "**Weather in Seattle, WA:**\n"
            "- Temperature: 58°F (14°C)\n"
            "- Humidity: 78%\n"
            "- Wind speed: 12 mph from the SW\n"
            "- Conditions: Overcast with light drizzle\n\n"
            "**24-hour forecast:** Expect continued overcast skies with occasional rain showers "
            "throughout the morning. Afternoon may see brief clearing, but temperatures remain "
            "cool in the mid-50s. Wind picking up to 18 mph by evening."
        ),
        (
            "The weather tool returned current conditions successfully. "
            "Seattle is experiencing typical Pacific Northwest weather — cool, humid, "
            "and overcast. Residents should bring a rain jacket for the next 24 hours."
        ),
    ],
    "tool_compound_interest": [
        (
            "I'll use the calculator tool with the compound interest formula.\n\n"
            "[Tool call: calculator(formula='compound_interest', principal=12000, "
            "rate=0.065, n=4, t=15)]\n\n"
            "**Result:** A = P(1 + r/n)^(nt)\n"
            "A = 12000 × (1 + 0.065/4)^(4×15)\n"
            "A = 12000 × (1.01625)^60\n"
            "A = 12000 × 2.6144 ≈ **$31,372.52**\n\n"
            "With a principal of $12,000, compounded quarterly at 6.5% over 15 years, "
            "the accumulated value is approximately **$31,372.52** — representing "
            "a gain of $19,372.52 (161% growth)."
        )
    ],
    "tool_search_api": [
        (
            "Calling the search tool for recent transformer papers.\n\n"
            "[Tool call: search(query='transformer attention mechanisms peer-reviewed 2023 2024')]\n\n"
            "**Results:**\n\n"
            "1. **Flash Attention 2: Faster Attention with Better Parallelism** (2023)\n"
            "   Authors: Dao et al. — Proposes IO-aware exact attention with 2-4x speedup "
            "over FlashAttention on A100 GPUs.\n\n"
            "2. **Longformer: The Long-Document Transformer** (2023 extended)\n"
            "   Authors: Beltagy et al. — Introduces sliding window + global attention "
            "for linear-complexity processing of long documents.\n\n"
            "3. **GQA: Training Generalized Multi-Query Transformer Models** (2023)\n"
            "   Authors: Ainslie et al. — Grouped-query attention reduces KV cache size "
            "while maintaining model quality at scale."
        ),
        (
            "The search tool successfully retrieved three relevant 2023-2024 papers on "
            "transformer attention mechanisms. Each entry includes title, authors, year, "
            "and a concise one-sentence summary of the contribution."
        ),
    ],
    "summarize_technical_article": [
        (
            "• HPA automatically scales Kubernetes pod replicas based on CPU or custom metrics every 15 seconds.\n"
            "• Scaling uses formula ceil(currentReplicas × ratio) with a ±10% tolerance band.\n"
            "• Supports memory, latency, and external metrics via the custom metrics API."
        )
    ],
    "summarize_meeting_notes": [
        (
            "**Action Items:**\n"
            "1. Sarah — Finalize API design document by June 20th\n"
            "2. Marcus — Schedule load testing for the week of June 24th\n"
            "3. Devon — Write and share migration runbook before July 1st\n"
            "4. All (Sarah, Marcus, Devon) — Attend architecture review on June 18th\n"
            "5. Sarah — Prepare slides for stakeholder demo on June 25th"
        )
    ],
    "summarize_explain_diff": [
        (
            "We made three security improvements to the authentication system. "
            "Login attempts are now rate-limited to prevent brute-force attacks, "
            "sessions automatically expire after inactivity, "
            "and all login and logout events are recorded in an audit log for compliance and monitoring."
        )
    ],
    "instruct_haiku": [
        (
            "Rows of data sleep\n"
            "Queries wake each record gently\n"
            "Index finds the key"
        )
    ],
    "instruct_pros_cons": [
        (
            "**Pros:**\n"
            "1. Independent deployment — teams can release services without coordinating a monolith deploy\n"
            "2. Technology flexibility — each service can use the best language or framework for its task\n"
            "3. Fault isolation — a failure in one service does not cascade to bring down the whole system\n"
            "4. Horizontal scalability — individual services scale independently based on their load\n"
            "5. Smaller, focused codebases — each team owns a bounded context, improving productivity\n\n"
            "**Cons:**\n"
            "1. Distributed system complexity — network calls replace in-process function calls, adding latency\n"
            "2. Operational overhead — requires orchestration, service discovery, and monitoring for each service\n"
            "3. Data consistency challenges — maintaining consistency across services requires eventual consistency patterns\n"
            "4. Increased testing surface — integration and contract testing become critical and expensive\n"
            "5. Service-to-service security — every inter-service call must be authenticated and authorized"
        )
    ],
    "factual_pagerank": [
        (
            "**PageRank Algorithm**\n\n"
            "**Intuition:** PageRank models a random web surfer who follows links at random. "
            "Pages linked to by many high-authority pages are themselves authoritative. "
            "The algorithm is a random walk over the web graph.\n\n"
            "**Formula:**\n"
            "PR(u) = (1 − d) / N + d × Σ (PR(v) / L(v)) for all v linking to u\n"
            "Where d is the damping factor, N is total pages, L(v) is out-degree of v.\n\n"
            "**Damping factor (d ≈ 0.85):** The probability that the surfer follows a link "
            "rather than teleporting to a random page. It prevents rank sinks and ensures convergence. "
            "The (1−d)/N term represents the teleportation probability distributed uniformly.\n\n"
            "**Limitation:** The original algorithm is susceptible to link spam — "
            "artificially created link farms can inflate page rank. "
            "It also struggles with dangling nodes (pages with no outbound links) "
            "and is computed via eigenvalue decomposition of the adjacency matrix, "
            "which is expensive at web scale."
        )
    ],
    "factual_cap_theorem": [
        (
            "**CAP Theorem (Brewer's Theorem)**\n\n"
            "A distributed system can satisfy at most 2 of the following 3 properties:\n\n"
            "- **Consistency (C):** Every read receives the most recent write or an error. "
            "All nodes see the same data at the same time.\n"
            "- **Availability (A):** Every request receives a response (not necessarily the most recent data). "
            "The system remains operational.\n"
            "- **Partition Tolerance (P):** The system continues operating despite network partition "
            "(message loss or delay between nodes).\n\n"
            "**The Claim:** In the presence of a network partition, you must choose between "
            "consistency and availability. Since partitions are inevitable in real networks, "
            "the practical trade-off is **CP vs AP**.\n\n"
            "**Real-world examples:**\n"
            "- **CA:** PostgreSQL (single node — no partition tolerance needed)\n"
            "- **CP:** HBase, ZooKeeper — sacrifices availability to guarantee consistent reads\n"
            "- **AP:** Cassandra, DynamoDB — remains available under partition, accepts stale reads"
        )
    ],
    "factual_transformer": [
        (
            "**Transformer Architecture ('Attention is All You Need', Vaswani et al. 2017)**\n\n"
            "**Encoder-Decoder Structure:**\n"
            "The encoder maps input tokens to continuous representations. "
            "The decoder auto-regressively generates output tokens, attending to encoder outputs.\n"
            "Both have N=6 identical layers.\n\n"
            "**Multi-Head Self-Attention:**\n"
            "Attention(Q,K,V) = softmax(QK^T / √d_k) × V\n"
            "Multi-head runs h=8 attention heads in parallel, projecting query, key, value "
            "into subspaces, then concatenating. This lets the model attend to different "
            "representation subspaces simultaneously.\n\n"
            "**Positional Encoding:**\n"
            "Since attention is permutation-invariant, sinusoidal positional encodings "
            "(sin/cos at different frequencies) inject position information.\n\n"
            "**Feed-Forward Sublayers:**\n"
            "Each layer has a position-wise FFN: FFN(x) = max(0, xW1 + b1)W2 + b2 "
            "(two linear transformations with ReLU).\n\n"
            "**Layer Normalization:**\n"
            "Applied after each sublayer (residual connection first). "
            "LayerNorm normalizes across features per token, stabilizing gradients and "
            "accelerating training convergence."
        )
    ],
}

# Model-B uses shorter, keyword-sparse variants of the same answers
_MODEL_B_RESPONSES: dict[str, list[str]] = {
    "code_json_parser": [
        (
            "Here's a function to flatten JSON:\n\n"
            "```python\nimport json\n\n"
            "def parse_nested_json(json_str):\n"
            "    data = json.loads(json_str)\n"
            "    result = {}\n"
            "    for k, v in data.items():\n"
            "        if isinstance(v, dict):\n"
            "            for nested_k, nested_v in v.items():\n"
            "                result[f'{k}.{nested_k}'] = nested_v\n"
            "        else:\n"
            "            result[k] = v\n"
            "    return result\n```\n\n"
            "This handles one level of nesting. Deep nesting would need more work."
        ),
        "You could extend it for deeper nesting with recursion if needed.",
    ],
    "code_bst": [
        (
            "Binary search tree implementation:\n\n"
            "```python\nclass Node:\n    def __init__(self, val):\n"
            "        self.value = val\n        self.left = None\n        self.right = None\n\n"
            "class BinarySearchTree:\n    def __init__(self):\n        self.root = None\n\n"
            "    def insert(self, value):\n"
            "        if not self.root:\n            self.root = Node(value)\n"
            "        else:\n            curr = self.root\n"
            "            while True:\n"
            "                if value < curr.value:\n"
            "                    if curr.left is None:\n                        curr.left = Node(value); break\n"
            "                    curr = curr.left\n"
            "                else:\n"
            "                    if curr.right is None:\n                        curr.right = Node(value); break\n"
            "                    curr = curr.right\n\n"
            "    def search(self, value):\n"
            "        curr = self.root\n"
            "        while curr:\n"
            "            if curr.value == value: return True\n"
            "            curr = curr.left if value < curr.value else curr.right\n"
            "        return False\n\n"
            "    def inorder(self):\n"
            "        result, stack, curr = [], [], self.root\n"
            "        while curr or stack:\n"
            "            while curr: stack.append(curr); curr = curr.left\n"
            "            curr = stack.pop()\n"
            "            result.append(curr.value)\n"
            "            curr = curr.right\n"
            "        return result\n\n"
            "    def delete(self, value):\n"
            "        pass  # deletion left as exercise\n```"
        ),
        "The delete method is complex and left unimplemented for now.",
    ],
    "code_async_http": [
        (
            "```python\nimport asyncio\nimport httpx\n\n"
            "async def fetch_all(urls):\n"
            "    results = []\n"
            "    async with httpx.AsyncClient() as client:\n"
            "        for url in urls:\n"
            "            try:\n"
            "                r = await client.get(url)\n"
            "                results.append({'url': url, 'status': r.status_code, 'body': r.text})\n"
            "            except Exception as e:\n"
            "                results.append({'url': url, 'status': 0, 'body': str(e)})\n"
            "    return results\n```\n\n"
            "Note: this is sequential, not concurrent — asyncio.gather would be needed for concurrency."
        )
    ],
    "code_lru_cache": [
        (
            "LRU Cache:\n\n"
            "```python\nclass LRUCache:\n"
            "    def __init__(self, capacity):\n"
            "        self.capacity = capacity\n"
            "        self.cache = {}\n        self.order = []\n\n"
            "    def get(self, key):\n"
            "        if key not in self.cache:\n            return -1\n"
            "        self.order.remove(key)\n        self.order.append(key)\n"
            "        return self.cache[key]\n\n"
            "    def put(self, key, value):\n"
            "        if key in self.cache:\n            self.order.remove(key)\n"
            "        elif len(self.cache) >= self.capacity:\n"
            "            lru = self.order.pop(0)\n            del self.cache[lru]\n"
            "        self.cache[key] = value\n        self.order.append(key)\n```\n\n"
            "This works but get/put are O(n) due to list.remove()."
        ),
        "O(1) requires OrderedDict or a doubly-linked list approach instead.",
    ],
    "reasoning_math_compound": [
        (
            "Compound interest calculation:\n\n"
            "Step 1: 5000 at 7% for 10 years monthly → approximately $10,000\n"
            "Step 2: Withdraw half → ~$5,000 remaining\n"
            "Step 3: 5% for 5 more years → approximately $6,400\n\n"
            "Final balance is roughly $6,400. "
            "The compound interest formula A = P(1+r/n)^(nt) was applied at each stage."
        )
    ],
    "reasoning_logical_fallacy": [
        (
            "The argument has some logical issues:\n\n"
            "1. Bandwagon appeal — just because everyone does it doesn't mean it's good.\n"
            "2. Using one friend's experience as proof is not strong evidence.\n"
            "3. Calling critics 'shills' doesn't address their arguments.\n\n"
            "A stronger argument would use data and objective analysis."
        )
    ],
    "reasoning_fermi": [
        (
            "Estimating piano tuners in Chicago:\n\n"
            "Chicago has about 2.7 million people. Maybe 10-15% of households have pianos. "
            "Each piano gets tuned roughly once a year. A tuner can do a few per day.\n\n"
            "Rough estimate: somewhere between 100 and 500 piano tuners.\n"
            "Order of magnitude: ~200."
        )
    ],
    "tool_weather": [
        (
            "Let me check the weather for Seattle.\n\n"
            "weather(location='Seattle')\n\n"
            "Seattle weather: around 58°F, cloudy, some rain. "
            "Wind is moderate. Expect more rain over the next day."
        ),
        "Seattle weather retrieved. Typical overcast conditions expected to continue.",
    ],
    "tool_compound_interest": [
        (
            "Using the calculator:\n\n"
            "calculate(principal=12000, rate=0.065, years=15)\n\n"
            "Result: approximately $31,000 after 15 years of compound growth at 6.5%.\n\n"
            "Note: exact compounding frequency affects the final number."
        )
    ],
    "tool_search_api": [
        (
            "search('transformer attention 2023 2024')\n\n"
            "Found some recent papers:\n\n"
            "1. Flash Attention 2 — faster attention computation (2023)\n"
            "2. Longformer — efficient attention for long documents\n"
            "3. GQA — grouped query attention\n\n"
            "These are significant 2023-2024 contributions to attention mechanisms."
        ),
        "Search returned relevant results on transformer attention research.",
    ],
    "summarize_technical_article": [
        (
            "• Kubernetes HPA scales pods based on CPU metrics.\n"
            "• It polls every 15 seconds and uses a scaling formula.\n"
            "• Custom metrics like memory are also supported."
        )
    ],
    "summarize_meeting_notes": [
        (
            "Action items from the meeting:\n"
            "1. Sarah — API document\n"
            "2. Marcus — load testing\n"
            "3. Devon — runbook\n"
            "4. All — attend review\n"
            "5. Sarah — demo slides"
        )
    ],
    "summarize_explain_diff": [
        (
            "Three security changes were made: rate limiting on login, "
            "session expiration, and audit logging of login events."
        )
    ],
    "instruct_haiku": [
        (
            "Data in tables sit\n"
            "Queries search through every row\n"
            "Results come back fast"
        )
    ],
    "instruct_pros_cons": [
        (
            "**Pros:**\n"
            "1. Services can be deployed independently\n"
            "2. Teams can pick their own technology stack\n"
            "3. Faults are isolated to individual services\n"
            "4. Each service scales independently\n"
            "5. Smaller codebases per team\n\n"
            "**Cons:**\n"
            "1. Network complexity increases\n"
            "2. Harder to operate and monitor\n"
            "3. Data consistency is tricky\n"
            "4. More testing required\n"
            "5. Security between services is complex"
        )
    ],
    "factual_pagerank": [
        (
            "PageRank ranks web pages by importance based on incoming links.\n\n"
            "The formula considers the rank of linking pages weighted by their out-degree.\n"
            "The damping factor (usually 0.85) accounts for the probability of following a link.\n\n"
            "A known limitation is that it can be gamed with link farms."
        )
    ],
    "factual_cap_theorem": [
        (
            "CAP theorem says distributed systems can only guarantee 2 of 3 properties:\n"
            "- Consistency: all nodes have same data\n"
            "- Availability: system always responds\n"
            "- Partition tolerance: works despite network issues\n\n"
            "Examples: PostgreSQL (CA), HBase (CP), Cassandra (AP)."
        )
    ],
    "factual_transformer": [
        (
            "The Transformer uses encoder-decoder architecture with self-attention.\n\n"
            "Multi-head attention lets the model attend to different positions. "
            "Positional encoding adds position information. "
            "Feed-forward layers process each position. "
            "Layer norm stabilizes training."
        )
    ],
}


def _get_response(model: str, scenario_id: str, turn_index: int) -> str:
    """Return a deterministic simulated response for the given model/scenario/turn."""
    corpus = _MODEL_A_RESPONSES if model == "model-a" else _MODEL_B_RESPONSES
    turns_for_scenario = corpus.get(scenario_id, [])
    if not turns_for_scenario:
        # Fallback generic response
        if model == "model-a":
            return (
                f"Here is a thorough response to the scenario '{scenario_id}'. "
                "I have addressed all the key requirements, included relevant keywords, "
                "and structured the answer clearly with examples and explanations."
            )
        else:
            return (
                f"Response to '{scenario_id}'. "
                "The main points are covered briefly."
            )
    idx = min(turn_index, len(turns_for_scenario) - 1)
    return turns_for_scenario[idx]


def _simulate_tool_calls(scenario: "Scenario", model: str) -> list[ToolCall]:
    """If the scenario requires a tool call, generate a simulated one."""
    if not scenario.tool_required:
        return []
    tool_name = scenario.expected_tool or "unknown_tool"
    if model == "model-a":
        # Correct tool call
        if tool_name == "weather":
            params = {"location": "Seattle, WA"}
        elif tool_name == "calculator":
            params = {"formula": "compound_interest", "principal": 12000, "rate": 0.065, "n": 4, "t": 15}
        elif tool_name == "search":
            params = {"query": "transformer attention mechanisms peer-reviewed 2023 2024"}
        else:
            params = {"query": "general search"}
        return [ToolCall(tool_name=tool_name, parameters=params, result="success", success=True)]
    else:
        # Slightly wrong parameters
        if tool_name == "weather":
            params = {"location": "Seattle"}   # missing state
        elif tool_name == "calculator":
            params = {"principal": 12000, "rate": 0.065, "years": 15}  # missing n, wrong key
        elif tool_name == "search":
            params = {"query": "transformer attention 2023 2024"}  # missing "peer-reviewed"
        else:
            params = {}
        return [ToolCall(tool_name=tool_name, parameters=params, result="partial", success=True)]


class ScenarioRunner:
    """Executes multi-turn scenario dialogues and captures execution traces."""

    def __init__(self, base_latency_ms: float = 120.0, token_rate: float = 4.0):
        self.base_latency_ms = base_latency_ms
        self.token_rate = token_rate  # chars per token approximation

    def _build_turn_prompt(self, scenario: "Scenario", turn_number: int) -> str:
        if turn_number == 0:
            return scenario.prompt
        # Follow-up prompts for multi-turn scenarios
        follow_ups = [
            "Can you elaborate on the edge cases?",
            "How would you test this solution?",
            "What are the performance implications?",
            "Can you simplify the explanation?",
        ]
        return follow_ups[(turn_number - 1) % len(follow_ups)]

    def run_scenario(
        self,
        scenario: "Scenario",
        model_name: str,
        run_id: str = "",
    ) -> ExecutionTrace:
        """Execute a multi-turn scenario and return the full execution trace."""
        if not run_id:
            run_id = str(uuid.uuid4())[:8]

        # Seed RNG deterministically for reproducibility
        seed_str = f"{scenario.id}:{model_name}"
        rng = random.Random(int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31))

        turns: list[Turn] = []
        all_tool_calls: list[ToolCall] = []
        total_tokens = 0
        start_time = time.time()

        for turn_idx in range(scenario.turns):
            user_input = self._build_turn_prompt(scenario, turn_idx)
            agent_response = _get_response(model_name, scenario.id, turn_idx)

            # Simulate tool calls on first turn if applicable
            turn_tool_calls: list[ToolCall] = []
            if turn_idx == 0 and scenario.tool_required:
                turn_tool_calls = _simulate_tool_calls(scenario, model_name)
                all_tool_calls.extend(turn_tool_calls)

            # Simulate timing
            response_chars = len(agent_response)
            tokens = max(10, int(response_chars / self.token_rate))
            jitter = rng.uniform(0.85, 1.15)
            latency = self.base_latency_ms * jitter + tokens * 0.5

            total_tokens += tokens
            turns.append(Turn(
                turn_number=turn_idx + 1,
                user_input=user_input,
                agent_response=agent_response,
                response_time_ms=round(latency, 1),
                tokens_used=tokens,
                tool_calls=turn_tool_calls,
            ))

        duration_ms = (time.time() - start_time) * 1000 + sum(t.response_time_ms for t in turns)

        return ExecutionTrace(
            run_id=run_id,
            scenario_id=scenario.id,
            model=model_name,
            turns=turns,
            total_tokens=total_tokens,
            duration_ms=round(duration_ms, 1),
            tool_calls=all_tool_calls,
        )

    def run_batch(
        self,
        scenarios: list["Scenario"],
        model: str,
        n_runs: int = 3,
        run_id: str = "",
    ) -> list[ExecutionTrace]:
        """Run each scenario n_runs times, returning all traces."""
        if not run_id:
            run_id = str(uuid.uuid4())[:8]
        traces: list[ExecutionTrace] = []
        for scenario in scenarios:
            for i in range(n_runs):
                trace = self.run_scenario(scenario, model, run_id=f"{run_id}-{i}")
                traces.append(trace)
        return traces
