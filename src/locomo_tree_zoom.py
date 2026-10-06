#!/usr/bin/env python3
"""Test an OptChat-style summary tree with a zoom tool on long-range LoCoMo.

Protocol: docs/tree-zoom-protocol.md (pre-registered). Strategies:
A full conversation, B tree view only, C tree + zoom, D BM25 at the median
context budget of C, E tree + zoom + search. Every model call goes through
the Claude Code CLI and is cached on disk by content hash.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from benchmark import Bm25
from graph_benchmark_common import write_result
from locomo_granularity import build_representations


SEED = 20261006
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
ANSWERABLE_CATEGORIES = (4, 1, 2)
PER_CATEGORY = 40
ADVERSARIAL_TOTAL = 20
VIEW_RATIO = 0.10
ZOOM_FANOUT = 16
MAX_STEPS = 8
MAX_OPERATIONS = 12
MAX_NODES_PER_STEP = 3
SEARCH_TOP_K = 5
SUMMARY_BATCH = 30
MODEL = "claude-sonnet-5-5"
STRATEGIES = ("A", "B", "C", "D", "E")
EVIDENCE_PATTERN = re.compile(r"D(\d+):(\d+)")


# --------------------------------------------------------------------------
# Corpus


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


def session_day(date_time: str) -> str:
    """'1:56 pm on 8 May, 2023' -> '8 May 2023'."""
    day = date_time.split(" on ", 1)[-1]
    return day.replace(",", "").strip()


def load_turns(sample: dict[str, Any]) -> list[dict[str, Any]]:
    conversation = sample["conversation"]
    sessions = sorted(
        (
            int(key.split("_")[1])
            for key, value in conversation.items()
            if re.fullmatch(r"session_\d+", key) and isinstance(value, list)
        )
    )
    turns: list[dict[str, Any]] = []
    for number in sessions:
        date = session_day(conversation.get(f"session_{number}_date_time", ""))
        for turn in conversation[f"session_{number}"]:
            text = turn["text"]
            caption = turn.get("blip_caption")
            if caption:
                text = f"{text} [shares a photo: {caption}]"
            turns.append(
                {
                    "id": turn["dia_id"],
                    "session": number,
                    "date": date,
                    "speaker": turn["speaker"],
                    "text": text,
                }
            )
    return turns


def render_turn(turn: dict[str, Any]) -> str:
    return f"[{turn['id']} | {turn['date']}] {turn['speaker']}: {turn['text']}"


def evidence_ids(question: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for raw in question.get("evidence", []) or []:
        for session, index in EVIDENCE_PATTERN.findall(str(raw)):
            found.append(f"D{session}:{index}")
    return found


def session_count(sample: dict[str, Any]) -> int:
    return sum(
        1
        for key, value in sample["conversation"].items()
        if re.fullmatch(r"session_\d+", key) and isinstance(value, list)
    )


def is_long_range(question: dict[str, Any], sessions: int) -> bool:
    ids = evidence_ids(question)
    if not ids:
        return False
    return all(int(EVIDENCE_PATTERN.match(item).group(1)) <= sessions // 2 for item in ids)


def gold_answer(question: dict[str, Any]) -> str:
    if question["category"] == 5:
        return "Not mentioned in the conversation"
    return str(question.get("answer", ""))


def sample_questions(dataset: list[dict[str, Any]], seed: int = SEED) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    per_conversation = PER_CATEGORY // len(dataset) if dataset else 0
    selected: list[dict[str, Any]] = []

    def row(sample: dict[str, Any], index: int, question: dict[str, Any]) -> dict[str, Any]:
        return {
            "qid": f"{sample['sample_id']}-q{index}",
            "sample_id": sample["sample_id"],
            "category": question["category"],
            "category_name": CATEGORY_NAMES[question["category"]],
            "question": question["question"],
            "gold": gold_answer(question),
            "evidence": evidence_ids(question),
        }

    for category in ANSWERABLE_CATEGORIES:
        pools: dict[str, list[dict[str, Any]]] = {}
        for sample in dataset:
            sessions = session_count(sample)
            pool = [
                row(sample, index, question)
                for index, question in enumerate(sample["qa"])
                if question["category"] == category and is_long_range(question, sessions)
            ]
            rng.shuffle(pool)
            pools[sample["sample_id"]] = pool
        chosen = []
        for sample in dataset:
            chosen.extend(pools[sample["sample_id"]][:per_conversation])
        leftovers = [
            item
            for sample in dataset
            for item in pools[sample["sample_id"]][per_conversation:]
        ]
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: PER_CATEGORY - len(chosen)])
        selected.extend(chosen)

    per_conversation_adv = ADVERSARIAL_TOTAL // len(dataset) if dataset else 0
    for sample in dataset:
        pool = [
            row(sample, index, question)
            for index, question in enumerate(sample["qa"])
            if question["category"] == 5
        ]
        rng.shuffle(pool)
        selected.extend(pool[:per_conversation_adv])
    return selected


# --------------------------------------------------------------------------
# Summary tree


class Tree:
    def __init__(self, turns: list[dict[str, Any]]) -> None:
        self.turns = turns
        self.nodes: dict[str, dict[str, Any]] = {}
        self._counter = 0
        self.root = self._build(0, len(turns))

    def _build(self, start: int, end: int) -> str:
        if end - start == 1:
            turn = self.turns[start]
            self.nodes[turn["id"]] = {
                "id": turn["id"],
                "leaf": True,
                "start": start,
                "end": end,
                "height": 0,
                "children": [],
            }
            return turn["id"]
        middle = (start + end) // 2
        left = self._build(start, middle)
        right = self._build(middle, end)
        self._counter += 1
        node_id = f"n{self._counter}"
        self.nodes[node_id] = {
            "id": node_id,
            "leaf": False,
            "start": start,
            "end": end,
            "height": 1 + max(self.nodes[left]["height"], self.nodes[right]["height"]),
            "children": [left, right],
            "summary": None,
        }
        return node_id

    def span(self, node_id: str) -> int:
        node = self.nodes[node_id]
        return node["end"] - node["start"]

    def turns_after(self, node_id: str) -> int:
        return len(self.turns) - self.nodes[node_id]["end"]

    def internal_by_height(self) -> list[list[str]]:
        levels: dict[int, list[str]] = {}
        for node in self.nodes.values():
            if not node["leaf"]:
                levels.setdefault(node["height"], []).append(node["id"])
        return [sorted(levels[h], key=lambda n: self.nodes[n]["start"]) for h in sorted(levels)]

    def line(self, node_id: str) -> str:
        node = self.nodes[node_id]
        if node["leaf"]:
            return render_turn(self.turns[node["start"]])
        first = self.turns[node["start"]]["date"]
        last = self.turns[node["end"] - 1]["date"]
        dates = first if first == last else f"{first} - {last}"
        return f"[{node_id} | {dates} | {self.span(node_id)} turns] {node['summary']}"

    def leaf_ids(self, node_ids: list[str]) -> set[str]:
        return {node_id for node_id in node_ids if self.nodes[node_id]["leaf"]}

    def to_json(self) -> dict[str, Any]:
        return {
            node_id: node["summary"]
            for node_id, node in self.nodes.items()
            if not node["leaf"]
        }

    def load_summaries(self, summaries: dict[str, str]) -> None:
        for node_id, summary in summaries.items():
            self.nodes[node_id]["summary"] = summary


def expand_frontier(
    tree: Tree,
    roots: list[str],
    priority: Callable[[str], float],
    accept: Callable[[list[str], str], bool],
) -> list[str]:
    """Greedy expansion: pop the highest-priority internal node and replace it
    by its children when `accept(frontier, node)` allows it."""
    frontier = list(roots)
    heap = [(-priority(n), tree.nodes[n]["start"], n) for n in roots if not tree.nodes[n]["leaf"]]
    heapq.heapify(heap)
    while heap:
        _, _, node_id = heapq.heappop(heap)
        if not accept(frontier, node_id):
            continue
        position = frontier.index(node_id)
        children = tree.nodes[node_id]["children"]
        frontier[position : position + 1] = children
        for child in children:
            if not tree.nodes[child]["leaf"]:
                heapq.heappush(heap, (-priority(child), tree.nodes[child]["start"], child))
    return frontier


def build_view(tree: Tree, budget: int) -> list[str]:
    tokens = {"total": estimate_tokens(tree.line(tree.root))}

    def accept(frontier: list[str], node_id: str) -> bool:
        delta = sum(estimate_tokens(tree.line(c)) for c in tree.nodes[node_id]["children"])
        delta -= estimate_tokens(tree.line(node_id))
        if tokens["total"] + delta > budget:
            return False
        tokens["total"] += delta
        return True

    return expand_frontier(
        tree,
        [tree.root],
        lambda n: tree.span(n) / (tree.turns_after(n) + 1),
        accept,
    )


def zoom(tree: Tree, node_id: str, fanout: int = ZOOM_FANOUT) -> list[str]:
    if tree.nodes[node_id]["leaf"]:
        return [node_id]

    def accept(frontier: list[str], _: str) -> bool:
        return len(frontier) + 1 <= fanout

    return expand_frontier(tree, [node_id], tree.span, accept)


def render(tree: Tree, node_ids: list[str]) -> str:
    return "\n".join(tree.line(n) for n in node_ids)


# --------------------------------------------------------------------------
# Model calls


class Claude:
    def __init__(self, cache_dir: Path, model: str = MODEL) -> None:
        self.cache_dir = cache_dir
        self.model = model
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.workdir = Path(tempfile.mkdtemp(prefix="tree-zoom-cli-"))
        self.lock = threading.Lock()
        self.calls = 0
        self.cached = 0

    def key(self, system: str, prompt: str) -> str:
        return hashlib.sha256(
            json.dumps([self.model, system, prompt]).encode()
        ).hexdigest()

    def __call__(self, system: str, prompt: str) -> dict[str, Any]:
        path = self.cache_dir / f"{self.key(system, prompt)}.json"
        if path.exists():
            with self.lock:
                self.cached += 1
            return json.loads(path.read_text())
        delay = 15.0
        for attempt in range(40):
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    [
                        "claude",
                        "-p",
                        "--model",
                        self.model,
                        "--system-prompt",
                        system,
                        "--tools",
                        "",
                        "--strict-mcp-config",
                        "--setting-sources",
                        "",
                        "--output-format",
                        "json",
                    ],
                    input=prompt,
                    capture_output=True,
                    text=True,
                    cwd=self.workdir,
                    timeout=600,
                )
                payload = json.loads(completed.stdout)
                if payload.get("is_error") or completed.returncode != 0:
                    raise RuntimeError(str(payload.get("result"))[:300])
            except (subprocess.TimeoutExpired, json.JSONDecodeError, RuntimeError) as error:
                print(f"claude call failed (attempt {attempt + 1}): {error}", file=sys.stderr)
                time.sleep(delay)
                delay = min(delay * 2, 900.0)
                continue
            usage = payload.get("usage", {})
            result = {
                "text": payload.get("result", ""),
                "input_tokens": usage.get("input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "list_cost_usd": payload.get("total_cost_usd"),
                "wall_seconds": round(time.perf_counter() - started, 3),
            }
            path.write_text(json.dumps(result))
            with self.lock:
                self.calls += 1
            return result
        raise RuntimeError("claude call failed after retries")


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


# --------------------------------------------------------------------------
# Tree summaries

SUMMARY_SYSTEM = (
    "You compress conversation memory. For each node you receive the content "
    "of its two children (verbatim turns or earlier summaries). Write ONE line "
    "of at most 30 words that summarizes the node. Keep the most specific "
    "facts: names, places, objects, events, plans, numbers. Do not add dates "
    "(they are shown separately). Return only a JSON object mapping each node "
    "id to its summary."
)


def summarize_tree(tree: Tree, claude: Claude) -> None:
    def child_content(child: str) -> str:
        node = tree.nodes[child]
        if node["leaf"]:
            return render_turn(tree.turns[node["start"]])
        return f"(summary of {tree.span(child)} turns) {node['summary']}"

    for level in tree.internal_by_height():
        pending = list(level)
        rounds = 0
        while pending:
            rounds += 1
            if rounds > 4:
                raise RuntimeError(f"summaries missing after retries: {pending[:5]}")
            batches = [
                pending[offset : offset + SUMMARY_BATCH]
                for offset in range(0, len(pending), SUMMARY_BATCH)
            ]

            def summarize_batch(batch: list[str]) -> list[str]:
                missing: list[str] = []
                blocks = []
                for node_id in batch:
                    left, right = tree.nodes[node_id]["children"]
                    blocks.append(
                        f"## {node_id}\n- {child_content(left)}\n- {child_content(right)}"
                    )
                prompt = (
                    "\n\n".join(blocks)
                    + f"\n\nReturn JSON with exactly these keys: {', '.join(batch)}"
                    + ("" if rounds == 1 else f"\n(retry {rounds})")
                )
                summaries = parse_json(claude(SUMMARY_SYSTEM, prompt)["text"])
                for node_id in batch:
                    summary = summaries.get(node_id)
                    if isinstance(summary, str) and summary.strip():
                        tree.nodes[node_id]["summary"] = " ".join(summary.split())
                    else:
                        missing.append(node_id)
                return missing

            with ThreadPoolExecutor(max_workers=3) as level_pool:
                pending = [n for found in level_pool.map(summarize_batch, batches) for n in found]


# --------------------------------------------------------------------------
# Readers

ANSWER_RULES = (
    "Answer the question about the conversation between two people. Use only "
    "the memory you are given. Each turn is shown as [dialogue id | session "
    "date] speaker: text. Resolve relative time expressions (yesterday, last "
    "week, next month) against the date of the session where they were said. "
    "Give a short answer (a few words or one sentence). If the memory does not "
    "contain the information, answer exactly: Not mentioned in the conversation."
)

DIRECT_SYSTEM = ANSWER_RULES + ' Return only JSON: {"answer": "..."}'

AGENT_TOOLS = {
    "C": (
        'Actions (return exactly one JSON object per step):\n'
        '- {"action": "zoom", "nodes": ["n12", ...]}  opens up to 3 summary '
        "nodes; each shows up to 16 finer lines, down to verbatim turns.\n"
        '- {"action": "answer", "answer": "..."}'
    ),
    "E": (
        'Actions (return exactly one JSON object per step):\n'
        '- {"action": "zoom", "nodes": ["n12", ...]}  opens up to 3 summary '
        "nodes; each shows up to 16 finer lines, down to verbatim turns.\n"
        '- {"action": "search", "query": "..."}  keyword search (BM25) over '
        "all verbatim turns; returns the top 5 turns.\n"
        '- {"action": "answer", "answer": "..."}'
    ),
}


def agent_system(strategy: str) -> str:
    return (
        ANSWER_RULES
        + " Your memory is shown as a view: recent turns verbatim, older turns "
        "folded into one-line summaries of tree nodes (node id, date range, turn "
        "count). Summaries are lossy: open nodes to read the verbatim turns "
        "before relying on a detail.\n\n"
        + AGENT_TOOLS[strategy]
    )


def direct_answer(claude: Claude, context: str, question: str) -> dict[str, Any]:
    prompt = f"# Memory\n{context}\n\n# Question\n{question}"
    response = claude(DIRECT_SYSTEM, prompt)
    answer = parse_json(response["text"]).get("answer") or response["text"].strip()
    return {
        "answer": str(answer),
        "context_tokens": estimate_tokens(context),
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
        "list_cost_usd": response["list_cost_usd"] or 0.0,
        "wall_seconds": response["wall_seconds"],
        "steps": 1,
        "operations": 0,
    }


def agent_answer(
    claude: Claude,
    tree: Tree,
    view: list[str],
    question: str,
    strategy: str,
    searcher: Bm25 | None,
) -> dict[str, Any]:
    view_text = render(tree, view)
    shown: set[str] = tree.leaf_ids(view)
    history: list[str] = []
    tool_tokens = 0
    operations = 0
    totals = {"input_tokens": 0, "output_tokens": 0, "list_cost_usd": 0.0, "wall_seconds": 0.0}
    trace: list[dict[str, Any]] = []
    answer = None
    for step in range(1, MAX_STEPS + 1):
        remaining = MAX_OPERATIONS - operations
        final = step == MAX_STEPS or remaining <= 0
        instruction = (
            "You must answer now with the answer action."
            if final
            else f"Step {step}/{MAX_STEPS}. Operations left: {remaining}. Choose one action."
        )
        prompt = (
            f"# Memory view\n{view_text}\n\n"
            + ("# Tool results\n" + "\n\n".join(history) + "\n\n" if history else "")
            + f"# Question\n{question}\n\n{instruction}"
        )
        response = claude(agent_system(strategy), prompt)
        for key in totals:
            totals[key] += response[key] or 0
        action = parse_json(response["text"])
        kind = action.get("action")
        trace.append({"step": step, "action": action or response["text"][:300]})
        if kind == "answer" or final or kind not in ("zoom", "search"):
            answer = action.get("answer") if kind == "answer" else None
            if answer is None:
                answer = action.get("answer") or response["text"].strip()
            break
        if kind == "search" and strategy == "E" and searcher is not None:
            query = str(action.get("query", ""))
            hits = [row["id"] for row in searcher.search(query, SEARCH_TOP_K)]
            operations += 1
            output = render(tree, hits) or "(no results)"
            shown.update(hits)
            history.append(f'search("{query}"):\n{output}')
        else:
            requested = [
                str(n) for n in (action.get("nodes") or []) if str(n) in tree.nodes
            ][: min(MAX_NODES_PER_STEP, remaining)]
            if not requested:
                history.append("zoom: no valid node id; use ids shown in brackets.")
                operations += 1
                continue
            for node_id in requested:
                lines = zoom(tree, node_id)
                shown.update(tree.leaf_ids(lines))
                history.append(f"zoom({node_id}):\n{render(tree, lines)}")
                operations += 1
        tool_tokens = estimate_tokens("\n\n".join(history))
    return {
        "answer": str(answer),
        "context_tokens": estimate_tokens(view_text) + tool_tokens,
        "steps": len(trace),
        "operations": operations,
        "shown_turns": sorted(shown),
        "trace": trace,
        **totals,
    }


def bm25_context(
    sample: dict[str, Any],
    turns: list[dict[str, Any]],
    question: str,
    budget: int,
) -> tuple[str, set[str]]:
    windows = build_representations(sample, 4, 2)["window4"]
    by_id = {turn["id"]: turn for turn in turns}
    ranked = Bm25(windows).search(question, len(windows))
    selected: list[str] = []
    seen: set[str] = set()
    used = 0
    for row in ranked:
        new = [i for i in row["source_ids"] if i not in seen and i in by_id]
        cost = sum(estimate_tokens(render_turn(by_id[i])) for i in new)
        if not new or used + cost > budget:
            continue
        selected.extend(new)
        seen.update(new)
        used += cost
    order = {turn["id"]: index for index, turn in enumerate(turns)}
    selected.sort(key=order.__getitem__)
    return "\n".join(render_turn(by_id[i]) for i in selected), seen


# --------------------------------------------------------------------------
# Judge

JUDGE_SYSTEM = (
    "You grade answers to questions about a long conversation. You get the "
    "question, the gold answer and a candidate answer. Be lenient on wording "
    "and date format (e.g. '7 May 2023' matches 'May 7, 2023'; a relative "
    "date that resolves to the gold date is correct) and accept extra detail "
    "if it does not contradict the gold answer. Be strict on facts: a wrong "
    "entity, date, number or a missing key part is WRONG. If the gold answer "
    "is 'Not mentioned in the conversation', the candidate is CORRECT only if "
    "it says the information is not available or not mentioned. Return only "
    'JSON: {"verdict": "CORRECT" | "WRONG"}'
)


def judge(claude: Claude, question: str, gold: str, answer: str) -> bool:
    prompt = (
        f"Question: {question}\nGold answer: {gold}\nCandidate answer: {answer}"
    )
    verdict = parse_json(claude(JUDGE_SYSTEM, prompt)["text"]).get("verdict", "")
    return str(verdict).upper() == "CORRECT"


# --------------------------------------------------------------------------
# Statistics


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarize(rows: list[dict[str, Any]], strategies: tuple[str, ...]) -> dict[str, Any]:
    answerable = [r for r in rows if r["category"] != 5]
    adversarial = [r for r in rows if r["category"] == 5]

    def accuracy(items: list[dict[str, Any]], strategy: str) -> float | None:
        if not items:
            return None
        return round(sum(r["results"][strategy]["correct"] for r in items) / len(items), 3)

    def median(items: list[dict[str, Any]], strategy: str, key: str) -> float | None:
        values = [r["results"][strategy].get(key) for r in items]
        values = [v for v in values if v is not None]
        return round(statistics.median(values), 3) if values else None

    series = []
    for strategy in strategies:
        reached = [
            bool(set(r["evidence"]) & set(r["results"][strategy].get("shown_turns", [])))
            for r in answerable
        ]
        series.append(
            {
                "strategy": strategy,
                "answerable_accuracy": accuracy(answerable, strategy),
                "by_category": {
                    CATEGORY_NAMES[c]: accuracy([r for r in answerable if r["category"] == c], strategy)
                    for c in ANSWERABLE_CATEGORIES
                },
                "adversarial_abstention": accuracy(adversarial, strategy),
                "evidence_reach": round(sum(reached) / len(reached), 3) if reached else None,
                "median_context_tokens": median(rows, strategy, "context_tokens"),
                "median_input_tokens": median(rows, strategy, "input_tokens"),
                "median_operations": median(rows, strategy, "operations"),
                "median_steps": median(rows, strategy, "steps"),
                "median_wall_seconds": median(rows, strategy, "wall_seconds"),
                "total_list_cost_usd": round(
                    sum(r["results"][strategy].get("list_cost_usd", 0.0) for r in rows), 2
                ),
            }
        )
    pairs = []
    for left, right in (("A", "B"), ("C", "B"), ("A", "C"), ("C", "D"), ("E", "C"), ("E", "D")):
        if left not in strategies or right not in strategies:
            continue
        b = sum(r["results"][left]["correct"] and not r["results"][right]["correct"] for r in answerable)
        c = sum(r["results"][right]["correct"] and not r["results"][left]["correct"] for r in answerable)
        pairs.append(
            {"left": left, "right": right, "left_only": b, "right_only": c, "p_value": round(mcnemar_exact(b, c), 4)}
        )
    return {
        "answerable_questions": len(answerable),
        "adversarial_questions": len(adversarial),
        "series": series,
        "paired_answerable": pairs,
    }


# --------------------------------------------------------------------------
# Runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/tree-zoom"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, help="smoke test: first N sampled questions")
    parser.add_argument("--conversations", nargs="*", help="restrict to sample ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_bytes = args.dataset.read_bytes()
    dataset = json.loads(dataset_bytes)
    questions = sample_questions(dataset)
    if args.conversations:
        questions = [q for q in questions if q["sample_id"] in args.conversations]
    if args.limit:
        questions = questions[: args.limit]
    needed = {q["sample_id"] for q in questions}
    samples = {s["sample_id"]: s for s in dataset if s["sample_id"] in needed}
    claude = Claude(args.cache_dir)
    pool = ThreadPoolExecutor(max_workers=args.workers)
    started = time.perf_counter()

    turns = {sid: load_turns(sample) for sid, sample in samples.items()}
    trees = {sid: Tree(turns[sid]) for sid in samples}
    list(pool.map(lambda sid: summarize_tree(trees[sid], claude), trees))
    full_text = {sid: "\n".join(render_turn(t) for t in turns[sid]) for sid in samples}
    budgets = {sid: int(estimate_tokens(full_text[sid]) * VIEW_RATIO) for sid in samples}
    views = {sid: build_view(trees[sid], budgets[sid]) for sid in samples}
    searchers = {
        sid: Bm25([{"id": t["id"], "text": f"{t['speaker']}: {t['text']}"} for t in turns[sid]])
        for sid in samples
    }
    print(f"trees ready in {time.perf_counter() - started:.0f}s", file=sys.stderr)

    rows = [{**q, "results": {}} for q in questions]

    def run(row: dict[str, Any], strategy: str) -> None:
        sid = row["sample_id"]
        tree = trees[sid]
        if strategy == "A":
            result = direct_answer(claude, full_text[sid], row["question"])
            result["shown_turns"] = [t["id"] for t in turns[sid]]
        elif strategy == "B":
            result = direct_answer(claude, render(tree, views[sid]), row["question"])
            result["shown_turns"] = sorted(tree.leaf_ids(views[sid]))
        elif strategy in ("C", "E"):
            result = agent_answer(
                claude, tree, views[sid], row["question"], strategy,
                searchers[sid] if strategy == "E" else None,
            )
        else:
            context, shown = bm25_context(samples[sid], turns[sid], row["question"], row["d_budget"])
            result = direct_answer(claude, context, row["question"])
            result["shown_turns"] = sorted(shown)
            result["budget"] = row["d_budget"]
        result["correct"] = judge(claude, row["question"], row["gold"], result["answer"])
        row["results"][strategy] = result

    jobs = [(row, s) for s in ("A", "B", "C", "E") for row in rows]
    for done, _ in enumerate(pool.map(lambda job: run(*job), jobs), 1):
        if done % 20 == 0:
            print(f"{done}/{len(jobs)} answers ({claude.calls} calls)", file=sys.stderr)
    d_budget = int(statistics.median(r["results"]["C"]["context_tokens"] for r in rows))
    for row in rows:
        row["d_budget"] = d_budget
    list(pool.map(lambda row: run(row, "D"), rows))

    payload = {
        "id": "locomo-tree-zoom-20261006",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/tree-zoom-protocol.md",
        "dataset": {
            "name": "LoCoMo locomo10.json",
            "sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        },
        "config": {
            "model": MODEL,
            "view_ratio": VIEW_RATIO,
            "zoom_fanout": ZOOM_FANOUT,
            "max_steps": MAX_STEPS,
            "max_operations": MAX_OPERATIONS,
            "search_top_k": SEARCH_TOP_K,
            "d_budget_tokens": d_budget,
            "seed": SEED,
        },
        "trees": {
            sid: {
                "turns": len(turns[sid]),
                "full_tokens": estimate_tokens(full_text[sid]),
                "view_budget": budgets[sid],
                "view_lines": len(views[sid]),
                "view_verbatim_turns": len(trees[sid].leaf_ids(views[sid])),
                "summaries": trees[sid].to_json(),
            }
            for sid in samples
        },
        "summary": summarize(rows, STRATEGIES),
        "model_calls": claude.calls,
        "cached_calls": claude.cached,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "rows": rows,
    }
    write_result(args.output, payload)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
