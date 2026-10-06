from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark import Bm25
from locomo_tree_zoom import (
    Tree,
    agent_answer,
    bm25_context,
    build_view,
    estimate_tokens,
    evidence_ids,
    is_long_range,
    load_turns,
    mcnemar_exact,
    parse_action,
    parse_json,
    render,
    sample_questions,
    zoom,
)


def make_sample(sessions: int = 4, per_session: int = 8) -> dict:
    conversation = {}
    for s in range(1, sessions + 1):
        conversation[f"session_{s}_date_time"] = f"1:00 pm on {s} May, 2023"
        conversation[f"session_{s}"] = [
            {
                "dia_id": f"D{s}:{i}",
                "speaker": "A" if i % 2 else "B",
                "text": f"session {s} turn {i} topic{s}x{i}",
            }
            for i in range(1, per_session + 1)
        ]
    conversation["session_2"][0]["blip_caption"] = "a red bike"
    return {"sample_id": "conv-x", "conversation": conversation, "qa": []}


def summarized_tree(sample: dict) -> Tree:
    tree = Tree(load_turns(sample))
    for node in tree.nodes.values():
        if not node["leaf"]:
            node["summary"] = f"summary {node['id']}"
    return tree


class FakeClaude:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, system: str, prompt: str) -> dict:
        self.prompts.append(prompt)
        return {
            "text": self.responses.pop(0),
            "input_tokens": 10,
            "output_tokens": 2,
            "list_cost_usd": 0.0,
            "wall_seconds": 0.1,
        }


class TreeZoomTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = make_sample()
        self.tree = summarized_tree(self.sample)

    def test_turns_carry_session_date_and_caption(self) -> None:
        turns = load_turns(self.sample)
        self.assertEqual(len(turns), 32)
        self.assertEqual(turns[0]["date"], "1 May 2023")
        self.assertIn("[shares a photo: a red bike]", turns[8]["text"])

    def test_tree_is_balanced_binary(self) -> None:
        internal = [n for n in self.tree.nodes.values() if not n["leaf"]]
        self.assertEqual(len(internal), 31)
        self.assertEqual(self.tree.span(self.tree.root), 32)
        self.assertEqual(self.tree.nodes[self.tree.root]["height"], 5)

    def test_view_respects_budget_and_favours_recent_turns(self) -> None:
        budget = 250
        view = build_view(self.tree, budget)
        self.assertLessEqual(estimate_tokens(render(self.tree, view)), budget + 5)
        self.assertEqual(
            [self.tree.nodes[n]["start"] for n in view],
            sorted(self.tree.nodes[n]["start"] for n in view),
        )
        self.assertTrue(self.tree.nodes[view[-1]]["leaf"])
        self.assertFalse(self.tree.nodes[view[0]]["leaf"])
        self.assertGreater(self.tree.span(view[0]), self.tree.span(view[-1]))

    def test_zoom_fanout_and_leaf(self) -> None:
        lines = zoom(self.tree, self.tree.root, fanout=4)
        self.assertEqual(len(lines), 4)
        self.assertEqual(sum(self.tree.span(n) for n in lines), 32)
        self.assertEqual(zoom(self.tree, "D1:1"), ["D1:1"])

    def test_evidence_parsing_and_long_range(self) -> None:
        self.assertEqual(evidence_ids({"evidence": ["D8:6; D9:17", "D:11:26"]}), ["D8:6", "D9:17"])
        self.assertTrue(is_long_range({"evidence": ["D1:3", "D2:1"]}, 4))
        self.assertFalse(is_long_range({"evidence": ["D1:3", "D3:1"]}, 4))
        self.assertFalse(is_long_range({"evidence": []}, 4))

    def test_sampling_is_deterministic_and_long_range(self) -> None:
        sample = make_sample()
        sample["qa"] = [
            {"question": "q1", "answer": "a", "evidence": ["D1:1"], "category": 4},
            {"question": "q2", "answer": "a", "evidence": ["D4:1"], "category": 4},
            {"question": "q3", "evidence": ["D1:2"], "category": 5, "adversarial_answer": "x"},
        ]
        first = sample_questions([sample])
        self.assertEqual(first, sample_questions([sample]))
        self.assertEqual([q["question"] for q in first], ["q1", "q3"])
        self.assertEqual(first[1]["gold"], "Not mentioned in the conversation")

    def test_bm25_context_stays_within_budget(self) -> None:
        turns = load_turns(self.sample)
        context, shown = bm25_context(self.sample, turns, "topic3x5", 60)
        self.assertIn("D3:5", shown)
        self.assertLessEqual(sum(estimate_tokens(line) for line in context.splitlines()), 60)

    def test_agent_zooms_then_answers(self) -> None:
        view = build_view(self.tree, 120)
        target = view[0]
        claude = FakeClaude(
            [
                f'{{"action": "zoom", "nodes": ["{target}", "bogus"]}}',
                '```json\n{"action": "answer", "answer": "red bike"}\n```',
            ]
        )
        result = agent_answer(claude, self.tree, view, "What?", "C", None)
        self.assertEqual(result["answer"], "red bike")
        self.assertEqual(result["operations"], 1)
        self.assertIn(f"zoom({target})", claude.prompts[1])
        self.assertGreater(result["context_tokens"], estimate_tokens(render(self.tree, view)))

    def test_agent_search_and_forced_answer(self) -> None:
        turns = load_turns(self.sample)
        searcher = Bm25([{"id": t["id"], "text": t["text"]} for t in turns])
        claude = FakeClaude(['{"action": "search", "query": "topic2x3"}'] * 7 + ['{"answer": "x"}'])
        result = agent_answer(claude, self.tree, build_view(self.tree, 120), "Q", "E", searcher)
        self.assertIn("D2:3", result["shown_turns"])
        self.assertEqual(result["steps"], 8)
        self.assertIn("must answer now", claude.prompts[-1])

    def test_parse_json_and_mcnemar(self) -> None:
        self.assertEqual(parse_json('noise {"a": 1} tail'), {"a": 1})
        self.assertEqual(parse_json("no json"), {})
        self.assertEqual(
            parse_json('<invoke>["n1"]</invoke>\n{"action": "zoom"}\n{"action": "answer"}'),
            {"action": "zoom"},
        )

    def test_parse_action_accepts_native_markup(self) -> None:
        self.assertEqual(
            parse_action('<invoke name="zoom">\n<parameter name="nodes">["n1","n2"]</parameter>\n</invoke>'),
            {"action": "zoom", "nodes": ["n1", "n2"]},
        )
        self.assertEqual(
            parse_action('Because X.\n<invoke name="answer">\n<parameter name="answer">Paris</parameter>\n</invoke>'),
            {"action": "answer", "answer": "Paris"},
        )
        self.assertEqual(parse_action('{"action": "search", "query": "q"} <invoke name="zoom"></invoke>'), {"action": "search", "query": "q"})
        self.assertEqual(parse_action('<invoke name="bash"><parameter name="command">true</parameter></invoke>'), {})

    def test_agent_recovers_from_format_error(self) -> None:
        claude = FakeClaude(["I think I should zoom.", '{"action": "answer", "answer": "ok"}'])
        result = agent_answer(claude, self.tree, build_view(self.tree, 120), "Q", "C", None)
        self.assertEqual(result["answer"], "ok")
        self.assertEqual(result["operations"], 0)
        self.assertIn("format error", claude.prompts[1])
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 6), 0.03125)


if __name__ == "__main__":
    unittest.main()
