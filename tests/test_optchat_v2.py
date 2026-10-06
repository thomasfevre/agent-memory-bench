from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optchat_v2 import (
    ABSTAIN,
    NODE,
    SCALE,
    Memory,
    Tree,
    agent_answer,
    bm25_context,
    build_tree,
    check_quote,
    cohen_kappa,
    fold,
    holm,
    message_searcher,
    nbytes,
    parse_action,
    parse_lines,
    render_context,
    render_view,
    zoom,
)


def make_memory(count: int, long_every: int = 3) -> Memory:
    messages = []
    for i in range(count):
        text = f"message {i} about topic{i}"
        if i % long_every == 0:
            text += " " + "x" * 600
        messages.append({"kind": "user" if i % 2 == 0 else "talk", "text": text, "date": f"2023/05/{1 + i // 10:02d}"})
    return Memory("mem", messages)


class FakeCli:
    """Answers compactor batches with short lines and agent steps from a script."""

    def __init__(self, agent_replies: list[str] | None = None) -> None:
        self.agent_replies = list(agent_replies or [])
        self.prompts: list[str] = []

    def __call__(self, system: str, prompt: str, model: str | None = None, timeout: int = 0) -> dict:
        self.prompts.append(prompt)
        if system.startswith("You write the memory of OptChat"):
            steps = prompt.count("\nStep ") or 1
            lines = [f"line {len(self.prompts)}-{k}" for k in range(steps)]
            text = lines[0] if steps == 1 else json.dumps(lines)
        else:
            text = self.agent_replies.pop(0)
        return {"text": text, "input_tokens": 1, "output_tokens": 1, "list_cost_usd": 0.0, "wall_seconds": 0.0}


class OptChatV2Test(unittest.TestCase):
    def test_scale_is_exactly_node_bytes(self) -> None:
        self.assertEqual(nbytes(SCALE), NODE)

    def test_tree_addressing_and_free_nodes(self) -> None:
        tree = Tree(make_memory(10))
        self.assertTrue(tree.exists((3, 0)))
        self.assertFalse(tree.exists((3, 1)))
        self.assertEqual(tree.label((2, 1)), "4+4")
        self.assertEqual(tree.free_text((0, 1)), "talk: message 1 about topic1")
        self.assertIsNone(tree.free_text((0, 0)))
        tree.text[(0, 1)] = "a"
        tree.text[(0, 0)] = "b"
        self.assertEqual(tree.free_text((1, 0)), "b\na")

    def test_build_tree_is_complete_and_has_no_ids_in_compactor_input(self) -> None:
        memory = make_memory(37)
        tree = Tree(memory)
        cli = FakeCli()
        stats = build_tree(cli, tree, budget=200)
        self.assertTrue(tree.complete())
        self.assertGreater(stats["calls"], 0)
        for prompt in cli.prompts:
            self.assertNotRegex(prompt.split("For scale")[0], r"\d+\+\d+\|")
        view = fold(tree, len(memory), 200)
        covered = [i for node in view for i in range(tree.start(node), tree.end(node))]
        self.assertEqual(covered, list(range(len(memory))))

    def test_fold_merges_oldest_first_and_respects_budget(self) -> None:
        memory = Memory("m", [{"kind": "user", "text": f"m{i}", "date": "d"} for i in range(16)])
        tree = Tree(memory)
        build_tree(FakeCli(), tree, budget=10_000)
        self.assertEqual(len(fold(tree, 16, budget=10_000)), 16)
        view = fold(tree, 16, budget=100)
        self.assertLess(len(view), 16)
        levels = [n[0] for n in view]
        self.assertEqual(levels, sorted(levels, reverse=True))
        covered = [i for node in view for i in range(tree.start(node), tree.end(node))]
        self.assertEqual(covered, list(range(16)))
        self.assertIn("<chat>", render_context(tree, view))
        self.assertNotIn("+", render_context(tree, view).replace("<chat>", ""))
        self.assertIn("|", render_view(tree, view))

    def test_zoom(self) -> None:
        tree = Tree(make_memory(8))
        build_tree(FakeCli(), tree, budget=10_000)
        self.assertEqual(zoom(tree, 4, 3), "No line 4+3.")
        self.assertEqual(zoom(tree, 2, 4), "No line 2+4.")
        self.assertEqual(zoom(tree, 8, 1), "No line 8+1.")
        self.assertTrue(zoom(tree, 4, 1).startswith("4+0|2023/05/01|user: message 4"))
        lines = zoom(tree, 0, 8).splitlines()
        self.assertEqual([l.split("|")[0] for l in lines], ["0+4", "4+4"])

    def test_parse_lines(self) -> None:
        self.assertEqual(parse_lines('x ["a", "b"] y', 2), ["a", "b"])
        self.assertIsNone(parse_lines('["a"]', 2))
        self.assertEqual(parse_lines(" one line ", 1), ["one line"])

    def test_parse_action_json_and_invoke(self) -> None:
        self.assertEqual(parse_action('{"action": "zoom", "lines": ["8+8", "3+1"]}')["lines"], [(8, 8), (3, 1)])
        native = '<invoke name="zoom"><parameter name="id">16</parameter><parameter name="n">4</parameter></invoke>'
        self.assertEqual(parse_action(native)["lines"], [(16, 4)])
        self.assertEqual(parse_action('{"action": "date", "ids": [3, "5"]}')["ids"], [3, 5])
        self.assertEqual(parse_action('{"answer": "x"}')["action"], "answer")

    def test_check_quote(self) -> None:
        read = {3: "user: I graduated in Business Administration."}
        self.assertIsNone(check_quote({"answer": "BA", "quote": "graduated in business administration"}, read))
        self.assertIsNotNone(check_quote({"answer": "BA", "quote": "graduated in law"}, read))
        self.assertIsNotNone(check_quote({"answer": ABSTAIN}, {}))
        self.assertIsNone(check_quote({"answer": ABSTAIN}, read))

    def test_agent_verify_rejects_then_accepts(self) -> None:
        memory = make_memory(8)
        tree = Tree(memory)
        build_tree(FakeCli(), tree, budget=10_000)
        view = fold(tree, 8, 100)
        cli = FakeCli([
            '{"action": "answer", "answer": "topic5", "quote": "about topic5", "source": 5}',
            '{"action": "zoom", "lines": ["5+1"]}',
            '{"action": "answer", "answer": "topic5", "quote": "about topic5", "source": 5}',
        ])
        row = {"dataset": "longmemeval_s", "question": "q", "question_date": "2023/06/01"}
        result = agent_answer(cli, row, "CV", tree, view, None)
        self.assertTrue(result["verified"])
        self.assertEqual(result["read_in_full"], [5])
        self.assertEqual(result["steps"], 3)
        self.assertIn("Current date: 2023/06/01", cli.prompts[-1])

    def test_search_agent_has_no_view(self) -> None:
        memory = make_memory(8)
        tree = Tree(memory)
        build_tree(FakeCli(), tree, budget=10_000)
        cli = FakeCli(['{"action": "search", "query": "topic6"}', '{"action": "answer", "answer": "6"}'])
        row = {"dataset": "locomo", "question": "q", "question_date": None}
        result = agent_answer(cli, row, "F", tree, [], message_searcher(memory))
        self.assertNotIn("<chat>", cli.prompts[0])
        self.assertIn(6, result["read_in_full"])

    def test_bm25_context_budget(self) -> None:
        memory = make_memory(20, long_every=100)
        context, ids = bm25_context(memory, "topic7", budget=200)
        self.assertLessEqual(nbytes(context), 200)
        self.assertIn(7, ids)

    def test_stats(self) -> None:
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertAlmostEqual(cohen_kappa([True, False, True, False], [True, False, True, False]), 1.0)


if __name__ == "__main__":
    unittest.main()
