from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memoryagentbench_detectiveqa_ablation import parse_target


class MemoryAgentBenchDetectiveQaTests(unittest.TestCase):
    def test_parse_target_removes_demonstration_and_keeps_options(self) -> None:
        question = """
Example: irrelevant demonstration.
Now Answer the Question: Who did it?
A. Alice
B. Bob
C. Carol
D. Dan
Output:
"""
        target, stem, options = parse_target(question)
        self.assertEqual(stem, "Who did it?")
        self.assertIn("A. Alice", target)
        self.assertEqual(options, ["A. Alice", "B. Bob", "C. Carol", "D. Dan"])


if __name__ == "__main__":
    unittest.main()
