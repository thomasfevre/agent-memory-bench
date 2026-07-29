from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from locomo_granularity import (
    build_representations,
    materialize_top_items,
    materialize_word_budget,
)


class LoCoMoGranularityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = {
            "conversation": {
                "session_1_date_time": "date",
                "session_1": [
                    {"dia_id": "d1", "speaker": "A", "text": "one two"},
                    {"dia_id": "d2", "speaker": "B", "text": "three four"},
                    {"dia_id": "d3", "speaker": "A", "text": "five six"},
                    {"dia_id": "d4", "speaker": "B", "text": "seven eight"},
                    {"dia_id": "d5", "speaker": "A", "text": "nine ten"},
                ],
            }
        }

    def test_builds_turn_window_and_session_views(self) -> None:
        representations = build_representations(self.sample, 4, 2)
        self.assertEqual(len(representations["turn"]), 5)
        self.assertEqual(len(representations["session"]), 1)
        self.assertEqual(len(representations["window4"]), 2)
        self.assertEqual(
            representations["window4"][0]["source_ids"],
            ["d1", "d2", "d3", "d4"],
        )

    def test_top_items_deduplicate_overlapping_turns(self) -> None:
        windows = build_representations(self.sample, 4, 2)["window4"]
        selected, _ = materialize_top_items(windows, 2)
        self.assertEqual(selected, {"d1", "d2", "d3", "d4", "d5"})

    def test_word_budget_keeps_whole_turns(self) -> None:
        turns = build_representations(self.sample, 4, 2)["turn"]
        selected, words = materialize_word_budget(turns, 6)
        self.assertEqual(selected, {"d1", "d2"})
        self.assertEqual(words, 6)


if __name__ == "__main__":
    unittest.main()
