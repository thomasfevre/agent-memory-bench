from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memoryagentbench_redial_retrieval import (
    extract_movie_name,
    movie_is_mentioned,
    parse_dialogues,
)


class MemoryAgentBenchRedialRetrievalTests(unittest.TestCase):
    def test_parse_dialogues(self) -> None:
        dialogues = parse_dialogues(
            "Dialogue 1:\n\nFirst exchange.\n\nDialogue 2:\n\nSecond exchange."
        )
        self.assertEqual(
            dialogues,
            [
                {"id": "dialogue-1", "text": "First exchange."},
                {"id": "dialogue-2", "text": "Second exchange."},
            ],
        )

    def test_extract_movie_name_from_dbpedia_entity(self) -> None:
        self.assertEqual(
            extract_movie_name(
                "<http://dbpedia.org/resource/Water_(1985_film)>"
            ),
            "Water",
        )

    def test_movie_mention_ignores_case_and_punctuation(self) -> None:
        self.assertTrue(
            movie_is_mentioned(
                "This Is Spinal Tap",
                "I recommend This Is Spinal Tap (1984).",
            )
        )


if __name__ == "__main__":
    unittest.main()
