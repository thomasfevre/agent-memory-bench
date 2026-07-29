from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memoryagentbench_ruler_retrieval import parse_documents


class MemoryAgentBenchRulerRetrievalTests(unittest.TestCase):
    def test_parse_documents_preserves_document_boundaries(self) -> None:
        documents = parse_documents(
            "Document 1:\nAlpha text.\n\nDocument 2:\nBeta text."
        )
        self.assertEqual(
            documents,
            [
                {"id": "document-1", "text": "Alpha text."},
                {"id": "document-2", "text": "Beta text."},
            ],
        )


if __name__ == "__main__":
    unittest.main()
