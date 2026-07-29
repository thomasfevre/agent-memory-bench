from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.download_datasets import download, verify


class DownloadDatasetsTests(unittest.TestCase):
    def test_download_and_reverify_pinned_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"public benchmark fixture")
            entry = {
                "name": "fixture",
                "url": source.as_uri(),
                "destination": "nested/fixture.bin",
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            destination = root / "cache"
            download(entry, destination)
            downloaded = destination / "nested" / "fixture.bin"
            self.assertEqual(source.read_bytes(), downloaded.read_bytes())
            download(entry, destination)

    def test_verify_rejects_corrupt_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.bin"
            path.write_bytes(b"wrong")
            entry = {
                "name": "fixture",
                "bytes": 5,
                "sha256": hashlib.sha256(b"right").hexdigest(),
            }
            with self.assertRaisesRegex(RuntimeError, "sha256 mismatch"):
                verify(path, entry)
