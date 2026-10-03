"""Artifact protection checks, including a competing-writers regression."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from threading import Barrier
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("lesson_runtime", ROOT / "scripts" / "lesson_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class OutputProtection(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="maedaily_output_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_default_runs_are_distinct_and_record_a_receipt(self):
        first = runtime.reserve_output_dir(None, "first", root=self.root)
        second = runtime.reserve_output_dir(None, "first", root=self.root)
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, self.root / "study_runs")
        receipt = json.loads((first / runtime.CLAIM_FILE).read_text(encoding="utf-8"))
        self.assertEqual(receipt["lesson"], "first")
        self.assertIn("reserved_at_utc", receipt)

    def test_nonempty_directory_and_file_are_preserved(self):
        output = self.root / "results"
        output.mkdir()
        sentinel = output / "summary.json"
        sentinel.write_bytes(b'{"teaching_reference": true}\n')
        before = hashlib.sha256(sentinel.read_bytes()).hexdigest()
        with self.assertRaises((ValueError, OSError)):
            runtime.reserve_output_dir(output, "first")
        self.assertEqual(hashlib.sha256(sentinel.read_bytes()).hexdigest(), before)
        self.assertEqual(list(output.iterdir()), [sentinel])
        with self.assertRaises((ValueError, OSError)):
            runtime.reserve_output_dir(sentinel, "first")
        self.assertEqual(hashlib.sha256(sentinel.read_bytes()).hexdigest(), before)

    def test_empty_directory_can_only_be_claimed_once(self):
        output = self.root / "empty"
        output.mkdir()
        self.assertEqual(runtime.reserve_output_dir(output, "first"), output)
        with self.assertRaises((ValueError, OSError)):
            runtime.reserve_output_dir(output, "first")

    def test_competing_writers_cannot_both_claim_the_same_directory(self):
        output = self.root / "shared"
        output.mkdir()
        barrier = Barrier(4)

        def claim():
            barrier.wait(timeout=10)
            try:
                runtime.reserve_output_dir(output, "first")
                return True
            except (ValueError, OSError):
                return False

        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(lambda _: claim(), range(4)))
        self.assertEqual(sum(outcomes), 1)
        self.assertTrue((output / runtime.CLAIM_FILE).is_file())


if __name__ == "__main__":
    unittest.main()
