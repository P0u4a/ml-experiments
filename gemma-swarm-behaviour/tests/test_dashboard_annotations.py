import json
import tempfile
import unittest
from pathlib import Path

from dashboard.server import AnnotationStore


class AnnotationStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.runs_dir = Path(self.temp.name)
        self.run_id = "eval-run-1"
        run_dir = self.runs_dir / self.run_id
        run_dir.mkdir()
        (run_dir / "complete.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        (run_dir / "manifest.json").write_text(
            json.dumps({"condition": "four-agent-private-board-swarm-framing"}),
            encoding="utf-8",
        )
        self.store = AnnotationStore(self.runs_dir)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_create_list_and_delete_are_append_only(self) -> None:
        annotation = self.store.create(
            self.run_id,
            {
                "comment": "Important decision point",
                "quote": "report it",
                "color": "yellow",
                "target": {
                    "item_key": "turn-4",
                    "field": "reasoning",
                    "start": 10,
                    "end": 19,
                    "label": "Reasoning / CoT",
                },
            },
        )
        self.assertEqual(self.store.list(self.run_id)[0]["comment"], "Important decision point")
        self.store.delete(self.run_id, annotation["id"])
        self.assertEqual(self.store.list(self.run_id), [])
        lines = (self.runs_dir / self.run_id / "annotations.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(json.loads(lines[1])["deleted"])

    def test_rejects_path_traversal_and_invalid_range(self) -> None:
        with self.assertRaises(ValueError):
            self.store.list("../outside")
        with self.assertRaises(ValueError):
            self.store.create(
                self.run_id,
                {
                    "comment": "Bad",
                    "quote": "abc",
                    "target": {"item_key": "turn-0", "field": "reasoning", "start": 0, "end": 2},
                },
            )


if __name__ == "__main__":
    unittest.main()
