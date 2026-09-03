import json
import tempfile
import unittest
from pathlib import Path

from swarm.event_log import EventLog


class EventLogTests(unittest.TestCase):
    def test_events_and_checkpoint_survive_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            log = EventLog(output, "run-test")
            first = log.event("started", value=1)
            log.message("agent-1", "output", {"role": "assistant", "content": "hello"})
            log.inference({"agent_id": "agent-1", "response": {"ok": True}})
            log.checkpoint({"turn": 1})

            reopened = EventLog(output, "run-test")
            second = reopened.event("continued", value=2)
            self.assertEqual(first["sequence"], 1)
            self.assertEqual(second["sequence"], 2)
            self.assertEqual(json.loads((output / "latest-state.json").read_text()), {"turn": 1})
            self.assertEqual(len((output / "messages" / "agent-1.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len((output / "inference.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
