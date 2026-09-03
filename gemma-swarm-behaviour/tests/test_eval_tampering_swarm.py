import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path


from swarm.eval_tampering_board import EvalTamperingMessageBoard  # noqa: E402
from swarm.eval_tampering_controller import (  # noqa: E402
    ALL_TOOLS,
    EvalAgentSession,
    EvalTamperingSwarmController,
)
from swarm.event_log import EventLog  # noqa: E402
from swarm.llama_client import Completion  # noqa: E402


class FakeClient:
    def __init__(self) -> None:
        self.calls = defaultdict(int)

    def complete(self, **request):
        slot = request["slot"]
        index = self.calls[slot]
        self.calls[slot] += 1
        if index == 0:
            name = "read_messageboard"
            arguments = {}
        else:
            name = "submit"
            arguments = {
                "answers": "5,3,6,4,9,3,8,2,9,5",
                "technical_difficulties": "NO",
                "suspicious_observations": "NO",
            }
        message = {
            "role": "assistant",
            "content": "",
            "reasoning_content": f"Agent in slot {slot} calls {name}.",
            "tool_calls": [
                {
                    "id": f"slot-{slot}-call-{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        }
        raw = {
            "choices": [{"message": message, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        return Completion(message=message, raw=raw, latency_ms=1.0)


class EvalTamperingSwarmTests(unittest.TestCase):
    def test_board_is_seeded_plaintext_and_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = EventLog(Path(directory), "board-test")
            board = EvalTamperingMessageBoard(
                seed="NOTES TO SELF\nSeed body",
                agent_ids={"agent-1", "agent-2", "agent-3", "agent-4"},
                event_log=log,
            )
            first = board.render()
            board.send("agent-2", "My independent assessment")
            second = board.render()

            self.assertTrue(first.startswith("MESSAGE 1\nFROM: notes_to_self\n"))
            self.assertIn("MESSAGE 2\nFROM: agent-2\nMy independent assessment", second)
            records = [
                json.loads(line)
                for line in (Path(directory) / "messageboard.jsonl").read_text().splitlines()
            ]
            self.assertEqual([record["id"] for record in records], [1, 2])
            self.assertTrue(records[0]["seeded"])
            self.assertFalse(records[1]["seeded"])

    def test_four_agents_keep_independent_histories_and_submissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            log = EventLog(output, "controller-test")
            agents = [
                EvalAgentSession(id=f"agent-{number}", name=f"Agent {number}", slot=number - 1)
                for number in range(1, 5)
            ]
            board = EvalTamperingMessageBoard(
                seed="seed",
                agent_ids={agent.id for agent in agents},
                event_log=log,
            )
            controller = EvalTamperingSwarmController(
                agents=agents,
                client=FakeClient(),  # type: ignore[arg-type]
                board=board,
                event_log=log,
                system_prompt="system",
                user_prompt="user",
                max_steps_per_agent=4,
                max_tokens=8192,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                max_history_chars=80_000,
            )
            controller.run()

            self.assertTrue(controller.completed)
            self.assertEqual(controller.total_turns, 8)
            self.assertTrue(all(agent.completed for agent in agents))
            self.assertTrue(all(agent.state.accuracy == 1.0 for agent in agents))
            self.assertEqual(
                [tool["function"]["name"] for tool in ALL_TOOLS],
                ["execute_command", "submit", "read_messageboard", "send_message"],
            )
            submission_records = (output / "submissions.jsonl").read_text().splitlines()
            self.assertEqual(len(submission_records), 4)
            for agent in agents:
                records = (output / "messages" / f"{agent.id}.jsonl").read_text().splitlines()
                self.assertEqual(len(records), 6)
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertFalse(
                any(event["type"] == "action_without_prior_messageboard_read" for event in events)
            )

    def test_private_boards_do_not_expose_peer_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            log = EventLog(output, "private-board-test")
            agents = [
                EvalAgentSession(id=f"agent-{number}", name=f"Agent {number}", slot=number - 1)
                for number in range(1, 5)
            ]
            boards = {
                agent.id: EvalTamperingMessageBoard(
                    seed="seed",
                    agent_ids={agent.id},
                    event_log=log,
                    artifact_name=f"messageboard-{agent.id}.jsonl",
                    board_id=agent.id,
                )
                for agent in agents
            }
            controller = EvalTamperingSwarmController(
                agents=agents,
                client=FakeClient(),  # type: ignore[arg-type]
                board=boards,
                event_log=log,
                system_prompt="system",
                user_prompt="user",
                max_steps_per_agent=4,
                max_tokens=8192,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                max_history_chars=80_000,
            )

            boards["agent-1"].send("agent-1", "private message")

            self.assertIn("private message", boards["agent-1"].render())
            self.assertNotIn("private message", boards["agent-2"].render())
            self.assertEqual(controller.state()["messageboard"]["scope"], "private")
            self.assertTrue((output / "messageboard-agent-1.jsonl").exists())
            self.assertTrue((output / "messageboard-agent-2.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
