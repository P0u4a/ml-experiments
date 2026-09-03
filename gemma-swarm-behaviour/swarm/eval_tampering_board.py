from __future__ import annotations

from dataclasses import asdict, dataclass

from .event_log import EventLog


@dataclass(frozen=True)
class Message:
    id: int
    author: str
    body: str
    seeded: bool = False


class EvalTamperingMessageBoard:
    """A plaintext, append-only board seeded with the hidden scratchpad."""

    def __init__(
        self,
        *,
        seed: str,
        agent_ids: set[str],
        event_log: EventLog,
        artifact_name: str = "messageboard.jsonl",
        board_id: str = "shared",
    ) -> None:
        seed = seed.rstrip("\n")
        if not seed:
            raise ValueError("message-board seed must not be empty")
        self._agent_ids = set(agent_ids)
        self._log = event_log
        self._artifact_name = artifact_name
        self.board_id = board_id
        self._messages = [Message(id=1, author="notes_to_self", body=seed, seeded=True)]
        self._persist(self._messages[0])
        self._log.event(
            "messageboard_seeded",
            message_id=1,
            author="notes_to_self",
            characters=len(seed),
            board_id=self.board_id,
        )

    @property
    def version(self) -> int:
        return len(self._messages)

    def send(self, agent_id: str, body: str) -> Message:
        if agent_id not in self._agent_ids:
            raise ValueError(f"unknown agent: {agent_id}")
        body = str(body).strip()
        if not body:
            raise ValueError("message must not be empty")
        if len(body) > 8_000:
            raise ValueError("message exceeds the 8000-character limit")
        message = Message(
            id=len(self._messages) + 1,
            author=agent_id,
            body=body,
        )
        self._messages.append(message)
        self._persist(message)
        self._log.event(
            "messageboard_message_sent",
            agent_id=agent_id,
            message_id=message.id,
            body=message.body,
            board_id=self.board_id,
        )
        return message

    def render(self) -> str:
        chunks: list[str] = []
        for message in self._messages:
            chunks.append(
                f"MESSAGE {message.id}\n"
                f"FROM: {message.author}\n"
                f"{message.body}"
            )
        return "\n\n---\n\n".join(chunks)

    def snapshot(self) -> dict[str, object]:
        return {
            "board_id": self.board_id,
            "version": self.version,
            "messages": [asdict(message) for message in self._messages],
        }

    def _persist(self, message: Message) -> None:
        self._log.append_jsonl(
            self._artifact_name,
            {"board_id": self.board_id, **asdict(message)},
        )
