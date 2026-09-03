from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLog:
    """Crash-resilient, append-only event log backed by a host bind mount."""

    def __init__(self, output_dir: Path, run_id: str) -> None:
        self.output_dir = output_dir
        self.run_id = run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.output_dir, 0o700)
        self._events_path = self.output_dir / "events.jsonl"
        self._messages_dir = self.output_dir / "messages"
        self._checkpoints_dir = self.output_dir / "checkpoints"
        self._messages_dir.mkdir(exist_ok=True)
        self._checkpoints_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._sequence = self._existing_sequence()

    def _existing_sequence(self) -> int:
        if not self._events_path.exists():
            return 0
        with self._events_path.open("rb") as handle:
            return sum(1 for _ in handle)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _append_fsync(path: Path, record: dict[str, Any]) -> None:
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, payload.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        tmp = path.with_name(f".{path.name}.tmp")
        payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, payload.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def event(self, kind: str, *, agent_id: str | None = None, **data: Any) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            record = {
                "sequence": self._sequence,
                "timestamp": self._timestamp(),
                "monotonic_ns": time.monotonic_ns(),
                "run_id": self.run_id,
                "type": kind,
                "agent_id": agent_id,
                "data": data,
            }
            self._append_fsync(self._events_path, record)
            return record

    def message(self, agent_id: str, direction: str, message: dict[str, Any]) -> None:
        record = {
            "timestamp": self._timestamp(),
            "direction": direction,
            "message": message,
        }
        with self._lock:
            self._append_fsync(self._messages_dir / f"{agent_id}.jsonl", record)

    def inference(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._append_fsync(
                self.output_dir / "inference.jsonl",
                {"timestamp": self._timestamp(), **record},
            )

    def append_jsonl(self, name: str, record: dict[str, Any]) -> None:
        """Append and fsync a named JSONL artifact under the run directory."""
        if Path(name).name != name:
            raise ValueError("append-only artifact name must be a basename")
        with self._lock:
            self._append_fsync(
                self.output_dir / name,
                {"timestamp": self._timestamp(), **record},
            )

    def checkpoint(self, state: dict[str, Any], sequence: int | None = None) -> Path:
        sequence = sequence if sequence is not None else self._sequence
        path = self._checkpoints_dir / f"state-{sequence:08d}.json"
        self._atomic_json(path, state)
        self._atomic_json(self.output_dir / "latest-state.json", state)
        return path

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self._atomic_json(self.output_dir / "manifest.json", manifest)

    def write_json(self, name: str, value: Any) -> None:
        self._atomic_json(self.output_dir / name, value)

    def write_text(self, name: str, value: str) -> None:
        path = self.output_dir / name
        tmp = path.with_name(f".{path.name}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, value.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def fsync_artifact(self, name: str) -> None:
        path = self.output_dir / name
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def mark_complete(self, summary: dict[str, Any]) -> None:
        self._atomic_json(
            self.output_dir / "complete.json",
            {"timestamp": self._timestamp(), "run_id": self.run_id, **summary},
        )
