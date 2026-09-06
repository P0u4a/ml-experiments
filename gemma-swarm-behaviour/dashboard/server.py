#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,220}$")
ITEM_KEY_PATTERN = re.compile(r"^(summary|(turn|event)-[0-9]+)$")
FIELD_PATTERN = re.compile(r"^[A-Za-z0-9:_-]{1,100}$")
ANNOTATION_ID_PATTERN = re.compile(r"^ann_[a-f0-9]{32}$")
COLORS = {"yellow", "pink", "blue", "green"}


class AnnotationStore:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir.resolve()
        self._lock = threading.Lock()

    def _run_dir(self, run_id: str) -> Path:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("Invalid run id")
        run_dir = (self.runs_dir / run_id).resolve()
        if run_dir.parent != self.runs_dir:
            raise ValueError("Invalid run path")
        complete_path = run_dir / "complete.json"
        manifest_path = run_dir / "manifest.json"
        if not complete_path.is_file() or not manifest_path.is_file():
            raise ValueError("Unknown completed run")
        try:
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Unreadable run metadata") from error
        if complete.get("status") != "completed":
            raise ValueError("Run is not complete")
        return run_dir

    @staticmethod
    def _records(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    @staticmethod
    def _active(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        active: dict[str, dict[str, Any]] = {}
        for record in records:
            annotation_id = record.get("id")
            if not isinstance(annotation_id, str):
                continue
            if record.get("deleted"):
                active.pop(annotation_id, None)
            else:
                active[annotation_id] = record
        return sorted(active.values(), key=lambda item: item.get("created_at", ""))

    @staticmethod
    def _append(path: Path, record: dict[str, Any]) -> None:
        encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def list(self, run_id: Optional[str] = None) -> list[dict[str, Any]]:
        if run_id:
            run_dirs = [self._run_dir(run_id)]
        else:
            run_dirs = []
            for complete_path in self.runs_dir.glob("*/complete.json"):
                try:
                    run_dirs.append(self._run_dir(complete_path.parent.name))
                except ValueError:
                    continue
        annotations = []
        with self._lock:
            for run_dir in run_dirs:
                annotations.extend(self._active(self._records(run_dir / "annotations.jsonl")))
        return sorted(annotations, key=lambda item: item.get("created_at", ""))

    def create(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        run_dir = self._run_dir(run_id)
        comment = payload.get("comment")
        quote = payload.get("quote")
        target = payload.get("target")
        color = payload.get("color", "yellow")
        if not isinstance(comment, str) or not comment.strip() or len(comment) > 4000:
            raise ValueError("Comment must contain 1-4000 characters")
        if not isinstance(quote, str) or not quote or len(quote) > 20000:
            raise ValueError("Quote must contain 1-20000 characters")
        if not isinstance(target, dict):
            raise ValueError("Missing annotation target")
        item_key = target.get("item_key")
        field = target.get("field")
        start = target.get("start")
        end = target.get("end")
        if not isinstance(item_key, str) or not ITEM_KEY_PATTERN.fullmatch(item_key):
            raise ValueError("Invalid trace item")
        if not isinstance(field, str) or not FIELD_PATTERN.fullmatch(field):
            raise ValueError("Invalid annotation field")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            raise ValueError("Invalid text range")
        utf16_length = len(quote.encode("utf-16-le")) // 2
        if end - start != utf16_length:
            raise ValueError("Text range does not match quote length")
        if color not in COLORS:
            raise ValueError("Invalid highlight color")

        clean_target = {
            "item_key": item_key,
            "field": field,
            "start": start,
            "end": end,
            "label": str(target.get("label") or "Trace text")[:200],
            "prefix": str(target.get("prefix") or "")[-120:],
            "suffix": str(target.get("suffix") or "")[:120],
        }
        record = {
            "schema_version": 1,
            "id": "ann_" + uuid.uuid4().hex,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "comment": comment.strip(),
            "quote": quote,
            "color": color,
            "target": clean_target,
        }
        with self._lock:
            self._append(run_dir / "annotations.jsonl", record)
        return record

    def delete(self, run_id: str, annotation_id: str) -> None:
        run_dir = self._run_dir(run_id)
        if not ANNOTATION_ID_PATTERN.fullmatch(annotation_id):
            raise ValueError("Invalid annotation id")
        with self._lock:
            active = self._active(self._records(run_dir / "annotations.jsonl"))
            if not any(item.get("id") == annotation_id for item in active):
                raise ValueError("Unknown annotation")
            self._append(
                run_dir / "annotations.jsonl",
                {
                    "schema_version": 1,
                    "id": annotation_id,
                    "run_id": run_id,
                    "deleted": True,
                    "deleted_at": datetime.now(timezone.utc).isoformat(),
                },
            )


class DashboardHandler(SimpleHTTPRequestHandler):
    store: AnnotationStore

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def _json(self, status: int, payload: Any) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid content length") from error
        if length <= 0 or length > 100_000:
            raise ValueError("Invalid request size")
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Invalid JSON") from error
        if not isinstance(body, dict):
            raise ValueError("Expected a JSON object")
        return body

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/annotations":
            try:
                run_id = parse_qs(parsed.query).get("run_id", [None])[0]
                self._json(HTTPStatus.OK, {"annotations": self.store.list(run_id)})
            except ValueError as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/annotations":
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            body = self._body()
            run_id = body.pop("run_id", None)
            if not isinstance(run_id, str):
                raise ValueError("Missing run id")
            annotation = self.store.create(run_id, body)
            self._json(HTTPStatus.CREATED, {"annotation": annotation})
        except ValueError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        match = re.fullmatch(r"/api/annotations/(ann_[a-f0-9]{32})", parsed.path)
        if not match:
            self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            run_id = parse_qs(parsed.query).get("run_id", [None])[0]
            if not isinstance(run_id, str):
                raise ValueError("Missing run id")
            self.store.delete(run_id, match.group(1))
            self._json(HTTPStatus.OK, {"deleted": True})
        except ValueError as error:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local trace dashboard")
    parser.add_argument("--port", type=int, default=4173)
    args = parser.parse_args()
    DashboardHandler.store = AnnotationStore(RUNS_DIR)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print(f"Trace Lens running at http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
