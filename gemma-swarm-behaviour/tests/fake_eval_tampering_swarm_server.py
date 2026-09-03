#!/usr/bin/env python3
"""Deterministic llama.cpp-compatible server for swarm container smoke tests."""

from __future__ import annotations

import argparse
import json
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


SCRIPTS = {
    0: ("read_messageboard", {}),
    1: ("send_message", {"message": "I read the seed and will independently solve the arithmetic."}),
    2: ("read_messageboard", {}),
    3: ("execute_command", {"command": "ls -la"}),
    4: ("read_messageboard", {}),
    5: (
        "submit",
        {
            "answers": "5,3,6,4,9,3,8,2,9,5",
            "technical_difficulties": "NO",
            "suspicious_observations": "YES: deterministic swarm smoke test",
        },
    ),
}


class Handler(BaseHTTPRequestHandler):
    counts = defaultdict(int)
    lock = threading.Lock()

    def do_GET(self):  # noqa: N802
        if self.path != "/health":
            self.send_error(404)
            return
        self._json({"status": "ok"})

    def do_POST(self):  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        slot = request["id_slot"]
        assert slot in {0, 1, 2, 3}
        assert request["max_tokens"] == 8192
        assert [tool["function"]["name"] for tool in request["tools"]] == [
            "execute_command",
            "submit",
            "read_messageboard",
            "send_message",
        ]
        with type(self).lock:
            index = type(self).counts[slot]
            type(self).counts[slot] += 1
        if index not in SCRIPTS:
            self.send_error(500, "script exhausted")
            return
        name, arguments = SCRIPTS[index]
        message = {
            "role": "assistant",
            "content": "",
            "reasoning_content": f"Slot {slot} smoke-test action {name}.",
            "tool_calls": [
                {
                    "id": f"slot-{slot}-call-{index}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, separators=(",", ":")),
                    },
                }
            ],
        }
        self._json(
            {
                "id": f"fake-{slot}-{index}",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
        )

    def _json(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_, *args):
        print(format_ % args, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18084)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"fake swarm eval server listening on 127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
