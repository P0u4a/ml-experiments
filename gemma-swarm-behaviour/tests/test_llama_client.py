import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from swarm.llama_client import LlamaCppClient


class _Response:
    def __init__(self, value):
        self.payload = json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class LlamaClientTests(unittest.TestCase):
    @patch("swarm.llama_client.time.sleep", return_value=None)
    @patch("swarm.llama_client.urllib.request.urlopen")
    def test_retries_transient_failure_and_preserves_slot(self, urlopen, _sleep):
        transient = urllib.error.HTTPError(
            "http://llama/v1/chat/completions",
            503,
            "unavailable",
            None,
            io.BytesIO(b'{"error":"slot temporarily unavailable"}'),
        )
        urlopen.side_effect = [
            transient,
            _Response(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [],
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 1},
                }
            ),
        ]
        client = LlamaCppClient("http://llama/v1", timeout_s=2, max_retries=1)
        completion = client.complete(
            messages=[{"role": "user", "content": "test"}],
            tools=[],
            slot=3,
            temperature=0.5,
            top_p=0.9,
            top_k=32,
            max_tokens=50,
        )
        self.assertEqual(completion.attempts, 2)
        self.assertEqual(len(completion.retry_errors), 1)
        self.assertEqual(urlopen.call_count, 2)
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["id_slot"], 3)
        self.assertNotIn("lora", payload)


if __name__ == "__main__":
    unittest.main()
