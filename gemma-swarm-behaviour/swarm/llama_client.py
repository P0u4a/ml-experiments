from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class Completion:
    message: dict[str, Any]
    raw: dict[str, Any]
    latency_ms: float
    attempts: int = 1
    retry_errors: list[str] | None = None


class LlamaCppClient:
    def __init__(
        self,
        base_url: str,
        model: str = "local",
        timeout_s: int = 900,
        api_key: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.api_key = api_key
        self.max_retries = max(0, max_retries)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def health(self) -> dict[str, Any]:
        url = self.base_url.removesuffix("/v1") + "/health"
        request = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        slot: int,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "stream": False,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "id_slot": slot,
            "cache_prompt": True,
        }
        started = time.perf_counter()
        retry_errors: list[str] = []
        raw: dict[str, Any] | None = None
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=self._headers(),
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    candidate = json.loads(response.read())
                if not isinstance(candidate, dict):
                    raise RuntimeError("llama.cpp returned a non-object JSON response")
                raw = candidate
                attempts = attempt
                break
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                detail = f"HTTP {error.code}: {body}"
                retryable = error.code in {408, 429, 500, 502, 503, 504}
                retry_errors.append(detail)
                if not retryable or attempt > self.max_retries:
                    raise RuntimeError(
                        f"llama.cpp request failed after {attempt} attempt(s): {detail}"
                    ) from error
            except (urllib.error.URLError, TimeoutError) as error:
                detail = f"{type(error).__name__}: {error}"
                retry_errors.append(detail)
                if attempt > self.max_retries:
                    raise RuntimeError(
                        f"llama.cpp request failed after {attempt} attempt(s): {detail}"
                    ) from error
            if attempt <= self.max_retries:
                time.sleep(min(2 ** (attempt - 1), 4))

        if raw is None:
            raise RuntimeError("llama.cpp request exhausted retries without a response")
        latency_ms = (time.perf_counter() - started) * 1000
        choices = raw.get("choices") or []
        if not choices or not isinstance(choices[0].get("message"), dict):
            raise RuntimeError(f"llama.cpp response had no assistant message: {raw}")
        return Completion(
            message=choices[0]["message"],
            raw=raw,
            latency_ms=latency_ms,
            attempts=attempts,
            retry_errors=retry_errors,
        )
