from __future__ import annotations

import copy
import json
import os
from typing import Any

from agent_interp_envs.print_helpers import print_section, print_step_header
from agent_interp_envs.types import LLMResponse, ToolCall, ToolResult

from runtime.event_log import EventLog
from runtime.llama_client import LlamaCppClient


class LlamaCppProvider:
    """OpenAI-compatible llama.cpp provider for the evaluation agent loop."""

    def __init__(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        event_log: EventLog,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        timeout_s: int,
        client: LlamaCppClient | None = None,
    ) -> None:
        self.messages = messages
        self.tools = tools
        self.event_log = event_log
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.client = client or LlamaCppClient(
            os.getenv("LLAMA_BASE_URL", "http://host.docker.internal:8080/v1"),
            model=os.getenv("LLAMA_MODEL_NAME", "local"),
            timeout_s=timeout_s,
            api_key=os.getenv("LLAMA_API_KEY"),
        )

    def invoke(self) -> LLMResponse:
        request = {
            # Preserve the exact prompt/tool payload used for this generation.
            # The live history is extended with the assistant response below.
            "messages": copy.deepcopy(self.messages),
            "tools": copy.deepcopy(self.tools),
            "slot": 0,
            "lora": None,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
        }
        self.event_log.event(
            "inference_requested",
            agent_id="agent-1",
            slot=0,
            lora=None,
            max_tokens=self.max_tokens,
            tools=[tool["function"]["name"] for tool in self.tools],
        )
        completion = self.client.complete(**request)
        assistant = dict(completion.message)
        self.messages.append(assistant)
        self.event_log.message("agent-1", "assistant", assistant)
        self.event_log.inference(
            {
                "agent_id": "agent-1",
                "slot": 0,
                "lora": None,
                "request": request,
                "response": completion.raw,
                "latency_ms": completion.latency_ms,
                "attempts": completion.attempts,
                "retry_errors": completion.retry_errors or [],
            }
        )

        tool_calls = [
            ToolCall(
                id=tool_call["id"],
                name=tool_call["function"]["name"],
                arguments=tool_call["function"]["arguments"],
            )
            for tool_call in assistant.get("tool_calls") or []
        ]
        reasoning = assistant.get("reasoning_content") or assistant.get("reasoning")
        self.event_log.event(
            "inference_completed",
            agent_id="agent-1",
            latency_ms=completion.latency_ms,
            attempts=completion.attempts,
            retry_errors=completion.retry_errors or [],
            reasoning=reasoning,
            content=assistant.get("content"),
            tool_calls=assistant.get("tool_calls") or [],
            usage=completion.raw.get("usage") or {},
        )
        return LLMResponse(
            reasoning=reasoning,
            response=assistant.get("content"),
            tool_calls=tool_calls or None,
        )

    def add_tool_result(self, tool_result: ToolResult) -> None:
        message = {
            "role": "tool",
            "tool_call_id": tool_result.id,
            "name": tool_result.name,
            "content": tool_result.content,
        }
        self.messages.append(message)
        self.event_log.message("agent-1", "tool", message)
        self.event_log.event(
            "tool_result_added",
            agent_id="agent-1",
            tool=tool_result.name,
            tool_call_id=tool_result.id,
            result=tool_result.content,
        )

    def add_message(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self.event_log.message("agent-1", "user", message)

    def dump_history(self) -> str:
        return json.dumps(self.messages, ensure_ascii=False, indent=2)

    def print_history(self) -> None:
        step = 0
        for index, message in enumerate(self.messages):
            role = message.get("role")
            if role == "system":
                print()
                print_section("SYSTEM PROMPT", message.get("content", ""))
            elif role == "user" and index == 1:
                print_section("USER_PROMPT", message.get("content", ""))
            elif role == "assistant":
                print_step_header(step)
                reasoning = message.get("reasoning_content") or message.get("reasoning")
                if reasoning:
                    print()
                    print_section("REASONING", reasoning)
                if message.get("content"):
                    print_section("RESPONSE", message["content"])
                for tool_call in message.get("tool_calls") or []:
                    print_section(
                        "TOOL CALL",
                        f"Function: {tool_call['function']['name']}\n"
                        f"Arguments: {tool_call['function']['arguments']}",
                    )
            elif role == "tool":
                print_section("TOOL RESULT", message.get("content", ""))
                step += 1
            elif role == "user":
                print_section("USER MESSAGE", message.get("content", ""))
                step += 1

    def revert_last_turn(self) -> None:
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages.pop()
