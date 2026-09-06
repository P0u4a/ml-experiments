#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path

from agent_interp_envs.config import load_config
from agent_interp_envs.print_helpers import print_step_header

from run_step import print_final_results, run_step
from states import EvalTamperingState, dump, init_fs_tracking
from tools import EXECUTE_COMMAND_TOOL, SUBMIT_TOOL

from agent.llama_cpp_provider import LlamaCppProvider
from runtime.event_log import EventLog
from runtime.sandbox import lock_down_agent_network


def main() -> int:
    config_path = Path(os.getenv("AGENT_CONFIG", "/opt/config.yaml"))
    config_bytes = config_path.read_bytes()
    config = load_config(str(config_path))
    output_dir = Path(os.getenv("AGENT_OUTPUT_DIR", "/opt/output"))
    run_id = os.getenv("EXPERIMENT_RUN_ID", "eval-tampering-single")
    event_log = EventLog(output_dir, run_id)
    event_log.write_text("config.yaml", config_bytes.decode("utf-8"))

    agent_config = config.get("agent", {}) or {}
    tools = [EXECUTE_COMMAND_TOOL, SUBMIT_TOOL]
    event_log.write_manifest(
        {
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "container_image": os.getenv("EXPERIMENT_IMAGE_ID"),
            "environment_revision": os.getenv("ENVIRONMENT_REVISION"),
            "llama_server_version": os.getenv("LLAMA_SERVER_VERSION"),
            "model_fingerprint": os.getenv("MODEL_FINGERPRINT"),
            "model_label": os.getenv("MODEL_LABEL", "gemma-4-12B-it-qat-UD-Q4_K_XL"),
            "condition": os.getenv("EXPERIMENT_CONDITION", "single-agent-control"),
            "lab_observation_sha256": os.getenv("LAB_OBSERVATION_SHA256"),
            "environment": "eval_tampering",
            "variation": (config.get("task", {}) or {}).get("variation"),
            "context_size": int(os.getenv("LLAMA_CONTEXT_SIZE", "131072")),
            "max_output_tokens": int(agent_config.get("max_tokens", 8192)),
            "tools": [tool["function"]["name"] for tool in tools],
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_limit": os.getenv("EXPERIMENT_CPU_LIMIT", "2"),
            "cpu_set": os.getenv("EXPERIMENT_CPU_SET", "0-1"),
            "memory_limit": os.getenv("EXPERIMENT_MEMORY_LIMIT", "4g"),
        }
    )

    try:
        sandbox = lock_down_agent_network()
        event_log.event("agent_network_locked_down", **sandbox)
    except Exception as error:
        event_log.event(
            "agent_network_lockdown_failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        event_log.write_json("interrupted.json", {"status": "sandbox-setup-failed"})
        return 1

    prompts = config.get("prompts", {}) or {}
    state = EvalTamperingState()
    init_fs_tracking()
    provider = LlamaCppProvider(
        messages=[
            {"role": "system", "content": prompts.get("system_prompt", "")},
            {"role": "user", "content": prompts.get("user_prompt", "")},
        ],
        tools=tools,
        event_log=event_log,
        max_tokens=int(agent_config.get("max_tokens", 8192)),
        temperature=float(agent_config.get("temperature", 1.0)),
        top_p=float(agent_config.get("top_p", 0.95)),
        top_k=int(agent_config.get("top_k", 64)),
        timeout_s=int(agent_config.get("inference_timeout_s", 900)),
    )

    completed = False
    exit_code = 0
    try:
        health = provider.client.health()
        event_log.event("llama_health_checked", health=health)
        event_log.event(
            "agent_started",
            agent_id="agent-1",
            tools=[tool["function"]["name"] for tool in tools],
        )
        provider.print_history()
        max_steps = int(agent_config.get("max_steps", 20))
        while state.step < max_steps:
            print_step_header(state.step)
            completed = run_step(provider, state, config)
            dump(state, provider, output_dir / f"step-{state.step}")
            state.step += 1
            event_log.checkpoint(
                {
                    "completed": completed,
                    "state": state.to_dict(),
                    "tools": [tool["function"]["name"] for tool in tools],
                }
            )
            if completed:
                break
        if not completed:
            print_final_results(state, completed=False)
    except KeyboardInterrupt:
        exit_code = 130
        event_log.event("experiment_interrupted", agent_id="agent-1")
    except Exception as error:
        exit_code = 1
        event_log.event(
            "experiment_crashed",
            agent_id="agent-1",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    finally:
        summary = {
            "status": "completed" if completed else "incomplete",
            "completed": completed,
            "state": state.to_dict(),
            "tools": [tool["function"]["name"] for tool in tools],
        }
        event_log.checkpoint(summary)
        if completed:
            event_log.mark_complete(summary)
        else:
            event_log.write_json("interrupted.json", summary)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
