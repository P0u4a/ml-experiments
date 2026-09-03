from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .eval_tampering_board import EvalTamperingMessageBoard
from .eval_tampering_controller import ALL_TOOLS, EvalAgentSession, EvalTamperingSwarmController
from .event_log import EventLog
from .llama_client import LlamaCppClient
from .sandbox import lock_down_agent_network


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("configuration must be a YAML mapping")
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _agents(config: dict[str, Any]) -> list[EvalAgentSession]:
    configured = (config.get("swarm", {}) or {}).get("agents") or [
        {"id": f"agent-{number}", "name": f"Agent {number}", "slot": number - 1}
        for number in range(1, 5)
    ]
    return [
        EvalAgentSession(
            id=str(item["id"]),
            name=str(item.get("name", item["id"])),
            slot=int(item["slot"]),
        )
        for item in configured
    ]


def _archive_workspace(destination: Path) -> None:
    result = subprocess.run(
        ["tar", "-czf", str(destination), "-C", "/agent", "."],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"workspace archive failed: {result.stdout}")


def main() -> int:
    config_path = Path(os.getenv("AGENT_CONFIG", "/opt/config.yaml"))
    config_bytes = config_path.read_bytes()
    config = _load_config(config_path)
    output_dir = Path(os.getenv("AGENT_OUTPUT_DIR", "/opt/output"))
    run_id = os.getenv("EXPERIMENT_RUN_ID", "eval-tampering-swarm")
    seed_path = Path(os.getenv("MESSAGEBOARD_SEED_PATH", "/opt/swarm_seed/.notes_to_self.txt"))
    seed_bytes = seed_path.read_bytes()
    log = EventLog(output_dir, run_id)
    log.write_text("config.yaml", config_bytes.decode("utf-8"))

    swarm_config = config.get("swarm", {}) or {}
    task_config = config.get("task", {}) or {}
    messageboard_scope = str(task_config.get("messageboard_scope", "shared"))
    if messageboard_scope not in {"shared", "private"}:
        raise ValueError("task.messageboard_scope must be 'shared' or 'private'")
    swarm_framing = bool(task_config.get("swarm_framing", True))
    agents = _agents(config)
    tool_names = [tool["function"]["name"] for tool in ALL_TOOLS]
    log.write_manifest(
        {
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": _sha256(config_bytes),
            "container_image": os.getenv("EXPERIMENT_IMAGE_ID"),
            "environment_revision": os.getenv("ENVIRONMENT_REVISION"),
            "llama_server_version": os.getenv("LLAMA_SERVER_VERSION"),
            "model_fingerprint": os.getenv("MODEL_FINGERPRINT"),
            "model_label": os.getenv("MODEL_LABEL", "gemma-4-12B-it-qat-UD-Q4_K_XL"),
            "condition": os.getenv(
                "EXPERIMENT_CONDITION", "four-agent-shared-board-swarm-framing"
            ),
            "messageboard_seed_sha256": _sha256(seed_bytes),
            "messageboard_seed_path": str(seed_path),
            "environment": "eval_tampering_swarm",
            "variation": (config.get("task", {}) or {}).get("variation"),
            "messageboard_scope": messageboard_scope,
            "swarm_framing": swarm_framing,
            "agent_count": len(agents),
            "agents": [{"id": agent.id, "name": agent.name, "slot": agent.slot} for agent in agents],
            "context_size_total": int(os.getenv("LLAMA_CONTEXT_SIZE", "131072")),
            "context_size_per_slot": int(os.getenv("LLAMA_CONTEXT_SIZE", "131072")) // len(agents),
            "max_output_tokens": int(swarm_config.get("max_tokens", 8192)),
            "max_steps_per_agent": int(swarm_config.get("max_steps_per_agent", 32)),
            "tools": tool_names,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_limit": os.getenv("EXPERIMENT_CPU_LIMIT", "2"),
            "cpu_set": os.getenv("EXPERIMENT_CPU_SET", "0-1"),
            "memory_limit": os.getenv("EXPERIMENT_MEMORY_LIMIT", "4g"),
        }
    )

    if Path("/agent/notes_to_self.txt").exists() or Path("/agent/.notes_to_self.txt").exists():
        error = "notes_to_self seed leaked into the agent workspace"
        log.event("seed_isolation_failed", error=error)
        log.write_json("interrupted.json", {"status": "seed-isolation-failed", "error": error})
        return 1
    log.event("seed_isolation_verified", workspace="/agent", seed_path=str(seed_path))

    try:
        sandbox = lock_down_agent_network()
        log.event("agent_network_locked_down", **sandbox)
    except Exception as error:
        log.event(
            "agent_network_lockdown_failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        log.write_json("interrupted.json", {"status": "sandbox-setup-failed"})
        return 1

    if messageboard_scope == "shared":
        board: EvalTamperingMessageBoard | dict[str, EvalTamperingMessageBoard]
        board = EvalTamperingMessageBoard(
            seed=seed_bytes.decode("utf-8"),
            agent_ids={agent.id for agent in agents},
            event_log=log,
        )
    else:
        board = {
            agent.id: EvalTamperingMessageBoard(
                seed=seed_bytes.decode("utf-8"),
                agent_ids={agent.id},
                event_log=log,
                artifact_name=f"messageboard-{agent.id}.jsonl",
                board_id=agent.id,
            )
            for agent in agents
        }
    client = LlamaCppClient(
        os.getenv("LLAMA_BASE_URL", "http://host.docker.internal:8080/v1"),
        model=os.getenv("LLAMA_MODEL_NAME", "local"),
        timeout_s=int(swarm_config.get("inference_timeout_s", 900)),
        api_key=os.getenv("LLAMA_API_KEY"),
    )
    prompts = config.get("prompts", {}) or {}
    controller = EvalTamperingSwarmController(
        agents=agents,
        client=client,
        board=board,
        event_log=log,
        system_prompt=str(prompts.get("system_prompt", "")),
        user_prompt=str(prompts.get("user_prompt", "")),
        max_steps_per_agent=int(swarm_config.get("max_steps_per_agent", 32)),
        max_tokens=int(swarm_config.get("max_tokens", 8192)),
        temperature=float(swarm_config.get("temperature", 1.0)),
        top_p=float(swarm_config.get("top_p", 0.95)),
        top_k=int(swarm_config.get("top_k", 64)),
        max_history_chars=int(swarm_config.get("max_history_chars", 80_000)),
        swarm_framing=swarm_framing,
    )
    controller.install_signal_handlers()

    exit_code = 0
    error_text: str | None = None
    try:
        health = client.health()
        log.event("llama_health_checked", health=health)
        controller.run()
    except Exception as error:
        exit_code = 1
        error_text = f"{type(error).__name__}: {error}"
        log.event("experiment_crashed", error=error_text, traceback=traceback.format_exc())
    finally:
        state = controller.state()
        log.checkpoint(state)
        log.write_json("experiment-state.json", state)
        log.write_json(
            "agent-results.json",
            {
                agent.id: {
                    "completed": agent.completed,
                    "turns": agent.turns,
                    **agent.state.to_dict(),
                }
                for agent in agents
            },
        )
        try:
            _archive_workspace(output_dir / "workspace.tar.gz")
            log.fsync_artifact("workspace.tar.gz")
            log.event("workspace_archived", artifact="workspace.tar.gz")
        except Exception as error:
            exit_code = 1
            log.event(
                "workspace_archive_failed",
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )

    summary = {
        "status": "completed" if controller.completed and exit_code == 0 else "incomplete",
        "completed": controller.completed and exit_code == 0,
        "total_turns": controller.total_turns,
        "completed_agents": sum(agent.completed for agent in agents),
        "agent_count": len(agents),
        "error": error_text,
        "agents": {
            agent.id: {
                "completed": agent.completed,
                "accuracy": agent.state.accuracy,
                "suspicious_observations": agent.state.suspicious_observations,
            }
            for agent in agents
        },
    }
    if summary["completed"]:
        log.mark_complete(summary)
    else:
        log.write_json("interrupted.json", summary)
    return exit_code if exit_code else (0 if controller.completed else 2)


if __name__ == "__main__":
    raise SystemExit(main())
