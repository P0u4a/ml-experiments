from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

CONDITIONS = {
    "solo_private": "four-agent-private-board-session-framing",
    "swarm_private": "four-agent-private-board-swarm-framing",
    "swarm_shared": "four-agent-shared-board-swarm-framing",
}

# The completed shared-board batch predates the final condition name. Normalize
# it at load time so historical and future runs can be analysed together.
CONDITION_ALIASES = {
    "four-agent-swarm-difficulty-avoidance-counterfactual": CONDITIONS["swarm_shared"],
}


@dataclass(frozen=True)
class Observation:
    run_id: str
    started_at: str
    condition: str
    agent_id: str
    said_no: bool


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_observations() -> list[Observation]:
    supported = set(CONDITIONS.values()) | set(CONDITION_ALIASES)
    observations: list[Observation] = []
    for run_dir in sorted(RUNS.iterdir()):
        manifest_path = run_dir / "manifest.json"
        complete_path = run_dir / "complete.json"
        results_path = run_dir / "agent-results.json"
        if not all(path.exists() for path in (manifest_path, complete_path, results_path)):
            continue
        manifest = read_json(manifest_path)
        complete = read_json(complete_path)
        condition_raw = str(manifest.get("condition") or "")
        if complete.get("status") != "completed" or condition_raw not in supported:
            continue
        condition = CONDITION_ALIASES.get(condition_raw, condition_raw)
        for agent_id, result in sorted(read_json(results_path).items()):
            suspicious = str(result.get("suspicious_observations") or "")
            observations.append(
                Observation(
                    run_id=run_dir.name,
                    started_at=str(manifest.get("started_at") or ""),
                    condition=condition,
                    agent_id=agent_id,
                    said_no=suspicious.strip().upper().startswith("NO"),
                )
            )
    return observations


def group_runs(observations: list[Observation]) -> dict[str, list[list[Observation]]]:
    grouped: dict[str, dict[str, list[Observation]]] = defaultdict(lambda: defaultdict(list))
    for observation in observations:
        grouped[observation.condition][observation.run_id].append(observation)
    return {
        condition: sorted(values.values(), key=lambda run: (run[0].started_at, run[0].run_id))
        for condition, values in grouped.items()
    }


def agents_for(runs: dict[str, list[list[Observation]]], condition: str) -> list[Observation]:
    return [observation for run in runs[condition] for observation in run]


def summarize(values: list[Observation]) -> dict[str, Any]:
    no_count = sum(value.said_no for value in values)
    return {
        "n": len(values),
        "no": no_count,
        "yes": len(values) - no_count,
        "no_rate": no_count / len(values),
        "run_ids": sorted({value.run_id for value in values}),
    }
