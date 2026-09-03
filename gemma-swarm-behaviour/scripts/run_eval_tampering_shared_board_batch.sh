#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repetitions="${1:-10}"
batch_id="${EVAL_BATCH_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
  echo "Repetitions must be a positive integer" >&2
  exit 1
fi

for repetition in $(seq 1 "$repetitions"); do
  EVAL_RUN_ID="eval-tampering-shared-board-swarm-${batch_id}-r${repetition}" \
  EVAL_CONFIG_PATH="$project_root/configs/eval_tampering_shared_board_swarm.yaml" \
  EXPERIMENT_CONDITION="four-agent-shared-board-swarm-framing" \
    "$project_root/scripts/run_eval_tampering_swarm.sh"
done
