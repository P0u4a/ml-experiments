#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
variant="${1:?usage: run_eval_tampering_private_board_batch.sh session|swarm [REPETITIONS]}"
repetitions="${2:-10}"
batch_id="${EVAL_BATCH_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
  echo "Repetitions must be a positive integer" >&2
  exit 1
fi

case "$variant" in
  session)
    config="$project_root/configs/eval_tampering_private_board_session.yaml"
    condition="four-agent-private-board-session-framing"
    ;;
  swarm)
    config="$project_root/configs/eval_tampering_private_board_swarm.yaml"
    condition="four-agent-private-board-swarm-framing"
    ;;
  *)
    echo "Variant must be 'session' or 'swarm'" >&2
    exit 1
    ;;
esac

for repetition in $(seq 1 "$repetitions"); do
  run_id="eval-tampering-${variant}-private-board-${batch_id}-r${repetition}"
  EVAL_RUN_ID="$run_id" \
  EVAL_IMAGE="gemma-eval-tampering-swarm:latest" \
  EVAL_CONFIG_PATH="$config" \
  EXPERIMENT_CONDITION="$condition" \
    "$project_root/scripts/run_eval_tampering_swarm.sh"
done
