#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repetitions="${1:-10}"
batch_id="${EVAL_BATCH_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

EVAL_BATCH_ID="${batch_id}-session" \
  "$project_root/scripts/run_eval_tampering_private_board_batch.sh" session "$repetitions"
EVAL_BATCH_ID="${batch_id}-swarm" \
  "$project_root/scripts/run_eval_tampering_private_board_batch.sh" swarm "$repetitions"
