#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker build \
  --file "$project_root/docker/eval_tampering_swarm.Dockerfile" \
  --tag gemma-eval-tampering-swarm:latest \
  "$project_root"
