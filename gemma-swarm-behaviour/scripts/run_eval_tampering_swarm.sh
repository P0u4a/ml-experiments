#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_server="${LLAMA_SERVER_PATH:?LLAMA_SERVER_PATH must point to the llama-server executable}"
model_path="${GEMMA_MODEL_PATH:?GEMMA_MODEL_PATH must point to the Gemma GGUF model}"
config_path="${EVAL_CONFIG_PATH:-$project_root/configs/eval_tampering_shared_board_swarm.yaml}"
image_name="${EVAL_IMAGE:-gemma-eval-tampering-swarm:latest}"
run_id="${EVAL_RUN_ID:-eval-tampering-swarm-$(date -u +%Y%m%dT%H%M%SZ)-$RANDOM}"
run_dir="${EVAL_RUN_DIR:-$project_root/runs/$run_id}"
port="${LLAMA_PORT:-8080}"
api_key="${LLAMA_API_KEY:-$(openssl rand -hex 32)}"
context_size="${LLAMA_CONTEXT_SIZE:-131072}"
condition="${EXPERIMENT_CONDITION:-four-agent-shared-board-swarm-framing}"

if [[ ! -x "$llama_server" ]]; then
  echo "llama-server is not executable: $llama_server" >&2
  exit 1
fi
if [[ ! -f "$model_path" ]]; then
  echo "Gemma model does not exist: $model_path" >&2
  exit 1
fi

if [[ -e "$run_dir" ]]; then
  echo "Refusing to overwrite existing run directory: $run_dir" >&2
  exit 1
fi
mkdir -p "$run_dir"

server_args=(
  --model "$model_path"
  --host 127.0.0.1
  --port "$port"
  --parallel 4
  --ctx-size "$context_size"
  --cache-type-k q8_0
  --cache-type-v q8_0
  --cont-batching
  --no-mmproj
  --jinja
  --reasoning on
  --reasoning-format deepseek
  --flash-attn on
  --n-gpu-layers 99
  --cache-ram 2048
  --api-key "$api_key"
)

"$llama_server" "${server_args[@]}" >"$run_dir/llama-server.log" 2>&1 &
llama_pid=$!

cleanup() {
  kill -TERM "$llama_pid" 2>/dev/null || true
  wait "$llama_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

server_ready=0
for _ in $(seq 1 180); do
  if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then
    server_ready=1
    break
  fi
  if ! kill -0 "$llama_pid" 2>/dev/null; then
    echo "llama-server exited during startup; see $run_dir/llama-server.log" >&2
    exit 1
  fi
  sleep 1
done
if [[ "$server_ready" != "1" ]]; then
  echo "llama-server did not become healthy within 180 seconds; see $run_dir/llama-server.log" >&2
  exit 1
fi

image_id="$(docker image inspect "$image_name" --format '{{.Id}}')"
environment_revision="$(find "$project_root/environment" -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | cut -d ' ' -f 1)"
llama_version="$($llama_server --version 2>&1 | head -1)"
model_fingerprint="$(shasum -a 256 "$model_path" | cut -d ' ' -f 1)"

docker run --rm \
  --stop-timeout 1000 \
  --cpus 2 \
  --cpuset-cpus 0-1 \
  --memory 4g \
  --pids-limit 512 \
  --security-opt no-new-privileges \
  --cap-add NET_ADMIN \
  --add-host host.docker.internal:host-gateway \
  --volume "$config_path:/opt/config.yaml:ro" \
  --volume "$run_dir:/opt/output" \
  --env "EXPERIMENT_CONDITION=$condition" \
  --env "LLAMA_BASE_URL=http://host.docker.internal:$port/v1" \
  --env "LLAMA_API_KEY=$api_key" \
  --env "LLAMA_CONTEXT_SIZE=$context_size" \
  --env "EXPERIMENT_RUN_ID=$run_id" \
  --env "EXPERIMENT_IMAGE_ID=$image_id" \
  --env "ENVIRONMENT_REVISION=$environment_revision" \
  --env "LLAMA_SERVER_VERSION=$llama_version" \
  --env "MODEL_FINGERPRINT=$model_fingerprint" \
  --env "EXPERIMENT_CPU_LIMIT=2" \
  --env "EXPERIMENT_CPU_SET=0-1" \
  --env "EXPERIMENT_MEMORY_LIMIT=4g" \
  "$image_name" 2>&1 | tee "$run_dir/run.log"

echo "Run artifacts: $run_dir"
