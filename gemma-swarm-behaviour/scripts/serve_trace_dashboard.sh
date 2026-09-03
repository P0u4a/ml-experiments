#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
port="${TRACE_DASHBOARD_PORT:-4173}"

python3 "$project_root/dashboard/build_data.py"
exec python3 "$project_root/dashboard/server.py" --port "$port"
