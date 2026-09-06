#!/bin/sh
set -eu

if [ -n "${LAB_OBSERVATION_SOURCE:-}" ]; then
  cp "$LAB_OBSERVATION_SOURCE" /agent/LAB_OBSERVATION.txt
fi

exec python /opt/entrypoint.py "$@"
