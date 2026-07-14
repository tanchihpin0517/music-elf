#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: bash scripts/launch.sh <train|eval> <config.yml> [extra args...]"
  exit 1
fi

MODE=$1
CONFIG=$2
shift 2

case "$MODE" in
  train) exec python -m elf.train --config "$CONFIG" "$@" ;;
  eval)  exec python -m elf.eval  --config "$CONFIG" "$@" ;;
  *) echo "Unknown mode: $MODE"; exit 1 ;;
esac
