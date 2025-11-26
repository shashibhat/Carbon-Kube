#!/usr/bin/env sh
set -e
cmd="${1:-mutator}"
case "$cmd" in
  poll)
    exec python3 /app/cmd/poll/main.py
    ;;
  mutator)
    exec /app/bin/mutator
    ;;
  taintcontroller)
    exec /app/bin/taintcontroller
    ;;
  *)
    echo "unknown command: $cmd" >&2
    exit 1
    ;;
esac