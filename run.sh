#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV_DIR="$ROOT_DIR/.venv"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV_DIR" --python 3.11
    uv pip install --python "$VENV_DIR/bin/python" -e "$ROOT_DIR[test]"
  else
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install -e "$ROOT_DIR[test]"
  fi
fi

exec "$VENV_DIR/bin/python" "$ROOT_DIR/src/benchmark.py" "$@"
