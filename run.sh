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

MODEL_DIR=${AMB_MINILM_DIR:-"$ROOT_DIR/.cache/models/all-MiniLM-L6-v2"}
if [ ! -f "$MODEL_DIR/model.onnx" ] || [ ! -f "$MODEL_DIR/tokenizer.json" ]; then
  "$VENV_DIR/bin/python" "$ROOT_DIR/tools/download_minilm.py" --destination "$MODEL_DIR"
fi

"$VENV_DIR/bin/python" "$ROOT_DIR/src/benchmark.py" --minilm-dir "$MODEL_DIR" "$@"
"$VENV_DIR/bin/python" "$ROOT_DIR/tools/update_common_registry.py"
"$VENV_DIR/bin/python" "$ROOT_DIR/tools/publish_safe_results.py"
"$VENV_DIR/bin/python" "$ROOT_DIR/tools/build_evidence_manifest.py"
"$VENV_DIR/bin/python" "$ROOT_DIR/tools/build_dashboard_data.py"
