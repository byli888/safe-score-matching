#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON_BIN:-python}"
"$PY" "$ROOT/train.py" --config "$ROOT/configs/quad2d_stabilize_stage_a.yaml"
"$PY" "$ROOT/train.py" --config "$ROOT/configs/quad2d_stabilize_stage_b.yaml"
