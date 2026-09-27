#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON_BIN:-python}" "$ROOT/train.py" --config "$ROOT/configs/f16_stage_b.yaml" "$@"
