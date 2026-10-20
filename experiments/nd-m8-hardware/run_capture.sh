#!/usr/bin/env bash
set -eo pipefail
export QWEN_NPU_DEVICES=6
source "$QWEN_MODEL_ROOT/env-npu.sh"
export PYTHONPATH="$DFLASH_RUN"
exec "$DFLASH_RUN/runtime/venv/bin/python" -u "$DFLASH_RUN/capture_client.py" "$DFLASH_RUN"
