#!/usr/bin/env bash
set -eo pipefail
: "${QWEN_MODEL_ROOT:?}"
: "${AUTHOR_RUN:?}"
export QWEN_NPU_DEVICES=6,7
source "$QWEN_MODEL_ROOT/env-npu.sh"
export PATH="$AUTHOR_RUN/runtime/venv/bin:$PATH"
exec "$AUTHOR_RUN/runtime/venv/bin/python" -u "$AUTHOR_RUN/ascend_client.py" "$AUTHOR_RUN" "${1:?mode}"
