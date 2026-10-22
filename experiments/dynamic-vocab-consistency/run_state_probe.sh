#!/usr/bin/env bash
set -eo pipefail
: "${DEPLOY_ENV:?Set the private deployment environment script}"
: "${SERVE_PYTHON:?Set the model-serving virtualenv Python}"
export QWEN_NPU_DEVICES=7
source "$DEPLOY_ENV"
export ASCEND_RT_VISIBLE_DEVICES=7
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VENDOR=$("$SERVE_PYTHON" -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("vllm_ascend").origin).parent / "_cann_ops_custom/vendors/custom_transformer")')
test -d "$VENDOR"
export ASCEND_CUSTOM_OPP_PATH="$VENDOR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$VENDOR/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$SERVE_PYTHON" "$ROOT/state_probe.py"
