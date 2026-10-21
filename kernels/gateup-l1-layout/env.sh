#!/usr/bin/env bash
set -eo pipefail
: "${DEPLOY_ENV:?Set the private deployment environment script}"
: "${QWEN_MODEL_PATH:?Set the private model checkpoint directory}"
export QWEN_NPU_DEVICES=7
source "$DEPLOY_ENV"
export ASCEND_RT_VISIBLE_DEVICES=7
cd "$(dirname "${BASH_SOURCE[0]}")"
exec "$@"
