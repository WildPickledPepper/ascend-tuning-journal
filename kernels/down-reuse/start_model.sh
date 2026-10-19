#!/usr/bin/env bash
set -eo pipefail
: "${QWEN_MODEL_ROOT:?Set the prepared model root}"
: "${DOWN_TUNING_ROOT:?Set the experiment directory}"
: "${DOWN_MODEL_RUN:?Set the result directory}"
mode=$1
case "$mode" in baseline-a|optimized-b|optimized-c|optimized-d|baseline-a2) ;; *) exit 2;; esac
export QWEN_NPU_DEVICES=7
source "$QWEN_MODEL_ROOT/env-npu.sh"
export PYTHONPATH="$DOWN_TUNING_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DOWN_TUNING_ENABLE=0
[[ $mode != optimized-* ]] || export DOWN_TUNING_ENABLE=1
export VLLM_SERVER_DEV_MODE=1
cache_mode=$mode
[[ $mode != baseline-a2 ]] || cache_mode=baseline-a
export VLLM_CACHE_ROOT="$DOWN_MODEL_RUN/cache-$cache_mode"
mkdir -p "$DOWN_MODEL_RUN/profiles-$mode"
profiler=$(python3 - "$DOWN_MODEL_RUN/profiles-$mode" <<'PY'
import json,sys
print(json.dumps({'profiler':'torch','torch_profiler_dir':sys.argv[1],
                 'torch_profiler_with_stack':False,'torch_profiler_use_gzip':False,'ignore_frontend':True}))
PY
)
exec python3 -m vllm.entrypoints.openai.api_server \
 --model "$QWEN_MODEL_ROOT/models/Qwen3.8-27B" --served-model-name Qwen3.8-27B \
 --host 127.0.0.1 --port 8038 --tensor-parallel-size 1 --dtype bfloat16 \
 --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 512 \
 --gpu-memory-utilization 0.90 --no-enable-prefix-caching \
 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}' \
 --worker-extension-cls down_model_extension.WorkerExtension \
 --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}' \
 --profiler-config "$profiler" --no-enable-log-requests
