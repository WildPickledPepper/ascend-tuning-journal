#!/usr/bin/env bash
set -eo pipefail
: "${QWEN_MODEL_ROOT:?Set private model root}"
: "${NZ_HARDWARE_RUN:?Set private result root}"
previous="$QWEN_MODEL_ROOT/experiments/nz-runtime-20261019"
export QWEN_NPU_DEVICES=6
source "$QWEN_MODEL_ROOT/env-npu.sh"
unset LD_PRELOAD DOWN_TUNING_ENABLE
export PYTHONPATH="$NZ_HARDWARE_RUN:$previous:$PYTHONPATH"
export VLLM_SERVER_DEV_MODE=1
export NZ_LAUNCH_TRACE_DIR="$NZ_HARDWARE_RUN/metadata"
export VLLM_CACHE_ROOT="$previous/model/cache-mtp"
mkdir -p "$NZ_LAUNCH_TRACE_DIR"
exec python3 -m vllm.entrypoints.openai.api_server \
 --model "$QWEN_MODEL_ROOT/models/Qwen3.8-27B" --served-model-name Qwen3.8-27B \
 --host 127.0.0.1 --port 8039 --tensor-parallel-size 1 --dtype bfloat16 \
 --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 512 \
 --gpu-memory-utilization 0.90 --no-enable-prefix-caching \
 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[2]}' \
 --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}' \
 --additional-config '{"weight_nz_mode":2}' \
 --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1,"enforce_eager":true}' \
 --worker-extension-cls hardware_observe.WorkerExtension --no-enable-log-requests
