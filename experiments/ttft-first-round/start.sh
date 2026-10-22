#!/usr/bin/env bash
set -eo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${DEPLOY_ENV:?Set the private deployment environment script}"
: "${SERVE_PYTHON:?Set the model-serving virtualenv Python}"
: "${QWEN_MODEL_PATH:?Set the private model checkpoint directory}"
export GATE_LAYOUT_ENABLE=1
export PACKED_PREFILL_ENABLE=1
export DYNAMIC_VOCAB_ENABLE=${1:?0 or 1}
export QWEN_NPU_DEVICES="${TTFT_PHYSICAL_CARD:-6}"
source "$DEPLOY_ENV"
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES="$QWEN_NPU_DEVICES"
unset DOWN_TUNING_ENABLE NZ_LAUNCH_TRACE_DIR LD_PRELOAD NATIVE_M8_RETILE NATIVE_M8_PRESERVE_L1 NATIVE_M8_BALANCE_L0
if [[ -n "${TTFT_HOST_ALLOCATOR:-}" ]]; then
  test -f "$TTFT_HOST_ALLOCATOR"
  export LD_PRELOAD="$TTFT_HOST_ALLOCATOR"
fi
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_SERVER_DEV_MODE=1
: "${COUNT_READY_BASE:?Set the count-ready source directory}"
export PYTHONPATH="$ROOT:$COUNT_READY_BASE${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export STATE_DIAGNOSTIC_ENABLE=${STATE_DIAGNOSTIC_ENABLE:-0}
export XDG_CACHE_HOME="${OVERLAP_CACHE_ROOT:-$ROOT/cache-fp32-$STATE_DIAGNOSTIC_ENABLE-$DYNAMIC_VOCAB_ENABLE}"
export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export ASCEND_WORK_PATH="$XDG_CACHE_HOME/ascend-work" ASCEND_CACHE_PATH="$XDG_CACHE_HOME/ascend"
export MAX_JOBS=4
exec "$SERVE_PYTHON" "$COUNT_READY_BASE/serve.py" \
 --model "$QWEN_MODEL_PATH" --served-model-name Qwen3.8-27B \
 --host 127.0.0.1 --port 8047 --tensor-parallel-size 1 --dtype bfloat16 \
 --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 512 \
 --gpu-memory-utilization 0.97 --no-enable-prefix-caching \
 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8]}' \
 --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}' \
 --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype float32 \
 --worker-extension-cls ttft_extension.GraphAudit \
 --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":7,"enforce_eager":false}' \
 --no-enable-log-requests \
 --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$ROOT/profiles\",\"torch_profiler_with_stack\":false,\"torch_profiler_with_memory\":false,\"ignore_frontend\":true}"
