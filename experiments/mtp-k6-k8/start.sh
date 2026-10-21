#!/usr/bin/env bash
set -eo pipefail
: "${QWEN_MODEL_ROOT:?}" "${DFLASH_RUN:?}" "${MTP_K:?}"
MTP_VERIFY_TOKENS=$((MTP_K + 1))
export QWEN_NPU_DEVICES=6
source "$QWEN_MODEL_ROOT/env-npu.sh"
unset LD_PRELOAD DOWN_TUNING_ENABLE DOWN_PREFETCH_M8_ENABLE NATIVE_M8_RETILE NATIVE_M8_PRESERVE_L1 NATIVE_M8_BALANCE_L0
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_SERVER_DEV_MODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$DFLASH_RUN${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="$DFLASH_RUN/cache/vllm" XDG_CACHE_HOME="$DFLASH_RUN/cache"
export TORCHINDUCTOR_CACHE_DIR="$DFLASH_RUN/cache/torchinductor" TRITON_CACHE_DIR="$DFLASH_RUN/cache/triton"
export ASCEND_WORK_PATH="$DFLASH_RUN/cache/ascend-work" ASCEND_CACHE_PATH="$DFLASH_RUN/cache/ascend"
export MAX_JOBS=4
exec "$DFLASH_RUN/runtime/venv/bin/python" "$DFLASH_RUN/serve.py" \
 --model "$QWEN_MODEL_ROOT/models/Qwen3.8-27B" --served-model-name Qwen3.8-27B \
 --host 127.0.0.1 --port 8040 --tensor-parallel-size 1 --dtype bfloat16 \
 --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 512 \
 --gpu-memory-utilization 0.97 --no-enable-prefix-caching \
 --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$MTP_VERIFY_TOKENS]}" \
 --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}' \
 --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16 \
 --worker-extension-cls model_extension.GraphAudit \
 --speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$MTP_K,\"enforce_eager\":false}" --no-enable-log-requests
