#!/usr/bin/env bash
set -eo pipefail
: "${QWEN_MODEL_ROOT:?Set private model root}"
: "${DFLASH_RUN:?Set private experiment directory}"
: "${DFLASH_DRAFT:?Set installed BF16 draft checkpoint}"
export QWEN_NPU_DEVICES=6
source "$QWEN_MODEL_ROOT/env-npu.sh"
unset LD_PRELOAD DOWN_TUNING_ENABLE DOWN_PREFETCH_M8_ENABLE NATIVE_M8_RETILE
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_SERVER_DEV_MODE=1
export PYTHONPATH="$DFLASH_RUN${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="$DFLASH_RUN/cache/vllm" XDG_CACHE_HOME="$DFLASH_RUN/cache"
export TORCHINDUCTOR_CACHE_DIR="$DFLASH_RUN/cache/torchinductor"
export TRITON_CACHE_DIR="$DFLASH_RUN/cache/triton"
export ASCEND_WORK_PATH="$DFLASH_RUN/cache/ascend-work" ASCEND_CACHE_PATH="$DFLASH_RUN/cache/ascend"
export MAX_JOBS=4
spec=$(python3 -c 'import json,os; print(json.dumps(dict(method="dflash",model=os.environ["DFLASH_DRAFT"],num_speculative_tokens=7,enforce_eager=False)))')
profile=$(python3 -c 'import json,os; print(json.dumps(dict(profiler="torch",torch_profiler_dir=os.path.join(os.environ["DFLASH_RUN"],"profiles"))))')
exec "$DFLASH_RUN/runtime/venv/bin/python" "$DFLASH_RUN/serve.py" \
  --model "$QWEN_MODEL_ROOT/models/Qwen3.8-27B" --served-model-name Qwen3.8-27B \
  --host 127.0.0.1 --port 8040 --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.97 --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8]}' \
  --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}' \
  --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16 \
  --worker-extension-cls graph_audit.GraphAudit --speculative-config "$spec" \
  --profiler-config "$profile" --no-enable-log-requests
