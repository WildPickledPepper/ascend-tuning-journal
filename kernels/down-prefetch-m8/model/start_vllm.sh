#!/usr/bin/env bash
set -eo pipefail
: "${QWEN_MODEL_ROOT:?Set private model root}"
: "${DFLASH_RUN:?Set private experiment directory}"
mode=${1:?mtp or dflash2}
[[ $mode == mtp || $mode == dflash2 ]] || exit 2
export QWEN_NPU_DEVICES=6
source "$QWEN_MODEL_ROOT/env-npu.sh"
unset DOWN_TUNING_ENABLE VLLM_SERVER_DEV_MODE NZ_LAUNCH_TRACE_DIR
unset LD_PRELOAD NATIVE_M8_RETILE NATIVE_M8_PRESERVE_L1 NATIVE_M8_BALANCE_L0
export DOWN_PREFETCH_M8_ENABLE=${DOWN_PREFETCH_M8_ENABLE:-0}
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_SERVER_DEV_MODE=1
export PYTHONPATH="$DFLASH_RUN${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT="$DFLASH_RUN/cache-prefetch-${DOWN_PREFETCH_M8_ENABLE}-$mode/vllm"
export XDG_CACHE_HOME="$DFLASH_RUN/cache-prefetch-${DOWN_PREFETCH_M8_ENABLE}-$mode"
export TORCHINDUCTOR_CACHE_DIR="$DFLASH_RUN/cache-prefetch-${DOWN_PREFETCH_M8_ENABLE}-$mode/torchinductor"
export TRITON_CACHE_DIR="$DFLASH_RUN/cache-prefetch-${DOWN_PREFETCH_M8_ENABLE}-$mode/triton"
export ASCEND_WORK_PATH="$DFLASH_RUN/cache-prefetch-${DOWN_PREFETCH_M8_ENABLE}-$mode/ascend-work"
export ASCEND_CACHE_PATH="$DFLASH_RUN/cache-prefetch-${DOWN_PREFETCH_M8_ENABLE}-$mode/ascend"
export MAX_JOBS=4
if [[ $mode == mtp ]]; then
  spec='{"method":"qwen3_5_mtp","num_speculative_tokens":7,"enforce_eager":false}'
  capture=8
else
  spec=$(python3 - "$DFLASH_RUN/models/Qwen3.8-27B-DFlash2" <<'PYCFG'
import json,sys
print(json.dumps(dict(method='dflash',model=sys.argv[1],num_speculative_tokens=7,enforce_eager=True)))
PYCFG
)
  capture=8
fi
exec "$DFLASH_RUN/runtime/venv/bin/python" "$DFLASH_RUN/serve.py" \
  --model "$QWEN_MODEL_ROOT/models/Qwen3.8-27B" --served-model-name Qwen3.8-27B \
  --host 127.0.0.1 --port 8040 --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.97 --no-enable-prefix-caching \
  --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$capture]}" \
  --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}' \
  --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16 \
  --worker-extension-cls model_extension.GraphAudit --speculative-config "$spec" --profiler-config "$(python3 -c 'import json,os; print(json.dumps(dict(profiler="torch", torch_profiler_dir=os.path.join(os.environ["DFLASH_RUN"],"profiles"))))')" --no-enable-log-requests "${@:2}"
