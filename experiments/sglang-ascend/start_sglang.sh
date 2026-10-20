#!/usr/bin/env bash
set -eo pipefail
: "${QWEN_MODEL_ROOT:?}"
: "${AUTHOR_RUN:?}"
mode=${1:?ar, mtp, or dflash2}
[[ $mode == ar || $mode == mtp || $mode == dflash2 ]] || exit 2
export QWEN_NPU_DEVICES=6,7
source "$QWEN_MODEL_ROOT/env-npu.sh"
source "$AUTHOR_RUN/runtime/custom-opp-2026.9.1/vendors/customize/bin/set_env.bash"
source "$AUTHOR_RUN/runtime/custom-opp-2026.9.1/vendors/custom_transformer/bin/set_env.bash"
unset LD_PRELOAD DOWN_TUNING_ENABLE VLLM_SERVER_DEV_MODE NZ_LAUNCH_TRACE_DIR
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SGLANG_USE_NPU=1
export PATH="$AUTHOR_RUN/runtime/venv/bin:$PATH"
export MAX_JOBS=4 OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$AUTHOR_RUN/cache"
export TRITON_CACHE_DIR="$AUTHOR_RUN/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$AUTHOR_RUN/cache/torchinductor"
export ASCEND_WORK_PATH="$AUTHOR_RUN/cache/ascend-work"
export ASCEND_CACHE_PATH="$AUTHOR_RUN/cache/ascend"
spec=()
if [[ $mode == dflash2 ]]; then
  spec=(--speculative-algorithm DFLASH
    --speculative-draft-model-path "$QWEN_MODEL_ROOT/experiments/dflash2-20261020/models/Qwen3.8-27B-DFlash2"
    --speculative-num-draft-tokens 8)
elif [[ $mode == mtp ]]; then
  spec=(--speculative-algorithm EAGLE --speculative-num-steps 7
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 8)
fi
exec "$AUTHOR_RUN/runtime/venv/bin/python" -u -m sglang.launch_server \
  --model-path "$QWEN_MODEL_ROOT/models/Qwen3.8-27B" \
  --served-model-name Qwen3.8-27B --device npu --tp-size 2 --dtype bfloat16 \
  --disable-radix-cache --mem-fraction-static 0.90 --max-running-requests 1 \
  --context-length 32768 --attention-backend ascend \
  --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer \
  --host 127.0.0.1 --port 8040 "${spec[@]}" "${@:2}"
