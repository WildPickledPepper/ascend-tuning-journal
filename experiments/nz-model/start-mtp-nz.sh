#!/usr/bin/env bash
set -eo pipefail
: "${QWEN_MODEL_ROOT:?Set model root}"
: "${MTP_RUN:?Set result directory}"
mode=${1:?baseline or mtp}
[[ $mode == baseline || $mode == mtp ]] || exit 2
if [[ $mode == baseline ]]; then device=7; port=8038; else device=6; port=8039; fi
export QWEN_NPU_DEVICES=$device
source "$QWEN_MODEL_ROOT/env-npu.sh"
unset DOWN_TUNING_ENABLE
export VLLM_CACHE_ROOT="$MTP_RUN/cache-$mode"
mkdir -p "$MTP_RUN/profiles-$mode"
config=$(python3 - "$MTP_RUN/profiles-$mode" <<'PYCFG'
import json,sys
print(json.dumps(dict(profiler='torch',torch_profiler_dir=sys.argv[1],torch_profiler_with_stack=False,torch_profiler_use_gzip=False,ignore_frontend=True)))
PYCFG
)
extra=()
capture=1
if [[ $mode == mtp ]]; then
 extra=(--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1,"enforce_eager":true}')
 capture=2
fi
exec python3 -m vllm.entrypoints.openai.api_server \
 --model "$QWEN_MODEL_ROOT/models/Qwen3.8-27B" --served-model-name Qwen3.8-27B \
 --host 127.0.0.1 --port "$port" --tensor-parallel-size 1 --dtype bfloat16 \
 --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 512 \
 --gpu-memory-utilization 0.90 --no-enable-prefix-caching \
 --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$capture]}" \
 --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":0,"video":0}' \
 --additional-config '{"weight_nz_mode":2}' \
 --profiler-config "$config" --no-enable-log-requests "${extra[@]}"
