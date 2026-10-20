#!/usr/bin/env bash
# Reuse official prebuilt framework code with the working private CANN/PyTorch.
# This does not install into the original runtime or compile an NPU kernel.
set -eo pipefail
: "${QWEN_MODEL_ROOT:?Set private model root}"
: "${DFLASH_RUN:?Set private experiment directory}"
image=quay.io/ascend/vllm-ascend:v0.27.1rc1
ctr=codex-dflash2-extract
mkdir -p "$DFLASH_RUN/runtime"
docker create --name "$ctr" --entrypoint true "$image" >/dev/null
trap 'docker rm "$ctr" >/dev/null 2>&1 || true' EXIT
for name in vllm vllm-ascend; do
  if [[ ! -d "$DFLASH_RUN/runtime/$name" ]]; then
    docker cp "$ctr:/vllm-workspace/$name" "$DFLASH_RUN/runtime/"
  fi
done
export QWEN_NPU_DEVICES=6
source "$QWEN_MODEL_ROOT/env-npu.sh"
python3 -m venv --system-site-packages "$DFLASH_RUN/runtime/venv"
site=$("$DFLASH_RUN/runtime/venv/bin/python" -c 'import site;print(site.getsitepackages()[0])')
docker cp "$ctr:/usr/local/python3.12.13/lib/python3.12/site-packages/vllm-0.27.1+empty.dist-info" "$site/"
docker cp "$ctr:/usr/local/python3.12.13/lib/python3.12/site-packages/vllm_ascend-0.27.1rc1.dist-info" "$site/"
"$DFLASH_RUN/runtime/venv/bin/python" - "$DFLASH_RUN/runtime" "$site" <<'PY'
import pathlib,sys
root=pathlib.Path(sys.argv[1]);site=pathlib.Path(sys.argv[2])
paths=[str(root/'vllm'),str(root/'vllm-ascend')]
(site/'dflash_runtime.pth').write_text('import sys; sys.path[:0] = '+repr(paths)+'\n')
PY
