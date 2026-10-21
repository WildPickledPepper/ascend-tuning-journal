#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"
: "${ASCEND_HOME_PATH:?Activate the matching private CANN 9.1 environment first}"
bisheng -xcce -O2 -g --cce-aicore-arch=dav-c220-cube \
  -fPIC -shared -std=c++17 -I"$ASCEND_HOME_PATH/include" \
  -I"$ASCEND_HOME_PATH/aarch64-linux/asc/include" \
  -I"$ASCEND_HOME_PATH/aarch64-linux/asc/include/basic_api" \
  -I"$ASCEND_HOME_PATH/aarch64-linux/asc/impl" \
  -I"$ASCEND_HOME_PATH/aarch64-linux/asc" \
  -isystem /usr/include/c++/11 -isystem /usr/include/aarch64-linux-gnu/c++/11 \
  -isystem /usr/include/aarch64-linux-gnu -isystem /usr/include \
  -L"$ASCEND_HOME_PATH/lib64" -lascendcl -lruntime \
  down_prefetch.asc -o libdown_prefetch_m8.so
python3 build_bridge.py
