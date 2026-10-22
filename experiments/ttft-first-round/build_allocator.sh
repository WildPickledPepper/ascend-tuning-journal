#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${JEMALLOC_ARCHIVE:?Set the official jemalloc-5.3.0.tar.bz2 archive}"
: "${SERVE_PYTHON:?Set a Python 3.12+ executable for safe tar extraction}"
"$SERVE_PYTHON" - "$JEMALLOC_ARCHIVE" "$ROOT" <<'PY'
import hashlib,sys,tarfile
from pathlib import Path
p,root=map(Path,sys.argv[1:])
assert hashlib.sha256(p.read_bytes()).hexdigest()=='2db82d1e7119df3e71b7640219b6dfe84789bc0537983c3b7ac4f7189aecfeaa'
assert not (root/'jemalloc-5.3.0').exists()
with tarfile.open(p) as t:t.extractall(root,filter='data')
PY
cd "$ROOT/jemalloc-5.3.0"
./configure --prefix="$ROOT/allocator" --disable-stats --disable-prof
make -j2
make install_lib_shared
