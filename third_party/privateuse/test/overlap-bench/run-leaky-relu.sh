#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
runtime="$root/backend/runtime"
bin="${TMPDIR:-/tmp}/pumem_leaky_relu_bench"
g++ -std=gnu++17 -O2 -pthread -DNDEBUG -I "$runtime" \
  "$here/leaky_relu_bench.cpp" \
  "$runtime/pumem_async_runtime.cpp" \
  "$runtime/cpu_backend_runtime.cpp" \
  -o "$bin"
exec "$bin" "$@"