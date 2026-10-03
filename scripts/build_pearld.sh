#!/usr/bin/env bash
# Build pearld (+ prlctl) from vendor/pearl without go-task.
# Mirrors Taskfile.yml targets build:zk-cache -> build:zk-gobind -> build:libxmss -> build:pearld/prlctl.
# Requires: Go, Rust (cargo), a C/C++ compiler (clang on macOS).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PEARL="$ROOT/vendor/pearl"
export CGO_ENABLED=1
export CGO_LDFLAGS_ALLOW=".*zk_pow_ffi.*"

cd "$PEARL/zk-pow"
if [ ! -s src/circuit/v2_cache.bin ] || [ ! -s src/v1/v1_cache.bin ]; then
  echo "== build:zk-cache"
  cargo run --release --no-default-features --bin build_cache src/circuit/v2_cache.bin src/v1/v1_cache.bin
fi

echo "== build:zk-gobind"
(cd "$PEARL/zk-pow/bindings/go" && cargo build --release)

echo "== build:libxmss"
if [ ! -f "$PEARL/xmss/libxmss.a" ]; then
  (cd "$PEARL/xmss" && make CC="${CC:-clang}" CXX="${CXX:-clang++}")
fi

echo "== build:pearld / prlctl"
cd "$PEARL"
mkdir -p bin
go build -tags xmss,zkpow -o bin/pearld ./node
go build -tags xmss,zkpow -o bin/prlctl ./node/cmd/prlctl
ls -la bin/pearld bin/prlctl
