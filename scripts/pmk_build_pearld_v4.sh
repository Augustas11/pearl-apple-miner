#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
# Build pinned Pearl v4 pearld/prlctl from vendor/pearl-fp8 in a derived tree.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIN="${PMK_PEARL_V4_PIN:-f696760b259500ecb608469ea3953aeabbe78948}"
VENDOR="${PMK_PEARL_V4_VENDOR:-$ROOT/vendor/pearl-fp8}"
BUILD_ROOT="${PMK_PEARL_V4_BUILD_ROOT:-$ROOT/bench/v4_emulation/pearl-build}"
SRC_ROOT="$BUILD_ROOT/src"
PEARL="$SRC_ROOT/pearl-$PIN"
STAMP="$PEARL/.pmk-v4-pin"

usage() {
  cat <<EOF
usage: scripts/pmk_build_pearld_v4.sh [--refresh] [--no-build] [--run-tests]

Build outputs:
  $BUILD_ROOT/bin/pearld
  $BUILD_ROOT/bin/prlctl

Environment:
  PMK_PEARL_V4_PIN          Pearl commit to export (default: $PIN)
  PMK_PEARL_V4_VENDOR       local pinned source (default: vendor/pearl-fp8)
  PMK_PEARL_V4_BUILD_ROOT   derived build/cache root
  PMK_V4_CARGO_JOBS         cargo jobs (default: 2)
  PMK_V4_GO_P               go package parallelism (default: 2)
  RAYON_NUM_THREADS         Rust rayon threads (default: 2)
EOF
}

REFRESH=0
NO_BUILD=0
RUN_TESTS=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --refresh)
      REFRESH=1
      shift
      ;;
    --no-build)
      NO_BUILD=1
      shift
      ;;
    --run-tests)
      RUN_TESTS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ ! -d "$VENDOR/.git" ]; then
  echo "missing Pearl vendor git checkout: $VENDOR" >&2
  exit 1
fi

ACTUAL_PIN="$(git -C "$VENDOR" rev-parse HEAD)"
if [ "$ACTUAL_PIN" != "$PIN" ]; then
  echo "vendor pin mismatch: got $ACTUAL_PIN, expected $PIN" >&2
  exit 1
fi

mkdir -p "$SRC_ROOT" "$BUILD_ROOT/bin" "$BUILD_ROOT/cache/go-build" \
  "$BUILD_ROOT/cache/go-mod" "$BUILD_ROOT/cache/cargo-home"

if [ "$REFRESH" -eq 1 ] || [ ! -f "$STAMP" ] || [ "$(cat "$STAMP" 2>/dev/null || true)" != "$PIN" ]; then
  TMP="$SRC_ROOT/.pearl-$PIN.tmp.$$"
  rm -rf "$TMP"
  mkdir -p "$TMP"
  git -C "$VENDOR" archive --format=tar "$PIN" | tar -xf - -C "$TMP"
  rm -rf "$PEARL"
  mv "$TMP" "$PEARL"
  printf '%s\n' "$PIN" > "$STAMP"
fi

if [ "$NO_BUILD" -eq 1 ]; then
  echo "$PEARL"
  exit 0
fi

export CGO_ENABLED=1
export CGO_LDFLAGS_ALLOW=".*zk_pow_ffi.*"
export CARGO_HOME="$BUILD_ROOT/cache/cargo-home"
export CARGO_BUILD_JOBS="${PMK_V4_CARGO_JOBS:-2}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-2}"
export GOCACHE="$BUILD_ROOT/cache/go-build"
export GOMODCACHE="$BUILD_ROOT/cache/go-mod"
export GOFLAGS="${GOFLAGS:-} -p=${PMK_V4_GO_P:-2}"
export MAKEFLAGS="${MAKEFLAGS:--j${PMK_V4_MAKE_JOBS:-2}}"

cd "$PEARL/zk-pow"
if [ ! -s src/api/fp8/fp8_cache.bin ] || [ ! -s src/v2/circuit/v2_cache.bin ] || [ ! -s src/v1/v1_cache.bin ]; then
  echo "== build:zk-cache fp8/v2/v1"
  cargo run --release --jobs "$CARGO_BUILD_JOBS" --no-default-features --bin build_cache \
    src/api/fp8/fp8_cache.bin src/v2/circuit/v2_cache.bin src/v1/v1_cache.bin
else
  echo "== build:zk-cache fp8/v2/v1 (cached)"
fi

echo "== build:zk-gobind"
(cd "$PEARL/zk-pow/bindings/go" && cargo build --release --jobs "$CARGO_BUILD_JOBS")

echo "== build:libxmss"
if [ ! -f "$PEARL/xmss/libxmss.a" ]; then
  (cd "$PEARL/xmss" && make CC="${CC:-clang}" CXX="${CXX:-clang++}")
else
  echo "== build:libxmss (cached)"
fi

echo "== build:pearld / prlctl"
cd "$PEARL"
mkdir -p bin
go build -tags xmss,zkpow -trimpath -o bin/pearld ./node
go build -tags xmss,zkpow -trimpath -o bin/prlctl ./node/cmd/prlctl
ln -sfn "$PEARL/bin/pearld" "$BUILD_ROOT/bin/pearld"
ln -sfn "$PEARL/bin/prlctl" "$BUILD_ROOT/bin/prlctl"
ls -la "$BUILD_ROOT/bin/pearld" "$BUILD_ROOT/bin/prlctl"

if [ "$RUN_TESTS" -eq 1 ]; then
  echo "== targeted node/wire/activation tests"
  go test -tags xmss,zkpow -run 'Test(CertificateV4Wire|MsgCertificate|Fp8ForkActivation|ShippedNetworksFp8ForkHeights|NewBlockTemplateForkRulesActive|BlockTemplateResultRequiredCertVersion)$' \
    ./node/wire ./node/chaincfg ./node/mining ./node
  echo "== targeted v4 verifier tests"
  go test -tags xmss,zkpow -run 'TestVerifyCertificateV4|TestCertificateV4AncestorValidation' \
    ./node/zkpow ./node/blockchain
fi
