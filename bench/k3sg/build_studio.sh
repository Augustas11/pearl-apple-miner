#!/usr/bin/env bash
# Build the self-contained Studio package bench/k3sg/studio/ (arm64 binary + runtime-compiled Metal sources + oracle
# vectors + manifest), then self-test the package on this Mac (probe + every vector, every cfg; correctness only).
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
swiftc -O -target arm64-apple-macos14.0 k3sg.swift -o studio/k3sg
cp k3sg.metal int8ref.metal mlx_ref.py k3sg.swift studio/
[ -d studio/vectors ] || "$ROOT/.venv/bin/python" make_vectors.py
find studio/vectors -name 'out_*.bin' -delete; rm -f studio/k3sg.local
(cd studio && find . -type f ! -name MANIFEST.sha256 | sort | xargs shasum -a 256 > MANIFEST.sha256)
lipo -archs studio/k3sg; otool -l studio/k3sg | grep -A3 LC_BUILD_VERSION | grep -E "minos|sdk"
du -sh studio
T=$(mktemp -d); cp -R studio/ "$T/"; (cd "$T" && shasum -a 256 -c MANIFEST.sha256 | { grep -vc ': OK$' || true; } | sed 's/^/manifest mismatches: /')
"$T/k3sg" probe | tail -1
for v in "$T"/vectors/*/; do "$T/k3sg" run "$v" "cfgs=64x64x16x2x2x1,64x64x16x2x2x0,64x64x16x2x2x2,64x64x32x2x2x1,64x64x8x2x2x2,128x64x16x4x2x1,128x64x16x4x2x2,64x128x16x2x4x1,128x128x16x4x4x1" | tail -1; done
rm -rf "$T"
echo "sync: rsync -a --delete $PWD/studio/ <target-host>:~/k3sg/"
