#!/bin/bash
# All compilation and Python packaging writes stay below dist/studio_b4.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT="$ROOT/dist/studio_b4"
WORK="$OUT/.build-work"
export MACOSX_DEPLOYMENT_TARGET=14.0
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2}
export GOOS=darwin GOARCH=arm64 CGO_ENABLED=1
export CFLAGS="${CFLAGS:-} -mmacosx-version-min=14.0"
export CXXFLAGS="${CXXFLAGS:-} -mmacosx-version-min=14.0"
export PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
# A dedicated interpreter owns every build backend and its executable PATH.
# Never resolve a local package name against the working directory or index.
[ "$(uname -m)" = arm64 ] || { echo 'arm64 build host required'; exit 1; }
mkdir -p "$WORK" "$OUT/wheels" "$OUT/bin" "$OUT/scripts"
exec > >(tee "$OUT/build.log") 2>&1
# Do not retain obsolete wheels across rebuilds.
rm -f "$OUT/wheels/"*.whl
rm -f "$OUT/MANIFEST.sha256"
# Copy inputs; never run build tools in the source checkout.
for part in pmkcore libpmk miner vendor/pearl; do
  mkdir -p "$WORK/$part"
  rsync -a --delete --exclude target --exclude .build --exclude .git --exclude .venv --exclude __pycache__ --exclude .pytest_cache --exclude .pmk_regtest --exclude '*.egg-info' --exclude build --exclude bin "$ROOT/$part/" "$WORK/$part/"
done
mkdir -p "$WORK/scripts"
cp "$ROOT/scripts/build_pearld.sh" "$WORK/scripts/"
cp "$ROOT/scripts/studio_b4/build-requirements.lock" "$WORK/build-requirements.lock"
cp "$ROOT/scripts/studio_b4/test-requirements.lock" "$WORK/test-requirements.lock"
if [ ! -x "$WORK/build-venv/bin/python" ]; then uv venv --python "$ROOT/.venv/bin/python" --seed "$WORK/build-venv"; fi
PY="$WORK/build-venv/bin/python"
export PATH="$WORK/build-venv/bin:$PATH"
"$PY" -m pip install --only-binary=:all: --require-hashes -r "$WORK/build-requirements.lock"
(cd "$WORK/pmkcore" && cargo build --offline --locked --release -j "$CARGO_BUILD_JOBS")
# SwiftPM embeds a build-host fallback; resolve the shipped resource bundle first.
"$PY" - "$WORK/libpmk/Sources/PMK" <<'PYCODE'
import pathlib, sys
for name in ('Host.swift', 'Probe.swift'):
    p = pathlib.Path(sys.argv[1]) / name
    p.write_text(p.read_text().replace('Bundle.module.resourceURL', '(Bundle(path: ProcessInfo.processInfo.environment["PMK_RESOURCE_BUNDLE"] ?? "") ?? Bundle.module).resourceURL').replace('Bundle.module.url', '(Bundle(path: ProcessInfo.processInfo.environment["PMK_RESOURCE_BUNDLE"] ?? "") ?? Bundle.module).url'))
PYCODE
swift build --package-path "$WORK/libpmk" -c release -j 2
rm -f "$WORK/vendor/pearl/xmss/libxmss.a"
MACOSX_DEPLOYMENT_TARGET=26.0 CFLAGS="-mmacosx-version-min=26.0" CXXFLAGS="-mmacosx-version-min=26.0" bash "$WORK/scripts/build_pearld.sh"
for patch in "$WORK/miner/native_patches/"*.patch; do
  [ -e "$patch" ] || continue
  (cd "$WORK/vendor/pearl" && patch -p1 < "$patch")
done
(cd "$WORK/vendor/pearl/py-pearl-mining" && "$WORK/build-venv/bin/maturin" build --release --locked --interpreter "$PY" --out "$OUT/wheels")
for pkg in "$WORK/miner" "$WORK/vendor/pearl/miner/miner-utils" "$WORK/vendor/pearl/miner/miner-base" "$WORK/vendor/pearl/miner/pearl-gateway"; do
  "$PY" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$OUT/wheels" "$pkg"
done
# Runtime versions and accepted upstream hashes come from B7, not fresh resolution.
DOWNLOAD=("$PY" -m pip download --dest "$OUT/wheels" --only-binary=:all:
  --python-version 312 --implementation cp --abi cp312 --abi abi3 --abi none
  --platform macosx_14_0_arm64 --platform macosx_13_0_arm64
  --platform macosx_12_0_arm64 --platform macosx_11_0_arm64)
# B7's torch hashes come from the PyTorch CPU index, whose arm64 wheel differs
# from PyPI's same-version wheel. Keep the shipped lock byte-for-byte unchanged;
# use only a build-time direct-URL mapping, preserving every approved hash.
"$PY" - "$WORK/miner/requirements.lock" "$WORK/download-requirements.lock" <<'PYCODE'
import pathlib, re, sys
source = pathlib.Path(sys.argv[1]).read_text()
version = re.search(r"^torch==([^ \\n]+)", source, re.M).group(1)
url = f"https://download-r2.pytorch.org/whl/cpu/torch-{version}-cp312-cp312-macosx_11_0_arm64.whl"
source = source.replace(f"torch=={version}", f"torch @ {url}", 1)
pathlib.Path(sys.argv[2]).write_text(source)
PYCODE
"${DOWNLOAD[@]}" --require-hashes --no-deps -r "$WORK/download-requirements.lock"
"${DOWNLOAD[@]}" --require-hashes --no-deps -r "$WORK/test-requirements.lock"

# Late miner-only fixes may land while the native stack is compiling. Refresh
# the copied miner source and rebuild its local wheel immediately before the
# packaging snapshot so the installed wheel and shipped source agree.
rsync -a --delete --exclude .git --exclude .venv --exclude __pycache__ --exclude .pytest_cache --exclude .pmk_regtest --exclude '*.egg-info' --exclude build "$ROOT/miner/" "$WORK/miner/"
rm -f "$OUT/wheels/pmk_miner-"*.whl
"$PY" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$OUT/wheels" "$WORK/miner"

"$PY" "$ROOT/scripts/studio_b4/package_bundle.py" "$ROOT" "$OUT"
echo "Bundle ready: $OUT"
du -sh "$OUT" "$OUT/wheels"
