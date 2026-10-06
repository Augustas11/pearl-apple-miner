#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VERSION="${PMK_RELEASE_VERSION:-0.2.0-beta.16}"
NAME="pmk-macos-arm64-$VERSION"
DIST="$ROOT/dist"
WORK="$DIST/release-work"
STAGE="$WORK/$NAME"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20261003/cpython-3.12.15%2B20261003-aarch64-apple-darwin-install_only.tar.gz"
PY_SHA256="316a463172740e71d8dca1f2730784e325f3f720941137b5d674d5801a632213"
V4_SOURCE_MANIFEST="${PMK_V4_G3_MANIFEST:-}"

die() { echo "build_release: $*" >&2; exit 1; }
[ "$(uname -s)" = Darwin ] || die "release build requires macOS"
[ "$(uname -m)" = arm64 ] || die "release build requires arm64"
[ -n "$V4_SOURCE_MANIFEST" ] || die "set PMK_V4_G3_MANIFEST to the trusted full v4 G3 manifest"
[ -f "$V4_SOURCE_MANIFEST" ] || die "v4 G3 manifest not found"
for tool in cargo swift curl shasum tar codesign file otool install_name_tool uv; do
  command -v "$tool" >/dev/null 2>&1 || die "missing build tool: $tool"
done

echo "==> Building native release artifacts"
"$ROOT/scripts/fetch_vendor.sh" --fp8
(cd "$ROOT/pmkcore" && cargo build --release --locked)
(cd "$ROOT/pmkcore/v4" && cargo build --release --locked --lib --bins)
(cd "$ROOT/libpmk" && swift build -c release)

rm -rf "$WORK"
mkdir -p "$WORK" "$STAGE/bin" "$STAGE/miner" "$STAGE/scripts/release" \
  "$STAGE/pmkcore/target/release" "$STAGE/pmkcore/v4/target/release" \
  "$STAGE/libpmk/.build/release" "$STAGE/vendor/pearl-fp8"

echo "==> Fetching pinned Python 3.12 runtime"
archive="$WORK/python.tar.gz"
curl -fL --retry 3 -o "$archive" "$PY_URL"
[ "$(shasum -a 256 "$archive" | awk '{print $1}')" = "$PY_SHA256" ] || die "Python runtime checksum mismatch"
tar -xzf "$archive" -C "$WORK"
mv "$WORK/python" "$STAGE/python"

echo "==> Building and hash-installing the sole non-stdlib dependency"
mkdir -p "$WORK/wheels"
RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--remap-path-prefix=/Users=/build-users --remap-path-prefix=/opt/homebrew=/toolchain" \
  CARGO_PROFILE_RELEASE_STRIP=symbols \
  "$ROOT/scripts/pmk_build_pool_binding.sh" --python "$ROOT/.venv/bin/python" --out-dir "$WORK/wheels"
wheel="$(ls -t "$WORK/wheels"/py_pearl_mining-*.whl | head -1)"
wheel_sha="$(shasum -a 256 "$wheel" | awk '{print $1}')"
printf 'py-pearl-mining @ file://%s --hash=sha256:%s\n' "$wheel" "$wheel_sha" > "$WORK/runtime-requirements.lock"
"$STAGE/python/bin/python3" -m pip install --no-deps --require-hashes -r "$WORK/runtime-requirements.lock"
rm -rf "$STAGE/python/lib/python3.12/site-packages/pip" \
  "$STAGE/python/lib/python3.12/site-packages/pip-"*.dist-info \
  "$STAGE/python/lib/python3.12/site-packages/py_pearl_mining-"*.dist-info
rm -f "$STAGE/python/bin/pip" "$STAGE/python/bin/pip3" "$STAGE/python/bin/pip3.12"

echo "==> Staging relocatable miner and admission assets"
rsync -a --delete --exclude __pycache__ "$ROOT/miner/pmk_miner/" "$STAGE/miner/pmk_miner/"
cp "$ROOT/scripts/pmk_mine.py" "$ROOT/scripts/pmk_quickstart.py" "$STAGE/scripts/"
cp "$ROOT/scripts/release/v4_slim_g3.py" "$STAGE/scripts/release/"
cp "$ROOT/scripts/release/hb_once.py" "$STAGE/scripts/release/"
cp "$ROOT/pmkcore/target/release/libpmkcore.dylib" "$STAGE/pmkcore/target/release/"
cp "$ROOT/pmkcore/v4/target/release/libpmkcore_v4.dylib" "$ROOT/pmkcore/v4/target/release/pmkcore-v4-oracle" "$STAGE/pmkcore/v4/target/release/"
cp "$ROOT/libpmk/.build/arm64-apple-macosx/release/libpmk.dylib" "$STAGE/libpmk/.build/release/"
install_name_tool -id @rpath/libpmkcore.dylib "$STAGE/pmkcore/target/release/libpmkcore.dylib"
install_name_tool -id @rpath/libpmkcore_v4.dylib "$STAGE/pmkcore/v4/target/release/libpmkcore_v4.dylib"
rsync -a "$ROOT/libpmk/.build/arm64-apple-macosx/release/libpmk_PMK.bundle" "$STAGE/libpmk/.build/release/"
printf '%s\n' 'f696760b259500ecb608469ea3953aeabbe78948' > "$STAGE/vendor/pearl-fp8/.pmk-v4-pin"

mkdir -p "$STAGE/dist/quickstart/.build-work/libpmk" "$STAGE/dist/quickstart/vectors/g3" "$STAGE/dist/quickstart/bin" "$STAGE/resources/v4-g3-slim"
rsync -a --delete --exclude .build "$ROOT/libpmk/" "$STAGE/dist/quickstart/.build-work/libpmk/"
rsync -a --delete "$ROOT/bench/k3sg/studio/vectors/" "$STAGE/dist/quickstart/vectors/g3/"
"$ROOT/.venv/bin/python" "$ROOT/scripts/studio_b4/build_g3_helper.py" "$ROOT" "$STAGE/dist/quickstart"

cat > "$STAGE/bin/python3" <<'SH'
#!/bin/sh
DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export PYTHONPATH="$DIR/miner:$DIR/scripts:$DIR${PYTHONPATH:+:$PYTHONPATH}"
export PMK_RESOURCE_BUNDLE="$DIR/libpmk/.build/release/libpmk_PMK.bundle"
exec "$DIR/python/bin/python3" "$@"
SH
cat > "$STAGE/bin/pearl-agent" <<'SH'
#!/bin/sh
DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
exec "$DIR/bin/python3" -m pmk_miner.beta_agent "$@"
SH
cat > "$STAGE/bin/pearl-miner" <<'SH'
#!/bin/sh
DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
exec "$DIR/bin/python3" -m pmk_miner.beta_cli "$@"
SH
chmod +x "$STAGE/bin/python3" "$STAGE/bin/pearl-agent" "$STAGE/bin/pearl-miner" "$STAGE/scripts/release/v4_slim_g3.py"
printf '%s\n' "$VERSION" > "$STAGE/VERSION"

# Frozen stdlib examples and third-party panic strings can retain build-host
# paths even when Mach-O load commands are clean. Same-length replacements keep
# binary offsets intact; signing happens afterward.
"$ROOT/.venv/bin/python" - "$STAGE" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
build_user = Path.home().name.encode()
anonymous_user = (b"builder" + b"x" * len(build_user))[:len(build_user)]
replacements = ((b"/" + b"Users/", b"/" + b"users/"),
                (b"/opt/homebrew", b"/build/toolss"),
                (build_user, anonymous_user))
for path in root.rglob("*"):
    if not path.is_file() or path.is_symlink():
        continue
    data = path.read_bytes()
    changed = data
    for old, new in replacements:
        changed = changed.replace(old, new)
    if changed != data:
        path.write_bytes(changed)
PY

echo "==> Ad-hoc signing and auditing every Mach-O"
while IFS= read -r path; do
  if file "$path" | grep -q 'Mach-O'; then
    codesign -s - -f "$path" >/dev/null
    codesign --verify --strict "$path"
    if otool -L "$path" | tail -n +2 | grep -Eq '/(Users|home)/|/opt/homebrew/'; then
      die "non-relocatable dependency in ${path#$STAGE/}"
    fi
  fi
done < <(find "$STAGE" -type f -print)
"$ROOT/.venv/bin/python" "$ROOT/scripts/release/make_v4_slim_manifest.py" \
  "$V4_SOURCE_MANIFEST" "$STAGE/resources/v4-g3-slim/manifest.json" \
  "$STAGE/pmkcore/v4/target/release/pmkcore-v4-oracle"
private_prefix='/''Users/'
if grep -R -a -l "$private_prefix" "$STAGE" | grep -q .; then
  die "private build path remained in release payload"
fi

echo "==> Creating release archive"
tarball="$DIST/$NAME.tar.gz"
tar -C "$WORK" -czf "$tarball" "$NAME"
sha="$(shasum -a 256 "$tarball" | awk '{print $1}')"
printf '%s  %s\n' "$sha" "$(basename "$tarball")" > "$tarball.sha256"
release_url="${PMK_RELEASE_URL_TEMPLATE:-https://github.com/Augustas11/pearl-apple-miner/releases/download/v$VERSION/$(basename "$tarball")}" 
sed -e "s#__PMK_VERSION__#$VERSION#g" -e "s#__PMK_RELEASE_URL__#$release_url#g" \
  -e "s#__PMK_RELEASE_SHA256__#$sha#g" "$ROOT/scripts/release/install.sh" > "$DIST/install.sh"
chmod +x "$DIST/install.sh"

size_bytes="$(stat -f %z "$tarball")"
[ "$size_bytes" -lt $((150 * 1024 * 1024)) ] || die "compressed tarball exceeds 150 MB"
echo "release_tarball=$(basename "$tarball")"
echo "compressed_size_bytes=$size_bytes"
du -sh "$STAGE/python" "$STAGE/dist/quickstart/vectors" "$STAGE/libpmk" "$STAGE/pmkcore" "$STAGE" | sed 's/^/content_size /'
