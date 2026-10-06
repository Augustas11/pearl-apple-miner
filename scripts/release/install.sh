#!/usr/bin/env sh
# SPDX-License-Identifier: Apache-2.0
set -eu

VERSION="__PMK_VERSION__"
RELEASE_URL="__PMK_RELEASE_URL__"
RELEASE_SHA256="__PMK_RELEASE_SHA256__"

fail_plain() {
  printf '%s\n' "$1" >&2
  exit "${2:-1}"
}

usage() {
  fail_plain "Your wallet address is missing. Get your personal install line at https://pearl.malibu.tech"
}

bad_wallet() {
  fail_plain "That doesn't look like a Pearl wallet address. Copy it again from your wallet, or get your install line at https://pearl.malibu.tech"
}

wallet=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --wallet) [ "$#" -ge 2 ] || usage; wallet="$2"; shift 2 ;;
    --token) [ "$#" -ge 2 ] || usage; shift 2 ;;
    --install-id) [ "$#" -ge 2 ] || usage; shift 2 ;;
    -h|--help) usage ;;
    *) usage ;;
  esac
done

[ -n "$wallet" ] || usage
# Accept addresses pasted in capitals or with stray spaces.
wallet="$(printf '%s' "$wallet" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')"

[ "$(uname -s)" = "Darwin" ] || fail_plain "This installer works on Apple Silicon Macs running macOS 14 or newer."
[ "$(uname -m)" = "arm64" ] || fail_plain "This installer works on Apple Silicon Macs running macOS 14 or newer."
macos_version="$(sw_vers -productVersion 2>/dev/null || true)"
macos_major="${macos_version%%.*}"
case "$macos_major" in
  ''|*[!0-9]*) fail_plain "This installer works on Apple Silicon Macs running macOS 14 or newer." ;;
esac
[ "$macos_major" -ge 14 ] || fail_plain "This installer works on Apple Silicon Macs running macOS 14 or newer."

case "$wallet" in
  prl1*) ;;
  *) bad_wallet ;;
esac
[ "${#wallet}" -ge 34 ] && [ "${#wallet}" -le 94 ] || bad_wallet
case "$wallet" in *[!0-9a-z]*) bad_wallet ;; esac
if [ -n "${PMK_RELEASE_URL:-}" ]; then
  RELEASE_URL="$PMK_RELEASE_URL"
fi
case "$RELEASE_URL" in
  ''|"__PMK_""RELEASE_URL__") fail_plain "The installer is missing its release URL." ;;
esac
[ "${#RELEASE_SHA256}" -eq 64 ] || fail_plain "The installer is missing its release checksum."
case "$RELEASE_SHA256" in *[!0-9a-f]*) fail_plain "The installer is missing its release checksum." ;; esac

app_root="$HOME/Library/Application Support/MalibuPearl"
logs_root="$HOME/Library/Logs/MalibuPearl"
agents_root="$HOME/Library/LaunchAgents"
label="tech.malibu.pearl"
plist="$agents_root/$label.plist"
tmp="${TMPDIR:-/tmp}/pmk-install.$$"
mkdir -p "$tmp"
cleanup_tmp() { rm -rf "$tmp"; }
trap cleanup_tmp EXIT HUP INT TERM

archive="$tmp/release.tar.gz"
case "$RELEASE_URL" in
  file://*) cp "${RELEASE_URL#file://}" "$archive" ;;
  *)
    printf '%s\n' "Downloading the Malibu Pearl miner..."
    curl -fsL --retry 3 --connect-timeout 10 -o "$archive" "$RELEASE_URL" \
      || fail_plain "Couldn't download the miner. Check your internet connection and run the same line again."
    ;;
esac
actual_sha="$(shasum -a 256 "$archive" | awk '{print $1}')"
[ "$actual_sha" = "$RELEASE_SHA256" ] || fail_plain "The download was damaged on the way. Run the same line again."

mkdir -p "$app_root" "$logs_root" "$agents_root"
chmod 700 "$app_root" "$logs_root"
extract="$tmp/extract"
mkdir -p "$extract"
tar -xzf "$archive" -C "$extract"
payload="$(find "$extract" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
[ -n "$payload" ] || fail_plain "The download was damaged on the way. Run the same line again."

# Catch a typo in the wallet (bech32/bech32m checksum) before anything is installed.
if ! "$payload/bin/python3" - "$wallet" <<'PY' >/dev/null 2>&1
import sys
CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
def polymod(values):
    gen = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        top = chk >> 25
        chk = (chk & 0x1ffffff) << 5 ^ v
        for i in range(5):
            chk ^= gen[i] if ((top >> i) & 1) else 0
    return chk
addr = sys.argv[1]
hrp, _, data = addr.rpartition("1")
ok = hrp == "prl" and len(data) >= 6 and all(c in CHARSET for c in data)
if ok:
    values = [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp] + [CHARSET.find(c) for c in data]
    ok = polymod(values) in (1, 0x2bc830a3)
sys.exit(0 if ok else 1)
PY
then
  rmdir "$app_root" "$logs_root" 2>/dev/null || true
  fail_plain "That wallet address has a typo. Copy it again from your wallet, or get your install line at https://pearl.malibu.tech"
fi

version_dir="$app_root/$VERSION"
rm -rf "$version_dir.tmp"
mkdir -p "$version_dir.tmp"
cp -R "$payload"/. "$version_dir.tmp/"
rm -rf "$version_dir"
mv "$version_dir.tmp" "$version_dir"
ln -sfn "$version_dir" "$app_root/current"

uid="$(id -u)"
launchctl bootout "gui/$uid" "$plist" >/dev/null 2>&1 || true

umask 077
printf '%s\n' "Checking your Mac computes exactly right..."
if ! PMK_HOME="$app_root/pmk-home" "$version_dir/bin/python3" "$version_dir/scripts/pmk_quickstart.py" admission >/dev/null 2>&1; then
  rm -rf "$app_root" "$logs_root"
  rm -f "$plist"
  printf '%s\n' "This Mac didn't pass the exactness check, so it won't mine. Nothing was left running."
  exit 1
fi
v4_admission="$app_root/pmk-home/v4-g3-admission.json"
v4_current=0
if "$version_dir/bin/python3" - "$v4_admission" "$version_dir/libpmk/.build/release/libpmk.dylib" <<'PY' >/dev/null 2>&1
import hashlib, sys
from pmk_miner.v4_admission import validate_v4_g3_admission_file
record = validate_v4_g3_admission_file(sys.argv[1])
actual = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
raise SystemExit(record["library_sha256"] != actual)
PY
then
  v4_current=1
fi
if [ "$v4_current" -eq 1 ] || PMK_HOME="$app_root/pmk-home" "$version_dir/bin/python3" "$version_dir/scripts/release/v4_slim_g3.py" \
    "$version_dir/resources/v4-g3-slim/manifest.json" \
    --library "$version_dir/libpmk/.build/release/libpmk.dylib" \
    --oracle "$version_dir/pmkcore/v4/target/release/pmkcore-v4-oracle" \
    --admission "$v4_admission" >/dev/null 2>&1; then
  :
else
  rm -f "$v4_admission"
fi

pool="${PMK_POOL_URL:-}"
if [ -z "$pool" ]; then
  pool="$("$version_dir/bin/python3" -c 'from pmk_miner.beta_agent import choose_pool_region,pool_url; print(pool_url(choose_pool_region()))' || printf '%s\n' 'stratum+tcp://sg.pearl.herominers.com:1200')"
fi
label_name="$(scutil --get ComputerName 2>/dev/null || hostname -s)"
worker="$(printf '%s' "$label_name" | tr -c 'A-Za-z0-9_-' '-' | sed 's/^-*//;s/-*$//' | cut -c1-32)"
[ -n "$worker" ] || worker="Mac"

printf '%s' "$wallet" > "$tmp/wallet"
printf '%s' "$pool" > "$tmp/pool"
printf '%s' "$worker" > "$tmp/worker"
printf '%s' "$label_name" > "$tmp/label"
"$version_dir/bin/python3" - "$app_root/config.json.tmp" "$app_root/config.json" "$tmp" "$VERSION" "$v4_admission" <<'PY'
import json, os, pathlib, re, secrets, sys
destination, previous_path, raw_dir, version, admission = sys.argv[1:]
raw = pathlib.Path(raw_dir)
local_secret = ""
try:
    previous = json.load(open(previous_path, encoding="utf-8"))
    candidate = previous.get("local_secret", "")
    if isinstance(candidate, str) and re.fullmatch(r"[0-9a-f]{64}", candidate):
        local_secret = candidate
except (OSError, ValueError, TypeError):
    pass
if not local_secret:
    local_secret = secrets.token_hex(32)
value = dict(version=version, v4_admission=admission, local_secret=local_secret,
    wallet=(raw / "wallet").read_text(), pool_url=(raw / "pool").read_text(),
    worker=(raw / "worker").read_text(), label=(raw / "label").read_text())
with open(destination, "w", encoding="utf-8") as stream:
    json.dump(value, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
os.chmod(destination, 0o600)
PY
mv "$app_root/config.json.tmp" "$app_root/config.json"
chmod 600 "$app_root/config.json"

"$version_dir/bin/python3" - "$plist.tmp" "$app_root" "$logs_root" "$HOME" <<'PY'
import os, plistlib, sys
path, app, logs, home = sys.argv[1:]
environment = {"HOME": home}
for name in ("PMK_TEST_FORCE_GPU_SECONDS", "PMK_TEST_FORCE_GPU_SHAPE"):
    if name in os.environ:
        environment[name] = os.environ[name]
value = {"Label": "tech.malibu.pearl",
 "ProgramArguments": [app + "/current/bin/pearl-agent", "--config", app + "/config.json"],
 "RunAtLoad": True, "KeepAlive": {"SuccessfulExit": False}, "ThrottleInterval": 10,
 "EnvironmentVariables": environment,
 "StandardOutPath": logs + "/agent.log", "StandardErrorPath": logs + "/agent.err.log",
 "WorkingDirectory": app + "/current"}
open(path, "wb").write(plistlib.dumps(value, sort_keys=True))
PY
mv "$plist.tmp" "$plist"
chmod 644 "$plist"

launchctl bootstrap "gui/$uid" "$plist"

if [ -d "$HOME/.local/bin" ]; then
  ln -sfn "$app_root/current/bin/pearl-miner" "$HOME/.local/bin/pearl-miner"
fi

"$app_root/current/bin/pearl-miner" open --wait 30
port="$("$app_root/current/bin/python3" -c 'import json,pathlib,sys
p=pathlib.Path(sys.argv[1])
try: print(int(json.loads(p.read_text()).get("local_port") or 47811))
except Exception: print(47811)' "$app_root/state.json" 2>/dev/null || printf '%s\n' 47811)"
printf '%s\n' "Done. Your Mac is mining Pearl."
printf '%s\n' "Your control page just opened in your browser."
printf '%s\n' "To open it again any time, go to http://localhost:$port"
