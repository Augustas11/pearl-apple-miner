#!/usr/bin/env python3
"""Assemble relocatable runtime files; pin and hash every shipped wheel."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

root, out = map(Path, sys.argv[1:])
work = out / '.build-work'
def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', '.pmk_regtest', '*.egg-info', 'build'))
    else:
        shutil.copy2(src, dst)
for p in ['miner', 'vendor/pearl/miner/pearl-gateway']:
    copy(work / p, out / p)
for name in ['pearld', 'prlctl']:
    copy(work / 'vendor/pearl/bin' / name, out / 'vendor/pearl/bin' / name)
    copy(work / 'vendor/pearl/bin' / name, out / 'bin' / name)
copy(work / 'pmkcore/target/release/libpmkcore.dylib', out / 'pmkcore/target/release/libpmkcore.dylib')
copy(work / 'pmkcore/target/release/libpmkcore.dylib', out / 'bin/libpmkcore.dylib')
for name in ['libpmk.dylib', 'libpmk_PMK.bundle']:
    copy(work / 'libpmk/.build/release' / name, out / 'libpmk/.build/release' / name)
    copy(work / 'libpmk/.build/release' / name, out / 'bin' / name)
for p in ['libpmk/resources/probe', 'libpmk/tests/reference/vectors', 'libpmk/metal']:
    copy(root / p, out / p)
for name in ['pmk_native_check.py', 'pmk_pool_mock_e2e.py', 'pmk_production_smoke.py', 'pmk_regtest_e2e.py', 'pmk_regtest_e2e.sh', 'pmk_regtest_gateway_tap.py', 'regtest_gateway_tap.py']:
    copy(root / 'scripts' / name, out / 'scripts' / name)
copy(root / 'scripts/studio_b4', out / 'scripts/studio_b4')
for window in ['b4_window.sh', 'pool_window.sh']:
    if (root / 'scripts/studio_b4' / window).exists():
        copy(root / 'scripts/studio_b4' / window, out / window)
sys.path.insert(0, str(out / 'miner'))
from pmk_miner.gateway_launcher import patch_gateway_copy
patched = patch_gateway_copy(out / 'vendor/pearl/miner/pearl-gateway', out / 'miner/gateway_patches/0001-b3-safe-async-proving.patch', out / '.patch-work')
copy(patched.root_dir, out / 'patched-gateway')
shutil.rmtree(out / '.patch-work')
subprocess.run(['xcrun', 'swiftc', '-O', '-target', 'arm64-apple-macos14.0', str(root / 'scripts/studio_b4/k3_alone.swift'), '-o', str(out / 'bin/k3-alone')], check=True)
copy(root / 'bench/k3sg/studio/vectors', out / 'vectors/g3')
subprocess.run([sys.executable, str(root / 'scripts/studio_b4/build_g3_helper.py'), str(root), str(out)], check=True)
# Enforce deployment compatibility and reject build-host dynamic dependencies.
audit = {}
for path in [out / 'bin/g3-admit', out / 'bin/k3-alone', out / 'bin/libpmkcore.dylib', out / 'bin/libpmk.dylib', out / 'bin/pearld', out / 'bin/prlctl', out / 'pmkcore/target/release/libpmkcore.dylib', out / 'libpmk/.build/release/libpmk.dylib']:
    text = subprocess.check_output(['otool', '-l', str(path)], text=True)
    minimums = re.findall(r'minos\s+(\d+\.\d+(?:\.\d+)?)', text)
    limit = (26, 4) if path.name in {'pearld', 'prlctl'} else (14, 0)
    assert minimums and all(tuple(map(int, v.split('.')[:2])) <= limit for v in minimums), (path, minimums)
    deps = subprocess.check_output(['otool', '-L', str(path)], text=True)
    linked = deps.splitlines()[2 if path.suffix == '.dylib' else 1:]
    assert '/opt/homebrew/' not in '\n'.join(linked) and not re.search(r'/(Users|home)/', '\n'.join(linked)), deps
    assert 'arm64' in subprocess.check_output(['lipo', '-archs', str(path)], text=True)
    audit[str(path.relative_to(out))] = {'minos':minimums, 'dependencies':deps.splitlines()[1:]}
requirements=[]
local_requirements=[]
runtime_names = set(re.findall(r'^([A-Za-z0-9_-]+)==', (out / 'miner/requirements.lock').read_text(), re.M))
runtime_names = {name.lower().replace('_', '-') for name in runtime_names}
for whl in sorted((out/'wheels').glob('*.whl')):
    with zipfile.ZipFile(whl) as z:
        metadata=z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))).decode()
        name=re.search(r'^Name: (.+)$',metadata,re.M)[1]
        version=re.search(r'^Version: (.+)$',metadata,re.M)[1]
        for n in z.namelist():
            if n.endswith(('.so','.dylib')):
                temporary=work/'wheel-audit'/Path(n).name
                temporary.parent.mkdir(exist_ok=True)
                temporary.write_bytes(z.read(n))
                text=subprocess.check_output(['otool','-l',str(temporary)],text=True)
                mins=re.findall(r'minos\s+(\d+\.\d+(?:\.\d+)?)',text)
                limit = (14, 0) if name == 'py-pearl-mining' else (26, 4)
                assert mins and all(tuple(map(int,v.split('.')[:2])) <= limit for v in mins), (whl,n,mins)
                deps=subprocess.check_output(['otool','-L',str(temporary)],text=True)
                assert '/opt/homebrew/' not in deps and not re.search(r'/(Users|home)/', '\n'.join(deps.splitlines()[1:])), (whl,n,deps)
                assert 'arm64' in subprocess.check_output(['lipo','-archs',str(temporary)],text=True), (whl,n)
                audit[f'wheels/{whl.name}!/{n}'] = {'minos': mins, 'dependencies': deps.splitlines()[1:]}
    requirement = f'{name}=={version} --hash=sha256:{hashlib.sha256(whl.read_bytes()).hexdigest()}'
    requirements.append(requirement)
    if name.lower().replace('_', '-') not in runtime_names:
        local_requirements.append(requirement)
(out/'local-wheels.lock').write_text('\n'.join(local_requirements)+'\n')
(out/'requirements.lock').write_text('\n'.join(requirements)+'\n')
(out/'binary-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
# Build scratch is intentionally excluded from the transferable payload/manifest.
lines=[]
for p in sorted(out.rglob('*')):
    rel=p.relative_to(out)
    if rel.parts[0] in {'.build-work','.venv','.venv-b4','.b4_run','.b4_logs','results','bench'} or any(part in {'__pycache__', '.pytest_cache'} for part in rel.parts) or p.name in {'MANIFEST.sha256','build.log','b4_summary.json'}: continue
    if p.is_file(): lines.append(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {rel}')
(out/'MANIFEST.sha256').write_text('\n'.join(lines)+'\n')
print(f'Pinned {len(requirements)} wheels; hashed {len(lines)} files')
