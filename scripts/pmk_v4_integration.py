#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
from __future__ import annotations
import argparse, base64, contextlib, ctypes as C, hashlib, json, os, subprocess, time
from pathlib import Path

from pmk_v4_paths import bundle_root, verify_pearl_pin

ROOT = bundle_root(__file__)
PIN = "f696760b259500ecb608469ea3953aeabbe78948"
U32, U64, PTR = C.c_uint32, C.c_uint64, C.c_void_p

class Slot(C.Structure):
    _fields_ = [("row", U32), ("col", U32), ("message", U32 * 16), ("hash", U32 * 8)]

class Stats(C.Structure):
    _fields_ = [("abi_version", U32), ("flags", U32)] + [(x, U64) for x in (
        "fallback_groups", "total_groups", "quantized_a", "quantized_b",
        "quant_saturated_a", "quant_saturated_b", "quant_nan_a", "quant_nan_b")
    ] + [("layout_failures", U32), ("fallback_alert", U32)]

class Result(C.Structure):
    _fields_ = [("abi_version", U32), ("status", C.c_int32), ("job_id", U64)] + [
        (x, U32) for x in ("block_count", "share_count", "block_stored", "share_stored", "overflow", "recovered")
    ] + [("blocks", C.POINTER(Slot)), ("shares", C.POINTER(Slot)), ("stats", Stats),
         ("c_bits", PTR), ("c_count", U64), ("gpu_start_time", C.c_double), ("gpu_end_time", C.c_double)]

class QuantCodes(C.Structure):
    _fields_ = [("abi_version", U32), ("reserved", U32), ("a_codes", PTR), ("bt_codes", PTR),
                ("a_code_count", U64), ("bt_code_count", U64)]

class Operand(C.Structure):
    _fields_ = [("clean_values", PTR), ("clean_value_count", U64),
                ("noise_e_codes", PTR), ("noise_e_count", U64),
                ("noise_f_codes", PTR), ("noise_f_count", U64),
                ("alpha_bf16", PTR), ("beta_bf16", PTR), ("scale_count", U64)]

class JobDesc(C.Structure):
    _fields_ = [("abi_version", U32), ("m", U32), ("n", U32), ("k", U32), ("r", U32),
                ("a", Operand), ("bt", Operand),
                ("jackpot_key", U32 * 8), ("block_bound", U32 * 8), ("share_bound", U32 * 8),
                ("block_capacity", U32), ("share_capacity", U32), ("job_id", U64)]

class GpuJobDesc(C.Structure):
    _fields_ = [(x, U32) for x in ("m", "n", "k", "rank", "tile_rows", "tile_cols", "row_period", "col_period")] + [
        ("a_values", PTR), ("a_scales", PTR), ("bt_values", PTR), ("bt_scales", PTR),
        ("a_noised", PTR), ("bt_noised", PTR), ("a_noise_e", PTR), ("a_noise_f", PTR),
        ("bt_noise_e", PTR), ("bt_noise_f", PTR), ("a_alpha", PTR), ("a_beta", PTR), ("a_l2", PTR),
        ("bt_alpha", PTR), ("bt_beta", PTR), ("bt_l2", PTR),
        ("key_a", C.c_uint8 * 32), ("key_b", C.c_uint8 * 32), ("hash_a", C.c_uint8 * 32), ("hash_b", C.c_uint8 * 32),
        ("noise_seed_a", C.c_uint8 * 32), ("noise_seed_b", C.c_uint8 * 32), ("jackpot_key", C.c_uint8 * 32)]

class TileResult(C.Structure):
    _fields_ = [("t_rows", U32), ("t_cols", U32), ("message", C.c_uint8 * 64), ("hash", C.c_uint8 * 32),
                ("policy_pass", U32), ("is_share", U32), ("is_block", U32)]

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def bytes_at(ptr, n):
    return C.string_at(ptr, n)

def words_from_bytes32(b):
    return (U32 * 8).from_buffer_copy(bytes(b))

def slot_words_to_bytes(words):
    return b''.join(int(w).to_bytes(4, 'little') for w in words)


@contextlib.contextmanager
def gpu_lock():
    path = Path("/tmp/pmm-gpu-bench.lock")
    owned = os.environ.get("PMK_GPU_LOCK_HELD") != "1"
    if owned:
        while True:
            try:
                path.mkdir()
                break
            except FileExistsError:
                print("GPU lock occupied; retrying in 15 seconds.", flush=True)
                time.sleep(15)
    try:
        yield
    finally:
        if owned:
            path.rmdir()

class Core:
    def __init__(self, path: Path):
        self.path = path.resolve(); self.lib = C.CDLL(str(self.path))
        L = self.lib
        L.pmkcore_v4_init.argtypes = [U32]; L.pmkcore_v4_init.restype = C.c_int32
        L.pmkcore_v4_job_create_grid_b200.argtypes = [PTR, PTR, PTR, U64, U32, U32, U32, C.POINTER(PTR)]; L.pmkcore_v4_job_create_grid_b200.restype = C.c_int32
        L.pmkcore_v4_job_free.argtypes = [PTR]
        L.pmkcore_v4_prepare_oracle_noised.argtypes = [PTR]; L.pmkcore_v4_prepare_oracle_noised.restype = C.c_int32
        L.pmkcore_v4_gpu_descriptor.argtypes = [PTR, C.POINTER(GpuJobDesc)]; L.pmkcore_v4_gpu_descriptor.restype = C.c_int32
        L.pmkcore_v4_tile_cpu_oracle.argtypes = [PTR, U32, U32, PTR, PTR, C.POINTER(TileResult)]; L.pmkcore_v4_tile_cpu_oracle.restype = C.c_int32
        L.pmkcore_v4_build_plain_proof.argtypes = [PTR, U32, U32, PTR, U64, C.POINTER(U64)]; L.pmkcore_v4_build_plain_proof.restype = C.c_int32
        L.pmkcore_v4_verify_plain_proof.argtypes = [PTR, PTR, U64, PTR, C.POINTER(C.c_uint8)]; L.pmkcore_v4_verify_plain_proof.restype = C.c_int32
        L.pmkcore_v4_mutate_plain_proof.argtypes = [PTR, U64, U32, PTR, U64, C.POINTER(U64)]; L.pmkcore_v4_mutate_plain_proof.restype = C.c_int32
        rc = L.pmkcore_v4_init(4)
        if rc not in (0, -9):
            raise RuntimeError(f"pmkcore_v4_init failed {rc}")
    def check(self, rc, what):
        if rc != 0: raise RuntimeError(f"{what} failed rc={rc}")

class Metal:
    def __init__(self, path: Path):
        self.path = path.resolve(); self.lib = C.CDLL(str(self.path)); L = self.lib
        L.pmk_v4_init_diagnostic.argtypes = [C.POINTER(PTR), PTR, U64]; L.pmk_v4_init_diagnostic.restype = C.c_int32
        L.pmk_v4_admission_metadata.argtypes = [PTR, PTR, U64]; L.pmk_v4_admission_metadata.restype = C.c_int32
        L.pmk_v4_run_job.argtypes = [PTR, C.POINTER(JobDesc), PTR, PTR, C.POINTER(PTR)]; L.pmk_v4_run_job.restype = C.c_int32
        L.pmk_v4_poll.argtypes = [PTR, C.POINTER(Result)]; L.pmk_v4_poll.restype = C.c_int32
        L.pmk_v4_job_quantized_codes.argtypes = [PTR, C.POINTER(QuantCodes)]; L.pmk_v4_job_quantized_codes.restype = C.c_int32
        L.pmk_v4_job_wait_callback.argtypes = [PTR]; L.pmk_v4_job_wait_callback.restype = C.c_int32
        L.pmk_v4_job_release.argtypes = [PTR]; L.pmk_v4_job_release.restype = C.c_int32
        L.pmk_v4_destroy.argtypes = [PTR]
        err = C.create_string_buffer(4096); self.ctx = PTR()
        rc = L.pmk_v4_init_diagnostic(C.byref(self.ctx), err, len(err))
        if rc != 0: raise RuntimeError(err.value.decode(errors='replace'))
    def metadata(self):
        out = C.create_string_buffer(16384); rc = self.lib.pmk_v4_admission_metadata(self.ctx, out, len(out))
        if rc != 0: raise RuntimeError("metadata failed")
        return json.loads(out.value)
    def close(self):
        if self.ctx: self.lib.pmk_v4_destroy(self.ctx); self.ctx = PTR()

def ensure_fixture(oracle: Path, out: Path, m: int, n: int, k: int, seed: int):
    out.mkdir(parents=True, exist_ok=True)
    seed_hex = hashlib.sha256(f"B9-integration/{m}/{n}/{k}/{seed}".encode()).hexdigest()
    meta = out / 'metadata.json'
    if not meta.exists():
        subprocess.run([str(oracle), 'gen', str(m), str(n), str(k), seed_hex, str(out)], check=True)
    return json.loads(meta.read_text())

def ensure_cabi_ref(oracle: Path, out: Path, m: int, n: int, k: int, a_codes: bytes, b_codes: bytes) -> bytes:
    out.mkdir(parents=True, exist_ok=True)
    stamp = hashlib.sha256(a_codes + b"\0" + b_codes).hexdigest()
    stamp_path = out / 'codes.sha256'
    c_path = out / 'c_b200.bin'
    if (not c_path.exists()) or (not stamp_path.exists()) or stamp_path.read_text().strip() != stamp:
        (out / 'a.bin').write_bytes(a_codes)
        (out / 'b.bin').write_bytes(b_codes)
        subprocess.run([str(oracle), 'ref', str(out), str(m), str(n), str(k)], check=True)
        stamp_path.write_text(stamp + '\n')
    return c_path.read_bytes()

def run_case(core: Core, metal: Metal, oracle: Path, case_dir: Path, m: int, n: int, k: int, idx: int, equal_bounds: bool = False):
    meta = ensure_fixture(oracle, case_dir, m, n, k, idx)
    proposed = base64.b64decode(meta['proposed_header_b64']); ancestor = base64.b64decode(meta['ancestor_header_b64'])
    job = PTR(); core.check(core.lib.pmkcore_v4_job_create_grid_b200(proposed, ancestor, None, 0, m, n, k, C.byref(job)), 'create_grid')
    try:
        core.check(core.lib.pmkcore_v4_prepare_oracle_noised(job), 'prepare_noised')
        gd = GpuJobDesc(); core.check(core.lib.pmkcore_v4_gpu_descriptor(job, C.byref(gd)), 'gpu_descriptor')
        tiles = (m // 16) * (n // 16)
        share_bound = bytes([0xff]) * 32; block_bound = bytes(32)
        if equal_bounds:
            block_bound = share_bound
        jd = JobDesc(); jd.abi_version = 1; jd.m = m; jd.n = n; jd.k = k; jd.r = 32
        jd.a = Operand(gd.a_values, m*k, gd.a_noise_e, m*32, gd.a_noise_f, k*32, gd.a_alpha, gd.a_beta, m)
        jd.bt = Operand(gd.bt_values, n*k, gd.bt_noise_e, n*32, gd.bt_noise_f, k*32, gd.bt_alpha, gd.bt_beta, n)
        jd.jackpot_key = words_from_bytes32(gd.jackpot_key)
        jd.block_bound = words_from_bytes32(block_bound); jd.share_bound = words_from_bytes32(share_bound)
        jd.block_capacity = 4; jd.share_capacity = max(64, tiles); jd.job_id = idx
        metal_job = PTR(); rc = metal.lib.pmk_v4_run_job(metal.ctx, C.byref(jd), None, None, C.byref(metal_job))
        if rc != 0: raise RuntimeError(f"pmk_v4_run_job failed rc={rc}")
        try:
            deadline = time.monotonic() + 300
            result = Result()
            while True:
                rc = metal.lib.pmk_v4_poll(metal_job, C.byref(result))
                if rc != 1:
                    if rc != 0: raise RuntimeError(f"poll failed rc={rc} status={result.status}")
                    break
                if time.monotonic() > deadline: raise TimeoutError("libpmk v4 job timed out")
                time.sleep(0.002)
            q = QuantCodes(); rc = metal.lib.pmk_v4_job_quantized_codes(metal_job, C.byref(q))
            if rc != 0: raise RuntimeError(f"quantized_codes failed rc={rc}")
            got_a = bytes_at(q.a_codes, q.a_code_count); got_b = bytes_at(q.bt_codes, q.bt_code_count)
            exp_a = bytes_at(gd.a_noised, m*k); exp_b = bytes_at(gd.bt_noised, n*k)
            qa_mis = sum(x != y for x, y in zip(got_a, exp_a)); qb_mis = sum(x != y for x, y in zip(got_b, exp_b))
            exp_c = ensure_cabi_ref(oracle, case_dir / 'cabi_ref', m, n, k, exp_a, exp_b)
            got_c = bytes_at(result.c_bits, m*n*4)
            c_mis = 0 if got_c == exp_c else sum(a != b for a, b in zip(got_c, exp_c))
            expected_blocks = tiles if equal_bounds else 0
            if result.block_count != expected_blocks or result.block_stored != expected_blocks:
                raise RuntimeError(f"expected {expected_blocks} blocks, got count={result.block_count} stored={result.block_stored} tiles={tiles}")
            if result.share_count != tiles or result.share_stored != tiles:
                raise RuntimeError(f"expected one share per tile, got count={result.share_count} stored={result.share_stored} tiles={tiles}")
            fold_mis = 0
            for stream_name, ptr, stored in [('block', result.blocks, result.block_stored), ('share', result.shares, result.share_stored)]:
                for i in range(stored):
                    slot = ptr[i]
                    tr = int(slot.row); tc = int(slot.col)
                    oracle_tile = TileResult()
                    core.check(core.lib.pmkcore_v4_tile_cpu_oracle(job, tr, tc, share_bound, block_bound, C.byref(oracle_tile)), 'tile_oracle')
                    if bytes(oracle_tile.message) != slot_words_to_bytes(slot.message) or bytes(oracle_tile.hash) != slot_words_to_bytes(slot.hash):
                        fold_mis += 1
            proof_len = U64(0); core.check(core.lib.pmkcore_v4_build_plain_proof(job, 0, 0, None, 0, C.byref(proof_len)), 'proof_len')
            proof = (C.c_uint8 * proof_len.value)(); core.check(core.lib.pmkcore_v4_build_plain_proof(job, 0, 0, proof, proof_len, C.byref(proof_len)), 'proof')
            accepted = C.c_uint8(0); nbits = (C.c_uint8 * 4).from_buffer_copy((0x207fffff).to_bytes(4, 'little'))
            core.check(core.lib.pmkcore_v4_verify_plain_proof(proposed, proof, proof_len, nbits, C.byref(accepted)), 'verify_proof')
            if accepted.value != 1: raise RuntimeError('proof verifier returned accepted=0')
            corrupt_rejected = 0
            for mut in range(4):
                out_len = U64(0); core.check(core.lib.pmkcore_v4_mutate_plain_proof(proof, proof_len, mut, None, 0, C.byref(out_len)), 'mut_len')
                out = (C.c_uint8 * out_len.value)(); core.check(core.lib.pmkcore_v4_mutate_plain_proof(proof, proof_len, mut, out, out_len, C.byref(out_len)), 'mut')
                acc = C.c_uint8(0); vrc = core.lib.pmkcore_v4_verify_plain_proof(proposed, out, out_len, nbits, C.byref(acc))
                if vrc != 0 or acc.value == 0: corrupt_rejected += 1
            mismatches = qa_mis + qb_mis + c_mis + fold_mis
            rec = {
                'case': case_dir.name, 'm': m, 'n': n, 'k': k, 'tiles': tiles, 'equal_bounds': equal_bounds, 'mismatches': mismatches,
                'quant_a_mismatches': qa_mis, 'quant_b_mismatches': qb_mis, 'c_byte_mismatches': c_mis,
                'fold_tile_mismatches': fold_mis, 'fallback_groups': int(result.stats.fallback_groups),
                'total_groups': int(result.stats.total_groups), 'layout_failures': int(result.stats.layout_failures),
                'fallback_alert': int(result.stats.fallback_alert), 'quantized_a': int(result.stats.quantized_a),
                'quantized_b': int(result.stats.quantized_b), 'quant_saturated_a': int(result.stats.quant_saturated_a),
                'quant_saturated_b': int(result.stats.quant_saturated_b), 'quant_nan_a': int(result.stats.quant_nan_a),
                'quant_nan_b': int(result.stats.quant_nan_b), 'gpu_seconds': result.gpu_end_time - result.gpu_start_time,
                'proofs_accepted': 1, 'corrupt_proofs_rejected': corrupt_rejected,
            }
            expected_total = m*n*(k//32)
            if mismatches or result.stats.layout_failures or result.overflow or result.stats.quantized_a != m*k or result.stats.quantized_b != n*k or result.stats.total_groups != expected_total or corrupt_rejected != 4:
                raise RuntimeError(json.dumps(rec, sort_keys=True))
            print(json.dumps(rec, sort_keys=True), flush=True)
            return rec
        finally:
            metal.lib.pmk_v4_job_wait_callback(metal_job)
            metal.lib.pmk_v4_job_release(metal_job)
    finally:
        core.lib.pmkcore_v4_job_free(job)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--library', type=Path, default=ROOT/'libpmk/.build/release/libpmk.dylib')
    ap.add_argument('--core-library', type=Path, default=ROOT/'pmkcore/v4/target/release/libpmkcore_v4.dylib')
    ap.add_argument('--oracle', type=Path, default=ROOT/'pmkcore/v4/target/release/pmkcore-v4-oracle')
    ap.add_argument('--output', type=Path, default=ROOT/'bench/v4_emulation/vectors/b9_integration.json')
    ap.add_argument('--fixture-root', type=Path, default=ROOT/'bench/v4_emulation/vectors/b9_integration')
    args = ap.parse_args()
    verify_pearl_pin(ROOT, PIN)
    cases = [(32,64,1024,False),(64,32,1024,False),(32,64,4096,False),(64,32,4096,False),(32,64,16384,False),(64,32,16384,False),(32,32,1024,True)]
    core = Core(args.core_library)
    with gpu_lock():
        metal = Metal(args.library)
        try:
            metadata = metal.metadata(); cache_key = metadata['cache_key']
            library_sha = sha(args.library.resolve())
            if metadata.get('library_sha256') != library_sha:
                raise RuntimeError('loaded native library identity differs from integration binary before run')
            records = []
            base = args.fixture_root.resolve()
            for i,(m,n,k,equal_bounds) in enumerate(cases, 1):
                suffix = '_equalbounds' if equal_bounds else ''
                records.append(run_case(core, metal, args.oracle.resolve(), base/f'{m}x{n}_k{k}_s{i}{suffix}', m, n, k, i, equal_bounds=equal_bounds))
            metadata_after = metal.metadata()
            library_sha_after = sha(args.library.resolve())
            if metadata_after.get('library_sha256') != library_sha_after or metadata_after.get('library_sha256') != metadata.get('library_sha256'):
                raise RuntimeError('loaded native library identity drifted during integration run')
            out = {
                'schema': 'pmk-v4-integration-v1', 'upstream_pin': PIN, 'passed': True, 'cache_key': cache_key,
                'core_library': str(args.core_library.resolve()), 'core_sha256': sha(args.core_library.resolve()),
                'library': str(args.library.resolve()), 'library_sha256': library_sha_after,
                'quantized_cells_checked': sum(r['quantized_a'] + r['quantized_b'] for r in records),
                'fold_tiles_checked': sum(r['tiles'] for r in records),
                'proofs_accepted': sum(r['proofs_accepted'] for r in records),
                'corrupt_proofs_rejected': sum(r['corrupt_proofs_rejected'] for r in records),
                'mismatches': sum(r['mismatches'] for r in records), 'cases': records,
            }
            if not (out['quantized_cells_checked'] > 0 and out['fold_tiles_checked'] > 0 and out['proofs_accepted'] > 0 and out['corrupt_proofs_rejected'] > 0 and out['mismatches'] == 0):
                out['passed'] = False
                raise RuntimeError(json.dumps(out, sort_keys=True))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
            print(json.dumps(out, sort_keys=True), flush=True)
        finally:
            metal.close()

if __name__ == '__main__':
    main()
