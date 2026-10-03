#!/usr/bin/env python3
"""Load the actual cdylib, build a job/proof through C, and verify with Python Pearl.

Run after cargo build --offline --release: ../.venv/bin/python -B tests/ffi_smoke.py
"""
import base64
import ctypes as c
import pathlib
import sys

import pearl_mining as pm

U8P = c.POINTER(c.c_uint8)
U64P = c.POINTER(c.c_uint64)
HANDLE = c.c_void_p
U8_32 = c.c_uint8 * 32


class Tile(c.Structure):
    _fields_ = [('t_rows', c.c_uint32), ('t_cols', c.c_uint32),
                ('transcript', c.c_uint32 * 16), ('hash', U8_32),
                ('is_share', c.c_uint32), ('is_block', c.c_uint32)]


class Template(c.Structure):
    _fields_ = [('job_key', U8_32), ('raw_root_a', U8_32), ('salted_root_a', U8_32),
                ('m', c.c_uint32), ('n', c.c_uint32), ('k', c.c_uint32), ('reserved', c.c_uint32)]


class Job(c.Structure):
    _fields_ = [('raw_root_b', U8_32), ('b_noise_seed', U8_32), ('a_noise_seed', U8_32)]


def buf(data):
    return (c.c_uint8 * len(data)).from_buffer_copy(data)


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    suffix = 'dylib' if sys.platform == 'darwin' else 'so'
    lib = c.CDLL(str(root / 'target' / 'release' / f'libpmkcore.{suffix}'))
    declarations = {
        'pmkcore_build_config': [c.c_uint32] * 4 + [U8P],
        'pmkcore_build_config_diagnostic': [c.c_uint32] * 4 + [U8P],
        'pmkcore_oracle_job_create': [U8P, U8P, c.c_uint32, c.c_uint32, U8P, c.c_uint64,
                                     U8P, c.c_uint64, c.POINTER(HANDLE)],
        'pmkcore_oracle_tile': [HANDLE, c.c_uint32, c.c_uint32, U8P, U8P, c.POINTER(Tile)],
        'pmkcore_oracle_scan': [HANDLE, U8P, U8P, c.POINTER(Tile), c.c_uint64, U64P],
        'pmkcore_oracle_build_plain_proof': [HANDLE, c.c_uint32, c.c_uint32, U8P, c.c_uint64, U64P],
        'pmkcore_oracle_export_vectors': [HANDLE, U8P, c.c_uint64, U64P],
        'pmkcore_template_init': [U8P, U8P, c.c_uint32, c.c_uint32, U8P, c.c_uint64, c.c_uint8, c.POINTER(Template)],
        'pmkcore_commit_job': [c.POINTER(Template), U8P, c.c_uint64, c.POINTER(Job)],
    }
    for name, args in declarations.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, c.c_int32
    lib.pmkcore_oracle_job_free.argtypes = [HANDLE]
    lib.pmkcore_oracle_job_free.restype = None
    assert c.sizeof(Tile) == c.sizeof(Template) == 112 and c.sizeof(Job) == 96
    for pattern, m in [(0, 128), (1, 64)]:
        n, k = 64, 2048
        header = pm.IncompleteBlockHeader(version=1, prev_block=bytes(32), merkle_root=bytes(32),
                                          timestamp=123, nbits=0x207fffff)
        hb, cfg = buf(header.to_bytes()), (c.c_uint8 * 52)()
        assert lib.pmkcore_build_config(pattern, k, m, n, cfg) == 0
        a, bt = (c.c_uint8 * (m*k))(), (c.c_uint8 * (n*k))()
        handle = HANDLE()
        assert lib.pmkcore_oracle_job_create(hb, cfg, m, n, a, len(a), bt, len(bt), c.byref(handle)) == 0
        try:
            bound = buf(bytes([255]) * 32)
            tile = Tile()
            assert lib.pmkcore_oracle_tile(handle, 0, 0, bound, bound, c.byref(tile)) == 0
            assert tile.is_share == tile.is_block == 1
            count = c.c_uint64()
            assert lib.pmkcore_oracle_scan(handle, bound, bound, None, 0, c.byref(count)) == 0
            assert count.value == 128
            length = c.c_uint64()
            assert lib.pmkcore_oracle_build_plain_proof(handle, 0, 0, None, 0, c.byref(length)) == 0
            proof_data = (c.c_uint8 * length.value)()
            assert lib.pmkcore_oracle_build_plain_proof(handle, 0, 0, proof_data, len(proof_data), c.byref(length)) == 0
            proof = pm.PlainProof.from_base64(base64.b64encode(bytes(proof_data)).decode())
            ok, message = pm.verify_plain_proof_for_cert_version(3, header, proof)
            assert ok, message
            assert lib.pmkcore_oracle_export_vectors(handle, None, 0, c.byref(length)) == 0
            exported = (c.c_uint8 * length.value)()
            assert lib.pmkcore_oracle_export_vectors(handle, exported, len(exported), c.byref(length)) == 0
            assert bytes(exported[:8]) == b'PMKVEC01'
            template, job = Template(), Job()
            assert lib.pmkcore_template_init(hb, cfg, m, n, a, len(a), 0, c.byref(template)) == 0
            assert lib.pmkcore_commit_job(c.byref(template), bt, len(bt), c.byref(job)) == 0
            assert bytes(template.raw_root_a) == bytes(proof.a.root)
            assert bytes(job.raw_root_b) == bytes(proof.bt.root)
            print(f'C ABI pattern={pattern}: cdylib create/tile/scan/proof/export/free, F2 roots, Python cert-v3 verification PASS')
        finally:
            lib.pmkcore_oracle_job_free(handle)


if __name__ == '__main__':
    main()
