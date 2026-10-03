#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes as C
import pathlib
import sys
import threading
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench" / "k3sg"))
sys.path.insert(0, str(ROOT / "bench" / "f1_k3"))

import sg_oracle  # noqa: E402
from sg_oracle import oracle  # noqa: E402

PMK_SUCCESS = 0
PMK_PENDING = 1
PMK_BUSY = -105
U256_MAX = (1 << 256) - 1


class Slot(C.Structure):
    _fields_ = [
        ("t_rows", C.c_uint32),
        ("t_cols", C.c_uint32),
        ("transcript", C.c_uint32 * 16),
        ("hash", C.c_uint32 * 8),
    ]


class JobDesc(C.Structure):
    _fields_ = [
        ("abi_version", C.c_uint32),
        ("m", C.c_uint32),
        ("n", C.c_uint32),
        ("k", C.c_uint32),
        ("a", C.POINTER(C.c_int8)),
        ("bt", C.POINTER(C.c_int8)),
        ("a_bytes", C.c_uint64),
        ("bt_bytes", C.c_uint64),
        ("a_seed", C.c_uint32 * 8),
        ("b_seed", C.c_uint32 * 8),
        ("block_bound", C.c_uint32 * 8),
        ("share_bound", C.c_uint32 * 8),
        ("block_capacity", C.c_uint32),
        ("share_capacity", C.c_uint32),
        ("cert_version", C.c_uint32),
        ("rank", C.c_uint32),
        ("job_id", C.c_uint64),
    ]


class Result(C.Structure):
    _fields_ = [
        ("abi_version", C.c_uint32),
        ("status", C.c_int32),
        ("job_id", C.c_uint64),
        ("block_count", C.c_uint32),
        ("share_count", C.c_uint32),
        ("block_stored", C.c_uint32),
        ("share_stored", C.c_uint32),
        ("overflow", C.c_uint32),
        ("recovered", C.c_uint32),
        ("blocks", C.POINTER(Slot)),
        ("shares", C.POINTER(Slot)),
        ("gpu_start_time", C.c_double),
        ("gpu_end_time", C.c_double),
    ]


def words_le(x: int) -> list[int]:
    return [(x >> (32 * i)) & 0xFFFFFFFF for i in range(8)]


def seed_words(seed: bytes) -> C.Array:
    assert len(seed) == 32
    return (C.c_uint32 * 8).from_buffer_copy(seed)


def slot_words(s: Slot) -> tuple[int, ...]:
    return (int(s.t_rows), int(s.t_cols), *[int(x) for x in s.transcript], *[int(x) for x in s.hash])


def tile_words(t: dict) -> tuple[int, ...]:
    return (
        int(t["t_rows"]),
        int(t["t_cols"]),
        *[int(x) for x in t["jp"]],
        *np.frombuffer(t["hash"], dtype="<u4").astype(np.uint32).tolist(),
    )


class PMK:
    def __init__(self, dylib: pathlib.Path):
        self.lib = C.CDLL(str(dylib))
        self.lib.pmk_init.argtypes = [C.POINTER(C.c_void_p), C.c_char_p, C.c_uint64]
        self.lib.pmk_init.restype = C.c_int32
        self.lib.pmk_init_diagnostic.argtypes = [C.POINTER(C.c_void_p), C.c_char_p, C.c_uint64]
        self.lib.pmk_init_diagnostic.restype = C.c_int32
        self.lib.pmk_probe.argtypes = [C.c_void_p, C.c_char_p, C.c_uint64]
        self.lib.pmk_probe.restype = C.c_int32
        self.lib.pmk_probe_refresh.argtypes = [C.c_void_p, C.c_char_p, C.c_uint64]
        self.lib.pmk_probe_refresh.restype = C.c_int32
        self.lib.pmk_destroy.argtypes = [C.c_void_p]
        self.lib.pmk_buffer_alloc.argtypes = [C.c_void_p, C.c_uint64, C.POINTER(C.c_void_p)]
        self.lib.pmk_buffer_alloc.restype = C.c_int32
        self.lib.pmk_buffer_release.argtypes = [C.c_void_p, C.c_void_p]
        self.lib.pmk_buffer_release.restype = C.c_int32
        cb_t = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p)
        self.callback_type = cb_t
        self.lib.pmk_run_job.argtypes = [C.c_void_p, C.POINTER(JobDesc), cb_t, C.c_void_p, C.POINTER(C.c_void_p)]
        self.lib.pmk_run_job.restype = C.c_int32
        self.lib.pmk_run_job_diagnostic.argtypes = [C.c_void_p, C.POINTER(JobDesc), cb_t, C.c_void_p, C.POINTER(C.c_void_p)]
        self.lib.pmk_run_job_diagnostic.restype = C.c_int32
        self.lib.pmk_poll.argtypes = [C.c_void_p, C.POINTER(Result)]
        self.lib.pmk_poll.restype = C.c_int32
        self.lib.pmk_job_wait_callback.argtypes = [C.c_void_p]
        self.lib.pmk_job_wait_callback.restype = C.c_int32
        self.lib.pmk_job_release.argtypes = [C.c_void_p]
        self.lib.pmk_job_release.restype = C.c_int32
        self.ctx = C.c_void_p()
        err = C.create_string_buffer(4096)
        rc = self.lib.pmk_init_diagnostic(C.byref(self.ctx), err, len(err))
        assert rc == PMK_SUCCESS, (rc, err.value.decode(errors="replace"))
        key = C.create_string_buffer(128)
        rc = self.lib.pmk_probe(self.ctx, key, len(key))
        assert rc == PMK_SUCCESS, rc
        self.cache_key = key.value.decode()
        self._buffers: list[C.c_void_p] = []
        self._callbacks = []

    def close(self) -> None:
        for ptr in list(self._buffers):
            rc = self.lib.pmk_buffer_release(self.ctx, ptr)
            assert rc == PMK_SUCCESS, rc
            self._buffers.remove(ptr)
        self.lib.pmk_destroy(self.ctx)

    def alloc_array(self, data: np.ndarray) -> C.c_void_p:
        assert data.flags["C_CONTIGUOUS"]
        ptr = C.c_void_p()
        rc = self.lib.pmk_buffer_alloc(self.ctx, data.nbytes, C.byref(ptr))
        assert rc == PMK_SUCCESS, rc
        C.memmove(ptr, data.ctypes.data, data.nbytes)
        self._buffers.append(ptr)
        return ptr

    def release_buffer(self, ptr: C.c_void_p) -> None:
        rc = self.lib.pmk_buffer_release(self.ctx, ptr)
        assert rc == PMK_SUCCESS, rc
        self._buffers.remove(ptr)

    def run_job(
        self,
        A: np.ndarray,
        Bt: np.ndarray,
        a_seed: bytes,
        b_seed: bytes,
        block_bound: int,
        share_bound: int,
        cap_block: int,
        cap_share: int,
        job_id: int,
        keep_open: bool = False,
        diagnostic: bool = False,
        callback_body=None,
    ):
        m, k = A.shape
        n = Bt.shape[0]
        a_ptr = self.alloc_array(np.ascontiguousarray(A, dtype=np.int8))
        bt_ptr = self.alloc_array(np.ascontiguousarray(Bt, dtype=np.int8))
        desc = JobDesc()
        desc.abi_version = 1
        desc.m, desc.n, desc.k = m, n, k
        desc.a = C.cast(a_ptr, C.POINTER(C.c_int8))
        desc.bt = C.cast(bt_ptr, C.POINTER(C.c_int8))
        desc.a_bytes, desc.bt_bytes = A.size, Bt.size
        desc.a_seed = seed_words(a_seed)
        desc.b_seed = seed_words(b_seed)
        desc.block_bound = (C.c_uint32 * 8)(*words_le(block_bound))
        desc.share_bound = (C.c_uint32 * 8)(*words_le(share_bound))
        desc.block_capacity = cap_block
        desc.share_capacity = cap_share
        desc.cert_version = 3
        desc.rank = 128
        desc.job_id = job_id
        callback_count = C.c_uint64(0)

        def mark(job, _user):
            assert int(C.cast(job, C.c_void_p).value or 0) != 0
            callback_count.value += 1
            if callback_body is not None:
                callback_body(job)

        cb = self.callback_type(mark)
        self._callbacks.append(cb)
        job = C.c_void_p()
        run = self.lib.pmk_run_job_diagnostic if diagnostic else self.lib.pmk_run_job
        rc = run(self.ctx, C.byref(desc), cb, None, C.byref(job))
        if rc != PMK_SUCCESS:
            self.release_buffer(a_ptr)
            self.release_buffer(bt_ptr)
            return rc, None
        if keep_open:
            return PMK_SUCCESS, (job, a_ptr, bt_ptr, callback_count)
        result = Result()
        deadline = time.time() + 180
        while time.time() < deadline:
            rc = self.lib.pmk_poll(job, C.byref(result))
            if rc != PMK_PENDING:
                break
            time.sleep(0.005)
        assert rc == PMK_SUCCESS and result.status == PMK_SUCCESS, (rc, result.status)
        deadline = time.time() + 5
        while callback_count.value == 0 and time.time() < deadline:
            time.sleep(0.001)
        assert callback_count.value == 1, callback_count.value
        blocks = [slot_words(result.blocks[i]) for i in range(result.block_stored)]
        shares = [slot_words(result.shares[i]) for i in range(result.share_stored)]
        assert result.gpu_end_time >= result.gpu_start_time >= 0
        self.release_job(job)
        self.release_buffer(a_ptr)
        self.release_buffer(bt_ptr)
        return PMK_SUCCESS, (result, blocks, shares)

    def release_job(self, job: C.c_void_p) -> None:
        rc = self.lib.pmk_job_wait_callback(job)
        assert rc == PMK_SUCCESS, rc
        rc = self.lib.pmk_job_release(job)
        assert rc == PMK_SUCCESS, rc


def make_job(rng: np.random.Generator, m: int, n: int, k: int):
    A = rng.integers(-64, 65, size=(m, k), dtype=np.int16).astype(np.int8)
    Bt = rng.integers(-64, 65, size=(n, k), dtype=np.int16).astype(np.int8)
    a_seed = rng.bytes(32)
    b_seed = rng.bytes(32)
    na, nb = oracle.noise(k, 128, b_seed, a_seed, m, n)
    Ap = A.astype(np.int64) + na
    Bp = (Bt.astype(np.int64) + nb).T
    assert Ap.min() >= -127 and Ap.max() <= 127
    assert Bp.min() >= -127 and Bp.max() <= 127
    tiles = sg_oracle.all_tiles(Ap, Bp, a_seed)
    return A, Bt, a_seed, b_seed, tiles


def assert_case(pmk: PMK, case_name: str, A, Bt, a_seed, b_seed, tiles, block_bound, share_bound, cap_block, cap_share, job_id, diagnostic=False):
    rc, payload = pmk.run_job(A, Bt, a_seed, b_seed, block_bound, share_bound, cap_block, cap_share, job_id,
                              diagnostic=diagnostic)
    assert rc == PMK_SUCCESS, (case_name, rc)
    result, blocks, shares = payload
    exp_b = [tile_words(t) for t in tiles if t["hv"] <= block_bound]
    exp_s = [tile_words(t) for t in tiles if t["hv"] <= share_bound]
    assert result.block_count == len(exp_b), (case_name, "block_count", result.block_count, len(exp_b))
    assert result.share_count == len(exp_s), (case_name, "share_count", result.share_count, len(exp_s))
    assert len(blocks) == len(exp_b) and set(blocks) == set(exp_b), (case_name, "block_slots", len(blocks), len(exp_b))
    assert len(shares) == len(exp_s) and set(shares) == set(exp_s), (case_name, "share_slots", len(shares), len(exp_s))
    return result


def run_matrix_tests(pmk: PMK, quick: bool) -> None:
    rng = np.random.default_rng(0xB2B2_2026)
    shapes = [(64, 64, 2048), (64, 128, 4096), (128, 64, 8192), (64, 64, 65536)]
    if not quick:
        shapes += [(256, 256, 4096), (128, 128, 65536)]
        while len(shapes) < 50:
            shapes.append((64 * int(rng.integers(1, 5)), 64 * int(rng.integers(1, 5)), int(rng.choice([2048, 4096, 8192]))))
    for i, (m, n, k) in enumerate(shapes):
        A, Bt, a_seed, b_seed, tiles = make_job(rng, m, n, k)
        cap = len(tiles)
        assert_case(pmk, f"full-{i}-{m}x{n}x{k}", A, Bt, a_seed, b_seed, tiles, 0, U256_MAX, max(4, cap), max(64, cap), i + 1,
                    diagnostic=k > 8192)
    print(f"G3 full-slot compare: PASS {len(shapes)} jobs", flush=True)


def run_boundaries_and_overflow(pmk: PMK) -> None:
    rng = np.random.default_rng(0xB0A)
    A, Bt, a_seed, b_seed, tiles = make_job(rng, 128, 128, 4096)
    chosen = min(tiles, key=lambda t: t["hv"])
    cap = len(tiles)
    for delta in (-1, 0, 1):
        bound = max(0, chosen["hv"] + delta)
        assert_case(pmk, f"boundary-{delta}", A, Bt, a_seed, b_seed, tiles, bound, 0, max(4, cap), 64, 100 + delta)
    zero = assert_case(pmk, "zero-threshold", A, Bt, a_seed, b_seed, tiles, 0, 0, max(4, cap), 64, 104)
    assert zero.block_count == 0 and zero.share_count == 0
    simultaneous = assert_case(pmk, "simultaneous", A, Bt, a_seed, b_seed, tiles, U256_MAX, U256_MAX, max(4, cap), max(64, cap), 105)
    assert simultaneous.block_count == simultaneous.share_count == cap
    overflow = assert_case(pmk, "overflow", A, Bt, a_seed, b_seed, tiles, U256_MAX, U256_MAX, 4, 64, 106)
    assert overflow.overflow == 3 and overflow.recovered == 1
    print("boundary/zero/simultaneous/overflow: PASS")


def run_invalid_and_inflight(pmk: PMK) -> None:
    rng = np.random.default_rng(88)
    A, Bt, a_seed, b_seed, _tiles = make_job(rng, 64, 64, 2048)
    rc, _ = pmk.run_job(A[:, :1024], Bt[:, :1024], a_seed, b_seed, 0, 0, 4, 64, 201)
    assert rc != PMK_SUCCESS
    rc, _ = pmk.run_job(A, Bt, a_seed, b_seed, 0, 0, 4, 64, 202)
    assert rc == PMK_SUCCESS
    A64, Bt64, a64, b64, _ = make_job(rng, 64, 64, 65536)
    rc, _ = pmk.run_job(A64, Bt64, a64, b64, 0, 0, 4, 64, 203)
    assert rc != PMK_SUCCESS
    rc, _ = pmk.run_job(A64, Bt64, a64, b64, 0, 0, 4, 64, 204, diagnostic=True)
    assert rc == PMK_SUCCESS
    handles = []
    for i in range(3):
        rc, h = pmk.run_job(A, Bt, a_seed, b_seed, 0, 0, 4, 64, 210 + i, keep_open=True)
        assert rc == PMK_SUCCESS
        handles.append(h)
    err = C.create_string_buffer(512)
    assert pmk.lib.pmk_probe_refresh(pmk.ctx, err, len(err)) == PMK_BUSY
    first_job, first_a, first_bt, _first_count = handles[0]
    del first_job
    desc = JobDesc()
    desc.abi_version = 1
    desc.m, desc.n, desc.k = A.shape[0], Bt.shape[0], A.shape[1]
    desc.a = C.cast(first_a, C.POINTER(C.c_int8))
    desc.bt = C.cast(first_bt, C.POINTER(C.c_int8))
    desc.a_bytes, desc.bt_bytes = A.size, Bt.size
    desc.a_seed = seed_words(a_seed)
    desc.b_seed = seed_words(b_seed)
    desc.block_bound = (C.c_uint32 * 8)(*words_le(0))
    desc.share_bound = (C.c_uint32 * 8)(*words_le(0))
    desc.block_capacity = 4
    desc.share_capacity = 64
    desc.cert_version = 3
    desc.rank = 128
    desc.job_id = 214
    extra = C.c_void_p()
    noop = pmk.callback_type(lambda _job, _user: None)
    pmk._callbacks.append(noop)
    rc = pmk.lib.pmk_run_job(pmk.ctx, C.byref(desc), noop, None, C.byref(extra))
    assert rc == PMK_BUSY, rc
    releasable = []
    for job, a_ptr, bt_ptr, callback_count in handles:
        result = Result()
        deadline = time.time() + 60
        while time.time() < deadline:
            rc = pmk.lib.pmk_poll(job, C.byref(result))
            if rc != PMK_PENDING:
                break
            time.sleep(0.005)
        assert rc == PMK_SUCCESS and result.status == PMK_SUCCESS
        deadline = time.time() + 5
        while callback_count.value == 0 and time.time() < deadline:
            time.sleep(0.001)
        assert callback_count.value == 1
        pmk.release_job(job)
        releasable.append((a_ptr, bt_ptr))
    for a_ptr, bt_ptr in releasable:
        pmk.release_buffer(a_ptr)
        pmk.release_buffer(bt_ptr)
    print("invalid inputs and 3-inflight cap: PASS")


def run_callback_drain(pmk: PMK) -> None:
    """The external owner cannot release while a foreign callback is returning."""
    rng = np.random.default_rng(0xCBAC)
    A, Bt, a_seed, b_seed, _tiles = make_job(rng, 64, 64, 2048)
    callback_entered = threading.Event()
    allow_callback_return = threading.Event()
    wait_returned = threading.Event()
    wait_codes: list[int] = []

    def block_callback(_job) -> None:
        assert pmk.lib.pmk_job_wait_callback(_job) == PMK_BUSY
        callback_entered.set()
        assert allow_callback_return.wait(5)

    rc, payload = pmk.run_job(
        A, Bt, a_seed, b_seed, 0, 0, 4, 64, 250,
        keep_open=True, callback_body=block_callback,
    )
    assert rc == PMK_SUCCESS
    job, a_ptr, bt_ptr, callback_count = payload
    result = Result()
    deadline = time.time() + 60
    while time.time() < deadline:
        rc = pmk.lib.pmk_poll(job, C.byref(result))
        if rc != PMK_PENDING:
            break
        time.sleep(0.005)
    assert rc == PMK_SUCCESS and result.status == PMK_SUCCESS
    assert callback_entered.wait(5)

    def wait_for_callback_return() -> None:
        wait_codes.append(pmk.lib.pmk_job_wait_callback(job))
        wait_returned.set()

    waiter = threading.Thread(target=wait_for_callback_return)
    waiter.start()
    assert not wait_returned.wait(0.05), "callback wait returned before callback"
    allow_callback_return.set()
    waiter.join(5)
    assert not waiter.is_alive()
    assert wait_codes == [PMK_SUCCESS]
    assert callback_count.value == 1
    pmk.release_job(job)
    pmk.release_buffer(a_ptr)
    pmk.release_buffer(bt_ptr)
    print("callback-return lifetime barrier: PASS")


def run_e2e(pmk: PMK) -> None:
    import pearl_mining as pm
    import crosscheck_sg

    header, _raw_a, _raw_b, a_seed, _Ap, _Bp, tiles, bound = crosscheck_sg.oracle_job(64, 0x1E100000)
    A = np.full((crosscheck_sg.M, crosscheck_sg.K), 64, dtype=np.int8)
    Bt = np.full((crosscheck_sg.N, crosscheck_sg.K), 64, dtype=np.int8)
    # oracle_job returns a_seed after computing b_seed internally; recompute both seeds from the same public data.
    cfg = crosscheck_sg.make_config()
    jk = oracle.job_key(bytes(crosscheck_sg.make_header(0x1E100000).to_bytes()), bytes(cfg.to_bytes()))
    raw_a = oracle.blake3(oracle.pad_to_chunk(A.tobytes()), key=jk).digest()
    raw_b = oracle.blake3(oracle.pad_to_chunk(Bt.tobytes()), key=jk).digest()
    b_seed, a_seed = oracle.seeds(jk, raw_a, raw_b, crosscheck_sg.M, crosscheck_sg.N)
    rc, payload = pmk.run_job(A, Bt, a_seed, b_seed, bound, U256_MAX, 128, len(tiles), 300, keep_open=True)
    assert rc == PMK_SUCCESS
    job, a_ptr, bt_ptr, callback_count = payload
    result = Result()
    deadline = time.time() + 180
    while time.time() < deadline:
        poll_rc = pmk.lib.pmk_poll(job, C.byref(result))
        if poll_rc != PMK_PENDING:
            break
        time.sleep(0.005)
    assert poll_rc == PMK_SUCCESS and result.status == PMK_SUCCESS
    deadline = time.time() + 5
    while callback_count.value == 0 and time.time() < deadline:
        time.sleep(0.001)
    assert callback_count.value == 1
    blocks = [slot_words(result.blocks[i]) for i in range(result.block_stored)]
    assert result.block_count > 0 and result.block_stored == result.block_count
    gpu_tiles = {(s[0], s[1]) for s in blocks}
    expected = next(t for t in tiles if t["hv"] <= bound)
    assert (expected["t_rows"], expected["t_cols"]) in gpu_tiles

    a_tree = pm.MerkleTree(A.tobytes(), jk)
    bt_tree = pm.MerkleTree(Bt.tobytes(), jk)
    a_leaf_indices = pm.MerkleTree.compute_leaf_indices_from_rows(expected["rows"], A.shape)
    bt_leaf_indices = pm.MerkleTree.compute_leaf_indices_from_rows(expected["cols"], Bt.shape)
    a_proof = pm.MatrixMerkleProof(a_tree.get_multileaf_proof(a_leaf_indices), expected["rows"])
    bt_proof = pm.MatrixMerkleProof(bt_tree.get_multileaf_proof(bt_leaf_indices), expected["cols"])
    proof = pm.PlainProof(crosscheck_sg.M, crosscheck_sg.N, crosscheck_sg.K, 128, a_proof, bt_proof, None)
    ok, msg = pm.verify_plain_proof_for_cert_version(3, header, proof)
    assert ok, msg
    assert list(proof.a.row_indices) == expected["rows"]
    assert list(proof.bt.row_indices) == expected["cols"]
    pmk.release_job(job)
    pmk.release_buffer(a_ptr)
    pmk.release_buffer(bt_ptr)
    print("end-to-end GPU find -> PlainProof verify: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=pathlib.Path, default=ROOT / "libpmk" / ".build" / "release" / "libpmk.dylib")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-e2e", action="store_true")
    args = parser.parse_args()
    pmk = PMK(args.library)
    try:
        print(f"probe cache key {pmk.cache_key}")
        run_matrix_tests(pmk, args.quick)
        run_boundaries_and_overflow(pmk)
        run_invalid_and_inflight(pmk)
        run_callback_drain(pmk)
        if not args.no_e2e:
            run_e2e(pmk)
    finally:
        pmk.close()
    print("libpmk integration: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
