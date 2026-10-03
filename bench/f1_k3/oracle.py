"""CPU oracle for the F1 K3-NA prototype (numpy int64 + `blake3` keyed hashing).

Implements Pearl's jackpot exactly as zk-pow does (vendor/pearl @ 7039e66f):
  * tiles: threads_partition (mine.rs:485-497) using PeriodicPattern.offset_is_valid (proof_utils.rs:223-232)
  * transcript: mine.rs:87-107 (cumulative int32 acc, XOR fold per full rank chunk, slot (ll/r-1)%16, rotl13)
  * hash: blake3(le_bytes(jackpot[16]), key=a_noise_seed) (proof_utils.rs:1502-1505), U256 LE <= bound (mine.rs:108-110)
  * bound: extract_difficulty_bound = saturating nbits_to_difficulty(nbits) * h*w*dot_len (sanity_checks.rs:183-231)
  * job key / commitments / salted seeds / noise for the pearl_mining cross-check:
    mine.rs:431-479, seed.rs:11-59, circuit/pearl_noise.rs:19-153
No code from zk-pow is imported; pearl_mining is only used by crosscheck.py as the reference to compare against.
"""
from __future__ import annotations

import struct

import numpy as np
from blake3 import blake3

RANK = 128
JACKPOT_SIZE = 16
LROT = 13
ROWS_PATTERN = [0, 8, 64, 72]
COLS_PATTERN = [0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51]
U256_MAX = (1 << 256) - 1


# ---------------- PeriodicPattern (port of proof_utils.rs:100-245) ----------------
def pattern_shape(pattern: list[int]) -> list[tuple[int, int]]:
    """from_list: decompose a sorted, 0-based index list into <=3 (stride, length) dims, padded with (period, 1)."""
    assert pattern and pattern[0] == 0 and all(a < b for a, b in zip(pattern, pattern[1:]))
    p = list(pattern)
    shape: list[tuple[int, int]] = []
    while len(p) > 1:
        for period in range(1, len(p)):
            if len(p) % period == 0:
                s = p[period]
                if all(p[i] + s == p[i + period] for i in range(len(p) - period)):
                    shape.append((s, len(p) // period))
                    p = p[:period]
                    break
        else:
            raise ValueError("Pattern is not periodic")
    shape.reverse()
    period = shape[-1][0] * shape[-1][1] if shape else 1
    while len(shape) < 3:
        shape.append((period, 1))
    assert len(shape) == 3
    return shape


def offset_is_valid(shape: list[tuple[int, int]], offset: int) -> bool:
    for stride, length in reversed(shape):
        offset %= stride * length
        if offset >= stride:
            return False
    return True


def period(shape: list[tuple[int, int]]) -> int:
    return shape[-1][0] * shape[-1][1]


def threads_partition(pattern: list[int], total: int) -> list[list[int]]:
    """mine.rs:485-497: one index set per valid offset, offsets ascending."""
    shape = pattern_shape(pattern)
    assert total % period(shape) == 0, "total_dimension must be divisible by pattern period"
    return [[o + d for d in pattern] for o in range(total) if offset_is_valid(shape, o)]


# ---------------- jackpot ----------------
def rotl32(x: np.ndarray, r: int) -> np.ndarray:
    return ((x << np.uint32(r)) | (x >> np.uint32(32 - r))).astype(np.uint32)


def transcripts(Ap: np.ndarray, Bp: np.ndarray, rows_pattern=ROWS_PATTERN, cols_pattern=COLS_PATTERN, rank=RANK):
    """Ap: m x k noised A' (int), Bp: k x n noised B' (int).
    Returns (row_sets [Tr x h], col_sets [Tc x w], jackpot [Tr, Tc, 16] uint32), Pearl tile order = (row set, col set)."""
    m, k = Ap.shape
    k2, n = Bp.shape
    assert k == k2
    rs = np.array(threads_partition(rows_pattern, m), dtype=np.int64)
    cs = np.array(threads_partition(cols_pattern, n), dtype=np.int64)
    acc = np.zeros((m, n), dtype=np.int64)
    jp = np.zeros((len(rs), len(cs), JACKPOT_SIZE), dtype=np.uint32)
    A64 = Ap.astype(np.int64)
    B64 = Bp.astype(np.int64)
    for ll in range(rank, k + 1, rank):   # full rank chunks only; trailing k % r excluded
        acc += A64[:, ll - rank:ll] @ B64[ll - rank:ll, :]
        assert acc.min() >= -(1 << 31) and acc.max() < (1 << 31), "int32 overflow"
        u = acc.astype(np.int32).view(np.uint32)
        g = u[rs[:, :, None, None], cs[None, None, :, :]]          # [Tr, h, Tc, w]
        x = np.bitwise_xor.reduce(np.bitwise_xor.reduce(g, axis=3), axis=1)   # [Tr, Tc]
        t = (ll // rank - 1) % JACKPOT_SIZE
        jp[:, :, t] = rotl32(jp[:, :, t], LROT) ^ x
    return rs, cs, jp


def jackpot_hash(jp16: np.ndarray, key32: bytes) -> bytes:
    msg = np.asarray(jp16, dtype="<u4").tobytes()
    assert len(msg) == 64
    return blake3(msg, key=key32).digest()


def u256_le(h: bytes) -> int:
    return int.from_bytes(h, "little")


def words_le(x: int) -> list[int]:
    return [(x >> (32 * i)) & 0xFFFFFFFF for i in range(8)]


def all_tiles(Ap, Bp, key32: bytes):
    """List of dicts per tile in Pearl iteration order: t_rows, t_cols, jp (16 u32), hash (bytes), hv (int)."""
    rs, cs, jp = transcripts(Ap, Bp)
    out = []
    for i in range(len(rs)):
        for j in range(len(cs)):
            h = jackpot_hash(jp[i, j], key32)
            out.append({"t_rows": int(rs[i, 0]), "t_cols": int(cs[j, 0]), "rows": rs[i].tolist(), "cols": cs[j].tolist(),
                        "jp": [int(v) for v in jp[i, j]], "hash": h, "hv": u256_le(h)})
    return out


# ---------------- difficulty (proof_utils.rs:54-82, sanity_checks.rs:183-231) ----------------
def nbits_to_difficulty(nbits: int) -> int:
    e, mant = nbits >> 24, nbits & 0xFFFFFF
    if mant == 0 or e == 0 or mant & 0x800000:
        return 0
    t = mant >> (8 * (3 - e)) if e <= 3 else mant << (8 * (e - 3))
    return t & U256_MAX   # U256 shl truncates


def extract_difficulty_bound(nbits: int, h: int, w: int, dot_len: int) -> int:
    base, f = nbits_to_difficulty(nbits), h * w * dot_len
    return U256_MAX if base > U256_MAX // f else base * f


# ---------------- commitments, seeds, noise (cross-check only) ----------------
SEED_SALT_A = blake3(b"pearl/cert-v3/noise-seed/A").digest()
SEED_SALT_B = blake3(b"pearl/cert-v3/noise-seed/B").digest()
LABEL_A = b"A_tensor" + bytes(24)
LABEL_B = b"B_tensor" + bytes(24)


def pad_to_chunk(data: bytes) -> bytes:
    """pearl_blake3::pad_to_chunk_boundary: zero-pad to a multiple of the 1024-byte BLAKE3 chunk."""
    return data + bytes(-(-len(data) // 1024) * 1024 - len(data))


def job_key(header_bytes: bytes, config_bytes: bytes) -> bytes:
    return blake3(header_bytes + config_bytes).digest()


def bind_root(root: bytes, dim: int, salt: bytes) -> bytes:
    return blake3(root + struct.pack("<I", dim) + bytes(28), key=salt).digest()


def seeds(jk: bytes, raw_a: bytes, raw_b: bytes, m: int, n: int) -> tuple[bytes, bytes]:
    """Salted (cert v3) derivation; returns (b_noise_seed, a_noise_seed)."""
    ha, hb = bind_root(raw_a, m, SEED_SALT_A), bind_root(raw_b, n, SEED_SALT_B)
    b_seed = blake3(jk + hb).digest()
    a_seed = blake3(b_seed + ha).digest()
    return b_seed, a_seed


def _rand_hash(index: int, label: bytes, key: bytes, slot: int) -> bytes:
    msg = bytearray(64)
    msg[slot * 4:slot * 4 + 4] = struct.pack("<i", 1 + index)
    msg[32:64] = label
    return blake3(bytes(msg), key=key).digest()


def uniform_rows(label: bytes, key: bytes, rows: list[int], ncols: int) -> np.ndarray:
    out = np.zeros((len(rows), ncols), dtype=np.int64)
    for ri, row in enumerate(rows):
        start = row * ncols
        b0, b1 = start // 32, -(-(start + ncols) // 32)
        buf = b"".join(_rand_hash(b, label, key, 0) for b in range(b0, b1))
        seg = np.frombuffer(buf, dtype=np.uint8)[start - b0 * 32:start - b0 * 32 + ncols]
        out[ri] = (seg & 63).astype(np.int64) - 32
    return out


def perm_matrix(label: bytes, key: bytes, k: int, r: int) -> np.ndarray:
    res = np.zeros((k, 2), dtype=np.int64)
    for i in range(-(-k // 8)):
        h = _rand_hash(i, label, key, 1)
        for j in range(8):
            idx = i * 8 + j
            if idx >= k:
                break
            u = struct.unpack_from("<I", h, j * 4)[0]
            i0 = u & (r - 1)
            i1 = i0 ^ (1 + (((r - 1) * u) >> 32))
            res[idx] = (i0, i1)
    return res


def noise(k: int, r: int, b_seed: bytes, a_seed: bytes, m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Full noise: E_A (m x k) and E_B^T (n x k), values in [-63, 63]."""
    e_al = uniform_rows(LABEL_A, a_seed, list(range(m)), r)
    e_ar = perm_matrix(LABEL_A, a_seed, k, r)
    e_bl = perm_matrix(LABEL_B, b_seed, k, r)
    e_br = uniform_rows(LABEL_B, b_seed, list(range(n)), r)
    na = e_al[:, e_ar[:, 0]] - e_al[:, e_ar[:, 1]]
    nb = e_br[:, e_bl[:, 0]] - e_br[:, e_bl[:, 1]]
    return na, nb
