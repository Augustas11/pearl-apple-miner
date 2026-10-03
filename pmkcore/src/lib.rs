//! pmkcore: native job builder for the Pearl v3 Metal miner (SPEC §4 R-A1, gate F2).
//!
//! Per template (once): `job_key = blake3(header76 ‖ config52)` (C1), A (m×k, row-major) and its
//! raw keyed-BLAKE3 Merkle root, salted with m (C2, cert v3).
//! Per job: fresh Bᵀ (n×k row-major, int8 in [-64,64]) from AES-128-CTR keyed from OS entropy,
//! written straight into caller memory, its raw root, then the salted seed chain
//! (`zk-pow/src/ffi/mine.rs:442-479`, `api/seed.rs`). Generation and hashing are fused per
//! 64 KiB segment so the bytes are hashed while still in cache; the segment chaining values are
//! merged into the BLAKE3 tree root, which equals `pearl_blake3::MerkleTree::new(..).root()`
//! (proved by `tests/reference.rs`).

use aes::Aes128;
use blake3::hazmat::{
    left_subtree_len, merge_subtrees_non_root, merge_subtrees_root, ChainingValue, HasherExt, Mode,
};
use ctr::cipher::{KeyIvInit, StreamCipher};
use rayon::prelude::*;
use std::ffi::CStr;
use std::panic::{catch_unwind, AssertUnwindSafe};
use zk_pow::api::proof::{IncompleteBlockHeader, MiningConfiguration};
use zk_pow::api::seed::{bind_root_a, bind_root_b};

pub mod oracle;
pub mod oracle_ffi;
pub use oracle_ffi::*;

#[cfg(all(target_arch = "aarch64", not(aes_armv8)))]
compile_error!("build with RUSTFLAGS=\"--cfg aes_armv8\" (see .cargo/config.toml): without it AES-CTR runs in software at ~0.16 GB/s");

pub type Hash256 = [u8; 32];

pub const CHUNK_LEN: usize = blake3::CHUNK_LEN; // 1024
/// Fused generate+hash unit. A power of two number of chunks, so each segment is a BLAKE3 subtree.
pub const SEGMENT_LEN: usize = 64 * CHUNK_LEN;
/// Elements mapped per keystream refill (keystream is 2 bytes per element and stays in L1).
const SUB_ELEMS: usize = 4096;
static ZEROS: [u8; 2 * SUB_ELEMS] = [0u8; 2 * SUB_ELEMS];

/// v1 policy (SPEC §3.2).
pub const POLICY_RANK: u16 = 128;
const MAX_ABI_BUFFER_BYTES: usize = 2_000_000_000;
const MAX_ABI_THREADS: u32 = 1024;
pub const HEADER_LEN: usize = IncompleteBlockHeader::SERIALIZED_SIZE; // 76
pub const CONFIG_LEN: usize = MiningConfiguration::SERIALIZED_SIZE; // 52

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum PmkError {
    NullPointer = -1,
    BadHeader = -2,
    BadConfig = -3,
    Policy = -4,
    BadShape = -5,
    BufferTooSmall = -6,
    SignalOutOfRange = -7,
    Entropy = -8,
    ThreadPool = -9,
    IllegalOffset = -10,
    Proof = -11,
    Panic = -12,
    ResourceLimit = -13,
}

impl PmkError {
    const ALL: [PmkError; 13] = [
        PmkError::NullPointer,
        PmkError::BadHeader,
        PmkError::BadConfig,
        PmkError::Policy,
        PmkError::BadShape,
        PmkError::BufferTooSmall,
        PmkError::SignalOutOfRange,
        PmkError::Entropy,
        PmkError::ThreadPool,
        PmkError::IllegalOffset,
        PmkError::Proof,
        PmkError::Panic,
        PmkError::ResourceLimit,
    ];

    pub fn message(self) -> &'static CStr {
        match self {
            PmkError::NullPointer => c"null pointer argument",
            PmkError::BadHeader => {
                c"header bytes do not parse/round-trip as a 76-byte IncompleteBlockHeader"
            }
            PmkError::BadConfig => {
                c"config bytes do not parse/round-trip as a 52-byte MiningConfiguration"
            }
            PmkError::Policy => {
                c"config violates v1 production policy (rank 128, k % 128 == 0, 2048 <= k <= 8192, no MoE)"
            }
            PmkError::BadShape => c"m/n zero, above 2^24, or not multiples of the pattern periods",
            PmkError::BufferTooSmall => c"buffer shorter than pmkcore_padded_len(rows, k)",
            PmkError::SignalOutOfRange => c"caller-supplied matrix has a value outside [-64, 64]",
            PmkError::Entropy => c"OS entropy source failed",
            PmkError::ThreadPool => c"rayon global thread pool already initialised",
            PmkError::IllegalOffset => c"tile offset is illegal or outside the matrix",
            PmkError::Proof => c"proof construction or config reconstruction failed",
            PmkError::Panic => c"internal panic caught at C ABI boundary",
            PmkError::ResourceLimit => c"oracle resource limit exceeded",
        }
    }
}

/// Template state; plain data, owned by the caller (C: `PmkTemplate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Template {
    pub job_key: Hash256,
    /// Raw Merkle root of padded row-major A (what the proof carries).
    pub raw_root_a: Hash256,
    /// `bind_root_a(raw_root_a, m)` — salted, used for the seed chain only.
    pub salted_root_a: Hash256,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub _reserved: u32,
}

/// Per-job outputs (C: `PmkJob`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct Job {
    /// Raw Merkle root of padded row-major Bᵀ (what the proof carries).
    pub raw_root_b: Hash256,
    pub b_noise_seed: Hash256,
    pub a_noise_seed: Hash256,
}

/// `pearl_blake3::padded_chunk_len(rows * cols)`.
pub fn padded_len(rows: usize, cols: usize) -> usize {
    (rows * cols).div_ceil(CHUNK_LEN) * CHUNK_LEN
}

fn checked_padded_len(rows: usize, cols: usize) -> Result<usize, PmkError> {
    let used = rows.checked_mul(cols).ok_or(PmkError::ResourceLimit)?;
    let padded = used
        .checked_add(CHUNK_LEN - 1)
        .map(|v| v / CHUNK_LEN * CHUNK_LEN)
        .ok_or(PmkError::ResourceLimit)?;
    if padded > MAX_ABI_BUFFER_BYTES {
        return Err(PmkError::ResourceLimit);
    }
    Ok(padded)
}

/// C1: `blake3(header_bytes ‖ config_bytes)`.
pub fn job_key(header: &[u8; HEADER_LEN], config: &[u8; CONFIG_LEN]) -> Hash256 {
    let mut h = blake3::Hasher::new();
    h.update(header);
    h.update(config);
    *h.finalize().as_bytes()
}

/// Parse and policy-check header/config; returns k.
pub fn check_job_params(
    header: &[u8; HEADER_LEN],
    config: &[u8; CONFIG_LEN],
    m: u32,
    n: u32,
) -> Result<u32, PmkError> {
    let hdr = IncompleteBlockHeader::from_bytes(header).map_err(|_| PmkError::BadHeader)?;
    if hdr.to_bytes() != *header {
        return Err(PmkError::BadHeader);
    }
    let cfg = MiningConfiguration::from_bytes(config).map_err(|_| PmkError::BadConfig)?;
    let k = cfg.common_dim;
    if cfg.rank != POLICY_RANK
        || cfg.moe.is_some()
        || !k.is_multiple_of(128)
        || !(2048..=8192).contains(&k)
    {
        return Err(PmkError::Policy);
    }
    let (rp, cp) = (cfg.rows_pattern.period(), cfg.cols_pattern.period());
    if m == 0
        || n == 0
        || m > 1 << 24
        || n > 1 << 24
        || !m.is_multiple_of(rp)
        || !n.is_multiple_of(cp)
    {
        return Err(PmkError::BadShape);
    }
    Ok(k)
}

/// Map 2 keystream bytes per element to [-64, 64] (multiply-shift; max relative bias 1/508).
#[inline]
fn map_signal(out: &mut [u8], ks: &[u8]) {
    for (o, p) in out.iter_mut().zip(ks.as_chunks::<2>().0.iter()) {
        let u = u16::from_le_bytes([p[0], p[1]]) as u32;
        *o = (((u * 129) >> 16) as i32 - 64) as i8 as u8;
    }
}

/// Fill `seg` (element offset `elem_off` in the matrix) from AES-128-CTR(key) at keystream
/// byte offset 2*elem_off. Each element consumes a distinct 2-byte keystream slice.
fn fill_segment(seg: &mut [u8], elem_off: usize, key: &[u8; 16]) {
    let block_off = (2 * elem_off / 16) as u128;
    let mut c = ctr::Ctr128BE::<Aes128>::new(key.into(), &block_off.to_be_bytes().into());
    let mut ks = [0u8; 2 * SUB_ELEMS];
    for out in seg.chunks_mut(SUB_ELEMS) {
        let ks = &mut ks[..2 * out.len()];
        c.apply_keystream_b2b(&ZEROS[..ks.len()], ks)
            .expect("keystream length");
        map_signal(out, ks);
    }
}

fn subtree_cv(cvs: &[ChainingValue], start: u64, len: u64, mode: Mode) -> ChainingValue {
    if len <= SEGMENT_LEN as u64 {
        return cvs[(start / SEGMENT_LEN as u64) as usize];
    }
    let left = left_subtree_len(len);
    merge_subtrees_non_root(
        &subtree_cv(cvs, start, left, mode),
        &subtree_cv(cvs, start + left, len - left, mode),
        mode,
    )
}

/// Keyed BLAKE3 root of `buf` (whose length must be a multiple of CHUNK_LEN — i.e. already
/// chunk-padded). If `fill` is set, the first `used` bytes are first overwritten with fresh
/// [-64,64] signal from AES-128-CTR(fill) and `buf[used..]` is zeroed (the pad).
/// Runs on the current rayon pool.
pub fn commit_root(buf: &mut [u8], used: usize, key: &Hash256, fill: Option<&[u8; 16]>) -> Hash256 {
    assert!(!buf.is_empty() && buf.len().is_multiple_of(CHUNK_LEN) && used <= buf.len());
    if fill.is_some() {
        buf[used..].fill(0);
    }
    let seg_work = |i: usize, seg: &mut [u8]| {
        if let Some(rk) = fill {
            let start = i * SEGMENT_LEN;
            let gen = used.saturating_sub(start).min(seg.len());
            fill_segment(&mut seg[..gen], start, rk);
        }
    };
    if buf.len() <= SEGMENT_LEN {
        seg_work(0, buf);
        return *blake3::Hasher::new_keyed(key)
            .update(buf)
            .finalize()
            .as_bytes();
    }
    let cvs: Vec<ChainingValue> = buf
        .par_chunks_mut(SEGMENT_LEN)
        .enumerate()
        .map(|(i, seg)| {
            seg_work(i, seg);
            blake3::Hasher::new_keyed(key)
                .set_input_offset((i * SEGMENT_LEN) as u64)
                .update(seg)
                .finalize_non_root()
        })
        .collect();
    let total = buf.len() as u64;
    let mode = Mode::KeyedHash(key);
    let left = left_subtree_len(total);
    *merge_subtrees_root(
        &subtree_cv(&cvs, 0, left, mode),
        &subtree_cv(&cvs, left, total - left, mode),
        mode,
    )
    .as_bytes()
}

fn os_key() -> Result<[u8; 16], PmkError> {
    let mut k = [0u8; 16];
    getrandom::fill(&mut k).map_err(|_| PmkError::Entropy)?;
    Ok(k)
}

fn check_range(buf: &[u8]) -> Result<(), PmkError> {
    if buf
        .par_chunks(SEGMENT_LEN)
        .all(|c| c.iter().all(|&b| (-64..=64).contains(&(b as i8))))
    {
        Ok(())
    } else {
        Err(PmkError::SignalOutOfRange)
    }
}

/// Build the template. `a_buf` holds A (m×k row-major), at least `padded_len(m, k)` bytes.
/// `generate_a = true` fills A from the CSPRNG; otherwise the caller's A is range-checked.
/// The chunk pad `a_buf[m*k..padded]` is zeroed in both cases.
pub fn template_init(
    header: &[u8; HEADER_LEN],
    config: &[u8; CONFIG_LEN],
    m: u32,
    n: u32,
    a_buf: &mut [u8],
    generate_a: bool,
) -> Result<Template, PmkError> {
    let k = check_job_params(header, config, m, n)?;
    let used = (m as usize)
        .checked_mul(k as usize)
        .ok_or(PmkError::ResourceLimit)?;
    let plen = checked_padded_len(m as usize, k as usize)?;
    if a_buf.len() < plen {
        return Err(PmkError::BufferTooSmall);
    }
    let a = &mut a_buf[..plen];
    let jk = job_key(header, config);
    let raw = if generate_a {
        let rk = os_key()?;
        commit_root(a, used, &jk, Some(&rk))
    } else {
        check_range(&a[..used])?;
        a[used..].fill(0);
        commit_root(a, used, &jk, None)
    };
    Ok(Template {
        job_key: jk,
        raw_root_a: raw,
        salted_root_a: bind_root_a(&raw, m),
        m,
        n,
        k,
        _reserved: 0,
    })
}

/// cert-v3 seed chain from a raw B root (mine.rs:442-479 with SeedDerivation::Salted).
pub fn seeds(t: &Template, raw_root_b: &Hash256) -> Job {
    let hb = bind_root_b(raw_root_b, t.n);
    let mut h = blake3::Hasher::new();
    h.update(&t.job_key).update(&hb);
    let b_noise_seed = *h.finalize().as_bytes();
    let mut h = blake3::Hasher::new();
    h.update(&b_noise_seed).update(&t.salted_root_a);
    Job {
        raw_root_b: *raw_root_b,
        b_noise_seed,
        a_noise_seed: *h.finalize().as_bytes(),
    }
}

/// Per job: fresh Bᵀ (n×k row-major) into `bt_buf` (≥ padded_len(n, k)), root, seeds.
fn validate_template_shape(t: &Template) -> Result<(), PmkError> {
    let (m, n, k) = (t.m, t.n, t.k);
    if m == 0
        || n == 0
        || m > 1 << 24
        || n > 1 << 24
        || !m.is_multiple_of(64)
        || !n.is_multiple_of(64)
        || !k.is_multiple_of(128)
        || !(2048..=8192).contains(&k)
    {
        return Err(PmkError::Policy);
    }
    checked_padded_len(m as usize, k as usize)?;
    checked_padded_len(n as usize, k as usize)?;
    Ok(())
}

pub fn build_job_with_key(
    t: &Template,
    bt_buf: &mut [u8],
    rng_key: &[u8; 16],
) -> Result<Job, PmkError> {
    validate_template_shape(t)?;
    let (n, k) = (t.n as usize, t.k as usize);
    let used = n.checked_mul(k).ok_or(PmkError::ResourceLimit)?;
    let plen = checked_padded_len(n, k)?;
    if bt_buf.len() < plen {
        return Err(PmkError::BufferTooSmall);
    }
    let root = commit_root(&mut bt_buf[..plen], used, &t.job_key, Some(rng_key));
    Ok(seeds(t, &root))
}

pub fn build_job(t: &Template, bt_buf: &mut [u8]) -> Result<Job, PmkError> {
    build_job_with_key(t, bt_buf, &os_key()?)
}

/// Per job with caller-supplied Bᵀ (range-checked, pad zeroed).
pub fn commit_job(t: &Template, bt_buf: &mut [u8]) -> Result<Job, PmkError> {
    validate_template_shape(t)?;
    let (n, k) = (t.n as usize, t.k as usize);
    let used = n.checked_mul(k).ok_or(PmkError::ResourceLimit)?;
    let plen = checked_padded_len(n, k)?;
    if bt_buf.len() < plen {
        return Err(PmkError::BufferTooSmall);
    }
    let bt = &mut bt_buf[..plen];
    check_range(&bt[..used])?;
    bt[used..].fill(0);
    let root = commit_root(bt, used, &t.job_key, None);
    Ok(seeds(t, &root))
}

// ---------------------------------------------------------------- C ABI (include/pmkcore.h)

fn code(f: impl FnOnce() -> Result<(), PmkError>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => e as i32,
        Err(_) => PmkError::Panic as i32,
    }
}

fn len_to_usize(len: u64) -> Result<usize, PmkError> {
    let len = usize::try_from(len).map_err(|_| PmkError::ResourceLimit)?;
    if len > isize::MAX as usize {
        return Err(PmkError::ResourceLimit);
    }
    Ok(len)
}

fn checked_len_u64(rows: u64, cols: u64) -> Result<u64, PmkError> {
    let rows = len_to_usize(rows)?;
    let cols = len_to_usize(cols)?;
    let len = checked_padded_len(rows, cols)?;
    u64::try_from(len).map_err(|_| PmkError::ResourceLimit)
}

unsafe fn slice_mut_from_raw<'a>(
    ptr: *mut u8,
    supplied_len: u64,
    required_len: usize,
) -> Result<&'a mut [u8], PmkError> {
    if ptr.is_null() {
        return Err(PmkError::NullPointer);
    }
    let supplied = len_to_usize(supplied_len)?;
    if supplied < required_len {
        return Err(PmkError::BufferTooSmall);
    }
    Ok(std::slice::from_raw_parts_mut(ptr, required_len))
}

unsafe fn array_from_raw<'a, const N: usize>(ptr: *const u8) -> Result<&'a [u8; N], PmkError> {
    if ptr.is_null() {
        return Err(PmkError::NullPointer);
    }
    Ok(&*(ptr as *const [u8; N]))
}

/// Size the rayon global pool (0 = one thread per logical CPU). Call at most once, before any
/// other call; otherwise the pool is created lazily with the default size.
#[no_mangle]
pub extern "C" fn pmkcore_init(num_threads: u32) -> i32 {
    code(|| {
        if num_threads > MAX_ABI_THREADS {
            return Err(PmkError::ResourceLimit);
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads as usize)
            .build_global()
            .map_err(|_| PmkError::ThreadPool)?;
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn pmkcore_padded_len(rows: u64, cols: u64) -> u64 {
    catch_unwind(AssertUnwindSafe(|| {
        checked_len_u64(rows, cols).unwrap_or(0)
    }))
    .unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn pmkcore_strerror(code: i32) -> *const std::ffi::c_char {
    catch_unwind(AssertUnwindSafe(|| {
        match PmkError::ALL.into_iter().find(|e| *e as i32 == code) {
            Some(e) => e.message().as_ptr(),
            None if code == 0 => c"ok".as_ptr(),
            None => c"unknown error".as_ptr(),
        }
    }))
    .unwrap_or_else(|_| PmkError::Panic.message().as_ptr())
}

/// # Safety
/// `header` → 76 bytes, `config` → 52 bytes, `a_buf` → `a_len` writable bytes, `out` → PmkTemplate.
#[no_mangle]
pub unsafe extern "C" fn pmkcore_template_init(
    header: *const u8,
    config: *const u8,
    m: u32,
    n: u32,
    a_buf: *mut u8,
    a_len: u64,
    generate_a: u8,
    out: *mut Template,
) -> i32 {
    code(|| {
        if out.is_null() {
            return Err(PmkError::NullPointer);
        }
        let header = array_from_raw::<HEADER_LEN>(header)?;
        let config = array_from_raw::<CONFIG_LEN>(config)?;
        let k = check_job_params(header, config, m, n)?;
        let required = checked_padded_len(m as usize, k as usize)?;
        let a = slice_mut_from_raw(a_buf, a_len, required)?;
        *out = template_init(header, config, m, n, a, generate_a != 0)?;
        Ok(())
    })
}

/// Fresh Bᵀ into `bt_buf` (≥ pmkcore_padded_len(n, k)) plus root and seeds into `out`.
/// # Safety
/// `tmpl` from pmkcore_template_init; `bt_buf` → `bt_len` writable bytes; `out` → PmkJob.
#[no_mangle]
pub unsafe extern "C" fn pmkcore_build_job(
    tmpl: *const Template,
    bt_buf: *mut u8,
    bt_len: u64,
    out: *mut Job,
) -> i32 {
    code(|| {
        if tmpl.is_null() || out.is_null() {
            return Err(PmkError::NullPointer);
        }
        let tmpl = &*tmpl;
        validate_template_shape(tmpl)?;
        let required = checked_padded_len(tmpl.n as usize, tmpl.k as usize)?;
        let bt = slice_mut_from_raw(bt_buf, bt_len, required)?;
        *out = build_job(tmpl, bt)?;
        Ok(())
    })
}

/// Root and seeds for a caller-filled Bᵀ (range-checked; pad zeroed).
/// # Safety
/// As pmkcore_build_job.
#[no_mangle]
pub unsafe extern "C" fn pmkcore_commit_job(
    tmpl: *const Template,
    bt_buf: *mut u8,
    bt_len: u64,
    out: *mut Job,
) -> i32 {
    code(|| {
        if tmpl.is_null() || out.is_null() {
            return Err(PmkError::NullPointer);
        }
        let tmpl = &*tmpl;
        validate_template_shape(tmpl)?;
        let required = checked_padded_len(tmpl.n as usize, tmpl.k as usize)?;
        let bt = slice_mut_from_raw(bt_buf, bt_len, required)?;
        *out = commit_job(tmpl, bt)?;
        Ok(())
    })
}
