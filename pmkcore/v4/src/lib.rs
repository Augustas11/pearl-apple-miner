// SPDX-License-Identifier: Apache-2.0
// Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
#![allow(
    dead_code,
    unused_imports,
    unexpected_cfgs,
    unsafe_op_in_unsafe_fn,
    clippy::all
)]

#[path = "../../../vendor/pearl-fp8/zk-pow/src/api/mod.rs"]
pub mod api;
#[path = "../../../vendor/pearl-fp8/zk-pow/src/circuit/mod.rs"]
pub mod circuit;
#[path = "../../../vendor/pearl-fp8/zk-pow/src/ffi/mod.rs"]
pub mod ffi;
#[path = "../../../vendor/pearl-fp8/zk-pow/src/v1/mod.rs"]
pub mod v1;
#[path = "../../../vendor/pearl-fp8/zk-pow/src/v2/mod.rs"]
pub mod v2;

use anyhow::{Context, Result, bail, ensure};
use api::fp8::compute::bf16_max;
use api::fp8::dtype::f32_to_bf16;
use api::fp8::jackpot_policy::{JackpotMessage, JackpotPolicy, OperandStrip};
use api::fp8::noise::{OperandNoise, sample_noise};
use api::fp8::plain_proof::PlainProofV4;
use api::fp8::prequant::{BLOCK_SIZE, exact_norms, open_prequant};
use api::fp8::public_params::{
    CommonParams, Device, HashId, JackpotStatement, JobParams, OperandParams, PublicParams, Quant,
};
use api::fp8::quantization::{BuiltRows, Fp8E4M3Quant, NORM_FLOOR};
use api::fp8::transcript::{compute_jackpot_ticket, jackpot_key, key_a, key_b};
use api::fp8::utils::B200;
use api::fp8::zk::Fp8Prover;
use api::layout::AxisPattern;
use api::layout::DimType::{Blake, Fold};
use api::primitives::{BlockHeader, Hash256, IncompleteBlockHeader, Sides};
use api::proof_utils::{check_jackpot_difficulty, nbits_to_difficulty, operand_digest_fp10};
use ffi::plain_proof::MatrixMerkleProof;
use pearl_blake3::MerkleTree;
use primitive_types::U256;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::ffi::CStr;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;

pub const HEADER_LEN: usize = IncompleteBlockHeader::SERIALIZED_SIZE;
pub const BLOCK_HEADER_LEN: usize = BlockHeader::SERIALIZED_SIZE;
pub const RANK: u16 = 32;
pub const TILE_ROWS: u32 = 16;
pub const TILE_COLS: u32 = 16;
pub const SUPPORTED_K: [u32; 3] = [1024, 4096, 16384];
pub const MAX_ABI_BYTES: usize = 2_000_000_000;
pub const MAX_PRODUCTION_DIM: u32 = 8192;
const MAX_ABI_THREADS: u32 = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Pmk4Error {
    NullPointer = -1,
    BadHeader = -2,
    BadConfig = -3,
    Policy = -4,
    BadShape = -5,
    BufferTooSmall = -6,
    Range = -7,
    Entropy = -8,
    ThreadPool = -9,
    IllegalOffset = -10,
    Proof = -11,
    Panic = -12,
    ResourceLimit = -13,
    Verify = -14,
}

impl Pmk4Error {
    const ALL: [Self; 14] = [
        Self::NullPointer,
        Self::BadHeader,
        Self::BadConfig,
        Self::Policy,
        Self::BadShape,
        Self::BufferTooSmall,
        Self::Range,
        Self::Entropy,
        Self::ThreadPool,
        Self::IllegalOffset,
        Self::Proof,
        Self::Panic,
        Self::ResourceLimit,
        Self::Verify,
    ];

    fn message(self) -> &'static CStr {
        match self {
            Self::NullPointer => c"null pointer argument",
            Self::BadHeader => c"header bytes failed Pearl v4 parsing or ancestry validation",
            Self::BadConfig => c"v4 public/job parameters failed Pearl validation",
            Self::Policy => c"v4 policy rejected k/device/rank/layout or tile admissibility",
            Self::BadShape => c"matrix dimensions are invalid for v4 grid job",
            Self::BufferTooSmall => c"output buffer is too small",
            Self::Range => c"operand value out of supported range",
            Self::Entropy => c"OS entropy source failed",
            Self::ThreadPool => c"rayon global thread pool already initialised",
            Self::IllegalOffset => c"tile offset is invalid or outside the matrix",
            Self::Proof => c"v4 proof construction, parsing, or mutation failed",
            Self::Panic => c"internal panic caught at C ABI boundary",
            Self::ResourceLimit => c"resource limit exceeded",
            Self::Verify => c"Pearl v4 verifier rejected the proof",
        }
    }
}

impl From<anyhow::Error> for Pmk4Error {
    fn from(_: anyhow::Error) -> Self {
        Self::Proof
    }
}

struct Plane {
    bytes: Vec<u8>,
    tree: MerkleTree,
    root: Hash256,
}

struct OracleCache {
    a_built: BuiltRows,
    b_built: BuiltRows,
}

struct GpuCache {
    a_noise: OperandNoise,
    b_noise: OperandNoise,
    a_alpha: Vec<u16>,
    a_beta: Vec<u16>,
    a_l2: Vec<u16>,
    b_alpha: Vec<u16>,
    b_beta: Vec<u16>,
    b_l2: Vec<u16>,
}

pub struct FixturePayload {
    pub a_noised: Vec<u8>,
    pub b_noised: Vec<u8>,
    pub a_noise_e: Vec<u8>,
    pub a_noise_f: Vec<u8>,
    pub b_noise_e: Vec<u8>,
    pub b_noise_f: Vec<u8>,
    pub a_alpha: Vec<u16>,
    pub a_beta: Vec<u16>,
    pub a_l2: Vec<u16>,
    pub b_alpha: Vec<u16>,
    pub b_beta: Vec<u16>,
    pub b_l2: Vec<u16>,
}

pub struct Job {
    proposed_header: IncompleteBlockHeader,
    ancestor_header: BlockHeader,
    ancestor_chain: Vec<BlockHeader>,
    job: JobParams,
    key_a: Hash256,
    key_b: Hash256,
    hash_a: Hash256,
    hash_b: Hash256,
    seed_a: Hash256,
    seed_b: Hash256,
    jackpot_key: Hash256,
    a_values: Vec<i8>,
    b_values: Vec<i8>,
    a_scales: Vec<u16>,
    b_scales: Vec<u16>,
    a_value_plane: Plane,
    b_value_plane: Plane,
    a_scale_plane: Plane,
    b_scale_plane: Plane,
    gpu: GpuCache,
    oracle: Option<OracleCache>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuJobDesc {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub rank: u32,
    pub tile_rows: u32,
    pub tile_cols: u32,
    pub row_period: u32,
    pub col_period: u32,
    pub a_values: *const i8,
    pub a_scales: *const u16,
    pub bt_values: *const i8,
    pub bt_scales: *const u16,
    pub a_noised: *const u8,
    pub bt_noised: *const u8,
    pub a_noise_e: *const u8,
    pub a_noise_f: *const u8,
    pub bt_noise_e: *const u8,
    pub bt_noise_f: *const u8,
    pub a_alpha: *const u16,
    pub a_beta: *const u16,
    pub a_l2: *const u16,
    pub bt_alpha: *const u16,
    pub bt_beta: *const u16,
    pub bt_l2: *const u16,
    pub key_a: Hash256,
    pub key_b: Hash256,
    pub hash_a: Hash256,
    pub hash_b: Hash256,
    pub noise_seed_a: Hash256,
    pub noise_seed_b: Hash256,
    pub jackpot_key: Hash256,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TileResult {
    pub t_rows: u32,
    pub t_cols: u32,
    pub message: [u8; 64],
    pub hash: Hash256,
    pub policy_pass: u32,
    pub is_share: u32,
    pub is_block: u32,
}

impl Default for TileResult {
    fn default() -> Self {
        Self {
            t_rows: 0,
            t_cols: 0,
            message: [0; 64],
            hash: [0; 32],
            policy_pass: 0,
            is_share: 0,
            is_block: 0,
        }
    }
}

fn code(f: impl FnOnce() -> Result<(), Pmk4Error>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => e as i32,
        Err(_) => Pmk4Error::Panic as i32,
    }
}

fn dense_pattern() -> Result<AxisPattern> {
    AxisPattern::new(&[(4, Fold), (4, Blake)])
}

fn common(k: u32) -> CommonParams {
    CommonParams {
        k,
        r: RANK,
        quant: Quant::Fp8E4M3Prequant,
        device: Device::B200,
    }
}

fn check_dims(m: u32, n: u32, k: u32) -> Result<(), Pmk4Error> {
    if m == 0
        || n == 0
        || m > MAX_PRODUCTION_DIM
        || n > MAX_PRODUCTION_DIM
        || !m.is_multiple_of(TILE_ROWS)
        || !n.is_multiple_of(TILE_COLS)
        || !SUPPORTED_K.contains(&k)
    {
        return Err(Pmk4Error::BadShape);
    }
    let estimate = estimated_job_bytes(m as usize, n as usize, k as usize)?;
    if estimate > MAX_ABI_BYTES || estimate > physical_memory_bytes().saturating_div(4) {
        return Err(Pmk4Error::ResourceLimit);
    }
    Ok(())
}

fn checked_add(a: usize, b: usize) -> Result<usize, Pmk4Error> {
    a.checked_add(b).ok_or(Pmk4Error::ResourceLimit)
}

fn checked_mul(a: usize, b: usize) -> Result<usize, Pmk4Error> {
    a.checked_mul(b).ok_or(Pmk4Error::ResourceLimit)
}

fn estimated_job_bytes(m: usize, n: usize, k: usize) -> Result<usize, Pmk4Error> {
    let rows = checked_add(m, n)?;
    let operand_elems = checked_mul(rows, k)?;
    let scale_elems = checked_mul(rows, k / BLOCK_SIZE)?;
    let noise_e = checked_mul(rows, RANK as usize)?;
    let noise_f = checked_mul(checked_mul(2, k)?, RANK as usize)?;

    let mut total = 0usize;
    // Clean i8 operands plus the byte copies committed into Merkle trees.
    total = checked_add(total, checked_mul(operand_elems, 2)?)?;
    // BF16 scale arrays plus the byte copies committed into Merkle trees.
    total = checked_add(total, checked_mul(scale_elems, 4)?)?;
    // Full deterministic GPU noise factors.
    total = checked_add(total, noise_e)?;
    total = checked_add(total, noise_f)?;
    // Per-row alpha/beta/l2 for both sides, BF16.
    total = checked_add(total, checked_mul(rows, 6)?)?;
    // Optional oracle E4M3 noised buffers prepared explicitly for diagnostics/G3.
    total = checked_add(total, operand_elems)?;
    // Merkle tree nodes, Vec overheads, and transient chunk buffers.
    checked_mul(total, 2)
}

#[cfg(target_os = "macos")]
fn physical_memory_bytes() -> usize {
    use std::ffi::{c_char, c_int, c_void};
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    let mut mem = 0u64;
    let mut len = std::mem::size_of::<u64>();
    let name = b"hw.memsize\0";
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr().cast(),
            (&mut mem as *mut u64).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 { mem as usize } else { 0 }
}

#[cfg(target_os = "linux")]
fn physical_memory_bytes() -> usize {
    let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") else {
        return 0;
    };
    meminfo
        .lines()
        .find_map(|line| {
            let rest = line.strip_prefix("MemTotal:")?;
            let kb = rest.split_whitespace().next()?.parse::<usize>().ok()?;
            Some(kb.saturating_mul(1024))
        })
        .unwrap_or(0)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn physical_memory_bytes() -> usize {
    0
}

fn bytes_from_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn value_bytes(v: &[i8]) -> Vec<u8> {
    v.iter().map(|&x| x as u8).collect()
}

fn plane(bytes: Vec<u8>, hash_id: HashId, key: Hash256) -> Result<Plane> {
    let padded = hash_id.pad(&bytes);
    let tree = MerkleTree::with_chunk_len(&padded, key, hash_id.chunk_len())?;
    let root = tree.root();
    Ok(Plane { bytes, tree, root })
}

fn build_public(
    job: JobParams,
    hash_a: Hash256,
    hash_b: Hash256,
    t_rows: u32,
    t_cols: u32,
) -> Result<PublicParams> {
    PublicParams::try_new(
        job,
        JackpotStatement {
            tile_bases: Sides {
                a: t_rows,
                b: t_cols,
            },
            hash_jackpot: [0; 32],
            hash_a,
            hash_b,
        },
        None,
    )
}

fn derive_seeds(public: &PublicParams, proposed_header: &IncompleteBlockHeader) -> Sides<Hash256> {
    public.noise_seeds(proposed_header)
}

fn make_grid_values(
    rows: usize,
    k: usize,
    seed: Option<[u8; 32]>,
    tag: u64,
) -> Result<Vec<i8>, Pmk4Error> {
    let mut seed_bytes = [0u8; 32];
    match seed {
        Some(s) => seed_bytes = s,
        None => getrandom::fill(&mut seed_bytes).map_err(|_| Pmk4Error::Entropy)?,
    }
    for (i, b) in tag.to_le_bytes().iter().enumerate() {
        seed_bytes[i] ^= *b;
    }
    let mut rng = ChaCha20Rng::from_seed(seed_bytes);
    Ok((0..rows * k)
        .map(|_| if rng.random::<bool>() { 64 } else { -64 })
        .collect())
}

pub fn create_grid_b200(
    proposed_header: &[u8; HEADER_LEN],
    ancestor_header: &[u8; BLOCK_HEADER_LEN],
    ancestor_chain: &[u8],
    m: u32,
    n: u32,
    k: u32,
) -> Result<Job, Pmk4Error> {
    create_grid_b200_seeded(
        proposed_header,
        ancestor_header,
        ancestor_chain,
        m,
        n,
        k,
        None,
    )
}

pub fn create_grid_b200_seeded(
    proposed_header: &[u8; HEADER_LEN],
    ancestor_header: &[u8; BLOCK_HEADER_LEN],
    ancestor_chain: &[u8],
    m: u32,
    n: u32,
    k: u32,
    seed: Option<[u8; 32]>,
) -> Result<Job, Pmk4Error> {
    check_dims(m, n, k)?;
    if !ancestor_chain.len().is_multiple_of(BLOCK_HEADER_LEN) {
        return Err(Pmk4Error::BadHeader);
    }
    let proposed =
        IncompleteBlockHeader::from_bytes(proposed_header).map_err(|_| Pmk4Error::BadHeader)?;
    if proposed.to_bytes() != *proposed_header {
        return Err(Pmk4Error::BadHeader);
    }
    let ancestor = BlockHeader::from_bytes(ancestor_header).map_err(|_| Pmk4Error::BadHeader)?;
    if ancestor.to_bytes() != *ancestor_header {
        return Err(Pmk4Error::BadHeader);
    }
    let chain = BlockHeader::chain_from_bytes(ancestor_chain).map_err(|_| Pmk4Error::BadHeader)?;
    let pattern = dense_pattern().map_err(|_| Pmk4Error::BadConfig)?;
    let job = JobParams {
        ancestor_header: ancestor,
        common: common(k),
        operands: Sides {
            a: OperandParams {
                num_rows: m,
                hash_id: HashId::Blake3Chunk1024,
                pattern: pattern.clone(),
            },
            b: OperandParams {
                num_rows: n,
                hash_id: HashId::Blake3Chunk1024,
                pattern,
            },
        },
        moe: None,
    };
    job.check_ancestry(&proposed, &chain)
        .map_err(|_| Pmk4Error::BadHeader)?;

    let k_usize = k as usize;
    let a_values = make_grid_values(m as usize, k_usize, seed, 0xA4A4)?;
    let b_values = make_grid_values(n as usize, k_usize, seed, 0xB4B4)?;
    let one = api::fp8::dtype::f32_to_bf16(1.0).map_err(|_| Pmk4Error::BadConfig)?;
    let scale_cols = k_usize / BLOCK_SIZE;
    let a_scales = vec![one; m as usize * scale_cols];
    let b_scales = vec![one; n as usize * scale_cols];

    let ka = key_a(&proposed);
    let kb = key_b(&ancestor);
    let a_value_plane =
        plane(value_bytes(&a_values), HashId::Blake3Chunk1024, ka).map_err(|_| Pmk4Error::Proof)?;
    let b_value_plane =
        plane(value_bytes(&b_values), HashId::Blake3Chunk1024, kb).map_err(|_| Pmk4Error::Proof)?;
    let a_scale_plane = plane(bytes_from_u16(&a_scales), HashId::Blake3Chunk1024, ka)
        .map_err(|_| Pmk4Error::Proof)?;
    let b_scale_plane = plane(bytes_from_u16(&b_scales), HashId::Blake3Chunk1024, kb)
        .map_err(|_| Pmk4Error::Proof)?;
    let hash_a = operand_digest_fp10(&a_value_plane.root, &a_scale_plane.root, &ka);
    let hash_b = operand_digest_fp10(&b_value_plane.root, &b_scale_plane.root, &kb);
    let public =
        build_public(job.clone(), hash_a, hash_b, 0, 0).map_err(|_| Pmk4Error::BadConfig)?;
    let seeds = derive_seeds(&public, &proposed);
    let a_all_rows: Vec<usize> = (0..m as usize).collect();
    let b_all_rows: Vec<usize> = (0..n as usize).collect();
    let a_all_u32: Vec<u32> = a_all_rows.iter().map(|&x| x as u32).collect();
    let b_all_u32: Vec<u32> = b_all_rows.iter().map(|&x| x as u32).collect();
    let noise = sample_noise(k_usize, RANK, seeds, &a_all_u32, &b_all_u32);
    let (a_alpha, a_beta, a_l2) =
        row_scales(&a_values, &a_scales, m as usize, k_usize).map_err(|_| Pmk4Error::Policy)?;
    let (b_alpha, b_beta, b_l2) =
        row_scales(&b_values, &b_scales, n as usize, k_usize).map_err(|_| Pmk4Error::Policy)?;
    let gpu = GpuCache {
        a_noise: noise.a,
        b_noise: noise.b,
        a_alpha,
        a_beta,
        a_l2,
        b_alpha,
        b_beta,
        b_l2,
    };
    Ok(Job {
        proposed_header: proposed,
        ancestor_header: ancestor,
        ancestor_chain: chain,
        job,
        key_a: ka,
        key_b: kb,
        hash_a,
        hash_b,
        seed_a: seeds.a,
        seed_b: seeds.b,
        jackpot_key: jackpot_key(&seeds.a),
        a_values,
        b_values,
        a_scales,
        b_scales,
        a_value_plane,
        b_value_plane,
        a_scale_plane,
        b_scale_plane,
        gpu,
        oracle: None,
    })
}

fn indices(pattern: &AxisPattern, base: u32, limit: u32) -> Result<Vec<usize>, Pmk4Error> {
    if !pattern.offset_is_valid(base)
        || base
            .checked_add(pattern.tile_max())
            .is_none_or(|x| x >= limit)
    {
        return Err(Pmk4Error::IllegalOffset);
    }
    Ok(pattern
        .tile_offsets()
        .into_iter()
        .map(|x| (x + base) as usize)
        .collect())
}

fn concat_built_rows(parts: Vec<BuiltRows>) -> BuiltRows {
    let total_codes = parts.iter().map(|p| p.noised_part.len()).sum();
    let total_rows = parts.iter().map(|p| p.alpha.len()).sum();
    let mut out = BuiltRows {
        noised_part: Vec::with_capacity(total_codes),
        alpha: Vec::with_capacity(total_rows),
        beta: Vec::with_capacity(total_rows),
        l2: Vec::with_capacity(total_rows),
    };
    for mut part in parts {
        out.noised_part.append(&mut part.noised_part);
        out.alpha.append(&mut part.alpha);
        out.beta.append(&mut part.beta);
        out.l2.append(&mut part.l2);
    }
    out
}

fn strip_chunked(
    values: &[i8],
    scales: &[u16],
    rows: usize,
    k: usize,
    rank: usize,
    noise: &OperandNoise,
) -> Result<BuiltRows> {
    const ROW_CHUNK: usize = 16;
    let scale_cols = k / BLOCK_SIZE;
    let starts: Vec<usize> = (0..rows).step_by(ROW_CHUNK).collect();
    let parts: Result<Vec<BuiltRows>> = starts
        .into_par_iter()
        .map(|r0| {
            let r1 = (r0 + ROW_CHUNK).min(rows);
            let rows_len = r1 - r0;
            let row_noise = OperandNoise {
                e: noise.e[r0 * rank..r1 * rank].to_vec(),
                f: noise.f.clone(),
            };
            let row_values = &values[r0 * k..r1 * k];
            let row_scales = &scales[r0 * scale_cols..r1 * scale_cols];
            Ok(strip(row_values, row_scales, rows_len, k, &row_noise)?.built)
        })
        .collect();
    Ok(concat_built_rows(parts?))
}

fn row_select_i8(src: &[i8], rows: &[usize], k: usize) -> Vec<i8> {
    rows.iter()
        .flat_map(|&r| src[r * k..(r + 1) * k].iter().copied())
        .collect()
}

fn row_select_u16(src: &[u16], rows: &[usize], cols: usize) -> Vec<u16> {
    rows.iter()
        .flat_map(|&r| src[r * cols..(r + 1) * cols].iter().copied())
        .collect()
}

fn strip(
    values: &[i8],
    scales: &[u16],
    rows: usize,
    k: usize,
    noise: &OperandNoise,
) -> Result<OperandStrip> {
    let opened = open_prequant(values, scales, rows, k, BLOCK_SIZE)?;
    let norms = exact_norms(values, scales, rows, k, BLOCK_SIZE)?;
    let built = Fp8E4M3Quant::new(Device::B200).noisy_quantize(&opened, noise, &norms)?;
    Ok(OperandStrip {
        clean: opened,
        built,
    })
}

fn row_scales(
    values: &[i8],
    scales: &[u16],
    rows: usize,
    k: usize,
) -> Result<(Vec<u16>, Vec<u16>, Vec<u16>)> {
    let norms = exact_norms(values, scales, rows, k, BLOCK_SIZE)?;
    let quant = Fp8E4M3Quant::new(Device::B200);
    let floor = f32_to_bf16(NORM_FLOOR)?;
    let mut alpha = Vec::with_capacity(rows);
    let mut beta = Vec::with_capacity(rows);
    let mut l2s = Vec::with_capacity(rows);
    for (l2, linf) in norms {
        let l2 = bf16_max(l2, floor);
        let linf = bf16_max(linf, floor);
        let (a, b) = quant.derive_row_scales(l2, linf, RANK as usize)?;
        alpha.push(a);
        beta.push(b);
        l2s.push(l2);
    }
    Ok((alpha, beta, l2s))
}

impl Job {
    pub fn m(&self) -> u32 {
        self.job.operands.a.num_rows
    }
    pub fn n(&self) -> u32 {
        self.job.operands.b.num_rows
    }
    pub fn k(&self) -> u32 {
        self.job.common.k
    }

    pub fn public_for_tile(&self, t_rows: u32, t_cols: u32) -> Result<PublicParams> {
        build_public(self.job.clone(), self.hash_a, self.hash_b, t_rows, t_cols)
    }

    fn row_indices(&self, t_rows: u32, t_cols: u32) -> Result<(Vec<usize>, Vec<usize>), Pmk4Error> {
        Ok((
            indices(&self.job.operands.a.pattern, t_rows, self.m())?,
            indices(&self.job.operands.b.pattern, t_cols, self.n())?,
        ))
    }

    fn noise_for(&self, rows: &[usize], cols: &[usize]) -> api::fp8::noise::Noise {
        let rows: Vec<u32> = rows.iter().map(|&x| x as u32).collect();
        let cols: Vec<u32> = cols.iter().map(|&x| x as u32).collect();
        sample_noise(
            self.k() as usize,
            RANK,
            Sides {
                a: self.seed_a,
                b: self.seed_b,
            },
            &rows,
            &cols,
        )
    }

    pub fn tile_oracle(
        &self,
        t_rows: u32,
        t_cols: u32,
        share_bound: Hash256,
        block_bound: Hash256,
    ) -> Result<TileResult, Pmk4Error> {
        let (rows, cols) = self.row_indices(t_rows, t_cols)?;
        let k = self.k() as usize;
        let scale_cols = k / BLOCK_SIZE;
        let noise = self.noise_for(&rows, &cols);
        let a_vals = row_select_i8(&self.a_values, &rows, k);
        let b_vals = row_select_i8(&self.b_values, &cols, k);
        let a_scales = row_select_u16(&self.a_scales, &rows, scale_cols);
        let b_scales = row_select_u16(&self.b_scales, &cols, scale_cols);
        let a =
            strip(&a_vals, &a_scales, rows.len(), k, &noise.a).map_err(|_| Pmk4Error::Policy)?;
        let b =
            strip(&b_vals, &b_scales, cols.len(), k, &noise.b).map_err(|_| Pmk4Error::Policy)?;
        let Some(message) = JackpotPolicy::for_device(Device::B200)
            .evaluate(
                &a,
                &b,
                k,
                &self.job.operands.a.pattern,
                &self.job.operands.b.pattern,
            )
            .map_err(|_| Pmk4Error::Policy)?
        else {
            return Err(Pmk4Error::Policy);
        };
        Ok(self.classify_replayed_message(t_rows, t_cols, message, share_bound, block_bound))
    }

    fn classify_replayed_message(
        &self,
        t_rows: u32,
        t_cols: u32,
        message: JackpotMessage,
        share_bound: Hash256,
        block_bound: Hash256,
    ) -> TileResult {
        let ticket = compute_jackpot_ticket(&self.seed_a, &message);
        TileResult {
            t_rows,
            t_cols,
            message,
            hash: ticket.jackpot,
            policy_pass: 1,
            is_share: u32::from(leq256(&ticket.jackpot, &share_bound)),
            is_block: u32::from(leq256(&ticket.jackpot, &block_bound)),
        }
    }

    pub fn classify_message(
        &self,
        t_rows: u32,
        t_cols: u32,
        message: JackpotMessage,
        share_bound: Hash256,
        block_bound: Hash256,
    ) -> Result<TileResult, Pmk4Error> {
        let replayed = self.tile_oracle(t_rows, t_cols, [0xff; 32], [0xff; 32])?;
        if replayed.message != message {
            return Err(Pmk4Error::Policy);
        }
        Ok(self.classify_replayed_message(t_rows, t_cols, message, share_bound, block_bound))
    }

    pub fn prepare_oracle_noised(&mut self) -> Result<(), Pmk4Error> {
        if self.oracle.is_some() {
            return Ok(());
        }
        let k = self.k() as usize;
        let a_built = strip_chunked(
            &self.a_values,
            &self.a_scales,
            self.m() as usize,
            k,
            RANK as usize,
            &self.gpu.a_noise,
        )
        .map_err(|_| Pmk4Error::Policy)?;
        let b_built = strip_chunked(
            &self.b_values,
            &self.b_scales,
            self.n() as usize,
            k,
            RANK as usize,
            &self.gpu.b_noise,
        )
        .map_err(|_| Pmk4Error::Policy)?;
        self.oracle = Some(OracleCache { a_built, b_built });
        Ok(())
    }

    pub fn noised_codes(&mut self) -> Result<(&[u8], &[u8]), Pmk4Error> {
        self.prepare_oracle_noised()?;
        let oracle = self.oracle.as_ref().expect("prepared above");
        Ok((&oracle.a_built.noised_part, &oracle.b_built.noised_part))
    }

    fn oracle_cache(&mut self) -> Result<&OracleCache, Pmk4Error> {
        self.prepare_oracle_noised()?;
        Ok(self.oracle.as_ref().expect("prepared above"))
    }

    pub fn fixture_payload(&mut self) -> Result<FixturePayload, Pmk4Error> {
        let o = self.oracle_cache()?;
        Ok(FixturePayload {
            a_noised: o.a_built.noised_part.clone(),
            b_noised: o.b_built.noised_part.clone(),
            a_noise_e: self.gpu.a_noise.e.clone(),
            a_noise_f: self.gpu.a_noise.f.clone(),
            b_noise_e: self.gpu.b_noise.e.clone(),
            b_noise_f: self.gpu.b_noise.f.clone(),
            a_alpha: self.gpu.a_alpha.clone(),
            a_beta: self.gpu.a_beta.clone(),
            a_l2: self.gpu.a_l2.clone(),
            b_alpha: self.gpu.b_alpha.clone(),
            b_beta: self.gpu.b_beta.clone(),
            b_l2: self.gpu.b_l2.clone(),
        })
    }

    pub fn gpu_desc(&self) -> GpuJobDesc {
        GpuJobDesc {
            m: self.m(),
            n: self.n(),
            k: self.k(),
            rank: RANK as u32,
            tile_rows: TILE_ROWS,
            tile_cols: TILE_COLS,
            row_period: self.job.operands.a.pattern.total(),
            col_period: self.job.operands.b.pattern.total(),
            a_values: self.a_values.as_ptr(),
            a_scales: self.a_scales.as_ptr(),
            bt_values: self.b_values.as_ptr(),
            bt_scales: self.b_scales.as_ptr(),
            a_noised: self
                .oracle
                .as_ref()
                .map_or(ptr::null(), |o| o.a_built.noised_part.as_ptr()),
            bt_noised: self
                .oracle
                .as_ref()
                .map_or(ptr::null(), |o| o.b_built.noised_part.as_ptr()),
            a_noise_e: self.gpu.a_noise.e.as_ptr(),
            a_noise_f: self.gpu.a_noise.f.as_ptr(),
            bt_noise_e: self.gpu.b_noise.e.as_ptr(),
            bt_noise_f: self.gpu.b_noise.f.as_ptr(),
            a_alpha: self.gpu.a_alpha.as_ptr(),
            a_beta: self.gpu.a_beta.as_ptr(),
            a_l2: self.gpu.a_l2.as_ptr(),
            bt_alpha: self.gpu.b_alpha.as_ptr(),
            bt_beta: self.gpu.b_beta.as_ptr(),
            bt_l2: self.gpu.b_l2.as_ptr(),
            key_a: self.key_a,
            key_b: self.key_b,
            hash_a: self.hash_a,
            hash_b: self.hash_b,
            noise_seed_a: self.seed_a,
            noise_seed_b: self.seed_b,
            jackpot_key: self.jackpot_key,
        }
    }

    fn matrix_proof(
        &self,
        plane: &Plane,
        row_indices: &[usize],
        row_bytes: usize,
        rows: usize,
        hash_id: HashId,
    ) -> Result<MatrixMerkleProof, Pmk4Error> {
        let leaves = MerkleTree::compute_leaf_indices_from_rows(
            row_indices,
            (rows, row_bytes),
            hash_id.chunk_len(),
        )
        .map_err(|_| Pmk4Error::Proof)?;
        Ok(MatrixMerkleProof {
            proof: plane.tree.get_multileaf_proof(&leaves),
            row_indices: row_indices.to_vec(),
        })
    }

    pub fn build_plain_proof(&self, t_rows: u32, t_cols: u32) -> Result<PlainProofV4, Pmk4Error> {
        let (rows, cols) = self.row_indices(t_rows, t_cols)?;
        let k = self.k() as usize;
        let scale_bytes = 2 * (k / BLOCK_SIZE);
        Ok(PlainProofV4 {
            job: self.job.clone(),
            ancestor_chain: self.ancestor_chain.clone(),
            values: Sides {
                a: self.matrix_proof(
                    &self.a_value_plane,
                    &rows,
                    k,
                    self.m() as usize,
                    HashId::Blake3Chunk1024,
                )?,
                b: self.matrix_proof(
                    &self.b_value_plane,
                    &cols,
                    k,
                    self.n() as usize,
                    HashId::Blake3Chunk1024,
                )?,
            },
            scales: Sides {
                a: self.matrix_proof(
                    &self.a_scale_plane,
                    &rows,
                    scale_bytes,
                    self.m() as usize,
                    HashId::Blake3Chunk1024,
                )?,
                b: self.matrix_proof(
                    &self.b_scale_plane,
                    &cols,
                    scale_bytes,
                    self.n() as usize,
                    HashId::Blake3Chunk1024,
                )?,
            },
            moe_witness: None,
        })
    }

    pub fn bound_for_nbits(&self, nbits: u32) -> Hash256 {
        let target = nbits_to_difficulty(nbits);
        let adjustment = TILE_ROWS
            .checked_mul(TILE_COLS)
            .and_then(|x| x.checked_mul(self.k()))
            .unwrap_or(u32::MAX);
        let bound = if target > U256::MAX / adjustment {
            U256::MAX
        } else {
            target * adjustment
        };
        let mut out = [0; 32];
        bound.to_little_endian(&mut out);
        out
    }
}

pub fn b200_matmul_codes(a: &[u8], b: &[u8], m: usize, n: usize, k: usize) -> Result<Vec<f32>> {
    B200 {}.matmul_fp8(a, b, None, m, n, k)
}

pub fn diagnostic_headers(seed: [u8; 32]) -> (IncompleteBlockHeader, BlockHeader) {
    let mut rng = ChaCha20Rng::from_seed(seed);
    fn rand32(rng: &mut ChaCha20Rng) -> Hash256 {
        let mut out = [0u8; 32];
        rng.fill(&mut out);
        out
    }
    let ancestor = BlockHeader {
        incomplete: IncompleteBlockHeader {
            version: 4,
            prev_block: rand32(&mut rng),
            merkle_root: rand32(&mut rng),
            timestamp: rng.random(),
            nbits: 0x207f_ffff,
        },
        proof_commitment: rand32(&mut rng),
    };
    let proposed = IncompleteBlockHeader {
        version: 4,
        prev_block: ancestor.block_hash(),
        merkle_root: rand32(&mut rng),
        timestamp: ancestor.incomplete.timestamp.wrapping_add(1),
        nbits: 0x207f_ffff,
    };
    (proposed, ancestor)
}

fn leq256(a: &Hash256, b: &Hash256) -> bool {
    U256::from_little_endian(a) <= U256::from_little_endian(b)
}

fn len_to_usize(len: u64) -> Result<usize, Pmk4Error> {
    let len = usize::try_from(len).map_err(|_| Pmk4Error::ResourceLimit)?;
    if len > isize::MAX as usize {
        return Err(Pmk4Error::ResourceLimit);
    }
    Ok(len)
}

unsafe fn array_from_raw<'a, const N: usize>(ptr: *const u8) -> Result<&'a [u8; N], Pmk4Error> {
    if ptr.is_null() {
        return Err(Pmk4Error::NullPointer);
    }
    Ok(&*(ptr as *const [u8; N]))
}

unsafe fn bytes_from_raw<'a>(ptr: *const u8, len: u64) -> Result<&'a [u8], Pmk4Error> {
    let len = len_to_usize(len)?;
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(Pmk4Error::NullPointer);
    }
    Ok(std::slice::from_raw_parts(ptr, len))
}

unsafe fn write_bytes(
    bytes: &[u8],
    out: *mut u8,
    out_cap: u64,
    out_len: *mut u64,
) -> Result<(), Pmk4Error> {
    if out_len.is_null() {
        return Err(Pmk4Error::NullPointer);
    }
    *out_len = bytes.len() as u64;
    if out.is_null() {
        return if out_cap == 0 {
            Ok(())
        } else {
            Err(Pmk4Error::NullPointer)
        };
    }
    let cap = len_to_usize(out_cap)?;
    if cap < bytes.len() {
        return Err(Pmk4Error::BufferTooSmall);
    }
    ptr::copy_nonoverlapping(bytes.as_ptr(), out, bytes.len());
    Ok(())
}

unsafe fn job_ref<'a>(job: *const Pmk4Job) -> Result<&'a Job, Pmk4Error> {
    if job.is_null() {
        return Err(Pmk4Error::NullPointer);
    }
    Ok(&*(job as *const Job))
}

unsafe fn job_mut<'a>(job: *mut Pmk4Job) -> Result<&'a mut Job, Pmk4Error> {
    if job.is_null() {
        return Err(Pmk4Error::NullPointer);
    }
    Ok(&mut *(job as *mut Job))
}

pub enum Pmk4Job {}

#[unsafe(no_mangle)]
pub extern "C" fn pmkcore_v4_init(num_threads: u32) -> i32 {
    code(|| {
        if num_threads > MAX_ABI_THREADS {
            return Err(Pmk4Error::ResourceLimit);
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads as usize)
            .build_global()
            .map_err(|_| Pmk4Error::ThreadPool)?;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn pmkcore_v4_strerror(code_: i32) -> *const std::ffi::c_char {
    catch_unwind(AssertUnwindSafe(|| {
        match Pmk4Error::ALL.into_iter().find(|e| *e as i32 == code_) {
            Some(e) => e.message().as_ptr(),
            None if code_ == 0 => c"ok".as_ptr(),
            None => c"unknown error".as_ptr(),
        }
    }))
    .unwrap_or_else(|_| Pmk4Error::Panic.message().as_ptr())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pmkcore_v4_job_create_grid_b200(
    proposed_header: *const u8,
    ancestor_header: *const u8,
    ancestor_chain: *const u8,
    ancestor_chain_len: u64,
    m: u32,
    n: u32,
    k: u32,
    out_job: *mut *mut Pmk4Job,
) -> i32 {
    code(|| {
        if out_job.is_null() {
            return Err(Pmk4Error::NullPointer);
        }
        *out_job = ptr::null_mut();
        let proposed_header = array_from_raw::<HEADER_LEN>(proposed_header)?;
        let ancestor_header = array_from_raw::<BLOCK_HEADER_LEN>(ancestor_header)?;
        let ancestor_chain = bytes_from_raw(ancestor_chain, ancestor_chain_len)?;
        let job = create_grid_b200(proposed_header, ancestor_header, ancestor_chain, m, n, k)?;
        *out_job = Box::into_raw(Box::new(job)) as *mut Pmk4Job;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pmkcore_v4_job_free(job: *mut Pmk4Job) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if !job.is_null() {
            drop(Box::from_raw(job as *mut Job));
        }
    }));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pmkcore_v4_gpu_descriptor(
    job: *const Pmk4Job,
    out: *mut GpuJobDesc,
) -> i32 {
    code(|| {
        if out.is_null() {
            return Err(Pmk4Error::NullPointer);
        }
        *out = job_ref(job)?.gpu_desc();
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pmkcore_v4_prepare_oracle_noised(job: *mut Pmk4Job) -> i32 {
    code(|| {
        job_mut(job)?.prepare_oracle_noised()?;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pmkcore_v4_tile_cpu_oracle(
    job: *const Pmk4Job,
    t_rows: u32,
    t_cols: u32,
    share_bound: *const u8,
    block_bound: *const u8,
    out: *mut TileResult,
) -> i32 {
    code(|| {
        if out.is_null() {
            return Err(Pmk4Error::NullPointer);
        }
        let share_bound = *array_from_raw::<32>(share_bound)?;
        let block_bound = *array_from_raw::<32>(block_bound)?;
        *out = job_ref(job)?.tile_oracle(t_rows, t_cols, share_bound, block_bound)?;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pmkcore_v4_classify_tile_message(
    job: *const Pmk4Job,
    t_rows: u32,
    t_cols: u32,
    message: *const u8,
    share_bound: *const u8,
    block_bound: *const u8,
    out: *mut TileResult,
) -> i32 {
    code(|| {
        if out.is_null() {
            return Err(Pmk4Error::NullPointer);
        }
        let message = *array_from_raw::<64>(message)?;
        let share_bound = *array_from_raw::<32>(share_bound)?;
        let block_bound = *array_from_raw::<32>(block_bound)?;
        *out = job_ref(job)?.classify_message(t_rows, t_cols, message, share_bound, block_bound)?;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pmkcore_v4_scan_cpu_oracle(
    job: *const Pmk4Job,
    share_bound: *const u8,
    block_bound: *const u8,
    out: *mut TileResult,
    out_cap: u64,
    out_len: *mut u64,
) -> i32 {
    code(|| {
        if out_len.is_null() {
            return Err(Pmk4Error::NullPointer);
        }
        let job = job_ref(job)?;
        let rows: Vec<u32> = (0..job.m())
            .filter(|&r| job.job.operands.a.pattern.offset_is_valid(r))
            .collect();
        let cols: Vec<u32> = (0..job.n())
            .filter(|&c| job.job.operands.b.pattern.offset_is_valid(c))
            .collect();
        let count = rows
            .len()
            .checked_mul(cols.len())
            .ok_or(Pmk4Error::ResourceLimit)?;
        *out_len = count as u64;
        if out.is_null() {
            return if out_cap == 0 {
                Ok(())
            } else {
                Err(Pmk4Error::NullPointer)
            };
        }
        if len_to_usize(out_cap)? < count {
            return Err(Pmk4Error::BufferTooSmall);
        }
        let share_bound = *array_from_raw::<32>(share_bound)?;
        let block_bound = *array_from_raw::<32>(block_bound)?;
        let results: Result<Vec<_>, _> = rows
            .par_iter()
            .flat_map_iter(|&r| {
                cols.iter()
                    .map(move |&c| job.tile_oracle(r, c, share_bound, block_bound))
            })
            .collect();
        let results = results?;
        ptr::copy_nonoverlapping(results.as_ptr(), out, results.len());
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pmkcore_v4_bound_for_nbits(
    job: *const Pmk4Job,
    nbits: u32,
    out_bound: *mut u8,
) -> i32 {
    code(|| {
        if out_bound.is_null() {
            return Err(Pmk4Error::NullPointer);
        }
        let bound = job_ref(job)?.bound_for_nbits(nbits);
        ptr::copy_nonoverlapping(bound.as_ptr(), out_bound, 32);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pmkcore_v4_build_plain_proof(
    job: *const Pmk4Job,
    t_rows: u32,
    t_cols: u32,
    out: *mut u8,
    out_cap: u64,
    out_len: *mut u64,
) -> i32 {
    code(|| {
        let proof = job_ref(job)?.build_plain_proof(t_rows, t_cols)?;
        let bytes = proof.to_bytes().map_err(|_| Pmk4Error::Proof)?;
        write_bytes(&bytes, out, out_cap, out_len)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pmkcore_v4_verify_plain_proof(
    proposed_header: *const u8,
    proof: *const u8,
    proof_len: u64,
    nbits_override_le_u32: *const u8,
    accepted: *mut u8,
) -> i32 {
    code(|| {
        if accepted.is_null() {
            return Err(Pmk4Error::NullPointer);
        }
        *accepted = 0;
        let header =
            IncompleteBlockHeader::from_bytes(array_from_raw::<HEADER_LEN>(proposed_header)?)
                .map_err(|_| Pmk4Error::BadHeader)?;
        let proof = PlainProofV4::from_bytes(bytes_from_raw(proof, proof_len)?)
            .map_err(|_| Pmk4Error::Proof)?;
        let nbits = if nbits_override_le_u32.is_null() {
            None
        } else {
            Some(u32::from_le_bytes(*array_from_raw::<4>(
                nbits_override_le_u32,
            )?))
        };
        match api::verify::verify_plain_proof(&header, &proof, nbits) {
            Ok(()) => {
                *accepted = 1;
                Ok(())
            }
            Err(_) => Err(Pmk4Error::Verify),
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pmkcore_v4_mutate_plain_proof(
    proof: *const u8,
    proof_len: u64,
    mutation_id: u32,
    out: *mut u8,
    out_cap: u64,
    out_len: *mut u64,
) -> i32 {
    code(|| {
        let mut bytes = bytes_from_raw(proof, proof_len)?.to_vec();
        if bytes.is_empty() {
            return Err(Pmk4Error::Proof);
        }
        let idx = match mutation_id {
            0 => 0,
            1 => 8.min(bytes.len() - 1),
            2 => bytes.len() / 2,
            3 => bytes.len() - 1,
            _ => (mutation_id as usize) % bytes.len(),
        };
        bytes[idx] ^= 0x5a;
        write_bytes(&bytes, out, out_cap, out_len)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pmkcore_v4_build_certificate(
    proposed_header: *const u8,
    plain_proof: *const u8,
    plain_proof_len: u64,
    public_out: *mut u8,
    public_cap: u64,
    public_len: *mut u64,
    proof_out: *mut u8,
    proof_cap: u64,
    proof_len: *mut u64,
) -> i32 {
    code(|| {
        let header =
            IncompleteBlockHeader::from_bytes(array_from_raw::<HEADER_LEN>(proposed_header)?)
                .map_err(|_| Pmk4Error::BadHeader)?;
        let cert = build_certificate_from_plain_proof_bytes(
            &header,
            bytes_from_raw(plain_proof, plain_proof_len)?,
        )
        .map_err(|_| Pmk4Error::Proof)?;
        write_bytes(&cert.public_data, public_out, public_cap, public_len)?;
        write_bytes(&cert.proof_data, proof_out, proof_cap, proof_len)?;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn pmkcore_v4_export_vectors(
    job: *const Pmk4Job,
    out: *mut u8,
    out_cap: u64,
    out_len: *mut u64,
) -> i32 {
    code(|| {
        let job = job_ref(job)?;
        let mut data = Vec::new();
        data.extend_from_slice(b"PMK4VEC1");
        for v in [job.m(), job.n(), job.k(), RANK as u32, TILE_ROWS, TILE_COLS] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        for bytes in [
            &job.proposed_header.to_bytes()[..],
            &job.ancestor_header.to_bytes()[..],
            &job.key_a[..],
            &job.key_b[..],
            &job.hash_a[..],
            &job.hash_b[..],
            &job.seed_a[..],
            &job.seed_b[..],
            &job.jackpot_key[..],
            &value_bytes(&job.a_values)[..],
            &value_bytes(&job.b_values)[..],
            &bytes_from_u16(&job.a_scales)[..],
            &bytes_from_u16(&job.b_scales)[..],
        ] {
            data.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            data.extend_from_slice(bytes);
        }
        write_bytes(&data, out, out_cap, out_len)
    })
}

pub struct CertificateBytes {
    pub public_data: Vec<u8>,
    pub proof_data: Vec<u8>,
}

pub fn build_certificate_from_plain_proof_bytes(
    proposed_header: &IncompleteBlockHeader,
    plain_proof_bytes: &[u8],
) -> Result<CertificateBytes> {
    let plain = PlainProofV4::from_bytes(plain_proof_bytes)?;
    // Authenticate the witness before paying prover setup/prove cost.
    api::verify::verify_plain_proof(proposed_header, &plain, None)
        .context("plain proof rejected before fp8 certificate proving")?;
    let mut prover = Fp8Prover::setup(plain.job.common.device)?;
    let (public_data, proof_data) = prover.prove(proposed_header, &plain)?;
    Ok(CertificateBytes {
        public_data,
        proof_data,
    })
}

pub fn verify_proof_bytes(
    header: &IncompleteBlockHeader,
    bytes: &[u8],
    nbits: Option<u32>,
) -> Result<()> {
    let proof = PlainProofV4::from_bytes(bytes)?;
    api::verify::verify_plain_proof(header, &proof, nbits)
}
