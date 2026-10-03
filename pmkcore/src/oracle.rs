//! G1 CPU oracle and retained raw-matrix proofs. B operands are always n×k (Bᵀ).
use crate::{check_range, job_key, padded_len, Hash256, PmkError, CONFIG_LEN, HEADER_LEN};
use pearl_blake3::MerkleTree;
use rayon::prelude::*;
use zk_pow::api::proof::{
    IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, PublicProofParams,
    SeedDerivation,
};
use zk_pow::api::proof_utils::compute_jackpot_hash;
use zk_pow::api::sanity_checks::extract_difficulty_bound;
use zk_pow::api::seed::{bind_root_a, bind_root_b, SEED_SALT_A, SEED_SALT_B};
use zk_pow::circuit::pearl_noise::{generate_permutation_matrix, generate_uniform_random_matrix};
use zk_pow::ffi::plain_proof::{list_to_pattern, MatrixMerkleProof, PlainProof};

pub const NA_ROWS: &[u32] = &[0, 8, 64, 72];
pub const NA_COLS: &[u32] = &[0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51];
pub const SG_ROWS: &[u32] = &[0, 8, 16, 24];
pub const SG_COLS: &[u32] = &[0, 1, 8, 9, 16, 17, 24, 25];
/// A diagnostic oracle retains raw, noised, noise, and Merkle data. Bound allocation before copying.
/// Caller must additionally enforce SPEC's device-wide 25%-of-RAM limit across concurrent jobs.
pub const MAX_ORACLE_BYTES: usize = 1 << 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pattern {
    Na,
    Sg,
}

impl Pattern {
    pub fn rows(self) -> &'static [u32] {
        match self {
            Self::Na => NA_ROWS,
            Self::Sg => SG_ROWS,
        }
    }
    pub fn cols(self) -> &'static [u32] {
        match self {
            Self::Na => NA_COLS,
            Self::Sg => SG_COLS,
        }
    }
}

/// Strict production v1 builder: rank 128, 2048 ≤ k ≤ 8192, k divisible by 128,
/// and the kernel's shape rules. Both patterns use upstream PeriodicPattern::from_list.
pub fn build_config(
    pattern: Pattern,
    k: u32,
    m: u32,
    n: u32,
) -> Result<MiningConfiguration, PmkError> {
    if !(2048..=8192).contains(&k) || !k.is_multiple_of(128) {
        return Err(PmkError::Policy);
    }
    let cfg = MiningConfiguration {
        common_dim: k,
        rank: 128,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(pattern.rows())
            .map_err(|_| PmkError::BadConfig)?,
        cols_pattern: PeriodicPattern::from_list(pattern.cols())
            .map_err(|_| PmkError::BadConfig)?,
        moe: None,
    };
    let bytes = cfg.to_bytes();
    let restored = MiningConfiguration::from_bytes(&bytes).map_err(|_| PmkError::BadConfig)?;
    // Assert semantic and byte round-trips, including the exact normalized lists.
    if restored.to_bytes() != bytes
        || restored.rows_pattern.to_list() != pattern.rows()
        || restored.cols_pattern.to_list() != pattern.cols()
    {
        return Err(PmkError::BadConfig);
    }
    validate_config(&bytes, m, n)?;
    Ok(cfg)
}

/// Explicit diagnostic builder for G2/G3 fixtures. Production callers must use build_config.
pub fn build_config_diagnostic(
    pattern: Pattern,
    k: u32,
    m: u32,
    n: u32,
) -> Result<MiningConfiguration, PmkError> {
    let cfg = MiningConfiguration {
        common_dim: k,
        rank: 128,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(pattern.rows())
            .map_err(|_| PmkError::BadConfig)?,
        cols_pattern: PeriodicPattern::from_list(pattern.cols())
            .map_err(|_| PmkError::BadConfig)?,
        moe: None,
    };
    let bytes = cfg.to_bytes();
    let restored = MiningConfiguration::from_bytes(&bytes).map_err(|_| PmkError::BadConfig)?;
    if restored.to_bytes() != bytes {
        return Err(PmkError::BadConfig);
    }
    validate_config(&bytes, m, n)?;
    Ok(cfg)
}

/// G1 also accepts diagnostic k up to 65536 (G2/G3), without expanding the production builder policy.
/// Only the two specified patterns, dense int7 MMA and rank 128 are supported.
fn validate_config(
    bytes: &[u8; CONFIG_LEN],
    m: u32,
    n: u32,
) -> Result<MiningConfiguration, PmkError> {
    let cfg = MiningConfiguration::from_bytes(bytes).map_err(|_| PmkError::BadConfig)?;
    if cfg.to_bytes() != *bytes {
        return Err(PmkError::BadConfig);
    }
    if cfg.rank != 128
        || cfg.moe.is_some()
        || !cfg.common_dim.is_multiple_of(128)
        || !(2048..=65536).contains(&cfg.common_dim)
    {
        return Err(PmkError::Policy);
    }
    // Compare canonical encodings before expanding an untrusted pattern's index list.
    let pattern = [Pattern::Na, Pattern::Sg]
        .into_iter()
        .find(|p| {
            cfg.rows_pattern == PeriodicPattern::from_list(p.rows()).unwrap()
                && cfg.cols_pattern == PeriodicPattern::from_list(p.cols()).unwrap()
        })
        .ok_or(PmkError::Policy)?;
    let row_multiple = if pattern == Pattern::Na { 128 } else { 64 };
    if m == 0
        || n == 0
        || m > 1 << 24
        || n > 1 << 24
        || !m.is_multiple_of(row_multiple)
        || !n.is_multiple_of(64)
    {
        return Err(PmkError::BadShape);
    }
    let header =
        IncompleteBlockHeader::from_bytes(&[0; HEADER_LEN]).map_err(|_| PmkError::BadHeader)?;
    let params = PublicProofParams::new_dummy(header, SeedDerivation::Salted, cfg, m, n, 0, 0);
    params.sanity_check().map_err(|_| PmkError::Policy)?;
    if params.compile().0.degree_bits() > 19 {
        return Err(PmkError::Policy);
    }
    Ok(cfg)
}

pub fn matrix_lens(bytes: &[u8; CONFIG_LEN], m: u32, n: u32) -> Result<(usize, usize), PmkError> {
    let cfg = validate_config(bytes, m, n)?;
    let k = cfg.common_dim as usize;
    let a_len = (m as usize).checked_mul(k).ok_or(PmkError::ResourceLimit)?;
    let bt_len = (n as usize).checked_mul(k).ok_or(PmkError::ResourceLimit)?;
    if a_len
        .checked_add(bt_len)
        .and_then(|x| x.checked_mul(6))
        .ok_or(PmkError::ResourceLimit)?
        > MAX_ORACLE_BYTES
    {
        return Err(PmkError::ResourceLimit);
    }
    Ok((a_len, bt_len))
}

/// Complete immutable snapshot. Sparse matrices contain [positive index, negative index] pairs;
/// E_AR is represented transposed (k×2), E_BR transposed (n×128).
#[derive(Debug)]
pub struct Intermediates {
    pub header: [u8; HEADER_LEN],
    pub config: [u8; CONFIG_LEN],
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub padded_a: Vec<u8>,
    pub padded_bt: Vec<u8>,
    pub job_key: Hash256,
    pub raw_root_a: Hash256,
    pub raw_root_b: Hash256,
    pub salt_input_a: [u8; 64],
    pub salt_input_b: [u8; 64],
    pub salted_root_a: Hash256,
    pub salted_root_b: Hash256,
    pub seed_input_a: [u8; 64],
    pub seed_input_b: [u8; 64],
    pub a_noise_seed: Hash256,
    pub b_noise_seed: Hash256,
    pub e_al: Vec<i8>,
    pub e_ar_t: Vec<[u32; 2]>,
    pub e_bl: Vec<[u32; 2]>,
    pub e_br_t: Vec<i8>,
    pub noise_a: Vec<i8>,
    pub noise_bt: Vec<i8>,
    pub noised_a: Vec<i8>,
    pub noised_bt: Vec<i8>,
}

/// Independent flags: a tile may satisfy both, either, or neither bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct TileResult {
    pub t_rows: u32,
    pub t_cols: u32,
    pub transcript: [u32; 16],
    pub hash: Hash256,
    pub is_share: u32,
    pub is_block: u32,
}

/// Owned immutable job. No borrowing of caller buffers; proofs always open the retained raw data.
pub struct OracleJob {
    data: Intermediates,
    config: MiningConfiguration,
    header: IncompleteBlockHeader,
    trees: (MerkleTree, MerkleTree),
}

fn salt_message(root: &Hash256, dim: u32) -> [u8; 64] {
    let mut out = [0; 64];
    out[..32].copy_from_slice(root);
    out[32..36].copy_from_slice(&dim.to_le_bytes());
    out
}
fn pair(a: &Hash256, b: &Hash256) -> [u8; 64] {
    let mut out = [0; 64];
    out[..32].copy_from_slice(a);
    out[32..].copy_from_slice(b);
    out
}
fn noise(dense: &[i8], sparse: &[[u32; 2]]) -> Vec<i8> {
    dense
        .par_chunks(128)
        .flat_map_iter(|row| {
            sparse
                .iter()
                .map(move |&[p, q]| row[p as usize] - row[q as usize])
        })
        .collect()
}
fn add_noise(raw: &[u8], noise: &[i8]) -> Vec<i8> {
    raw.iter()
        .zip(noise)
        .map(|(&a, &e)| ((a as i8 as i16) + e as i16) as i8)
        .collect()
}

impl OracleJob {
    /// Input matrices must contain exactly m*k and n*k signed-int8 bytes, in row-major order.
    /// Diagnostic k range is [2048,65536], unlike the stricter production build_config.
    pub fn new(
        header: &[u8; HEADER_LEN],
        config: &[u8; CONFIG_LEN],
        m: u32,
        n: u32,
        a: &[u8],
        bt: &[u8],
    ) -> Result<Self, PmkError> {
        let hdr = IncompleteBlockHeader::from_bytes(header).map_err(|_| PmkError::BadHeader)?;
        if hdr.to_bytes() != *header {
            return Err(PmkError::BadHeader);
        }
        let cfg = validate_config(config, m, n)?;
        let k = cfg.common_dim as usize;
        let a_len = (m as usize).checked_mul(k).ok_or(PmkError::ResourceLimit)?;
        let b_len = (n as usize).checked_mul(k).ok_or(PmkError::ResourceLimit)?;
        // Six payload equivalents conservatively cover retained raw, trees, noise, noised,
        // dense/sparse factors, tree nodes and construction temporaries. Export/scan are additional.
        if a_len
            .checked_add(b_len)
            .and_then(|x| x.checked_mul(6))
            .ok_or(PmkError::ResourceLimit)?
            > MAX_ORACLE_BYTES
        {
            return Err(PmkError::ResourceLimit);
        }
        if a.len() != a_len || bt.len() != b_len {
            return Err(PmkError::BadShape);
        }
        check_range(a)?;
        check_range(bt)?;
        let jk = job_key(header, config);
        let mut padded_a = a.to_vec();
        padded_a.resize(padded_len(m as usize, k), 0);
        let mut padded_bt = bt.to_vec();
        padded_bt.resize(padded_len(n as usize, k), 0);
        let ta = MerkleTree::new(&padded_a, jk);
        let tb = MerkleTree::new(&padded_bt, jk);
        let raw_a = ta.root();
        let raw_b = tb.root();
        let salted_a = bind_root_a(&raw_a, m);
        let salted_b = bind_root_b(&raw_b, n);
        let seed_input_b = pair(&jk, &salted_b);
        let b_seed = *blake3::hash(&seed_input_b).as_bytes();
        let seed_input_a = pair(&b_seed, &salted_a);
        let a_seed = *blake3::hash(&seed_input_a).as_bytes();
        let mut label_a = [0; 32];
        label_a[..8].copy_from_slice(b"A_tensor");
        let mut label_b = [0; 32];
        label_b[..8].copy_from_slice(b"B_tensor");
        let e_al: Vec<i8> = generate_uniform_random_matrix(
            &label_a,
            &a_seed,
            &(0..m as usize).collect::<Vec<_>>(),
            128,
        )
        .into_iter()
        .flatten()
        .collect();
        let e_br_t: Vec<i8> = generate_uniform_random_matrix(
            &label_b,
            &b_seed,
            &(0..n as usize).collect::<Vec<_>>(),
            128,
        )
        .into_iter()
        .flatten()
        .collect();
        let e_ar_t = generate_permutation_matrix(&label_a, &a_seed, k, 128);
        let e_bl = generate_permutation_matrix(&label_b, &b_seed, k, 128);
        let noise_a = noise(&e_al, &e_ar_t);
        let noise_bt = noise(&e_br_t, &e_bl);
        let noised_a = add_noise(a, &noise_a);
        let noised_bt = add_noise(bt, &noise_bt);
        let data = Intermediates {
            header: *header,
            config: *config,
            m,
            n,
            k: k as u32,
            padded_a,
            padded_bt,
            job_key: jk,
            raw_root_a: raw_a,
            raw_root_b: raw_b,
            salt_input_a: salt_message(&raw_a, m),
            salt_input_b: salt_message(&raw_b, n),
            salted_root_a: salted_a,
            salted_root_b: salted_b,
            seed_input_a,
            seed_input_b,
            a_noise_seed: a_seed,
            b_noise_seed: b_seed,
            e_al,
            e_ar_t,
            e_bl,
            e_br_t,
            noise_a,
            noise_bt,
            noised_a,
            noised_bt,
        };
        Ok(Self {
            data,
            config: cfg,
            header: hdr,
            trees: (ta, tb),
        })
    }

    pub fn intermediates(&self) -> &Intermediates {
        &self.data
    }
    pub fn config(&self) -> MiningConfiguration {
        self.config
    }
    pub fn header(&self) -> IncompleteBlockHeader {
        self.header
    }

    /// Exact number of partition tiles; no matrix arithmetic or allocation.
    pub fn tile_count(&self) -> usize {
        (self.data.m as usize / self.config.rows_pattern.size() as usize)
            * (self.data.n as usize / self.config.cols_pattern.size() as usize)
    }

    /// Consensus saturating bound, serialized as uint256 LE. Rank128 makes the penalty neutral.
    pub fn bound(&self, nbits: u32) -> Hash256 {
        let mut bytes = [0; 32];
        extract_difficulty_bound(nbits, &self.config).to_little_endian(&mut bytes);
        bytes
    }

    fn indices(&self, tr: u32, tc: u32) -> Result<(Vec<usize>, Vec<usize>), PmkError> {
        let (rp, cp) = (&self.config.rows_pattern, &self.config.cols_pattern);
        if !rp.offset_is_valid(tr)
            || !cp.offset_is_valid(tc)
            || tr.checked_add(rp.max()).is_none_or(|x| x >= self.data.m)
            || tc.checked_add(cp.max()).is_none_or(|x| x >= self.data.n)
        {
            return Err(PmkError::IllegalOffset);
        }
        let mut rows: Vec<usize> = rp
            .indices_with_offset(tr)
            .into_iter()
            .map(|x| x as usize)
            .collect();
        let mut cols: Vec<usize> = cp
            .indices_with_offset(tc)
            .into_iter()
            .map(|x| x as usize)
            .collect();
        rows.sort_unstable();
        cols.sort_unstable();
        Ok((rows, cols))
    }

    pub fn tile(
        &self,
        t_rows: u32,
        t_cols: u32,
        share_bound: Hash256,
        block_bound: Hash256,
    ) -> Result<TileResult, PmkError> {
        let (rows, cols) = self.indices(t_rows, t_cols)?;
        let transcript = transcript(
            &self.data.noised_a,
            &self.data.noised_bt,
            self.data.k as usize,
            &rows,
            &cols,
        );
        let hash = compute_jackpot_hash(&transcript, self.data.a_noise_seed);
        Ok(TileResult {
            t_rows,
            t_cols,
            transcript,
            hash,
            is_share: u32::from(leq256(&hash, &share_bound)),
            is_block: u32::from(leq256(&hash, &block_bound)),
        })
    }

    /// All Pearl tiles in ascending row-offset, then column-offset order. Use tile() for bounded subsets.
    pub fn scan(&self, share_bound: Hash256, block_bound: Hash256) -> Vec<TileResult> {
        let cols: Vec<u32> = (0..self.data.n)
            .filter(|&c| self.config.cols_pattern.offset_is_valid(c))
            .collect();
        let rows: Vec<u32> = (0..self.data.m)
            .filter(|&r| self.config.rows_pattern.offset_is_valid(r))
            .collect();
        rows.par_iter()
            .flat_map_iter(|&r| {
                cols.iter().map(move |&c| {
                    self.tile(r, c, share_bound, block_bound)
                        .expect("validated partition")
                })
            })
            .collect()
    }

    /// Open a tile's RAW rows, with GLOBAL pattern minima. Threshold verification is the caller's
    /// responsibility: PlainProof carries no jackpot hash or claimed bound.
    pub fn build_plain_proof(&self, t_rows: u32, t_cols: u32) -> Result<Vec<u8>, PmkError> {
        let (rows, cols) = self.indices(t_rows, t_cols)?;
        let (rp, tr) = list_to_pattern(&rows.iter().map(|&x| x as u32).collect::<Vec<_>>())
            .map_err(|_| PmkError::Proof)?;
        let (cp, tc) = list_to_pattern(&cols.iter().map(|&x| x as u32).collect::<Vec<_>>())
            .map_err(|_| PmkError::Proof)?;
        let reconstructed = MiningConfiguration {
            rows_pattern: rp,
            cols_pattern: cp,
            ..self.config
        };
        if reconstructed.to_bytes() != self.data.config || tr != t_rows || tc != t_cols {
            return Err(PmkError::Proof);
        }
        let p = PublicProofParams::new_dummy(
            self.header,
            SeedDerivation::Salted,
            reconstructed,
            self.data.m,
            self.data.n,
            tr,
            tc,
        );
        p.sanity_check().map_err(|_| PmkError::Proof)?;
        if p.compile().0.degree_bits() > 19 {
            return Err(PmkError::Policy);
        }
        let matrix_proof = |tree: &MerkleTree, indices: Vec<usize>, dim: u32| {
            let leaves = MerkleTree::compute_leaf_indices_from_rows(
                &indices,
                (dim as usize, self.data.k as usize),
            );
            MatrixMerkleProof {
                proof: tree.get_multileaf_proof(&leaves),
                row_indices: indices,
            }
        };
        let proof = PlainProof {
            m: self.data.m as usize,
            n: self.data.n as usize,
            k: self.data.k as usize,
            noise_rank: 128,
            a: matrix_proof(&self.trees.0, rows, self.data.m),
            bt: matrix_proof(&self.trees.1, cols, self.data.n),
            moe: None,
        };
        bincode::serialize(&proof).map_err(|_| PmkError::Proof)
    }

    pub fn export_vectors_len(&self) -> Result<usize, PmkError> {
        let d = &self.data;
        let mut total = 8usize.checked_add(4).ok_or(PmkError::ResourceLimit)?;
        let mut add_field = |name: &str, data_len: usize| -> Result<(), PmkError> {
            total = total
                .checked_add(2)
                .and_then(|v| v.checked_add(name.len()))
                .and_then(|v| v.checked_add(8))
                .and_then(|v| v.checked_add(data_len))
                .ok_or(PmkError::ResourceLimit)?;
            if total > MAX_ORACLE_BYTES {
                return Err(PmkError::ResourceLimit);
            }
            Ok(())
        };
        add_field("dimensions", 16)?;
        for (name, len) in [
            ("header", d.header.len()),
            ("config", d.config.len()),
            ("padded_a", d.padded_a.len()),
            ("padded_bt", d.padded_bt.len()),
            ("job_key", d.job_key.len()),
            ("raw_root_a", d.raw_root_a.len()),
            ("raw_root_b", d.raw_root_b.len()),
            ("salt_input_a", d.salt_input_a.len()),
            ("salt_input_b", d.salt_input_b.len()),
            ("salted_root_a", d.salted_root_a.len()),
            ("salted_root_b", d.salted_root_b.len()),
            ("seed_input_a", d.seed_input_a.len()),
            ("seed_input_b", d.seed_input_b.len()),
            ("a_noise_seed", d.a_noise_seed.len()),
            ("b_noise_seed", d.b_noise_seed.len()),
            ("salt_key_a", SEED_SALT_A.len()),
            ("salt_key_b", SEED_SALT_B.len()),
            ("e_al", d.e_al.len()),
            ("e_br_t", d.e_br_t.len()),
            ("noise_a", d.noise_a.len()),
            ("noise_bt", d.noise_bt.len()),
            ("noised_a", d.noised_a.len()),
            ("noised_bt", d.noised_bt.len()),
        ] {
            add_field(name, len)?;
        }
        add_field(
            "e_ar_t",
            d.e_ar_t
                .len()
                .checked_mul(2)
                .and_then(|v| v.checked_mul(std::mem::size_of::<u32>()))
                .ok_or(PmkError::ResourceLimit)?,
        )?;
        add_field(
            "e_bl",
            d.e_bl
                .len()
                .checked_mul(2)
                .and_then(|v| v.checked_mul(std::mem::size_of::<u32>()))
                .ok_or(PmkError::ResourceLimit)?,
        )?;
        Ok(total)
    }

    /// Versioned named byte sections; see README's PMKVEC01 format. No native-endian serialization.
    pub fn export_vectors(&self) -> Vec<u8> {
        let d = &self.data;
        let mut fields: Vec<(&str, Vec<u8>)> = Vec::new();
        fields.push((
            "dimensions",
            [d.m, d.n, d.k, 128]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ));
        macro_rules! bytes { ($($field:ident),* $(,)?) => { $(fields.push((stringify!($field), d.$field.to_vec()));)* }; }
        bytes!(
            header,
            config,
            padded_a,
            padded_bt,
            job_key,
            raw_root_a,
            raw_root_b,
            salt_input_a,
            salt_input_b,
            salted_root_a,
            salted_root_b,
            seed_input_a,
            seed_input_b,
            a_noise_seed,
            b_noise_seed
        );
        fields.push(("salt_key_a", SEED_SALT_A.to_vec()));
        fields.push(("salt_key_b", SEED_SALT_B.to_vec()));
        for (name, data) in [
            ("e_al", &d.e_al),
            ("e_br_t", &d.e_br_t),
            ("noise_a", &d.noise_a),
            ("noise_bt", &d.noise_bt),
            ("noised_a", &d.noised_a),
            ("noised_bt", &d.noised_bt),
        ] {
            fields.push((name, data.iter().map(|&v| v as u8).collect()));
        }
        for (name, data) in [("e_ar_t", &d.e_ar_t), ("e_bl", &d.e_bl)] {
            fields.push((
                name,
                data.iter()
                    .flatten()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            ));
        }
        let mut out = b"PMKVEC01".to_vec();
        out.extend_from_slice(&(fields.len() as u32).to_le_bytes());
        for (name, data) in fields {
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(data.len() as u64).to_le_bytes());
            out.extend_from_slice(&data);
        }
        out
    }
}

/// uint256 little-endian comparison, including equality.
pub fn leq256(hash: &Hash256, bound: &Hash256) -> bool {
    hash.iter().rev().cmp(bound.iter().rev()).is_le()
}

/// Jackpot over already-noised row-major A and Bᵀ; allows consensus diagnostic k through 65536.
/// Panics on malformed slices/indices. This low-level Rust helper does not validate a mining job.
pub fn transcript(ap: &[i8], btp: &[i8], k: usize, rows: &[usize], cols: &[usize]) -> [u32; 16] {
    assert!(k > 0 && k <= 65536 && k.is_multiple_of(128));
    let mut acc = vec![0i32; rows.len() * cols.len()];
    let mut jackpot = [0u32; 16];
    for start in (0..k).step_by(128) {
        let mut x = 0u32;
        for (u, &row) in rows.iter().enumerate() {
            let a = &ap[row * k + start..row * k + start + 128];
            for (v, &col) in cols.iter().enumerate() {
                let b = &btp[col * k + start..col * k + start + 128];
                let val = &mut acc[u * cols.len() + v];
                *val += a
                    .iter()
                    .zip(b)
                    .map(|(&a, &b)| a as i32 * b as i32)
                    .sum::<i32>();
                x ^= *val as u32;
            }
        }
        let slot = (start / 128) % 16;
        jackpot[slot] = jackpot[slot].rotate_left(13) ^ x;
    }
    jackpot
}
