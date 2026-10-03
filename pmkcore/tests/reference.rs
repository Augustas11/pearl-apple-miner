//! F2 correctness: pmkcore roots and seeds are byte-identical to the zk-pow reference.

use pmkcore::*;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;
use zk_pow::api::proof::{
    IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, PublicProofParams,
    SeedDerivation,
};
use zk_pow::ffi::mine::try_mine_one;

fn policy_config(k: u32) -> MiningConfiguration {
    MiningConfiguration {
        common_dim: k,
        rank: 128,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&[
            0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51,
        ])
        .unwrap(),
        moe: None,
    }
}

fn random_header(rng: &mut impl RngCore) -> IncompleteBlockHeader {
    let mut prev_block = [0u8; 32];
    let mut merkle_root = [0u8; 32];
    rng.fill_bytes(&mut prev_block);
    rng.fill_bytes(&mut merkle_root);
    // Hard target so try_mine_one(.., wrong_jackpot_hash = true, ..) returns on the first tile.
    IncompleteBlockHeader {
        version: rng.next_u32(),
        prev_block,
        merkle_root,
        timestamp: rng.next_u32(),
        nbits: 0x1d00ffff,
    }
}

/// Reference-side seeds from zk-pow's verifier code path (`PublicProofParams::commitment_hash`).
fn reference_seeds(
    header: IncompleteBlockHeader,
    config: MiningConfiguration,
    hash_a: [u8; 32],
    hash_b: [u8; 32],
    m: u32,
    n: u32,
) -> ([u8; 32], [u8; 32], [u8; 32]) {
    let p = PublicProofParams {
        block_header: header,
        seed_derivation: SeedDerivation::Salted,
        mining_config: config,
        hash_a,
        hash_b,
        hash_jackpot: [0; 32],
        m,
        n,
        t_rows: 0,
        t_cols: 0,
        moe: None,
    };
    let jk = p.job_key();
    let (b, a) = p.commitment_hash(jk);
    (jk, b, a)
}

fn pearl_root(buf: &[u8], key: [u8; 32]) -> [u8; 32] {
    pearl_blake3::MerkleTree::new(buf, key).root()
}

#[test]
fn tree_root_matches_pearl_blake3() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    // chunk counts around the 64-chunk segment size and non-powers of two
    for chunks in [
        1usize, 2, 3, 63, 64, 65, 127, 128, 129, 191, 200, 257, 1000, 1024, 1500, 4097,
    ] {
        let mut buf = vec![0u8; chunks * CHUNK_LEN];
        rng.fill_bytes(&mut buf);
        let mut key = [0u8; 32];
        rng.fill_bytes(&mut key);
        let expect = pearl_root(&buf, key);
        assert_eq!(expect, pearl_blake3::blake3_digest(&buf, Some(key)));
        let len = buf.len();
        assert_eq!(
            commit_root(&mut buf, len, &key, None),
            expect,
            "chunks={chunks}"
        );
    }
    // unaligned payload: pad_to_chunk_boundary semantics
    let raw: Vec<u8> = (0..3 * 500 + 70_000).map(|i| (i % 251) as u8).collect();
    let padded = pearl_blake3::pad_to_chunk_boundary(&raw);
    let mut buf = vec![0xFFu8; padded.len()];
    buf[..raw.len()].copy_from_slice(&raw);
    buf[raw.len()..].fill(0);
    assert_eq!(
        commit_root(&mut buf, raw.len(), &[9; 32], None),
        pearl_root(&padded, [9; 32])
    );
}

/// ≥ 20 jobs whose A and B come from zk-pow's own reference miner (`try_mine_one`): the roots
/// carried by its PlainProof and the seeds its verifier derives must equal pmkcore's.
#[test]
fn reference_miner_jobs_match() {
    let shapes = [
        (128usize, 64usize, 2048usize),
        (256, 128, 2048),
        (128, 128, 4096),
        (256, 64, 2176),
    ];
    for job in 0..24u64 {
        let (m, n, k) = shapes[job as usize % shapes.len()];
        let mut rng = ChaCha8Rng::seed_from_u64(1000 + job);
        let header = random_header(&mut rng);
        let config = policy_config(k as u32);
        let mut regen = rng.clone();
        let proof = try_mine_one(
            &mut rng,
            m,
            n,
            k,
            header,
            config,
            None,
            true,
            SeedDerivation::Salted,
        )
        .expect("try_mine_one")
        .expect("wrong_jackpot_hash=true returns on the first tile");
        let (_, params) = proof
            .parse_proof(header, SeedDerivation::Salted)
            .expect("parse_proof");

        // Regenerate exactly what try_mine_one drew: A (m×k) then B (k×n), then Bᵀ.
        let a: Vec<u8> = (0..m * k)
            .map(|_| regen.random_range(-64i8..=64) as u8)
            .collect();
        let b: Vec<i8> = (0..k * n).map(|_| regen.random_range(-64i8..=64)).collect();
        let mut bt = vec![0u8; padded_len(n, k)];
        for c in 0..n {
            for l in 0..k {
                bt[c * k + l] = b[l * n + c] as u8;
            }
        }
        let mut a_buf = vec![0u8; padded_len(m, k)];
        a_buf[..m * k].copy_from_slice(&a);

        let t = template_init(
            &header.to_bytes(),
            &config.to_bytes(),
            m as u32,
            n as u32,
            &mut a_buf,
            false,
        )
        .unwrap();
        let j = commit_job(&t, &mut bt).unwrap();

        assert_eq!(t.raw_root_a, params.hash_a, "job {job}: raw root A");
        assert_eq!(j.raw_root_b, params.hash_b, "job {job}: raw root B");
        assert_eq!(t.job_key, params.job_key(), "job {job}: job_key");
        let (b_seed, a_seed) = params.commitment_hash(params.job_key());
        assert_eq!(j.b_noise_seed, b_seed, "job {job}: b_noise_seed");
        assert_eq!(j.a_noise_seed, a_seed, "job {job}: a_noise_seed");
    }
}

/// Fixed A per template, fresh B per job: every generated job is in range, its root equals
/// pearl_blake3's MerkleTree root of the bytes written, seeds equal the zk-pow verifier's, and
/// seeds differ across jobs.
#[test]
fn generated_jobs_fixed_a_fresh_b() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    for &(m, n, k, jobs) in &[
        (256usize, 128usize, 2048usize, 20usize),
        (512, 512, 4096, 6),
    ] {
        let header = random_header(&mut rng);
        let config = policy_config(k as u32);
        let mut a = vec![0xAAu8; padded_len(m, k)];
        let t = template_init(
            &header.to_bytes(),
            &config.to_bytes(),
            m as u32,
            n as u32,
            &mut a,
            true,
        )
        .unwrap();
        assert!(a[..m * k].iter().all(|&x| (-64..=64).contains(&(x as i8))));
        assert_eq!(t.raw_root_a, pearl_root(&a, t.job_key));

        let mut seen = HashSet::new();
        let mut bt = vec![0x7Fu8; padded_len(n, k) + 4096];
        for job in 0..jobs {
            let j = build_job(&t, &mut bt).unwrap();
            let used = &bt[..padded_len(n, k)];
            assert!(
                used[..n * k]
                    .iter()
                    .all(|&x| (-64..=64).contains(&(x as i8))),
                "job {job}: range"
            );
            assert!(used[n * k..].iter().all(|&x| x == 0), "job {job}: pad");
            assert_eq!(
                j.raw_root_b,
                pearl_root(used, t.job_key),
                "job {job}: root B"
            );
            let (jk, b_seed, a_seed) = reference_seeds(
                header,
                config,
                t.raw_root_a,
                j.raw_root_b,
                m as u32,
                n as u32,
            );
            assert_eq!(
                (t.job_key, j.b_noise_seed, j.a_noise_seed),
                (jk, b_seed, a_seed),
                "job {job}: seeds"
            );
            assert!(
                seen.insert(j.a_noise_seed) && seen.insert(j.b_noise_seed),
                "job {job}: seed repeated"
            );
        }
        assert!(
            bt[padded_len(n, k)..].iter().all(|&x| x == 0x7F),
            "wrote past padded_len"
        );
    }
}

#[test]
fn signal_distribution() {
    let mut buf = vec![0u8; 8 << 20];
    let len = buf.len();
    commit_root(&mut buf, len, &[1; 32], Some(&[3; 16]));
    let mut counts = [0u64; 256];
    for &b in &buf {
        counts[b as usize] += 1;
    }
    // Pearson chi-square over the 129 values (128 dof: mean 128, sd 16). The multiply-shift
    // map's own bias (≤ 1/508 relative) contributes < 1 at this sample size.
    let expect = len as f64 / 129.0;
    let chi2: f64 = (-64i8..=64)
        .map(|v| (counts[v as u8 as usize] as f64 - expect).powi(2) / expect)
        .sum();
    println!("chi2 = {chi2:.1} (128 dof)");
    assert!(chi2 < 200.0, "chi2 {chi2}");
    assert_eq!(
        counts.iter().sum::<u64>(),
        (-64i8..=64).map(|v| counts[v as u8 as usize]).sum::<u64>()
    );
}

#[test]
fn c_abi_and_errors() {
    let header = random_header(&mut ChaCha8Rng::seed_from_u64(3)).to_bytes();
    let config = policy_config(2048).to_bytes();
    let (m, n, k) = (128u32, 64u32, 2048u64);
    let mut a = vec![0u8; pmkcore_padded_len(m as u64, k) as usize];
    let mut t = std::mem::MaybeUninit::<Template>::zeroed();
    unsafe {
        assert_eq!(
            pmkcore_template_init(
                header.as_ptr(),
                config.as_ptr(),
                m,
                n,
                a.as_mut_ptr(),
                a.len() as u64,
                1,
                t.as_mut_ptr()
            ),
            0
        );
        let t = t.assume_init();
        let mut bt = vec![0u8; pmkcore_padded_len(n as u64, k) as usize];
        let mut j = Job::default();
        assert_eq!(
            pmkcore_build_job(&t, bt.as_mut_ptr(), bt.len() as u64, &mut j),
            0
        );
        let mut j2 = Job::default();
        assert_eq!(
            pmkcore_commit_job(&t, bt.as_mut_ptr(), bt.len() as u64, &mut j2),
            0
        );
        assert_eq!(j, j2, "build_job and commit_job agree on the same bytes");

        assert_eq!(
            pmkcore_build_job(&t, bt.as_mut_ptr(), bt.len() as u64 - 1, &mut j),
            PmkError::BufferTooSmall as i32
        );
        assert_eq!(
            pmkcore_build_job(std::ptr::null(), bt.as_mut_ptr(), bt.len() as u64, &mut j),
            PmkError::NullPointer as i32
        );
        bt[5] = 65;
        assert_eq!(
            pmkcore_commit_job(&t, bt.as_mut_ptr(), bt.len() as u64, &mut j),
            PmkError::SignalOutOfRange as i32
        );
        let mut t2 = Template {
            job_key: [0; 32],
            raw_root_a: [0; 32],
            salted_root_a: [0; 32],
            m: 0,
            n: 0,
            k: 0,
            _reserved: 0,
        };
        assert_eq!(
            pmkcore_template_init(
                header.as_ptr(),
                config.as_ptr(),
                100,
                n,
                a.as_mut_ptr(),
                a.len() as u64,
                1,
                &mut t2
            ),
            PmkError::BadShape as i32
        );
        let mut bad = config;
        bad[4] = 64; // rank 64
        assert_eq!(
            pmkcore_template_init(
                header.as_ptr(),
                bad.as_ptr(),
                m,
                n,
                a.as_mut_ptr(),
                a.len() as u64,
                1,
                &mut t2
            ),
            PmkError::Policy as i32
        );
        let msg = std::ffi::CStr::from_ptr(pmkcore_strerror(PmkError::Policy as i32));
        assert!(msg.to_str().unwrap().contains("rank 128"));
    }
}
