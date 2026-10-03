use pmkcore::oracle::{Pattern, TileResult};
use pmkcore::*;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern};

fn random_header(rng: &mut impl RngCore) -> [u8; HEADER_LEN] {
    let mut prev_block = [0u8; 32];
    let mut merkle_root = [0u8; 32];
    rng.fill_bytes(&mut prev_block);
    rng.fill_bytes(&mut merkle_root);
    IncompleteBlockHeader {
        version: rng.next_u32(),
        prev_block,
        merkle_root,
        timestamp: rng.next_u32(),
        nbits: 0x1d00ffff,
    }
    .to_bytes()
}

fn make_job(pattern: Pattern, m: u32, n: u32) -> *mut pmkcore::oracle_ffi::PmkOracleJob {
    let mut rng = ChaCha8Rng::seed_from_u64(match pattern {
        Pattern::Na => 11,
        Pattern::Sg => 12,
    });
    let header = random_header(&mut rng);
    let mut config = [0u8; CONFIG_LEN];
    let pattern_id = match pattern {
        Pattern::Na => 0,
        Pattern::Sg => 1,
    };
    unsafe {
        assert_eq!(
            pmkcore_build_config(pattern_id, 2048, m, n, config.as_mut_ptr()),
            0
        );
    }
    let mut a = vec![0u8; padded_len(m as usize, 2048)];
    let mut bt = vec![0u8; padded_len(n as usize, 2048)];
    for v in &mut a[..m as usize * 2048] {
        *v = rng.random_range(-64i8..=64) as u8;
    }
    for v in &mut bt[..n as usize * 2048] {
        *v = rng.random_range(-64i8..=64) as u8;
    }
    let mut job = std::ptr::null_mut();
    unsafe {
        assert_eq!(
            pmkcore_oracle_job_create(
                header.as_ptr(),
                config.as_ptr(),
                m,
                n,
                a.as_ptr(),
                a.len() as u64,
                bt.as_ptr(),
                bt.len() as u64,
                &mut job,
            ),
            0
        );
    }
    assert!(!job.is_null());
    job
}

#[test]
fn b1_config_builder_rejects_bad_inputs() {
    let mut cfg = [0u8; CONFIG_LEN];
    unsafe {
        assert_eq!(pmkcore_build_config(0, 2048, 128, 64, cfg.as_mut_ptr()), 0);
        assert_eq!(pmkcore_build_config(1, 2048, 64, 64, cfg.as_mut_ptr()), 0);
        assert_eq!(
            pmkcore_build_config(2, 2048, 32, 32, cfg.as_mut_ptr()),
            PmkError::Policy as i32
        );
        assert_eq!(
            pmkcore_build_config(0, 64, 128, 64, cfg.as_mut_ptr()),
            PmkError::Policy as i32
        );
        assert_eq!(
            pmkcore_build_config(0, 65536, 128, 64, cfg.as_mut_ptr()),
            PmkError::Policy as i32
        );
        assert_eq!(
            pmkcore_build_config_diagnostic(0, 65536, 128, 64, cfg.as_mut_ptr()),
            0
        );
        assert_eq!(
            pmkcore_build_config(0, 2048, 127, 64, cfg.as_mut_ptr()),
            PmkError::BadShape as i32
        );
        assert_eq!(
            pmkcore_build_config(0, 2048, 128, 64, std::ptr::null_mut()),
            PmkError::NullPointer as i32
        );
    }
}

#[test]
fn production_template_abi_rejects_diagnostic_config_and_bad_lengths() {
    let mut rng = ChaCha8Rng::seed_from_u64(44);
    let header = random_header(&mut rng);
    let diagnostic_config = MiningConfiguration {
        common_dim: 65536,
        rank: 128,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&[
            0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51,
        ])
        .unwrap(),
        moe: None,
    }
    .to_bytes();
    let mut template = Template {
        job_key: [0; 32],
        raw_root_a: [0; 32],
        salted_root_a: [0; 32],
        m: 0,
        n: 0,
        k: 0,
        _reserved: 0,
    };
    let mut a = [0u8; 1024];
    unsafe {
        assert_eq!(
            pmkcore_template_init(
                header.as_ptr(),
                diagnostic_config.as_ptr(),
                128,
                64,
                a.as_mut_ptr(),
                a.len() as u64,
                1,
                &mut template,
            ),
            PmkError::Policy as i32
        );
        assert_eq!(
            pmkcore_template_init(
                header.as_ptr(),
                diagnostic_config.as_ptr(),
                128,
                64,
                a.as_mut_ptr(),
                u64::MAX,
                1,
                &mut template,
            ),
            PmkError::Policy as i32
        );
    }
}

#[test]
fn production_job_abi_rejects_forged_template_before_raw_slice() {
    let forged_diagnostic = Template {
        job_key: [0; 32],
        raw_root_a: [0; 32],
        salted_root_a: [0; 32],
        m: 64,
        n: 64,
        k: 65536,
        _reserved: 0,
    };
    let forged_extreme = Template {
        n: u32::MAX,
        ..forged_diagnostic
    };
    let mut seeds = Job::default();
    let dummy = std::ptr::NonNull::<u8>::dangling().as_ptr();
    unsafe {
        assert_eq!(
            pmkcore_build_job(&forged_diagnostic, dummy, 1, &mut seeds),
            PmkError::Policy as i32
        );
        assert_eq!(
            pmkcore_commit_job(&forged_diagnostic, dummy, 1, &mut seeds),
            PmkError::Policy as i32
        );
        assert_eq!(
            pmkcore_build_job(&forged_extreme, dummy, 1, &mut seeds),
            PmkError::Policy as i32
        );
    }
}

#[test]
fn production_job_abi_rejects_null_and_oversized_lengths_before_slicing() {
    let mut rng = ChaCha8Rng::seed_from_u64(45);
    let header = random_header(&mut rng);
    let mut config = [0u8; CONFIG_LEN];
    let mut template = Template {
        job_key: [0; 32],
        raw_root_a: [0; 32],
        salted_root_a: [0; 32],
        m: 0,
        n: 0,
        k: 0,
        _reserved: 0,
    };
    unsafe {
        assert_eq!(
            pmkcore_build_config(1, 2048, 64, 64, config.as_mut_ptr()),
            0
        );
    }
    let mut a = vec![0u8; padded_len(64, 2048)];
    let mut bt = vec![0u8; padded_len(64, 2048)];
    let mut seeds = Job::default();
    unsafe {
        assert_eq!(
            pmkcore_template_init(
                header.as_ptr(),
                config.as_ptr(),
                64,
                64,
                a.as_mut_ptr(),
                a.len() as u64,
                1,
                &mut template,
            ),
            0
        );
        assert_eq!(
            pmkcore_build_job(&template, std::ptr::null_mut(), bt.len() as u64, &mut seeds),
            PmkError::NullPointer as i32
        );
        assert_eq!(
            pmkcore_build_job(&template, bt.as_mut_ptr(), u64::MAX, &mut seeds),
            PmkError::ResourceLimit as i32
        );
        assert_eq!(
            pmkcore_commit_job(&template, bt.as_mut_ptr(), bt.len() as u64 - 1, &mut seeds),
            PmkError::BufferTooSmall as i32
        );
    }
}

#[test]
fn b1_oracle_tile_and_buffer_conventions() {
    unsafe {
        let job = make_job(Pattern::Sg, 64, 64);
        let easy = [0xffu8; 32];
        let mut tile = std::mem::MaybeUninit::<TileResult>::zeroed();
        assert_eq!(
            pmkcore_oracle_tile(job, 0, 0, easy.as_ptr(), easy.as_ptr(), tile.as_mut_ptr()),
            0
        );
        let tile = tile.assume_init();
        assert_eq!(tile.t_rows, 0);
        assert_eq!(tile.t_cols, 0);
        assert_eq!(tile.is_share, 1);
        assert_eq!(tile.is_block, 1);

        let mut scan_len = 0u64;
        assert_eq!(
            pmkcore_oracle_scan(
                job,
                easy.as_ptr(),
                easy.as_ptr(),
                std::ptr::null_mut(),
                0,
                &mut scan_len
            ),
            0
        );
        assert!(scan_len > 0);
        let mut small_scan: Vec<TileResult> = std::iter::repeat_with(|| std::mem::zeroed())
            .take(scan_len as usize - 1)
            .collect();
        assert_eq!(
            pmkcore_oracle_scan(
                job,
                easy.as_ptr(),
                easy.as_ptr(),
                small_scan.as_mut_ptr(),
                small_scan.len() as u64,
                &mut scan_len
            ),
            PmkError::BufferTooSmall as i32
        );
        let mut scan: Vec<TileResult> = std::iter::repeat_with(|| std::mem::zeroed())
            .take(scan_len as usize)
            .collect();
        assert_eq!(
            pmkcore_oracle_scan(
                job,
                easy.as_ptr(),
                easy.as_ptr(),
                scan.as_mut_ptr(),
                scan.len() as u64,
                &mut scan_len
            ),
            0
        );
        assert_eq!(scan[0].t_rows, 0);

        let mut proof_len = 0u64;
        assert_eq!(
            pmkcore_oracle_build_plain_proof(job, 0, 0, std::ptr::null_mut(), 0, &mut proof_len),
            0
        );
        assert!(proof_len > 0);
        let mut one_byte_short = vec![0u8; proof_len as usize - 1];
        assert_eq!(
            pmkcore_oracle_build_plain_proof(
                job,
                0,
                0,
                one_byte_short.as_mut_ptr(),
                one_byte_short.len() as u64,
                &mut proof_len
            ),
            PmkError::BufferTooSmall as i32
        );
        let mut proof = vec![0u8; proof_len as usize];
        assert_eq!(
            pmkcore_oracle_build_plain_proof(
                job,
                0,
                0,
                proof.as_mut_ptr(),
                proof.len() as u64,
                &mut proof_len
            ),
            0
        );
        assert!(proof.iter().any(|&b| b != 0));

        let mut vectors_len = 0u64;
        assert_eq!(
            pmkcore_oracle_export_vectors(job, std::ptr::null_mut(), 0, &mut vectors_len),
            0
        );
        assert!(vectors_len > 0);

        let mut out = std::mem::MaybeUninit::<TileResult>::zeroed();
        assert_eq!(
            pmkcore_oracle_tile(job, 8, 0, easy.as_ptr(), easy.as_ptr(), out.as_mut_ptr()),
            PmkError::IllegalOffset as i32
        );
        assert_eq!(
            pmkcore_oracle_tile(
                std::ptr::null(),
                0,
                0,
                easy.as_ptr(),
                easy.as_ptr(),
                out.as_mut_ptr()
            ),
            PmkError::NullPointer as i32
        );
        assert_eq!(
            pmkcore_oracle_export_vectors(job, std::ptr::null_mut(), 0, std::ptr::null_mut()),
            PmkError::NullPointer as i32
        );
        pmkcore_oracle_job_free(job);
        pmkcore_oracle_job_free(std::ptr::null_mut());
    }
}
