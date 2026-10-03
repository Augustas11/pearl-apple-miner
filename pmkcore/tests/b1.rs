use bincode::Options;
use pmkcore::oracle::{self, OracleJob, Pattern, TileResult};
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::io::Write;
use std::process::{Command, Stdio};
use zk_pow::api::proof::{
    IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, SeedDerivation,
};
use zk_pow::api::sanity_checks::check_rank_penalty;
use zk_pow::api::verify::verify_plain_proof;
use zk_pow::ffi::mine::try_mine_one;
use zk_pow::ffi::plain_proof::PlainProof;

const EASY_NBITS: u32 = 0x207f_ffff;
const EASY_SHARE_NBITS: u32 = 0x1e10_0000;
const K: u32 = 2048;
const MAX_BOUND: [u8; 32] = [0xff; 32];
const ZERO_BOUND: [u8; 32] = [0; 32];

fn hex(bytes: &[u8]) -> String {
    const LUT: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(LUT[(b >> 4) as usize] as char);
        out.push(LUT[(b & 15) as usize] as char);
    }
    out
}

fn header(rng: &mut impl RngCore, nbits: u32) -> [u8; 76] {
    let mut prev_block = [0u8; 32];
    let mut merkle_root = [0u8; 32];
    rng.fill_bytes(&mut prev_block);
    rng.fill_bytes(&mut merkle_root);
    IncompleteBlockHeader {
        version: rng.next_u32(),
        prev_block,
        merkle_root,
        timestamp: rng.next_u32(),
        nbits,
    }
    .to_bytes()
}

fn signals(rng: &mut impl Rng, len: usize) -> Vec<u8> {
    (0..len)
        .map(|_| rng.random_range(-64i8..=64) as u8)
        .collect()
}

fn shape(pattern: Pattern) -> (u32, u32) {
    match pattern {
        Pattern::Na => (128, 64),
        Pattern::Sg => (64, 64),
    }
}

fn tile_count(pattern: Pattern, m: u32, n: u32) -> usize {
    let area = match pattern {
        Pattern::Na => 4 * 16,
        Pattern::Sg => 4 * 8,
    };
    m as usize * n as usize / area
}

fn pattern_rows(pattern: Pattern) -> &'static [u32] {
    match pattern {
        Pattern::Na => &[0, 8, 64, 72],
        Pattern::Sg => &[0, 8, 16, 24],
    }
}

fn pattern_cols(pattern: Pattern) -> &'static [u32] {
    match pattern {
        Pattern::Na => &[0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51],
        Pattern::Sg => &[0, 1, 8, 9, 16, 17, 24, 25],
    }
}

fn csv(values: impl IntoIterator<Item = u32>) -> String {
    values
        .into_iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn rank64_config_bytes(pattern: Pattern) -> [u8; 52] {
    MiningConfiguration {
        common_dim: K,
        rank: 64,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(pattern_rows(pattern)).unwrap(),
        cols_pattern: PeriodicPattern::from_list(pattern_cols(pattern)).unwrap(),
        moe: None,
    }
    .to_bytes()
}

fn config_bytes(pattern: Pattern, m: u32, n: u32) -> [u8; 52] {
    let cfg = oracle::build_config(pattern, K, m, n).expect("build_config");
    let bytes = cfg.to_bytes();
    let reparsed = MiningConfiguration::from_bytes(&bytes).expect("config roundtrip parse");
    assert_eq!(reparsed.to_bytes(), bytes);
    bytes
}

fn as_i8(bytes: &[u8]) -> Vec<i8> {
    bytes.iter().map(|&b| b as i8).collect()
}

fn parse_plain(bytes: &[u8]) -> PlainProof {
    PlainProof::deserialize_compat(bytes).expect("pmkcore proof bytes are PlainProof bincode")
}

fn serialize_plain(proof: &PlainProof) -> Vec<u8> {
    bincode::options()
        .with_fixint_encoding()
        .serialize(proof)
        .expect("serialize PlainProof")
}

fn assert_tile_eq(a: &TileResult, b: &TileResult) {
    assert_eq!((a.t_rows, a.t_cols), (b.t_rows, b.t_cols));
    assert_eq!(a.transcript, b.transcript);
    assert_eq!(a.hash, b.hash);
    assert_eq!((a.is_share, a.is_block), (b.is_share, b.is_block));
}

fn exported_field_names(buf: &[u8]) -> Vec<String> {
    assert!(buf.starts_with(b"PMKVEC01"));
    let mut off = 8usize;
    let fields = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
    off += 4;
    let mut names = Vec::with_capacity(fields);
    for _ in 0..fields {
        let name_len = u16::from_le_bytes(buf[off..off + 2].try_into().unwrap()) as usize;
        off += 2;
        let name = std::str::from_utf8(&buf[off..off + name_len])
            .unwrap()
            .to_string();
        off += name_len;
        let payload_len = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap()) as usize;
        off += 8 + payload_len;
        names.push(name);
    }
    assert_eq!(
        off,
        buf.len(),
        "PMKVEC01 parser must consume the whole blob"
    );
    names
}

#[test]
fn config_builders_enforce_v1_patterns_and_shapes() {
    let na = oracle::build_config(Pattern::Na, 4096, 128, 64).unwrap();
    assert_eq!(na.rank, 128);
    assert_eq!(na.common_dim, 4096);
    assert_eq!(na.rows_pattern.to_list(), vec![0, 8, 64, 72]);
    assert_eq!(
        na.cols_pattern.to_list(),
        vec![0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51]
    );
    assert_eq!(
        MiningConfiguration::from_bytes(&na.to_bytes())
            .unwrap()
            .to_bytes(),
        na.to_bytes()
    );

    let sg = oracle::build_config(Pattern::Sg, 4096, 64, 64).unwrap();
    assert_eq!(sg.rows_pattern.to_list(), vec![0, 8, 16, 24]);
    assert_eq!(sg.cols_pattern.to_list(), vec![0, 1, 8, 9, 16, 17, 24, 25]);
    assert_eq!(
        MiningConfiguration::from_bytes(&sg.to_bytes())
            .unwrap()
            .to_bytes(),
        sg.to_bytes()
    );

    assert!(oracle::build_config(Pattern::Na, 1024, 128, 64).is_err());
    assert!(oracle::build_config(Pattern::Na, 4097, 128, 64).is_err());
    assert!(oracle::build_config(Pattern::Na, 4096, 64, 64).is_err());
    assert!(oracle::build_config(Pattern::Sg, 4096, 32, 64).is_err());
}

#[test]
fn transcript_matches_stored_k3sg_vectors() {
    for (name, m, n, k) in [
        ("v1_256x256x4096", 256usize, 256usize, 4096usize),
        ("v2_256x256x4096_pm127", 256, 256, 4096),
        ("v3_128x128x65536", 128, 128, 65536),
        ("v4_pearl_c64", 128, 128, 2048),
    ] {
        let dir = format!(
            "{}/../bench/k3sg/studio/vectors/{name}",
            env!("CARGO_MANIFEST_DIR")
        );
        let a = as_i8(&std::fs::read(format!("{dir}/A.bin")).unwrap());
        let b = as_i8(&std::fs::read(format!("{dir}/B.bin")).unwrap());
        let tiles = std::fs::read(format!("{dir}/tiles.bin")).unwrap();
        // These checked-in files have a flat JSON uint32 key array. Read that single field
        // without adding a JSON dependency to the production crate.
        let metadata = std::fs::read_to_string(format!("{dir}/job.json")).unwrap();
        let key_words = metadata
            .split_once("\"key\"")
            .unwrap()
            .1
            .split_once('[')
            .unwrap()
            .1
            .split_once(']')
            .unwrap()
            .0;
        let key: Vec<u8> = key_words
            .split(',')
            .flat_map(|x| x.trim().parse::<u32>().unwrap().to_le_bytes())
            .collect();
        let key: [u8; 32] = key.try_into().unwrap();
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        let mut bt = vec![0i8; n * k];
        for row in 0..k {
            for col in 0..n {
                bt[col * k + row] = b[row * n + col];
            }
        }
        assert_eq!(tiles.len() % 104, 0);
        let count = tiles.len() / 104;
        assert_eq!(count, m * n / 32);
        // Exercise zero, local nonzero, next-group, and final GLOBAL offsets in each dataset.
        for index in [0, 1, count / 2, count - 1] {
            let record = &tiles[index * 104..(index + 1) * 104];
            let words: [u32; 26] = std::array::from_fn(|i| {
                u32::from_le_bytes(record[4 * i..4 * i + 4].try_into().unwrap())
            });
            let rows = pattern_rows(Pattern::Sg)
                .iter()
                .map(|&r| (words[0] + r) as usize)
                .collect::<Vec<_>>();
            let cols = pattern_cols(Pattern::Sg)
                .iter()
                .map(|&c| (words[1] + c) as usize)
                .collect::<Vec<_>>();
            let got = oracle::transcript(&a, &bt, k, &rows, &cols);
            assert_eq!(&got[..], &words[2..18], "{name}: tile {index}");
            let message: Vec<u8> = got.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(
                blake3::keyed_hash(&key, &message).as_bytes(),
                &record[72..104],
                "{name}: hash {index}"
            );
        }
        println!("K3SG {name}: 4 global-offset transcripts + hashes PASS");
    }
}

#[test]
fn oracle_matches_zk_pow_reference_first_tile() {
    for (pattern, seed) in [(Pattern::Na, 0x51A7u64), (Pattern::Sg, 0x59C7u64)] {
        let (m, n) = shape(pattern);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let header = IncompleteBlockHeader::from_bytes(&header(&mut rng, EASY_NBITS)).unwrap();
        let config = oracle::build_config(pattern, K, m, n).unwrap();

        let mut regen = rng.clone();
        let proof = try_mine_one(
            &mut rng,
            m as usize,
            n as usize,
            K as usize,
            header,
            config,
            None,
            false,
            SeedDerivation::Salted,
        )
        .expect("try_mine_one")
        .expect("easy target finds the first tile");

        let a = signals(&mut regen, m as usize * K as usize);
        let b: Vec<i8> = (0..K as usize * n as usize)
            .map(|_| regen.random_range(-64i8..=64))
            .collect();
        let mut bt = vec![0u8; n as usize * K as usize];
        for c in 0..n as usize {
            for l in 0..K as usize {
                bt[c * K as usize + l] = b[l * n as usize + c] as u8;
            }
        }

        let job = OracleJob::new(&header.to_bytes(), &config.to_bytes(), m, n, &a, &bt).unwrap();
        let scan = job.scan(MAX_BOUND, ZERO_BOUND);
        assert!(!scan.is_empty());
        assert_eq!((scan[0].t_rows, scan[0].t_cols), (0, 0));

        let ours = parse_plain(
            &job.build_plain_proof(scan[0].t_rows, scan[0].t_cols)
                .unwrap(),
        );
        let (theirs_private, theirs_public) =
            proof.parse_proof(header, SeedDerivation::Salted).unwrap();
        let compiled = theirs_public.compile().0;
        let noise = zk_pow::circuit::pearl_noise::compute_noise(&compiled);
        let transcript = zk_pow::circuit::chip::compute_jackpot(
            &compiled,
            &theirs_private.s_a,
            &theirs_private.s_b,
            &noise,
        );
        assert_eq!(scan[0].transcript, transcript);
        assert_eq!(
            scan[0].hash,
            zk_pow::api::proof_utils::compute_jackpot_hash(&transcript, compiled.a_noise_seed())
        );
        let (_, ours_public) = ours.parse_proof(header, SeedDerivation::Salted).unwrap();
        assert_eq!(ours_public.hash_a, theirs_public.hash_a);
        assert_eq!(ours_public.hash_b, theirs_public.hash_b);
        assert_eq!(ours_public.t_rows, theirs_public.t_rows);
        assert_eq!(ours_public.t_cols, theirs_public.t_cols);
        assert_eq!(
            ours_public.mining_config.to_bytes(),
            theirs_public.mining_config.to_bytes()
        );
    }
}

#[test]
fn random_finds_build_plain_proofs_that_python_accepts() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xB100);
    let mut cases = String::new();
    let mut proof_count = 0usize;
    let mut negative_seed: Option<([u8; 76], Vec<u8>)> = None;

    for pattern in [Pattern::Na, Pattern::Sg] {
        let (m, n) = shape(pattern);
        let expected_tiles = tile_count(pattern, m, n);
        let config = config_bytes(pattern, m, n);
        for job_idx in 0..20 {
            let header = header(&mut rng, EASY_NBITS);
            let a = signals(&mut rng, m as usize * K as usize);
            let bt = signals(&mut rng, n as usize * K as usize);
            let job = OracleJob::new(&header, &config, m, n, &a, &bt).unwrap();
            assert_eq!(job.header().to_bytes(), header);
            assert_eq!(job.config().to_bytes(), config);
            assert_eq!(job.bound(EASY_NBITS), MAX_BOUND);

            assert_eq!(
                job.tile_count(),
                expected_tiles,
                "{pattern:?} job {job_idx}: tile count"
            );
            let share_bound = job.bound(EASY_SHARE_NBITS);
            assert_ne!(share_bound, MAX_BOUND);
            let classified = job.scan(share_bound, ZERO_BOUND);
            assert_eq!(classified.len(), expected_tiles);
            let finds: Vec<_> = classified.into_iter().filter(|t| t.is_share == 1).collect();
            assert!(!finds.is_empty());
            for tile in &finds {
                assert_eq!(
                    tile.is_share, 1,
                    "every reported find must meet the share bound"
                );
                let one = job
                    .tile(tile.t_rows, tile.t_cols, MAX_BOUND, ZERO_BOUND)
                    .unwrap();
                assert_tile_eq(tile, &one);
                let proof = job.build_plain_proof(tile.t_rows, tile.t_cols).unwrap();
                if negative_seed.is_none() {
                    negative_seed = Some((header, proof.clone()));
                }
                let parsed = parse_plain(&proof);
                verify_plain_proof(
                    &IncompleteBlockHeader::from_bytes(&header).unwrap(),
                    &parsed,
                    Some(EASY_SHARE_NBITS),
                    SeedDerivation::Salted,
                )
                .expect("Rust verifier accepts proof at share bound");
                cases.push_str(&format!(
                    "{pattern:?}_{job_idx}_{}_{} ok {} {} {:08x}\n",
                    tile.t_rows,
                    tile.t_cols,
                    hex(&header),
                    hex(&proof),
                    EASY_SHARE_NBITS
                ));
                proof_count += 1;
            }
            let exported = job.export_vectors();
            let names = exported_field_names(&exported);
            assert_eq!(names.len(), 26);
            for required in [
                "dimensions",
                "header",
                "config",
                "job_key",
                "raw_root_a",
                "raw_root_b",
                "salted_root_a",
                "salted_root_b",
                "a_noise_seed",
                "b_noise_seed",
                "noised_a",
                "noised_bt",
            ] {
                assert!(
                    names.iter().any(|n| n == required),
                    "missing PMKVEC01 field {required}"
                );
            }
        }
    }

    assert!(
        proof_count >= 40,
        "expected at least one proof per random job"
    );

    for pattern in [Pattern::Na, Pattern::Sg] {
        let (m, n) = shape(pattern);
        let mine_header = header(&mut rng, EASY_NBITS);
        let mine_config = config_bytes(pattern, m, n);
        let a = vec![0u8; m as usize * K as usize];
        let bt = vec![0u8; n as usize * K as usize];
        let job = OracleJob::new(&mine_header, &mine_config, m, n, &a, &bt).unwrap();
        let tile = job.scan(MAX_BOUND, ZERO_BOUND).remove(0);
        let rows = csv(pattern_rows(pattern).iter().map(|&row| tile.t_rows + row));
        let cols = csv(pattern_cols(pattern).iter().map(|&col| tile.t_cols + col));
        cases.push_str(&format!(
            "mine_cmp pearl_mining_{pattern:?} {m} {n} {K} {} {} {} {} {} {}\n",
            hex(&mine_header),
            hex(&mine_config),
            rows,
            cols,
            hex(&job.intermediates().raw_root_a),
            hex(&job.intermediates().raw_root_b)
        ));
    }

    let (neg_header, neg_proof) = negative_seed.expect("at least one proof for negative cases");
    let mut wrong_header = neg_header;
    wrong_header[4] ^= 0x55;
    cases.push_str(&format!(
        "wrong_header err {} {} {:08x}\n",
        hex(&wrong_header),
        hex(&neg_proof),
        EASY_SHARE_NBITS
    ));
    cases.push_str(&format!(
        "malformed wrong_config_rank64 noise_rank64 {} {} {:08x} err\n",
        hex(&neg_header),
        hex(&neg_proof),
        EASY_SHARE_NBITS
    ));
    cases.push_str(&format!(
        "malformed wrong_config_rank256 noise_rank256 {} {} {:08x} err\n",
        hex(&neg_header),
        hex(&neg_proof),
        EASY_SHARE_NBITS
    ));
    let mut illegal = parse_plain(&neg_proof);
    let old_min = illegal.a.row_indices[0];
    for row in &mut illegal.a.row_indices {
        *row = *row - old_min + 8;
    }
    cases.push_str(&format!(
        "illegal_offset err {} {} {:08x}\n",
        hex(&neg_header),
        hex(&serialize_plain(&illegal)),
        EASY_SHARE_NBITS
    ));

    let (m, n) = shape(Pattern::Sg);
    let neg_config = config_bytes(Pattern::Sg, m, n);
    let legacy_header = header(&mut rng, EASY_SHARE_NBITS);
    let parsed_legacy_header = IncompleteBlockHeader::from_bytes(&legacy_header).unwrap();
    let parsed_neg_config = MiningConfiguration::from_bytes(&neg_config).unwrap();
    let mut legacy_proof = None;
    for _ in 0..200 {
        if let Some(proof) = try_mine_one(
            &mut rng,
            m as usize,
            n as usize,
            K as usize,
            parsed_legacy_header,
            parsed_neg_config,
            None,
            false,
            SeedDerivation::Legacy,
        )
        .unwrap()
        {
            legacy_proof = Some(proof);
            break;
        }
    }
    let legacy_proof = serialize_plain(&legacy_proof.expect("legacy proof under easy bound"));
    cases.push_str(&format!(
        "legacy_unsalted err {} {} {:08x}\n",
        hex(&legacy_header),
        hex(&legacy_proof),
        EASY_SHARE_NBITS
    ));
    let raw65_header = header(&mut rng, EASY_NBITS);
    cases.push_str(&format!(
        "raw65 raw_signal65 {m} {n} {K} {} {}\n",
        hex(&raw65_header),
        hex(&neg_config)
    ));

    let mut child = Command::new("../.venv/bin/python")
        .arg("-B")
        .arg("tests/verify_b1.py")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn Python verifier");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(cases.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "Python verifier failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    println!("40 random jobs (20 NA + 20 SG): {proof_count} finds, all Python cert-v3 accepted at non-saturated share bound");
    let stdout = String::from_utf8(output.stdout).unwrap();
    for line in stdout
        .lines()
        .filter(|line| !line.starts_with("Na_") && !line.starts_with("Sg_"))
    {
        println!("{line}");
    }
}

#[test]
fn verifier_rejects_negative_cases() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xBAD5EED);
    let (m, n) = shape(Pattern::Sg);
    let header = header(&mut rng, EASY_NBITS);
    let config = config_bytes(Pattern::Sg, m, n);
    let mut a = signals(&mut rng, m as usize * K as usize);
    let bt = signals(&mut rng, n as usize * K as usize);
    let job = OracleJob::new(&header, &config, m, n, &a, &bt).unwrap();
    let tile = job.scan(MAX_BOUND, ZERO_BOUND).remove(0);
    let proof_bytes = job.build_plain_proof(tile.t_rows, tile.t_cols).unwrap();
    let proof = parse_plain(&proof_bytes);
    let parsed_header = IncompleteBlockHeader::from_bytes(&header).unwrap();
    verify_plain_proof(&parsed_header, &proof, None, SeedDerivation::Salted).unwrap();

    let mut wrong_header = header;
    wrong_header[4] ^= 0x80;
    let wrong_header = IncompleteBlockHeader::from_bytes(&wrong_header).unwrap();
    assert!(verify_plain_proof(&wrong_header, &proof, None, SeedDerivation::Salted).is_err());

    let cfg = MiningConfiguration::from_bytes(&config).unwrap();
    let bad_row = (0..m)
        .find(|&offset| !cfg.rows_pattern.offset_is_valid(offset))
        .expect("SG has invalid row offsets");
    assert!(job.build_plain_proof(bad_row, tile.t_cols).is_err());

    let mut bad_proof = proof.clone();
    bad_proof.k = 64;
    assert!(verify_plain_proof(&parsed_header, &bad_proof, None, SeedDerivation::Salted).is_err());

    let mut bad_config = MiningConfiguration::from_bytes(&config).unwrap();
    bad_config.cols_pattern = PeriodicPattern::from_list(&[0, 1, 2, 3, 16, 17, 18, 19]).unwrap();
    let (_, mut public) = proof
        .parse_proof(parsed_header, SeedDerivation::Salted)
        .unwrap();
    public.mining_config = bad_config;
    assert!(public.sanity_check().is_ok());
    assert_ne!(public.job_key(), job.intermediates().job_key);

    a[0] = 65i8 as u8;
    assert!(OracleJob::new(&header, &config, m, n, &a, &bt).is_err());

    let legacy_header =
        IncompleteBlockHeader::from_bytes(&crate::header(&mut rng, 0x1e00_ffff)).unwrap();
    let good_config = MiningConfiguration::from_bytes(&config).unwrap();
    let mut legacy = None;
    for _ in 0..200 {
        if let Some(proof) = try_mine_one(
            &mut rng,
            m as usize,
            n as usize,
            K as usize,
            legacy_header,
            good_config,
            None,
            false,
            SeedDerivation::Legacy,
        )
        .unwrap()
        {
            legacy = Some(proof);
            break;
        }
    }
    let legacy = legacy.expect("legacy proof under non-saturated bound");
    verify_plain_proof(&legacy_header, &legacy, None, SeedDerivation::Legacy).unwrap();
    assert!(verify_plain_proof(&legacy_header, &legacy, None, SeedDerivation::Salted).is_err());

    let rank64 = MiningConfiguration::from_bytes(&rank64_config_bytes(Pattern::Sg)).unwrap();
    let zeros_a = vec![0u8; m as usize * K as usize];
    let zeros_bt = vec![0u8; n as usize * K as usize];
    assert!(OracleJob::new(&header, &rank64.to_bytes(), m, n, &zeros_a, &zeros_bt).is_err());
    let rank64_proof = try_mine_one(
        &mut rng,
        m as usize,
        n as usize,
        K as usize,
        parsed_header,
        rank64,
        None,
        false,
        SeedDerivation::Salted,
    )
    .unwrap()
    .unwrap();
    verify_plain_proof(&parsed_header, &rank64_proof, None, SeedDerivation::Salted).unwrap();
    let (_, public64) = rank64_proof
        .parse_proof(parsed_header, SeedDerivation::Salted)
        .unwrap();
    assert!(check_rank_penalty(&rank64, &public64.hash_jackpot(), EASY_NBITS).is_err());
}
