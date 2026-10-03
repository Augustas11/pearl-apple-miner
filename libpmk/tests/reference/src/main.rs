use pearl_blake3::blake3_digest;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};
use zk_pow::circuit::pearl_noise::compute_noise_for_indices;

const RANK: usize = 128;

#[derive(Clone)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn byte(&mut self) -> u8 {
        (self.next_u64() >> 56) as u8
    }

    fn signal(&mut self) -> i8 {
        (self.next_u64() % 129) as i8 - 64
    }
}

#[derive(Serialize)]
struct NoiseVector {
    id: String,
    m: usize,
    n: usize,
    k: usize,
    rank: usize,
    rng_seed: String,
    a_noise_seed: String,
    b_noise_seed: String,
    rows_sha256: String,
    cols_sha256: String,
    raw_a_sha256: String,
    raw_bt_sha256: String,
    noise_a_sha256: String,
    noise_bt_sha256: String,
    noised_a_sha256: String,
    noised_b_sha256: String,
    raw_a_prefix_hex: String,
    raw_bt_prefix_hex: String,
    noise_a_prefix_hex: String,
    noise_bt_prefix_hex: String,
    noised_a_prefix_hex: String,
    noised_b_prefix_hex: String,
}

#[derive(Serialize)]
struct Blake3Vector {
    id: String,
    key_hex: String,
    block_hex: String,
    pearl_digest_hex: String,
    crate_digest_hex: String,
}

fn sha_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn prefix_hex(bytes: &[u8]) -> String {
    hex::encode(&bytes[..bytes.len().min(256)])
}

fn i8_bytes(values: &[i8]) -> Vec<u8> {
    values.iter().map(|&v| v as u8).collect()
}

fn u32_bytes(values: &[usize]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&(v as u32).to_le_bytes());
    }
    out
}

fn make_signal(len: usize, rng: &mut Rng) -> Vec<i8> {
    (0..len).map(|_| rng.signal()).collect()
}

fn make_seed(rng: &mut Rng) -> [u8; 32] {
    let mut s = [0u8; 32];
    for b in &mut s {
        *b = rng.byte();
    }
    s
}

fn emit_noise_vectors(out_dir: &PathBuf) -> anyhow::Result<()> {
    let shapes = [
        (64, 64, 2048, 4),
        (128, 64, 2048, 4),
        (64, 128, 2048, 4),
        (256, 256, 4096, 4),
        (128, 128, 65536, 4),
        (8192, 8192, 4096, 1),
    ];
    let mut records = Vec::new();
    for (shape_idx, &(m, n, k, variants)) in shapes.iter().enumerate() {
        for variant in 0..variants {
            let seed = 0xB200_0000_0000_0000u64 ^ ((shape_idx as u64) << 32) ^ variant as u64;
            let mut rng = Rng::new(seed);
            let a_seed = make_seed(&mut rng);
            let b_seed = make_seed(&mut rng);
            let raw_a = make_signal(m * k, &mut rng);
            let raw_bt = make_signal(n * k, &mut rng);
            let rows: Vec<usize> = (0..m).collect();
            let cols: Vec<usize> = (0..n).collect();
            let noise = compute_noise_for_indices(k, RANK, (b_seed, a_seed), &rows, &cols);

            let mut noised_a = Vec::with_capacity(m * k);
            for row in 0..m {
                for col in 0..k {
                    noised_a.push(raw_a[row * k + col] + noise.a[row][col]);
                }
            }

            let mut noised_b = Vec::with_capacity(k * n);
            for kk in 0..k {
                for col in 0..n {
                    noised_b.push(raw_bt[col * k + kk] + noise.b[col][kk]);
                }
            }

            let noise_a_flat: Vec<i8> = noise.a.iter().flat_map(|row| row.iter().copied()).collect();
            let noise_bt_flat: Vec<i8> = noise.b.iter().flat_map(|col| col.iter().copied()).collect();
            let raw_a_bytes = i8_bytes(&raw_a);
            let raw_bt_bytes = i8_bytes(&raw_bt);
            let noise_a_bytes = i8_bytes(&noise_a_flat);
            let noise_bt_bytes = i8_bytes(&noise_bt_flat);
            let noised_a_bytes = i8_bytes(&noised_a);
            let noised_b_bytes = i8_bytes(&noised_b);

            records.push(NoiseVector {
                id: format!("g2_{m}x{n}x{k}_v{variant}"),
                m,
                n,
                k,
                rank: RANK,
                rng_seed: format!("{seed:016x}"),
                a_noise_seed: hex::encode(a_seed),
                b_noise_seed: hex::encode(b_seed),
                rows_sha256: sha_hex(&u32_bytes(&rows)),
                cols_sha256: sha_hex(&u32_bytes(&cols)),
                raw_a_sha256: sha_hex(&raw_a_bytes),
                raw_bt_sha256: sha_hex(&raw_bt_bytes),
                noise_a_sha256: sha_hex(&noise_a_bytes),
                noise_bt_sha256: sha_hex(&noise_bt_bytes),
                noised_a_sha256: sha_hex(&noised_a_bytes),
                noised_b_sha256: sha_hex(&noised_b_bytes),
                raw_a_prefix_hex: prefix_hex(&raw_a_bytes),
                raw_bt_prefix_hex: prefix_hex(&raw_bt_bytes),
                noise_a_prefix_hex: prefix_hex(&noise_a_bytes),
                noise_bt_prefix_hex: prefix_hex(&noise_bt_bytes),
                noised_a_prefix_hex: prefix_hex(&noised_a_bytes),
                noised_b_prefix_hex: prefix_hex(&noised_b_bytes),
            });
        }
    }
    let mut jsonl = String::new();
    for record in records {
        jsonl.push_str(&serde_json::to_string(&record)?);
        jsonl.push('\n');
    }
    fs::write(out_dir.join("noise_vectors.jsonl"), jsonl)?;
    Ok(())
}

fn emit_blake3_vectors(out_dir: &PathBuf) -> anyhow::Result<()> {
    let mut records = Vec::new();
    for id in 0..16u64 {
        let mut rng = Rng::new(0xB3A5_EED0_0000_0000 ^ id);
        let key = make_seed(&mut rng);
        let mut block = [0u8; 64];
        for b in &mut block {
            *b = rng.byte();
        }
        let pearl_digest = blake3_digest(&block, Some(key));
        let crate_digest = *blake3::Hasher::new_keyed(&key).update(&block).finalize().as_bytes();
        assert_eq!(pearl_digest, crate_digest);
        records.push(Blake3Vector {
            id: format!("single_block_keyed_{id:02}"),
            key_hex: hex::encode(key),
            block_hex: hex::encode(block),
            pearl_digest_hex: hex::encode(pearl_digest),
            crate_digest_hex: hex::encode(crate_digest),
        });
    }
    fs::write(out_dir.join("blake3_single_block_keyed.json"), serde_json::to_string_pretty(&records)?)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let out = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "vectors".to_string()));
    fs::create_dir_all(&out)?;
    emit_noise_vectors(&out)?;
    emit_blake3_vectors(&out)?;
    Ok(())
}
