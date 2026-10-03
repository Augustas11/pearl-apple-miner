//! Operand families for the v4 emulation experiment.
//!
//! Policy families go through Pearl's real miner path: int8 values + per-block
//! (BLOCK_SIZE = 8) BF16 scales -> `exact_norms` / `open_prequant` (Pearl's zk-pow
//! crate) -> rank-32 BLAKE3 noise lines (`stubs/noise.rs`) -> Pearl's
//! `Fp8E4M3Quant::noisy_quantize` (verbatim, B200 noise atom) -> E4M3 codes.
//! Adversarial families write E4M3 codes directly (they need not pass policy).

use anyhow::{Result, bail};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;

use crate::api::fp8::dtype::f32_to_bf16;
use crate::api::fp8::noise::{NoiseFactor, OperandNoise, Side, sample_line};
use crate::api::fp8::prequant::{BLOCK_SIZE, exact_norms, open_prequant};
use crate::api::fp8::public_params::Device;
use crate::api::fp8::quantization::{BuiltRows, Fp8E4M3Quant};

pub const RANK: u16 = 32; // PublicParams: "Rank must be exactly 32"

pub const POLICY_FAMILIES: [&str; 3] = ["const", "uniform", "gauss"];
pub const ADV_FAMILIES: [&str; 2] = ["adv_uniform", "adv_edge"];

/// Miner-chosen clean operand in Pearl's prequant format.
pub struct Clean {
    pub ints: Vec<i8>,
    pub scales: Vec<u16>,
    pub rows: usize,
    pub k: usize,
}

fn gauss(rng: &mut ChaCha20Rng) -> f64 {
    // Box-Muller.
    let u1: f64 = rng.random::<f64>().max(1e-300);
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

pub fn clean_family(name: &str, rows: usize, k: usize, seed: u64) -> Result<Clean> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let nb = k / BLOCK_SIZE;
    let one = f32_to_bf16(1.0)?;
    let (ints, scales): (Vec<i8>, Vec<u16>) = match name {
        // KB "miner-friendly": constant magnitude, random sign.
        "const" => (
            (0..rows * k).map(|_| if rng.random::<bool>() { 64 } else { -64 }).collect(),
            vec![one; rows * nb],
        ),
        // Uniform int8 values, unit scales.
        "uniform" => ((0..rows * k).map(|_| rng.random_range(-127i32..=127) as i8).collect(), vec![one; rows * nb]),
        // LLM-like: Gaussian ints (std 24) with log-normal per-block scales.
        "gauss" => {
            let ints = (0..rows * k).map(|_| (gauss(&mut rng) * 24.0).round().clamp(-127.0, 127.0) as i8).collect();
            let mut scales = Vec::with_capacity(rows * nb);
            for _ in 0..rows * nb {
                scales.push(f32_to_bf16((0.5 * gauss(&mut rng)).exp2() as f32)?);
            }
            (ints, scales)
        }
        _ => bail!("unknown policy family {name}"),
    };
    Ok(Clean { ints, scales, rows, k })
}

/// Pearl's miner path for one side: returns (clean BF16 rows, BuiltRows).
pub fn build_side(clean: &Clean, side: Side, key: &[u8; 32], device: Device) -> Result<(Vec<u16>, BuiltRows)> {
    let (rows, k) = (clean.rows, clean.k);
    let norms = exact_norms(&clean.ints, &clean.scales, rows, k, BLOCK_SIZE)?;
    let opened = open_prequant(&clean.ints, &clean.scales, rows, k, BLOCK_SIZE)?;
    let f: Vec<u8> = (0..k as u32)
        .into_par_iter()
        .flat_map_iter(|i| sample_line(key, side, NoiseFactor::F, i, RANK))
        .collect();
    let quant = Fp8E4M3Quant::new(device);
    const CH: usize = 16;
    let parts: Vec<Result<BuiltRows>> = (0..rows.div_ceil(CH))
        .into_par_iter()
        .map(|c| {
            let r0 = c * CH;
            let r1 = (r0 + CH).min(rows);
            let e: Vec<u8> = (r0..r1).flat_map(|r| sample_line(key, side, NoiseFactor::E, r as u32, RANK)).collect();
            let noise = OperandNoise { e, f: f.clone() };
            quant.noisy_quantize(&opened[r0 * k..r1 * k], &noise, &norms[r0..r1])
        })
        .collect();
    let mut out = BuiltRows {
        noised_part: Vec::with_capacity(rows * k),
        alpha: vec![],
        beta: vec![],
        l2: vec![],
    };
    for p in parts {
        let p = p?;
        out.noised_part.extend(p.noised_part);
        out.alpha.extend(p.alpha);
        out.beta.extend(p.beta);
        out.l2.extend(p.l2);
    }
    Ok((opened, out))
}

fn is_nan_code(c: u8) -> bool {
    c & 0x7F == 0x7F
}

fn rand_code(rng: &mut ChaCha20Rng) -> u8 {
    loop {
        let c: u8 = rng.random();
        if !is_nan_code(c) {
            return c;
        }
    }
}

/// Adversarial E4M3 codes (rows x k). `adv_edge` picks a mode per (row, 32-group)
/// so mixed-binade, zero/-0, saturated and cancelling groups appear in the same row.
pub fn adversarial(name: &str, rows: usize, k: usize, seed: u64, is_b: bool) -> Result<Vec<u8>> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed ^ if is_b { 0xB0B0 } else { 0xA0A0 });
    let mut out = vec![0u8; rows * k];
    match name {
        "adv_uniform" => out.iter_mut().for_each(|c| *c = rand_code(&mut rng)),
        "adv_edge" => {
            for r in 0..rows {
                for g in 0..k / 32 {
                    let mode = rng.random_range(0..8u32);
                    let base = rand_code(&mut rng);
                    for t in 0..32 {
                        let s: u8 = if rng.random::<bool>() { 0x80 } else { 0 };
                        let c = match mode {
                            0 => rand_code(&mut rng),
                            // mostly +0/-0
                            1 => {
                                if rng.random_range(0..8) == 0 {
                                    rand_code(&mut rng)
                                } else {
                                    s
                                }
                            }
                            // +-448 mixed with subnormals
                            2 => {
                                if rng.random::<bool>() {
                                    0x7E | s
                                } else {
                                    rng.random_range(1..8u8) | s
                                }
                            }
                            // two far binades: exponent field 15 vs 1..3
                            3 => {
                                let e: u8 = if rng.random::<bool>() { 14 + rng.random_range(0..2u8) } else { rng.random_range(1..4u8) };
                                let m: u8 = rng.random_range(0..8u8);
                                let c = (e << 3) | m;
                                if is_nan_code(c) { 0x7E | s } else { c | s }
                            }
                            // cancelling pairs x, -x (A side); B side constant per pair
                            4 => {
                                if is_b {
                                    base
                                } else if t % 2 == 0 {
                                    base
                                } else {
                                    base ^ 0x80
                                }
                            }
                            // one constant code
                            5 => base,
                            // exponent-field-15 only (largest binade), random signs
                            6 => {
                                let c = (15u8 << 3) | rng.random_range(0..7u8);
                                c | s
                            }
                            // subnormals and zeros only
                            _ => rng.random_range(0..8u8) | s,
                        };
                        out[r * k + g * 32 + t] = c;
                    }
                }
            }
        }
        _ => bail!("unknown adversarial family {name}"),
    }
    Ok(out)
}
