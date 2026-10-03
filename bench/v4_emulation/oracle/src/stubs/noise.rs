// SPDX-License-Identifier: ISC
// Derived from Pearl zk-pow (ISC), pearl-research-labs/pearl commit 25695462416f0eb069abe7a515d188882078bf70 (fp8-scheme).
// Copyright (c) 2025-2026 Pearl Research Labs; Copyright (c) 2015-2016 The Decred developers.
// Verbatim copies of the parts named below; see THIRD_PARTY_NOTICES.md.
//! Stub of `zk-pow/src/api/fp8/noise.rs` (Pearl, fp8-scheme 2569546).
//! `OperandNoise`, `sample_line` and `normalize_line` are copied verbatim except
//! that the keyed-BLAKE3 key is passed in directly instead of being derived from
//! the block-header seed via `transcript::subkey` (the transcript/header plumbing
//! is not compiled here). The resulting noise has the same distribution as a
//! real job's noise; it is simply not tied to a real header.

use crate::api::fp8::compute::{bf16_div, bf16_mul};
use crate::api::fp8::dtype::{bf16_to_f32, f32_to_bf16, f32_to_fp8_e4m3};
use crate::api::fp8::quantization::NOISE_TARGET_NORM;

const INT_SQRT_PREC: u64 = 32;

pub struct OperandNoise {
    pub e: Vec<u8>,
    pub f: Vec<u8>,
}

#[repr(u8)]
#[derive(Clone, Copy)]
pub enum Side {
    A = 0,
    B = 1,
}

#[repr(u8)]
#[derive(Clone, Copy)]
pub enum NoiseFactor {
    E = 0,
    F = 1,
}

pub fn sample_line(key: &[u8; 32], side: Side, factor: NoiseFactor, line: u32, rank: u16) -> Vec<u8> {
    let mut material = Vec::with_capacity(64);
    material.push(side as u8);
    material.push(factor as u8);
    material.extend_from_slice(&line.to_le_bytes());
    assert!(material.len() <= 64, "noise line material must fit one BLAKE3 block");
    material.resize(64, 0);

    let mut bytes = vec![0u8; usize::from(rank)];
    let mut hasher = blake3::Hasher::new_keyed(key);
    hasher.update(&material);
    hasher.finalize_xof().fill(&mut bytes);

    normalize_line(&bytes)
}

fn normalize_line(bytes: &[u8]) -> Vec<u8> {
    let x: Vec<i64> = bytes
        .iter()
        .map(|&b| {
            let sign = 1 - 2 * ((b >> 7) as i64); // +1 (bit 7 = 0) or -1
            let magnitude = ((b & 0x7F) as i64) + 1; // uniform in [1, 128], never 0
            sign * magnitude
        })
        .collect();

    let sumsq: u64 = x.iter().map(|&xi| (xi * xi) as u64).sum();
    let norm_scaled = (sumsq * (INT_SQRT_PREC * INT_SQRT_PREC)).isqrt();

    let numer = f32_to_bf16((NOISE_TARGET_NORM * INT_SQRT_PREC as f64) as f32).expect("8192 is representable");
    let denom = f32_to_bf16(norm_scaled as f32).expect("norm_scaled < 2^24 is representable");
    let scale = bf16_div(numer, denom).expect("noise-line scale is finite");

    x.iter()
        .map(|&xi| {
            let xb = f32_to_bf16(xi as f32).expect("|x_i| <= 128 is representable in bf16");
            let entry = bf16_mul(xb, scale).expect("noise entry is finite");
            f32_to_fp8_e4m3(bf16_to_f32(entry)).expect("noise entry is finite")
        })
        .collect()
}
