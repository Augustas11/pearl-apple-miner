// SPDX-License-Identifier: ISC
// Derived from Pearl zk-pow (ISC), pearl-research-labs/pearl commit 25695462416f0eb069abe7a515d188882078bf70 (fp8-scheme).
// Copyright (c) 2025-2026 Pearl Research Labs; Copyright (c) 2015-2016 The Decred developers.
// Verbatim copies of the parts named below; see THIRD_PARTY_NOTICES.md.
//! Stub of `zk-pow/src/api/fp8/public_params.rs`: only the `Device` enum and its
//! inherent impl, copied verbatim from Pearl (fp8-scheme 2569546, lines 140-195).
//! The rest of public_params (job/transcript plumbing) is not needed by the
//! matmul, quantization and jackpot-policy files this oracle compiles.

/// `pB`'s device byte.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[repr(u8)]
pub enum Device {
    H100 = 0,
    B200 = 1,
}

impl Device {
    pub const ALL: [Self; 2] = [Self::H100, Self::B200];

    pub const fn lg2_delta(self) -> i32 {
        match self {
            Self::H100 => 0,
            Self::B200 => -1,
        }
    }

    pub const fn delta(self) -> f64 {
        f64::from_bits(((self.lg2_delta() + 1023) as u64) << 52)
    }

    pub const fn fp8_window_bits(self) -> u32 {
        match self {
            Self::H100 => 13,
            Self::B200 => 25,
        }
    }

    pub const fn liveness_code_shift(self) -> u64 {
        ((3 + self.lg2_delta()) * 128) as u64
    }

    pub const fn sigma_encoding_offset(self) -> u64 {
        (14 + self.lg2_delta()) as u64
    }
}
