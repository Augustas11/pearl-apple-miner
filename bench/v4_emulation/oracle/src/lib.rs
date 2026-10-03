//! Bit-exact FP8 (certificate-v4) oracle built from Pearl's OWN source files.
//!
//! The files below are compiled verbatim from `vendor/pearl-fp8/zk-pow/src`
//! (branch `fp8-scheme`, HEAD 2569546) via `#[path]`, so `B200::matmul_fp8`,
//! the quantization path and the jackpot policy are Pearl's code, not a
//! re-implementation. Their `pub(crate)` items are reachable because they
//! live inside this crate's module tree. Only two small modules are stubbed
//! (see `stubs/`): `public_params` (just the `Device` enum, copied) and
//! `noise` (the line normalizer copied; the BLAKE3 key is a fixed test key
//! instead of the header-derived subkey).
#![allow(dead_code, unused_imports, unexpected_cfgs, clippy::all)]

pub mod api {
    pub mod fp8 {
        #[path = "../../../../../../vendor/pearl-fp8/zk-pow/src/api/fp8/compute.rs"]
        pub mod compute;
        #[path = "../../../../../../vendor/pearl-fp8/zk-pow/src/api/fp8/dtype.rs"]
        pub mod dtype;
        #[path = "../../../../../../vendor/pearl-fp8/zk-pow/src/api/fp8/jackpot_policy.rs"]
        pub mod jackpot_policy;
        #[path = "../../stubs/noise.rs"]
        pub mod noise;
        /// Pearl's public prequant module, taken from the zk-pow crate itself.
        pub mod prequant {
            pub use zk_pow::api::fp8::prequant::*;
        }
        #[path = "../../stubs/public_params.rs"]
        pub mod public_params;
        #[path = "../../../../../../vendor/pearl-fp8/zk-pow/src/api/fp8/quantization.rs"]
        pub mod quantization;
        #[path = "../../../../../../vendor/pearl-fp8/zk-pow/src/api/fp8/utils.rs"]
        pub mod utils;
    }
    #[path = "../../../../../vendor/pearl-fp8/zk-pow/src/api/layout.rs"]
    pub mod layout;
}

pub mod circuit {
    pub mod fp8 {
        #[path = "../../../../../../vendor/pearl-fp8/zk-pow/src/circuit/fp8/unpredictability.rs"]
        pub mod unpredictability;
    }
}

pub mod families;
pub mod cli;
