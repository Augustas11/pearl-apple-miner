// Prints STARK degree_bits / expected rows for the v1 production pattern, using zk-pow's own
// CompiledPublicParams (same code path generate_proof uses: prove.rs:62). Usage: degbits m n k [t_rows t_cols]
use zk_pow::api::proof::{IncompleteBlockHeader, MMAType, MiningConfiguration, PeriodicPattern, PublicProofParams, SeedDerivation};

fn main() {
    let a: Vec<u32> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let (m, n, k) = (a[0], a[1], a[2]);
    let (tr, tc) = (a.get(3).copied().unwrap_or(0), a.get(4).copied().unwrap_or(0));
    let cfg = MiningConfiguration {
        common_dim: k,
        rank: 128,
        mma_type: MMAType::Int7xInt7ToInt32,
        rows_pattern: PeriodicPattern::from_list(&[0, 8, 64, 72]).unwrap(),
        cols_pattern: PeriodicPattern::from_list(&[0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51]).unwrap(),
        moe: None,
    };
    let hdr = IncompleteBlockHeader { version: 0, prev_block: [0; 32], merkle_root: [0; 32], timestamp: 0, nbits: 0x1e010000 };
    let p = PublicProofParams::new_dummy(hdr, SeedDerivation::Salted, cfg, m, n, tr, tc);
    let (c, _, _) = p.compile();
    println!("m={m} n={n} k={k} t=({tr},{tc}) expected_rows={} degree_bits={}", c.expected_num_rows(), c.degree_bits());
}
