// SPDX-License-Identifier: Apache-2.0
// Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
use pmkcore_v4::{SUPPORTED_K, create_grid_b200_seeded, diagnostic_headers, verify_proof_bytes};
use primitive_types::U256;

fn seed(byte: u8) -> [u8; 32] {
    [byte; 32]
}

fn job(k: u32) -> pmkcore_v4::Job {
    let s = seed((k / 1024) as u8);
    let (proposed, ancestor) = diagnostic_headers(s);
    create_grid_b200_seeded(
        &proposed.to_bytes(),
        &ancestor.to_bytes(),
        &[],
        16,
        16,
        k,
        Some(s),
    )
    .expect("grid job")
}

#[test]
fn supported_k_policy_accepts_g3_set_and_rejects_65536() {
    assert_eq!(SUPPORTED_K, [1024, 4096, 16384]);
    for &k in &SUPPORTED_K {
        let mut j = job(k);
        j.prepare_oracle_noised().expect("policy/quantization");
        let proof = j.build_plain_proof(0, 0).expect("plain proof");
        let _bytes = proof.to_bytes().expect("proof bytes");
    }

    let s = seed(9);
    let (proposed, ancestor) = diagnostic_headers(s);
    assert!(
        create_grid_b200_seeded(
            &proposed.to_bytes(),
            &ancestor.to_bytes(),
            &[],
            16,
            16,
            65536,
            Some(s),
        )
        .is_err()
    );
}

fn assert_typed_mutation_rejects(
    proposed: &pmkcore_v4::api::primitives::IncompleteBlockHeader,
    proof: &pmkcore_v4::api::fp8::plain_proof::PlainProofV4,
    label: &str,
    mutate: impl FnOnce(&mut pmkcore_v4::api::fp8::plain_proof::PlainProofV4),
) {
    let mut bad = proof.clone();
    mutate(&mut bad);
    let bytes = bad.to_bytes().expect("mutated proof still serializes");
    assert!(
        verify_proof_bytes(proposed, &bytes, Some(0x207f_ffff)).is_err(),
        "typed mutation must reject: {label}"
    );
}

#[test]
fn proof_verifies_and_mutations_reject() {
    let s = seed(1);
    let (proposed, ancestor) = diagnostic_headers(s);
    let j = create_grid_b200_seeded(
        &proposed.to_bytes(),
        &ancestor.to_bytes(),
        &[],
        16,
        16,
        1024,
        Some(s),
    )
    .expect("job");
    let proof = j.build_plain_proof(0, 0).expect("plain proof");
    let bytes = proof.to_bytes().expect("proof bytes");
    verify_proof_bytes(&proposed, &bytes, Some(0x207f_ffff)).expect("honest proof verifies");
    for idx in [0usize, 8, bytes.len() / 2, bytes.len() - 1] {
        let mut bad = bytes.clone();
        bad[idx] ^= 0x41;
        assert!(
            verify_proof_bytes(&proposed, &bad, Some(0x207f_ffff)).is_err(),
            "mutation at {idx}"
        );
    }
}

#[test]
fn field_oriented_plain_proof_mutations_reject() {
    let s = seed(5);
    let (proposed, ancestor) = diagnostic_headers(s);
    let j = create_grid_b200_seeded(
        &proposed.to_bytes(),
        &ancestor.to_bytes(),
        &[],
        16,
        16,
        1024,
        Some(s),
    )
    .expect("job");
    let proof = j.build_plain_proof(0, 0).expect("plain proof");
    let bytes = proof.to_bytes().expect("proof bytes");
    verify_proof_bytes(&proposed, &bytes, Some(0x207f_ffff)).expect("honest proof verifies");

    assert_typed_mutation_rejects(&proposed, &proof, "job values / geometry", |p| {
        p.job.operands.a.num_rows = 32;
    });
    assert_typed_mutation_rejects(&proposed, &proof, "commitment root", |p| {
        p.values.a.proof.root[0] ^= 1;
    });
    assert_typed_mutation_rejects(&proposed, &proof, "noise config / rank", |p| {
        p.job.common.r = 16;
    });
    assert_typed_mutation_rejects(&proposed, &proof, "prequant value leaf", |p| {
        p.values.a.proof.leaf_data[0][0] ^= 1;
    });
    assert_typed_mutation_rejects(&proposed, &proof, "matrix row indices", |p| {
        p.values.a.row_indices[0] = 1;
        p.scales.a.row_indices[0] = 1;
    });
    assert_typed_mutation_rejects(&proposed, &proof, "ancestor header", |p| {
        p.job.ancestor_header.proof_commitment[0] ^= 1;
    });
    assert_typed_mutation_rejects(&proposed, &proof, "ancestor chain", |p| {
        p.ancestor_chain.push(p.job.ancestor_header);
    });
    assert_typed_mutation_rejects(&proposed, &proof, "merkle proof path", |p| {
        if let Some(first) = p.values.a.proof.siblings.first_mut() {
            first[0] ^= 1;
        } else {
            p.values.a.proof.total_leaves += 1;
        }
    });

    let mut bad_header = proposed;
    bad_header.merkle_root[0] ^= 1;
    assert!(
        verify_proof_bytes(&bad_header, &bytes, Some(0x207f_ffff)).is_err(),
        "proposed header mutation must reject"
    );
}

#[test]
fn ancestry_rejection() {
    let s = seed(3);
    let (proposed, ancestor) = diagnostic_headers(s);
    let mut bad_ancestor = ancestor;
    bad_ancestor.incomplete.prev_block[0] ^= 1;
    assert!(
        create_grid_b200_seeded(
            &proposed.to_bytes(),
            &bad_ancestor.to_bytes(),
            &[],
            16,
            16,
            1024,
            Some(s),
        )
        .is_err()
    );
}

#[test]
fn gpu_descriptor_has_factors_before_noised_oracle() {
    let j = job(1024);
    let d = j.gpu_desc();
    assert!(!d.a_values.is_null());
    assert!(!d.a_noise_e.is_null());
    assert!(!d.a_alpha.is_null());
    assert!(
        d.a_noised.is_null(),
        "full noised codes require explicit oracle preparation"
    );
}

#[test]
fn bound_helper_matches_upstream_predicate_edges() {
    let j = job(1024);
    for nbits in [0x0100_3456, 0x1d00_ffff, 0x207f_ffff, 0x2200_ffff] {
        let bound = j.bound_for_nbits(nbits);
        pmkcore_v4::api::proof_utils::check_jackpot_difficulty(&bound, nbits, 16, 16, 1024)
            .expect("returned bound must satisfy upstream predicate");
        let mut plus = U256::from_little_endian(&bound);
        if plus < U256::MAX {
            plus += U256::from(1u8);
            let mut plus_bytes = [0u8; 32];
            plus.to_little_endian(&mut plus_bytes);
            assert!(
                pmkcore_v4::api::proof_utils::check_jackpot_difficulty(
                    &plus_bytes,
                    nbits,
                    16,
                    16,
                    1024
                )
                .is_err(),
                "bound+1 should reject when not saturated for nbits {nbits:08x}"
            );
        }
    }
}
