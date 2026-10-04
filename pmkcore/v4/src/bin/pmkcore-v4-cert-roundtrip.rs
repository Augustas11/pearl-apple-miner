// SPDX-License-Identifier: Apache-2.0
// Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
use anyhow::{Context, Result, bail, ensure};
use plonky2::util::timing::TimingTree;
use pmkcore_v4::api::fp8::plain_proof::PlainProofV4;
use pmkcore_v4::api::fp8::public_params::PublicParams;
use pmkcore_v4::api::fp8::zk::Fp8Verifier;
use pmkcore_v4::api::primitives::IncompleteBlockHeader;
use pmkcore_v4::{HEADER_LEN, pmkcore_v4_build_certificate};
use serde::Deserialize;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Deserialize)]
struct Manifest {
    proposed_header: String,
}

fn hex_bytes(s: &str) -> Result<Vec<u8>> {
    ensure!(s.len().is_multiple_of(2), "hex length must be even");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).context("hex byte"))
        .collect()
}

fn read_inputs(dir: &Path) -> Result<([u8; HEADER_LEN], Vec<u8>, PlainProofV4)> {
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
    let header_vec = hex_bytes(&manifest.proposed_header)?;
    let proposed_header: [u8; HEADER_LEN] = header_vec
        .try_into()
        .map_err(|_| anyhow::anyhow!("proposed_header must be {HEADER_LEN} bytes"))?;
    let plain_bytes = std::fs::read(dir.join("plain_proof.bin"))?;
    let plain = PlainProofV4::from_bytes(&plain_bytes)?;
    Ok((proposed_header, plain_bytes, plain))
}

fn verify(
    proposed: &IncompleteBlockHeader,
    chain: &[pmkcore_v4::api::primitives::BlockHeader],
    public_data: &[u8],
    proof_data: &[u8],
) -> Result<()> {
    let params = PublicParams::from_bytes(public_data)?;
    let mut timing = TimingTree::default();
    let verifier = Fp8Verifier::generate(&params, proposed, &mut timing)?;
    verifier.verify_block(proposed, chain, public_data, proof_data)
}

fn main() -> Result<()> {
    eprintln!("stage: start");
    let _ = std::io::stderr().flush();
    let dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("libpmk/resources/v4_probe"));
    let (proposed_bytes, plain_bytes, plain) = read_inputs(&dir)?;
    eprintln!(
        "stage: loaded inputs plain_bytes={} chain_len={}",
        plain_bytes.len(),
        plain.ancestor_chain.len()
    );
    let _ = std::io::stderr().flush();
    let proposed = IncompleteBlockHeader::from_bytes(&proposed_bytes)?;

    let mut public = vec![0u8; 4096];
    let mut proof = vec![0u8; 64 * 1024 * 1024];
    let mut public_len = 0u64;
    let mut proof_len = 0u64;

    eprintln!("stage: calling pmkcore_v4_build_certificate");
    let _ = std::io::stderr().flush();
    let prove_started = Instant::now();
    let rc = unsafe {
        pmkcore_v4_build_certificate(
            proposed_bytes.as_ptr(),
            plain_bytes.as_ptr(),
            plain_bytes.len() as u64,
            public.as_mut_ptr(),
            public.len() as u64,
            &mut public_len,
            proof.as_mut_ptr(),
            proof.len() as u64,
            &mut proof_len,
        )
    };
    ensure!(rc == 0, "pmkcore_v4_build_certificate failed rc={rc}");
    public.truncate(public_len as usize);
    proof.truncate(proof_len as usize);
    eprintln!(
        "stage: certificate built rc=0 public={} proof={}",
        public.len(),
        proof.len()
    );
    let _ = std::io::stderr().flush();
    println!(
        "certificate: public={} proof={} prove_sec={:.3}",
        public.len(),
        proof.len(),
        prove_started.elapsed().as_secs_f64()
    );

    let verify_started = Instant::now();
    verify(&proposed, &plain.ancestor_chain, &public, &proof)
        .context("honest certificate verify")?;
    println!(
        "verify: honest ok verify_sec={:.3}",
        verify_started.elapsed().as_secs_f64()
    );

    let mut bad_public = public.clone();
    let idx = bad_public.len().saturating_sub(1);
    bad_public[idx] ^= 1;
    match verify(&proposed, &plain.ancestor_chain, &bad_public, &proof) {
        Ok(()) => bail!("corrupt public_data unexpectedly verified"),
        Err(err) => println!("verify: corrupt public rejected: {err}"),
    }

    let mut bad_proof = proof.clone();
    let idx = bad_proof.len() / 2;
    bad_proof[idx] ^= 1;
    match verify(&proposed, &plain.ancestor_chain, &public, &bad_proof) {
        Ok(()) => bail!("corrupt proof_data unexpectedly verified"),
        Err(err) => println!("verify: corrupt proof rejected: {err}"),
    }

    Ok(())
}
