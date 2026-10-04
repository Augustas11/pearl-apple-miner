// SPDX-License-Identifier: Apache-2.0
// Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
use anyhow::{Context, Result, bail, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use pmkcore_v4::{b200_matmul_codes, create_grid_b200_seeded, diagnostic_headers};
use rayon::prelude::*;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Serialize)]
struct Metadata {
    m: u32,
    n: u32,
    k: u32,
    seed_hex: String,
    proposed_header_b64: String,
    ancestor_header_b64: String,
    ancestor_chain_b64: String,
    rank: u32,
    device: &'static str,
    layout: &'static str,
    files: Vec<&'static str>,
}

fn hex32(s: &str) -> Result<[u8; 32]> {
    ensure!(s.len() == 64, "seed must be 32 bytes of hex");
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16)?;
    }
    Ok(out)
}

fn bytes_u16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn write(path: impl AsRef<Path>, bytes: &[u8]) -> Result<()> {
    fs::write(path.as_ref(), bytes).with_context(|| format!("write {}", path.as_ref().display()))
}

fn generate(args: &[String]) -> Result<()> {
    ensure!(
        args.len() == 7,
        "usage: pmkcore-v4-oracle gen <m> <n> <k> <seed-hex32> <outdir>"
    );
    let m: u32 = args[2].parse()?;
    let n: u32 = args[3].parse()?;
    let k: u32 = args[4].parse()?;
    let seed = hex32(&args[5])?;
    let out = PathBuf::from(&args[6]);
    fs::create_dir_all(&out)?;

    let (proposed, ancestor) = diagnostic_headers(seed);
    let mut job = create_grid_b200_seeded(
        &proposed.to_bytes(),
        &ancestor.to_bytes(),
        &[],
        m,
        n,
        k,
        Some(seed),
    )
    .map_err(|e| anyhow::anyhow!("create diagnostic job failed: {e:?}"))?;
    let payload = job
        .fixture_payload()
        .map_err(|e| anyhow::anyhow!("prepare oracle fixture failed: {e:?}"))?;

    write(out.join("a.bin"), &payload.a_noised)?;
    write(out.join("b.bin"), &payload.b_noised)?;
    write(out.join("a_noise_e.bin"), &payload.a_noise_e)?;
    write(out.join("a_noise_f.bin"), &payload.a_noise_f)?;
    write(out.join("b_noise_e.bin"), &payload.b_noise_e)?;
    write(out.join("b_noise_f.bin"), &payload.b_noise_f)?;
    write(out.join("a_alpha.bf16"), &bytes_u16(&payload.a_alpha))?;
    write(out.join("a_beta.bf16"), &bytes_u16(&payload.a_beta))?;
    write(out.join("a_l2.bf16"), &bytes_u16(&payload.a_l2))?;
    write(out.join("b_alpha.bf16"), &bytes_u16(&payload.b_alpha))?;
    write(out.join("b_beta.bf16"), &bytes_u16(&payload.b_beta))?;
    write(out.join("b_l2.bf16"), &bytes_u16(&payload.b_l2))?;

    let meta = Metadata {
        m,
        n,
        k,
        seed_hex: args[5].clone(),
        proposed_header_b64: STANDARD.encode(proposed.to_bytes()),
        ancestor_header_b64: STANDARD.encode(ancestor.to_bytes()),
        ancestor_chain_b64: STANDARD.encode([]),
        rank: 32,
        device: "B200",
        layout: "AxisPattern([(4,Fold),(4,Blake)]) x AxisPattern([(4,Fold),(4,Blake)])",
        files: vec![
            "a.bin",
            "b.bin",
            "a_noise_e.bin",
            "a_noise_f.bin",
            "b_noise_e.bin",
            "b_noise_f.bin",
            "a_alpha.bf16",
            "a_beta.bf16",
            "a_l2.bf16",
            "b_alpha.bf16",
            "b_beta.bf16",
            "b_l2.bf16",
        ],
    };
    write(
        out.join("metadata.json"),
        serde_json::to_string_pretty(&meta)?.as_bytes(),
    )?;
    println!(
        "gen: wrote deterministic v4 B200 fixture m={m} n={n} k={k} to {}",
        out.display()
    );
    Ok(())
}

fn ref_(args: &[String]) -> Result<()> {
    ensure!(
        args.len() == 6,
        "usage: pmkcore-v4-oracle ref <dir> <m> <n> <k>"
    );
    let dir = PathBuf::from(&args[2]);
    let m: usize = args[3].parse()?;
    let n: usize = args[4].parse()?;
    let k: usize = args[5].parse()?;
    let a = fs::read(dir.join("a.bin"))?;
    let b = fs::read(dir.join("b.bin"))?;
    ensure!(a.len() == m * k, "a.bin length {} != {}", a.len(), m * k);
    ensure!(b.len() == n * k, "b.bin length {} != {}", b.len(), n * k);
    let starts: Vec<usize> = (0..m).step_by(16).collect();
    let started = Instant::now();
    let mut parts: Vec<(usize, usize, Vec<f32>, f64)> = starts
        .into_par_iter()
        .map(|r0| -> Result<(usize, usize, Vec<f32>, f64)> {
            let r1 = (r0 + 16).min(m);
            let chunk_started = Instant::now();
            let part = b200_matmul_codes(&a[r0 * k..r1 * k], &b, r1 - r0, n, k)?;
            Ok((r0, r1, part, chunk_started.elapsed().as_secs_f64()))
        })
        .collect::<Result<Vec<_>>>()?;
    parts.sort_by_key(|(r0, _, _, _)| *r0);
    let mut out = Vec::with_capacity(m * n * 4);
    for (r0, r1, part, secs) in parts {
        let gmac = ((r1 - r0) as f64 * n as f64 * k as f64) / 1.0e9;
        println!("ref: rows {r0}..{r1}/{m} {secs:.3}s {gmac:.3} GMAC");
        for x in part {
            out.extend_from_slice(&x.to_bits().to_le_bytes());
        }
    }
    write(dir.join("c_b200.bin"), &out)?;
    let secs = started.elapsed().as_secs_f64();
    let gmac = (m as f64 * n as f64 * k as f64) / 1.0e9;
    println!(
        "ref: wrote {} cells to {} in {:.3}s ({:.3} GMAC, {:.3} GMAC/s)",
        m * n,
        dir.join("c_b200.bin").display(),
        secs,
        gmac,
        if secs > 0.0 { gmac / secs } else { 0.0 }
    );
    Ok(())
}

fn selftest() -> Result<()> {
    let seed = [7u8; 32];
    let (proposed, ancestor) = diagnostic_headers(seed);
    let job = create_grid_b200_seeded(
        &proposed.to_bytes(),
        &ancestor.to_bytes(),
        &[],
        16,
        16,
        1024,
        Some(seed),
    )
    .map_err(|e| anyhow::anyhow!("create job failed: {e:?}"))?;
    let proof = job
        .build_plain_proof(0, 0)
        .map_err(|e| anyhow::anyhow!("build proof failed: {e:?}"))?;
    pmkcore_v4::api::verify::verify_plain_proof(&proposed, &proof, Some(0x207f_ffff))?;
    let bytes = proof.to_bytes()?;
    let mut corrupt = bytes.clone();
    corrupt[0] ^= 1;
    ensure!(pmkcore_v4::verify_proof_bytes(&proposed, &corrupt, Some(0x207f_ffff)).is_err());
    println!("selftest: proof accepted and byte mutation rejected");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("gen") => generate(&args),
        Some("ref") => ref_(&args),
        Some("selftest") => selftest(),
        _ => bail!("usage: pmkcore-v4-oracle gen|ref|selftest ..."),
    }
}
