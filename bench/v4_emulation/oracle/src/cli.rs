//! v4 emulation oracle CLI (lives in the lib so Pearl's pub(crate) items are reachable).
//!
//!   v4-oracle gen <family> <m> <n> <k> <seed> <outdir>
//!       Writes <outdir>/a.bin (m x k E4M3 codes) and b.bin (n x k codes, B stored
//!       transposed exactly as Pearl's matmul_fp8 expects). Policy families also run
//!       Pearl's JackpotPolicy::evaluate on a few lottery tiles.
//!   v4-oracle ref <dir> <m> <n> <k> [b200|h100]
//!       Pearl's Device::matmul_fp8 (verbatim) -> <dir>/c_<dev>.bin (u32 FP32 bits).
//!   v4-oracle analyze <dir> <m> <n> <k> <cells>
//!       Per (cell, 32-group) census of where B200 arithmetic is non-linear, plus a
//!       check of the linearised identity used by kernel B against Pearl's output.
//!   v4-oracle cmp <c1.bin> <c2.bin>
//!       Bit-exact comparison.

use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;

use crate::api::fp8::jackpot_policy::{JackpotPolicy, OperandStrip};
use crate::api::fp8::noise::Side;
use crate::api::fp8::public_params::Device;
use crate::api::fp8::quantization::BuiltRows;
use crate::api::layout::{AxisPattern, DimType, check_lottery_layout};
use crate::families::{ADV_FAMILIES, POLICY_FAMILIES, adversarial, build_side, clean_family};

fn arg<T: std::str::FromStr>(a: &[String], i: usize, what: &str) -> Result<T> {
    a.get(i)
        .with_context(|| format!("missing <{what}>"))?
        .parse::<T>()
        .map_err(|_| anyhow::anyhow!("bad <{what}>"))
}

fn write_u32(path: &Path, v: &[u32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    fs::write(path, bytes)?;
    Ok(())
}

fn read_u32(path: &Path) -> Result<Vec<u32>> {
    let b = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn slice_built(b: &BuiltRows, r0: usize, r1: usize, k: usize) -> BuiltRows {
    BuiltRows {
        noised_part: b.noised_part[r0 * k..r1 * k].to_vec(),
        alpha: b.alpha[r0..r1].to_vec(),
        beta: b.beta[r0..r1].to_vec(),
        l2: b.l2[r0..r1].to_vec(),
    }
}

fn generate(a: &[String]) -> Result<()> {
    let fam: String = arg(a, 2, "family")?;
    let (m, n, k, seed): (usize, usize, usize, u64) = (arg(a, 3, "m")?, arg(a, 4, "n")?, arg(a, 5, "k")?, arg(a, 6, "seed")?);
    let out: String = arg(a, 7, "outdir")?;
    let out = Path::new(&out);
    fs::create_dir_all(out)?;
    ensure!(k % 32 == 0 && k >= 1024, "Pearl requires 32 | k and k >= 1024");
    let t = Instant::now();
    if POLICY_FAMILIES.contains(&fam.as_str()) {
        let dev = Device::B200;
        let key_a = *blake3::hash(format!("v4-emul A {seed}").as_bytes()).as_bytes();
        let key_b = *blake3::hash(format!("v4-emul B {seed}").as_bytes()).as_bytes();
        let ca = clean_family(&fam, m, k, seed * 2 + 1)?;
        let cb = clean_family(&fam, n, k, seed * 2 + 2)?;
        let (a_clean, a_built) = build_side(&ca, Side::A, &key_a, dev)?;
        let (b_clean, b_built) = build_side(&cb, Side::B, &key_b, dev)?;
        fs::write(out.join("a.bin"), &a_built.noised_part)?;
        fs::write(out.join("b.bin"), &b_built.noised_part)?;
        println!("gen {fam}: m={m} n={n} k={k} seed={seed} ({:.1}s, Pearl quantization path, device=B200)", t.elapsed().as_secs_f64());

        // Jackpot policy on lottery tiles: h=4 x w=64 (256 cells, the minimum tile).
        let rows_pat = AxisPattern::new(&[(4, DimType::Blake)])?;
        let cols_pat = AxisPattern::new(&[(4, DimType::Blake), (16, DimType::Fold)])?;
        check_lottery_layout(&rows_pat, &cols_pat)?;
        let (h, w) = (rows_pat.tile_size() as usize, cols_pat.tile_size() as usize);
        let full = JackpotPolicy::for_device(dev);
        // Checks 1-2 neutralised (as Pearl's own unit tests do) to isolate check 3.
        let mut check3_only = JackpotPolicy::for_device(dev);
        check3_only.eps_idle = 1.0;
        check3_only.sigma_min = 0.0;
        let tiles = 4usize.min(m / h).min(n / w);
        let mut pass = 0;
        let mut pass3 = 0;
        for t in 0..tiles {
            let (r0, c0) = (t * h, t * w);
            let sa = OperandStrip {
                clean: a_clean[r0 * k..(r0 + h) * k].to_vec(),
                built: slice_built(&a_built, r0, r0 + h, k),
            };
            let sb = OperandStrip {
                clean: b_clean[c0 * k..(c0 + w) * k].to_vec(),
                built: slice_built(&b_built, c0, c0 + w, k),
            };
            if full.evaluate(&sa, &sb, k, &rows_pat, &cols_pat)?.is_some() {
                pass += 1;
            }
            if check3_only.evaluate(&sa, &sb, k, &rows_pat, &cols_pat)?.is_some() {
                pass3 += 1;
            }
        }
        // Diagnostics for checks 1/2 (formulas from jackpot_policy.rs docs).
        let sig = |b: &BuiltRows| -> f64 {
            b.alpha
                .iter()
                .zip(&b.l2)
                .map(|(&al, &l2)| dev.delta() * zk_pow::api::fp8::dtype::bf16_to_f32(al) as f64 * zk_pow::api::fp8::dtype::bf16_to_f32(l2) as f64)
                .fold(f64::INFINITY, f64::min)
        };
        println!(
            "policy (Pearl JackpotPolicy::evaluate, B200, tile {h}x{w}, k={k}): full PASS {pass}/{tiles}; check3-only PASS {pass3}/{tiles}; min sigma A={:.2} B={:.2}",
            sig(&a_built),
            sig(&b_built)
        );
    } else if ADV_FAMILIES.contains(&fam.as_str()) {
        fs::write(out.join("a.bin"), adversarial(&fam, m, k, seed, false)?)?;
        fs::write(out.join("b.bin"), adversarial(&fam, n, k, seed, true)?)?;
        println!("gen {fam}: m={m} n={n} k={k} seed={seed} (adversarial codes, not policy-checked)");
    } else {
        bail!("unknown family {fam}");
    }
    // Code-level stats.
    let codes = fs::read(out.join("a.bin"))?;
    let zeros = codes.iter().filter(|&&c| c & 0x7F == 0).count();
    let sub = codes.iter().filter(|&&c| c & 0x78 == 0 && c & 7 != 0).count();
    let sat = codes.iter().filter(|&&c| c & 0x7F == 0x7E).count();
    let mut hist = [0usize; 16];
    for &c in &codes {
        hist[((c >> 3) & 15) as usize] += 1;
    }
    println!(
        "A codes: zero {:.4}% subnormal {:.4}% +-448 {:.4}% exp-field histogram {:?}",
        100.0 * zeros as f64 / codes.len() as f64,
        100.0 * sub as f64 / codes.len() as f64,
        100.0 * sat as f64 / codes.len() as f64,
        hist
    );
    Ok(())
}

fn reference(a: &[String]) -> Result<()> {
    let dir: String = arg(a, 2, "dir")?;
    let (m, n, k): (usize, usize, usize) = (arg(a, 3, "m")?, arg(a, 4, "n")?, arg(a, 5, "k")?);
    let dev = match a.get(6).map(String::as_str).unwrap_or("b200") {
        "b200" => Device::B200,
        "h100" => Device::H100,
        d => bail!("unknown device {d}"),
    };
    let dir = Path::new(&dir);
    let av = fs::read(dir.join("a.bin"))?;
    let bv = fs::read(dir.join("b.bin"))?;
    ensure!(av.len() == m * k && bv.len() == n * k, "operand sizes do not match m,n,k");
    let t = Instant::now();
    const CH: usize = 8;
    let parts: Vec<Result<Vec<f32>>> = (0..m.div_ceil(CH))
        .into_par_iter()
        .map(|c| {
            let r0 = c * CH;
            let r1 = (r0 + CH).min(m);
            dev.matmul_fp8(&av[r0 * k..r1 * k], &bv, None, r1 - r0, n, k)
        })
        .collect();
    let mut out = Vec::with_capacity(m * n);
    for p in parts {
        out.extend(p?.into_iter().map(f32::to_bits));
    }
    let name = if dev == Device::B200 { "c_b200.bin" } else { "c_h100.bin" };
    write_u32(&dir.join(name), &out)?;
    let secs = t.elapsed().as_secs_f64();
    println!(
        "ref {dev:?}: {m}x{n}x{k} in {secs:.1}s ({:.3} GMAC/s on CPU, Pearl matmul_fp8) -> {}",
        (m * n * k) as f64 / secs / 1e9,
        dir.join(name).display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Analysis (independent re-derivation of B200 group semantics, cross-checked
// against Pearl's output bits for every analysed cell).

#[derive(Clone, Copy)]
struct Dec {
    neg: bool,
    e: i32,   // effective exponent field (1..15)
    sig: u32, // 4-bit significand incl. implicit bit (0..15)
}

fn dec(c: u8) -> Dec {
    let ef = ((c >> 3) & 15) as i32;
    let m = (c & 7) as u32;
    Dec {
        neg: c & 0x80 != 0,
        e: ef.max(1),
        sig: if ef != 0 { m | 8 } else { m },
    }
}

/// f32 -> (neg, exponent, 24-bit significand) as Pearl's GFloat (zero => sig 0).
fn gf(x: f32) -> (bool, i32, u32) {
    let b = x.to_bits();
    let ef = ((b >> 23) & 0xFF) as i32;
    let m = b & 0x7FFFFF;
    if ef == 0 && m == 0 {
        return (b >> 31 != 0, -133, 0);
    }
    if ef == 0 { (b >> 31 != 0, 1 - 127, m) } else { (b >> 31 != 0, ef - 127, m | 0x800000) }
}

/// Round-toward-zero of an exact value v * 2^-18 (v: i128) to f32 (no subnormals reachable).
fn rz_f32_units18(v: i128) -> f32 {
    if v == 0 {
        return 0.0;
    }
    let neg = v < 0;
    let mut mag = v.unsigned_abs();
    let width = 128 - mag.leading_zeros() as i32;
    let mut e2 = -18; // value = mag * 2^e2
    if width > 24 {
        mag >>= width - 24;
        e2 += width - 24;
    }
    let x = (mag as f64) * (e2 as f64).exp2();
    let f = x as f32; // exact: mag < 2^24
    if neg { -f } else { f }
}

fn analyze(a: &[String]) -> Result<()> {
    let dir: String = arg(a, 2, "dir")?;
    let (m, n, k, cells): (usize, usize, usize, usize) = (arg(a, 3, "m")?, arg(a, 4, "n")?, arg(a, 5, "k")?, arg(a, 6, "cells")?);
    let dir = Path::new(&dir);
    let av = fs::read(dir.join("a.bin"))?;
    let bv = fs::read(dir.join("b.bin"))?;
    let cref = read_u32(&dir.join("c_b200.bin"))?;
    let ng = k / 32;
    // Per (row, group) exponent bounds over nonzero entries.
    // (max exponent, min exponent, min lowest-set-bit exponent e+ctz(sig), max |sig*2^(e-lsb_lo)|)
    let bounds = |v: &[u8], rows: usize| -> Vec<(i32, i32, i32, u32)> {
        let mut out = vec![(i32::MIN, i32::MAX, i32::MAX, 0u32); rows * ng];
        for r in 0..rows {
            for g in 0..ng {
                let mut hi = i32::MIN;
                let mut lo = i32::MAX;
                let mut lsb = i32::MAX;
                for t in 0..32 {
                    let d = dec(v[r * k + g * 32 + t]);
                    if d.sig != 0 {
                        hi = hi.max(d.e);
                        lo = lo.min(d.e);
                        lsb = lsb.min(d.e + d.sig.trailing_zeros() as i32);
                    }
                }
                let mut mx = 0u32;
                for t in 0..32 {
                    let d = dec(v[r * k + g * 32 + t]);
                    if d.sig != 0 {
                        mx = mx.max(d.sig << (d.e - lsb).max(0) >> (lsb - d.e).max(0));
                    }
                }
                out[r * ng + g] = (hi, lo, lsb, mx);
            }
        }
        out
    };
    let ab = bounds(&av, m);
    let bb = bounds(&bv, n);
    // (min lsb, argmin, 2nd min lsb over other positions, max e, argmax, 2nd max e)
    let two = |v: &[u8], rows: usize| -> Vec<(i32, usize, i32, i32, usize, i32)> {
        let mut out = Vec::with_capacity(rows * ng);
        for r in 0..rows {
            for g in 0..ng {
                let (mut l1, mut p1, mut l2) = (i32::MAX, 99usize, i32::MAX);
                let (mut h1, mut q1, mut h2) = (i32::MIN, 99usize, i32::MIN);
                for t in 0..32 {
                    let d = dec(v[r * k + g * 32 + t]);
                    if d.sig == 0 {
                        continue;
                    }
                    let l = d.e + d.sig.trailing_zeros() as i32;
                    if l < l1 { l2 = l1; l1 = l; p1 = t; } else if l < l2 { l2 = l; }
                    if d.e > h1 { h2 = h1; h1 = d.e; q1 = t; } else if d.e > h2 { h2 = d.e; }
                }
                out.push((l1, p1, l2, h1, q1, h2));
            }
        }
        out
    };
    let a2 = two(&av, m);
    let b2 = two(&bv, n);
    // Sample cells evenly.
    let stride = ((m * n) / cells.max(1)).max(1);
    let idx: Vec<usize> = (0..m * n).step_by(stride).take(cells).collect();
    #[derive(Default, Clone, Copy)]
    struct St {
        groups: u64,
        trunc_prod: u64,
        trunc_carry: u64,
        nonlinear: u64,
        lin_mismatch: u64,
        span_a_gt: [u64; 4], // row-group span > 9, 10, 11, 13
        fb_bound: u64,       // kernel-B bound-based fallback (span<=11 limbs, carry/prod conditions)
        fb_exact_e: u64,     // fallback if E were known exactly per cell-group
        fb_refined: u64,     // refined bound predicate (ctz-aware carry, lsb-aware products, |ia|<=32639)
        fb_ref_span: u64,
        fb_ref_carry: u64,
        fb_ref_prod: u64,
        fb_ref2: u64, // + second-min/second-max pairing refinement
        final_mismatch: u64,
        e_gt_ce2: u64,
        rz_rounded: u64,          // group result != C + exact dot (RZ dropped bits)
        win_exact: [u64; 5],      // windows of W=1,2,4,8,128 groups that are fully exact (no rounding/truncation)
        win_total: [u64; 5],
    }
    let stats: Vec<St> = idx
        .par_iter()
        .map(|&cell| {
            let (i, j) = (cell / n, cell % n);
            let mut st = St::default();
            let mut c = 0.0f32;
            let wins = [1usize, 2, 4, 8, 128];
            let mut win_ok = [true; 5];
            for g in 0..ng {
                st.groups += 1;
                let (cneg, ce, csig) = gf(c);
                let mut e = if csig != 0 { ce } else { i32::MIN };
                let mut prods = [(false, 0i32, 0u32); 32];
                let mut exact_units: i128 = 0; // exact sum of products, unit 2^-18
                for t in 0..32 {
                    let x = dec(av[i * k + g * 32 + t]);
                    let y = dec(bv[j * k + g * 32 + t]);
                    let p = x.sig * y.sig;
                    let pe = x.e + y.e - 14;
                    prods[t] = (x.neg ^ y.neg, pe, p);
                    if p != 0 {
                        e = e.max(pe);
                        let v = (p as i128) << (x.e + y.e - 2);
                        exact_units += if x.neg ^ y.neg { -v } else { v };
                    }
                }
                // B200 group (mirror of windowed_group_sum, width 26).
                let mut acc: i64 = 0;
                let mut tp = false;
                for &(neg, pe, p) in &prods {
                    if p == 0 {
                        continue;
                    }
                    let s = (e - pe) as u32;
                    let full = (p as u64) << 19;
                    let al = if s >= 32 { 0 } else { full >> s };
                    if s >= 32 || (al << s) != full {
                        tp = true;
                    }
                    acc += if neg { -(al as i64) } else { al as i64 };
                }
                let mut tc = false;
                if csig != 0 {
                    let s = (e - ce) as u32;
                    let full = (csig as u64) << 2;
                    let al = if s >= 32 { 0 } else { full >> s };
                    if s >= 32 || (al << s) != full {
                        tc = true;
                    }
                    if e > ce + 2 {
                        st.e_gt_ce2 += 1;
                    }
                    acc += if cneg { -(al as i64) } else { al as i64 };
                }
                let new_c = if acc == 0 {
                    0.0f32
                } else {
                    let sign = acc < 0;
                    let mut sg = acc.unsigned_abs() as u32;
                    let width = 32 - sg.leading_zeros() as i32;
                    let mut ex = e + width - 26;
                    if width > 26 { sg >>= width - 26 } else { sg <<= 26 - width }
                    if ex < -126 {
                        let sh = -126 - ex;
                        sg = if sh >= 32 { 0 } else { sg >> sh };
                        ex = -126;
                    }
                    sg >>= 2;
                    if sg == 0 {
                        if sign { -0.0 } else { 0.0 }
                    } else {
                        let mut eb = ex + 127;
                        if sg & 0x800000 == 0 {
                            eb -= 1;
                        }
                        f32::from_bits((sign as u32) << 31 | ((eb as u32) & 0xFF) << 23 | (sg & 0x7FFFFF))
                    }
                };
                // exact (unrounded) C + dot, in units of 2^-18
                let c_units_all: i128 = if csig == 0 { 0 } else {
                    let sh = ce - 23 + 18;
                    let v = if sh >= 0 { (csig as i128) << sh } else { (csig as i128) >> (-sh) };
                    if cneg { -v } else { v }
                };
                let exact_total = c_units_all + exact_units;
                let nc_units: i128 = {
                    let (nn, ne, ns) = gf(new_c);
                    if ns == 0 { 0 } else {
                        let sh = ne - 23 + 18;
                        let v = if sh >= 0 { (ns as i128) << sh } else { (ns as i128) >> (-sh) };
                        if nn { -v } else { v }
                    }
                };
                let group_exact = !tp && !tc && nc_units == exact_total;
                if nc_units != exact_total { st.rz_rounded += 1; }
                for (wi, &w) in wins.iter().enumerate() {
                    if !group_exact { win_ok[wi] = false; }
                    if (g + 1) % w == 0 || g + 1 == ng {
                        st.win_total[wi] += 1;
                        if win_ok[wi] { st.win_exact[wi] += 1; }
                        win_ok[wi] = true;
                    }
                }
                st.trunc_prod += tp as u64;
                st.trunc_carry += tc as u64;
                if tp || tc {
                    st.nonlinear += 1;
                    st.fb_exact_e += 1;
                } else {
                    // Linearised identity: RZ_f32(C + exact dot) must equal the group result.
                    let c_units: i128 = if csig == 0 {
                        0
                    } else {
                        let sh = ce - 23 + 18;
                        let v = if sh >= 0 { (csig as i128) << sh } else { (csig as i128) >> (-sh) };
                        if cneg { -v } else { v }
                    };
                    if rz_f32_units18(c_units + exact_units).to_bits() != new_c.to_bits() {
                        st.lin_mismatch += 1;
                    }
                }
                // Bound-based fast-path predicate (what kernel B evaluates per cell-group).
                let (ah, al, alsb, amx) = ab[i * ng + g];
                let (bh, bl, blsb, bmx) = bb[j * ng + g];
                let any = ah != i32::MIN && bh != i32::MIN;
                let sa = if ah == i32::MIN { 0 } else { ah - al };
                for (q, lim) in [9, 10, 11, 13].iter().enumerate() {
                    if sa > *lim {
                        st.span_a_gt[q] += 1;
                    }
                }
                if any {
                    let sb = bh - bl;
                    let pe_ub = ah + bh - 14;
                    let pe_lb = al + bl - 14;
                    let carry_ok = csig == 0 || pe_ub <= ce + 2;
                    let e_ub = if csig == 0 { pe_ub } else { pe_ub.max(ce) };
                    let prod_ok = e_ub - pe_lb <= 19;
                    let span_ok = sa <= 11 && sb <= 11;
                    if !(carry_ok && prod_ok && span_ok) {
                        st.fb_bound += 1;
                    }
                    let r_carry = csig == 0 || pe_ub - ce <= 2 + csig.trailing_zeros() as i32;
                    let r_prod = e_ub <= alsb + blsb - 14 + 19;
                    let r_span = amx <= 32639 && bmx <= 32639;
                    if !(r_carry && r_prod && r_span) {
                        st.fb_refined += 1;
                    }
                    let (al1, ap1, al2, ah1, aq1, ah2) = a2[i * ng + g];
                    let (bl1, bp1, bl2, bh1, bq1, bh2) = b2[j * ng + g];
                    let sat = |x: i32, y: i32| if x == i32::MAX || y == i32::MAX { i32::MAX } else { x + y };
                    let satm = |x: i32, y: i32| if x == i32::MIN || y == i32::MIN { i32::MIN } else { x + y };
                    let lsb_pair = if ap1 != bp1 { sat(al1, bl2).min(sat(al2, bl1)) } else { al1 + bl1 };
                    let e_pair = if aq1 != bq1 { satm(ah1, bh2).max(satm(ah2, bh1)) } else { ah1 + bh1 };
                    let pe_ub2 = e_pair - 14;
                    let e_ub2 = if csig == 0 { pe_ub2 } else { pe_ub2.max(ce) };
                    let r2_carry = csig == 0 || pe_ub2 - ce <= 2 + csig.trailing_zeros() as i32;
                    let r2_prod = e_ub2 <= lsb_pair - 14 + 19;
                    if !(r2_carry && r2_prod && r_span) {
                        st.fb_ref2 += 1;
                    }
                    st.fb_ref_span += !r_span as u64;
                    st.fb_ref_carry += !r_carry as u64;
                    st.fb_ref_prod += !r_prod as u64;
                }
                c = new_c;
            }
            if c.to_bits() != cref[cell] {
                st.final_mismatch += 1;
            }
            st
        })
        .collect();
    let mut s = St::default();
    for t in &stats {
        s.groups += t.groups;
        s.trunc_prod += t.trunc_prod;
        s.trunc_carry += t.trunc_carry;
        s.nonlinear += t.nonlinear;
        s.lin_mismatch += t.lin_mismatch;
        for q in 0..4 {
            s.span_a_gt[q] += t.span_a_gt[q];
        }
        s.fb_bound += t.fb_bound;
        s.fb_exact_e += t.fb_exact_e;
        s.fb_refined += t.fb_refined;
        s.fb_ref_span += t.fb_ref_span;
        s.fb_ref2 += t.fb_ref2;
        s.fb_ref_carry += t.fb_ref_carry;
        s.fb_ref_prod += t.fb_ref_prod;
        s.final_mismatch += t.final_mismatch;
        s.e_gt_ce2 += t.e_gt_ce2;
        s.rz_rounded += t.rz_rounded;
        for q in 0..5 { s.win_exact[q] += t.win_exact[q]; s.win_total[q] += t.win_total[q]; }
    }
    let pct = |x: u64| 100.0 * x as f64 / s.groups as f64;
    println!("analyze {}: {} cells x {} groups = {} cell-groups", dir.display(), idx.len(), ng, s.groups);
    println!("  re-derived B200 vs Pearl output bits: {} mismatching cells (must be 0)", s.final_mismatch);
    println!("  cell-groups with a truncated product term : {:.4}%", pct(s.trunc_prod));
    println!("  cell-groups with a truncated carry        : {:.4}%  (E > CE+2: {:.4}%)", pct(s.trunc_carry), pct(s.e_gt_ce2));
    println!("  cell-groups that are non-linear (either)  : {:.4}%", pct(s.nonlinear));
    println!("  linear groups where RZ_f32(C + exact dot) != B200 result: {} (must be 0)", s.lin_mismatch);
    println!(
        "  A row-group exponent span >9/>10/>11/>13  : {:.3}% / {:.3}% / {:.3}% / {:.3}%",
        pct(s.span_a_gt[0]),
        pct(s.span_a_gt[1]),
        pct(s.span_a_gt[2]),
        pct(s.span_a_gt[3])
    );
    println!("  naive bound fallback predicate (KB-style) : {:.4}%", pct(s.fb_bound));
    println!(
        "  refined fallback predicate (kernel B)     : {:.4}%  [limb-range {:.4}%, carry {:.4}%, product {:.4}%]",
        pct(s.fb_refined),
        pct(s.fb_ref_span),
        pct(s.fb_ref_carry),
        pct(s.fb_ref_prod)
    );
    println!("  refined + 2nd-extreme pairing predicate   : {:.4}%", pct(s.fb_ref2));
    println!("  groups whose RZ actually drops bits       : {:.4}%", pct(s.rz_rounded));
    println!(
        "  fully exact (no rounding/truncation) windows W=1/2/4/8/128 groups: {}",
        (0..5).map(|q| format!("{:.2}%", 100.0 * s.win_exact[q] as f64 / s.win_total[q].max(1) as f64)).collect::<Vec<_>>().join(" / ")
    );
    Ok(())
}

fn cmp(a: &[String]) -> Result<()> {
    let x = read_u32(Path::new(&arg::<String>(a, 2, "c1")?))?;
    let y = read_u32(Path::new(&arg::<String>(a, 3, "c2")?))?;
    ensure!(x.len() == y.len(), "length mismatch {} vs {}", x.len(), y.len());
    let bad: Vec<usize> = (0..x.len()).filter(|&i| x[i] != y[i]).collect();
    println!("cmp: {} cells, {} mismatches", x.len(), bad.len());
    for &i in bad.iter().take(10) {
        println!("  cell {i}: {:08x} vs {:08x}", x[i], y[i]);
    }
    if !bad.is_empty() {
        bail!("MISMATCH");
    }
    Ok(())
}

pub fn run() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("gen") => generate(&a),
        Some("ref") => reference(&a),
        Some("analyze") => analyze(&a),
        Some("cmp") => cmp(&a),
        _ => bail!("usage: v4-oracle gen|ref|analyze|cmp ... (see src/main.rs header)"),
    }
}
