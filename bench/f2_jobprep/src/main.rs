//! F2 CPU-side benchmark: pmkcore job-build throughput vs the K3 GPU time it must hide under.
//! Usage: cargo run --release   (from bench/f2_jobprep). See README.md.

use pmkcore::{padded_len, pmkcore_build_job, Job, Template};
use std::process::Command;
use std::time::{Duration, Instant};

const LOCK: &str = "/tmp/pmm-gpu-bench.lock";
const IDLE_TOPS: f64 = 19.0;
/// R6 V6 rk128 128x64 median TOPS (docs/kb/r6-result.md, loaded machine): the best measured
/// K3-like rate so far (fold, no BLAKE3/compare yet). 8192²×4096 was not measured.
const R6_TOPS: &[((usize, usize, usize), f64)] = &[
    ((4096, 4096, 2048), 10.70),
    ((4096, 4096, 4096), 9.77),
    ((4096, 4096, 8192), 8.38),
    ((8192, 8192, 2048), 5.63),
    ((8192, 8192, 8192), 7.76),
];
const THREADS: [usize; 3] = [1, 9, 10];
const NS: [usize; 4] = [1024, 2048, 4096, 8192];
const KS: [usize; 3] = [2048, 4096, 8192];

fn sh(cmd: &str) -> String {
    Command::new("/bin/sh").args(["-c", cmd]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}
fn load1() -> f64 {
    sh("sysctl -n vm.loadavg").trim_matches(|c| c == '{' || c == '}' || c == ' ').split_whitespace().next().and_then(|s| s.parse().ok()).unwrap_or(f64::NAN)
}

struct Lock;
impl Lock {
    fn acquire() -> Lock {
        let t0 = Instant::now();
        while load1() > 8.0 && t0.elapsed() < Duration::from_secs(30 * 60) {
            println!("  [load] 1-min loadavg {:.2} > 8, waiting (waited {} s)", load1(), t0.elapsed().as_secs());
            std::thread::sleep(Duration::from_secs(15));
        }
        if load1() > 8.0 {
            println!("  [load] FLAG: loadavg still {:.2} after 30 min wait; proceeding", load1());
        }
        let t1 = Instant::now();
        while let Err(e) = std::fs::create_dir(LOCK) {
            if e.kind() != std::io::ErrorKind::AlreadyExists {
                panic!("mkdir {LOCK}: {e}");
            }
            if t1.elapsed().as_secs() % 60 < 15 {
                println!("  [lock] {LOCK} busy, retrying every 15 s (waited {} s)", t1.elapsed().as_secs());
            }
            std::thread::sleep(Duration::from_secs(15));
        }
        println!("  [lock] acquired after {} s; loadavg before: {}", t1.elapsed().as_secs(), sh("sysctl -n vm.loadavg"));
        Lock
    }
}
impl Drop for Lock {
    fn drop(&mut self) {
        println!("  [lock] releasing; loadavg after: {}", sh("sysctl -n vm.loadavg"));
        let _ = std::fs::remove_dir(LOCK);
    }
}

/// Median / min / max seconds per call: 1 warm-up, then ≥ 5 reps and ≥ 0.6 s (≤ 200 reps).
fn time(mut f: impl FnMut()) -> (f64, f64, f64) {
    f();
    let mut v = Vec::new();
    let t0 = Instant::now();
    while v.len() < 5 || (t0.elapsed().as_secs_f64() < 0.6 && v.len() < 200) {
        let t = Instant::now();
        f();
        v.push(t.elapsed().as_secs_f64());
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0], v[v.len() - 1])
}

fn pool(t: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap()
}

fn policy_config(k: u32) -> [u8; 52] {
    let mut c = [0u8; 52];
    c[0..4].copy_from_slice(&k.to_le_bytes());
    c[4..6].copy_from_slice(&128u16.to_le_bytes());
    c[8..14].copy_from_slice(&[7, 1, 3, 1, 0, 0]); // rows [0,8,64,72]
    c[14..20].copy_from_slice(&[0, 3, 3, 3, 0, 0]); // cols [0..3,16..19,32..35,48..51]
    c
}
fn header() -> [u8; 76] {
    let mut h = [0u8; 76];
    for (i, b) in h.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(37).wrapping_add(11);
    }
    h
}

fn main() {
    println!("== F2 job prep (pmkcore) ==");
    println!("date: {}", sh("date"));
    println!(
        "host: {} | {} | {} P + {} E cores | macOS {} | {}",
        sh("sysctl -n hw.model"),
        sh("sysctl -n machdep.cpu.brand_string"),
        sh("sysctl -n hw.perflevel0.physicalcpu"),
        sh("sysctl -n hw.perflevel1.physicalcpu"),
        sh("sw_vers -productVersion"),
        sh("pmset -g batt | head -1"),
    );
    println!("low power mode: {}", sh("pmset -g | grep -i lowpowermode | awk '{print $2}'"));
    println!("RNG: AES-128-CTR (ARMv8 AES), OS-entropy key per job; 2 keystream bytes/element; BLAKE3 crate {}", "1.8 (NEON)");
    println!("timing: wall clock, 1 warm-up, median of >=5 reps and >=0.6 s; T = rayon threads (9 = one core reserved)");

    let _lock = Lock::acquire();

    // 1) BLAKE3 alone over 64 MiB
    println!("\n-- BLAKE3 keyed hash alone, 64 MiB buffer (GB/s, median [min..max]) --");
    let mut buf = vec![0u8; 64 << 20];
    for (i, b) in buf.iter_mut().enumerate() {
        *b = (i * 2654435761usize >> 13) as u8;
    }
    let key = [5u8; 32];
    let len = buf.len() as f64;
    let (t, lo, hi) = time(|| {
        std::hint::black_box(pearl_blake3::blake3_digest(&buf, Some(key)));
    });
    println!("  pearl_blake3::blake3_digest (blake3::keyed_hash, 1 thread)  {:6.2} [{:.2}..{:.2}]", len / t / 1e9, len / hi / 1e9, len / lo / 1e9);
    for &th in &THREADS {
        let p = pool(th);
        let (t, lo, hi) = p.install(|| {
            time(|| {
                std::hint::black_box(pearl_blake3::MerkleTree::new(&buf, key).root());
            })
        });
        println!("  pearl_blake3::MerkleTree::new().root()  T={th:2}            {:6.2} [{:.2}..{:.2}]", len / t / 1e9, len / hi / 1e9, len / lo / 1e9);
        let (t, lo, hi) = p.install(|| {
            time(|| {
                let l = buf.len();
                std::hint::black_box(pmkcore::commit_root(&mut buf, l, &key, None));
            })
        });
        println!("  pmkcore::commit_root (hash only)        T={th:2}            {:6.2} [{:.2}..{:.2}]", len / t / 1e9, len / hi / 1e9, len / lo / 1e9);
    }
    drop(buf);

    // 2) Template (A, once per template) at T = 9
    println!("\n-- template_init (generate A + root, once per template), T=9 --");
    let p9 = pool(9);
    for &m in &[4096usize, 8192] {
        for &k in &KS {
            let mut a = vec![0u8; padded_len(m, k)];
            let t0 = Instant::now();
            p9.install(|| pmkcore::template_init(&header(), &policy_config(k as u32), m as u32, m as u32, &mut a, true)).unwrap();
            let dt = t0.elapsed().as_secs_f64();
            println!("  m={m:5} k={k:5}  {:7.2} ms  ({:.2} GB/s)", dt * 1e3, (m * k) as f64 / dt / 1e9);
        }
    }

    // 3) build_job via the C ABI
    println!("\n-- pmkcore_build_job (C ABI): fresh B^T n x k + root + seeds --");
    println!("  {:>5} {:>5} {:>3} | {:>9} {:>9} {:>9} | {:>7} {:>8}", "n", "k", "T", "ms med", "ms min", "ms max", "GB/s", "jobs/s");
    let mut med = std::collections::HashMap::new();
    for &k in &KS {
        for &n in &NS {
            let mut a = vec![0u8; padded_len(128, k)];
            let t = pmkcore::template_init(&header(), &policy_config(k as u32), 128, n as u32, &mut a, true).unwrap();
            let mut bt = vec![0u8; padded_len(n, k)];
            for &th in &THREADS {
                let p = pool(th);
                let (tm, lo, hi) = p.install(|| {
                    time(|| {
                        let mut j = Job::default();
                        let rc = unsafe { pmkcore_build_job(&t as *const Template, bt.as_mut_ptr(), bt.len() as u64, &mut j) };
                        assert_eq!(rc, 0);
                        std::hint::black_box(j);
                    })
                });
                med.insert((n, k, th), tm);
                println!(
                    "  {n:5} {k:5} {th:3} | {:9.3} {:9.3} {:9.3} | {:7.2} {:8.1}",
                    tm * 1e3,
                    lo * 1e3,
                    hi * 1e3,
                    (n * k) as f64 / tm / 1e9,
                    1.0 / tm
                );
            }
        }
    }
    drop(_lock);

    // 4) vs GPU time
    println!("\n-- build (T=9) vs K3 GPU time for the same job; ratio = build / GPU (must be <= 0.8) --");
    println!("  {:>5} {:>5} {:>5} | {:>9} | {:>9} {:>7} | {:>9} {:>7} {:>6}", "m", "n", "k", "build ms", "GPU@19 ms", "ratio", "GPU@R6 ms", "ratio", "R6 TOPS");
    for &n in &[4096usize, 8192] {
        for &k in &KS {
            let m = n;
            let b = med[&(n, k, 9)];
            let ops = 2.0 * (m * n * k) as f64;
            let g19 = ops / (IDLE_TOPS * 1e12);
            let r6 = R6_TOPS.iter().find(|(s, _)| *s == (m, n, k)).map(|(_, r)| *r);
            let (gr, rr) = match r6 {
                Some(r) => (format!("{:9.2}", ops / (r * 1e12) * 1e3), format!("{:7.3}", b / (ops / (r * 1e12)))),
                None => (format!("{:>9}", "n/a"), format!("{:>7}", "n/a")),
            };
            println!(
                "  {m:5} {n:5} {k:5} | {:9.3} | {:9.2} {:7.3} | {gr} {rr} {:>6}",
                b * 1e3,
                g19 * 1e3,
                b / g19,
                r6.map(|r| format!("{r:.2}")).unwrap_or("n/a".into())
            );
        }
    }

    // 5) smallest shape per k with build <= 0.8 x GPU at 19 TOPS (T=9)
    println!("\n-- smallest m=n (from {NS:?}) per k with build(T=9) <= 0.8 x GPU@19 TOPS; and analytic m_min for fixed n --");
    for &k in &KS {
        let ok = NS.iter().find(|&&n| med[&(n, k, 9)] <= 0.8 * 2.0 * (n * n * k) as f64 / (IDLE_TOPS * 1e12));
        let mins: Vec<String> = NS
            .iter()
            .map(|&n| {
                // build cost depends on n*k only (A is fixed); GPU time scales with m, so m_min = build / (0.8 * 2nk / R)
                let m = med[&(n, k, 9)] / (0.8 * 2.0 * (n * k) as f64 / (IDLE_TOPS * 1e12));
                format!("n={n}: m>={}", ((m / 128.0).ceil() as usize).max(1) * 128)
            })
            .collect();
        println!("  k={k:5}: smallest square m=n = {:?}; {}", ok, mins.join(", "));
    }
}
