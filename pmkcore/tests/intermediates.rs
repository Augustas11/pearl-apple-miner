//! Independent Python byte-for-byte check of all intermediate sections, including jackpot wrapping.
use pmkcore::oracle::{build_config, OracleJob, Pattern};
use std::io::Write;
use std::process::{Command, Stdio};

fn decode_hash(s: &str) -> [u8; 32] {
    std::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
}

#[test]
fn every_intermediate_matches_independent_python_oracle() {
    for pattern in [Pattern::Na, Pattern::Sg] {
        let (m, n, k) = (if pattern == Pattern::Na { 128 } else { 64 }, 64, 4096);
        let config = build_config(pattern, k, m, n).unwrap();
        let header = std::array::from_fn(|i| i as u8);
        let a = (0..m as usize * k as usize)
            .map(|i| (((i * 17 + 3) % 129) as i16 - 64) as i8 as u8)
            .collect::<Vec<_>>();
        let bt = (0..n as usize * k as usize)
            .map(|i| (((i * 29 + 11) % 129) as i16 - 64) as i8 as u8)
            .collect::<Vec<_>>();
        let job = OracleJob::new(&header, &config.to_bytes(), m, n, &a, &bt).unwrap();
        let mut child = Command::new("../.venv/bin/python")
            .args(["-B", "tests/check_vectors.py"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(&job.export_vectors())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "Python vector checker: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        let mut lines = stdout.lines();
        let fingerprint = lines.next().unwrap();
        let pinned = match pattern {
            Pattern::Na => "08f2e0c3d6a8c44440499a337526a7afd3bc10a355e2fb0168805124a635ffbe",
            Pattern::Sg => "127d5b12c565edeec6cade3b474eb6e3b914dc58ab163ce76bfb210215dd3317",
        };
        assert_eq!(fingerprint, pinned, "pinned intermediate bytes changed");
        println!("{pattern:?}: all 26 intermediate fields independently verified; export BLAKE3={fingerprint}");
        for line in lines {
            let fields: Vec<_> = line.split_whitespace().collect();
            assert_eq!(fields.len(), 19);
            let tr = fields[0].parse().unwrap();
            let tc = fields[1].parse().unwrap();
            let hash = decode_hash(fields[18]);
            let tile = job.tile(tr, tc, hash, hash).unwrap();
            assert_eq!(tile.hash, hash);
            assert_eq!(
                tile.transcript,
                std::array::from_fn(|i| fields[i + 2].parse::<u32>().unwrap())
            );
            assert_eq!((tile.is_share, tile.is_block), (1, 1), "equality must win");
            let mut less = hash;
            for b in &mut less {
                let (v, borrow) = b.overflowing_sub(1);
                *b = v;
                if !borrow {
                    break;
                }
            }
            let mut more = hash;
            for b in &mut more {
                let (v, carry) = b.overflowing_add(1);
                *b = v;
                if !carry {
                    break;
                }
            }
            let boundary = job.tile(tr, tc, less, more).unwrap();
            assert_eq!((boundary.is_share, boundary.is_block), (0, 1));
        }
    }
}
