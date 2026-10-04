<!-- SPDX-License-Identifier: Apache-2.0 -->

# pmkcore v3 and v4 Rust tests

`pmkcore` and `pmkcore/v4` are independent Cargo packages rather than members
of one workspace. Run both release suites from the repository root with this exact
one-line command:

```sh
cd pmkcore && RAYON_NUM_THREADS=4 cargo test --release --offline -j 4 -- --test-threads=4 && RAYON_NUM_THREADS=4 cargo test --release --offline -j 4 --manifest-path v4/Cargo.toml -- --test-threads=4
```

The command enters `pmkcore/` before running Cargo so that Cargo loads
`pmkcore/.cargo/config.toml`, which supplies the ARMv8 AES configuration used
by both packages. The `&&` makes a v3 failure prevent a misleading v4 pass,
while the second manifest ensures that the v4 library, binaries, integration
test, and doc tests are all selected. The thread and build-job limits keep the
CPU workload bounded; they do not filter any tests.

The B9-fix run is recorded in
`bench/evidence/b9_fix_rust_v3_v4_release.txt`. It passed 16 v3 pmkcore tests,
436 upstream tests path-included by the v4 library, and 6
`pmkcore/v4/tests/v1_core.rs` integration tests. There were zero failures and
zero filtered tests. The v4 library reported six pre-existing ignored tests:
three memory-heavy or fixture-generation proof tests, one heavy LUT
precommitment test, and two v2 fixture generators. The command adds no skips or
filters. The binary harnesses and both packages' doc tests also ran, with zero
test cases defined.

The release build preserved both admission-relevant binaries byte-for-byte:
`pmkcore/v4/target/release/libpmkcore_v4.dylib` remained
`4a6055828cf9a554c519ed396d9f9d21f8535b7cdb60660910676c9ddda06f5c`, and
`libpmk/.build/arm64-apple-macosx/release/libpmk.dylib` remained
`5b14b60cbb59454f93b4c22075ab7bba75c93881ab5849115c5c0f5c80646675`.
