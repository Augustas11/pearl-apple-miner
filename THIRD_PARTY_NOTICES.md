# Third-party notices

pmk itself is Apache-2.0 (see `LICENSE`). It includes or derives from the following.

## OpenJarvis (Apache-2.0)

`upstream/openjarvis/` contains files derived from OpenJarvis
(https://github.com/open-jarvis/OpenJarvis), modified to run against current Pearl.
Its `LICENSE` and `NOTICE` are kept unchanged. Changes are listed in `docs/OPENJARVIS_UPGRADE.md`.

## Pearl Research Labs (ISC)

Source: https://github.com/pearl-research-labs/pearl. pmk builds against it (it is fetched into
`vendor/`, not redistributed here) and a few files copy small parts of it verbatim:

- `bench/v4_emulation/oracle/src/stubs/noise.rs` and
  `bench/v4_emulation/oracle/src/stubs/public_params.rs`: verbatim copies of parts of
  `zk-pow/src/api/fp8/noise.rs` and `public_params.rs` (fp8-scheme commit 25695462416f0eb069abe7a515d188882078bf70).
- `libpmk/metal/` BLAKE3 keyed compression: a Metal port of the BLAKE3 round/compression structure used in
  `miner/pearl-gemm/csrc/blake3/blake3.cuh` (pearl-gemm has no separate license; it falls under the repository
  root ISC license below).
- Test vectors in `libpmk/tests/reference/` and `pmkcore/tests/` are produced with Pearl's own `zk-pow` code.

```
ISC License

Copyright (c) 2025-2026 Pearl Research Labs
Copyright (c) 2015-2016 The Decred developers

Permission to use, copy, modify, and distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
```

## BLAKE3

The keyed compression is implemented per the BLAKE3 specification (https://github.com/BLAKE3-team/BLAKE3),
released under CC0-1.0 or Apache-2.0. `pmkcore` depends on the `blake3` crate (same dual license) from crates.io.

## Other crates and Python packages

Rust crates (`pmkcore/Cargo.lock`) and Python packages (`miner/requirements.lock`) keep their own licenses.
