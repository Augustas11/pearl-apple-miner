# Pearl pool protocol: how pmk submits shares to public Pearl pools (SPEC §7.6, §13.6, G5b)

Status: research, 2026-10-02. Read-only. Every claim has a citation; **UNVERIFIED** marks claims without primary evidence. Wallet addresses are omitted throughout.

**Source abbreviations.**
- `vendor/pearl/...` = the repo at 7039e66f.
- External repos, cited at the commit fetched on 2026-10-02:
  - NUSHY = github.com/nushypool/pearl_stratum_protocol_v1 @d79cf1b
  - SOAT = github.com/blindrun/soat-miner @48defc8
  - ASCEND = github.com/arabel1a/ascend_prl @9804fa8
  - OPM = github.com/Muskwak/Open-Pearl-Miner @ac3d992
  - CPP = github.com/1640675651/CPPminer @6785ad3
  - P3090 = github.com/rheza/prl-3090-miner @a8cd672
  - HMK = github.com/DQMining/DRQ-Miner-Testing @fe3bcc7
- Live pool APIs, fetched 2026-10-02:
  - HERO-API = https://pearl.herominers.com/api/stats
  - KRYPTEX-API = https://pool.kryptex.com/prl/api/v1/pool/stats
  - LUCKY-API = https://pearl.luckypool.io/api/stats
  - ALPHA-API = https://pearl.alphapool.tech/api/stats
  - 2M-API = https://prl.2miners.com/api/stats

## 0. Answer in brief

- **There is no single standard.** There is no official Pearl stratum. Pearl ships only the share-verify and prove primitives for pools (`vendor/pearl/zk-pow/bindings/go/src/plain.rs:1-3`). Pools speak at least two dialects:
  1. **Object dialect: HeroMiners, Kryptex, k1pool, LuckyPool.**
     - The miner chooses the MiningConfiguration, and it travels inside the PlainProof.
     - The pool pushes `mining.notify {job_id, header, target, height, cert_version}`.
     - The miner submits `mining.submit {job_id, plain_proof}`, where `plain_proof` is base64 of the bincode PlainProof.
  2. **AlphaPool / Hashmonkeys dialect.**
     - The pool dictates the config via `pearl.set_mining_params`.
     - It uses a `pearl.challenge` anti-spam PoW, positional arrays and vardiff.
- **A pool share is exactly a Pearl PlainProof checked with `nbits_override`.** The pool generates the ZK proof only for block-qualified shares. The miner never sends a ZK certificate.
- **Units: 1 pool H = 1 int7 MAC = 2 ops.**
  - So pool H/s = m·n·k/s = (2·m·n·k/s)/2.
  - The M3 Ultra at 16.6 TOPS is **8.3 TH/s in pool units**.
  - At HeroMiners' fixed difficulty 2^21 ("9.01 PH" = 2^53 H), that is **one share per ~18 min**.
  - Earnings are ≈ **0.18 PRL/day**.
- **HeroMiners difficulty is fixed** at 2^21. There is no vardiff-down and no `set_difficulty`; diff suffixes are ignored. It runs plain TCP only.
- **Acceptance of our h·w=32 SG pattern is UNVERIFIED.** The first share at the pool is the probe.

## 1. Message flow and fields

### 1.1 Object dialect (HeroMiners; also Kryptex, k1pool, LuckyPool)

The wire format is newline-delimited JSON-RPC over `stratum+tcp`. SOAT established the HeroMiners wire format by live probing (SOAT `src/core/pearl_pool.h:8-31`):

```
-> {"id":1,"method":"mining.authorize","params":{"wallet":"prl1...","worker":"rig1","pass":"x","agent":"<miner>/<ver>"}}
<- {"id":1,"error":null,"result":true}
<- {"id":null,"method":"mining.notify","params":{"job_id":"00000000_2097152",
      "header":"<152 hex = 76-byte IncompleteBlockHeader>","target":"<64 hex, big-endian>",
      "height":101009,"cert_version":3}}
-> {"id":9,"method":"mining.submit","params":{"job_id":"...","plain_proof":"<base64 bincode PlainProof>"}}
<- {"id":9,"result":true,"error":null}     (reject example: accepted:false "Jackpot condition not satisfied")
```

**HeroMiners specifics** (SOAT `pearl_pool.h:22-31`, `COORDINATION.md:504-506, 581-586`):
- **Params must be JSON objects.** Array params are rejected with `{"code":20,"msg":"params must be an object"}`.
- **The payout field must be `wallet`.** `login` and `user` "succeed" at authorize but then fail with code 24, "Wallet is missing".
- **Message order varies.** Authorize can arrive interleaved with the first notify (SOAT `pearl_pool.h:634`).
- **The pool only pushes.** There is no `mining.set_difficulty` or `set_target`; the notify `target` is authoritative. It decodes exactly as `0xFFFF·2^208 / 2097152`.

**What a job contains.**
- **header:** the 76-byte incomplete header (version, prev_block, merkle_root, timestamp, network nbits). Use it byte-exact; job_key = blake3(header ‖ config) binds it (CPP `docs/proof.md:13-22`; SPEC C1).
- **target:** the pool share target, 32 bytes big-endian.
- **job_id:** `<8-hex seq>_<difficulty>`. On HeroMiners the suffix 2097152 = 2^21 is the share difficulty (SOAT `COORDINATION.md:178-180`).
- **height** and **cert_version** (3 today).

It carries **no MiningConfiguration, no m/n/k, no pattern and no extranonce**. A job with no cert_version must be treated as invalid (SOAT `pearl_pool.h:500-504`).

**Job rotation.**
- HeroMiners pushes a new job every ~14–19 s; SOAT saw 5 job ids in 71 s with the target unchanged (`COORDINATION.md:491-494`).
- A proof mined against an older header but submitted under the newest job_id is rejected with "Jackpot condition not satisfied" (`COORDINATION.md:504-513`). So the pool validates against the header of the job_id the miner names.
- How long HeroMiners keeps older job_ids valid is UNVERIFIED; no clean_jobs flag was observed.

**What a share submission carries.**
- `plain_proof` = base64(bincode PlainProof). This is the same object pearl-gateway's `submitPlainProof` takes (SOAT `pearl_pool.h:28-31`; NUSHY README:136-142, 162-175; `vendor/pearl/zk-pow/bindings/go/src/plain.rs:16-19`, "bincode-serialized PlainProof").
- The proof carries:
  - m, n, k and the noise rank (the full MiningConfiguration);
  - the A-row and Bᵀ-column strips as Merkle multiproofs: leaf data, indices, root and siblings (NUSHY:166-175).
- There are no separate tile coordinates; the rows and columns inside the proof imply the tile.
- The proof is not ZK and is uncompressed by default.
  - Kryptex optionally negotiates gzip via `"type":"v2"` in authorize (ASCEND `src/pools/kryptex.c:46-60`). pmk should not use it.
- Optional, pool-specific extra fields must not affect validity:
  - `hs` (telemetry H/s) and `lpm_shape` (NUSHY:106-158);
  - `wallet` and `worker` in the submit object on k1pool and LuckyPool (ASCEND `src/pools/k1.c:100-106`; OPM `python/pool_common.py:115-119`).
- **Size.**
  - Observed: 90,180 bytes for a SOAT share (`COORDINATION.md:464-467`).
  - Our SG estimate (h=4, w=8, k=4096): ≈48 KiB of strips plus Merkle siblings, so ≈60 KB raw and ≈80 KB base64 (UNVERIFIED).

**Pool-side check.** The pool runs `verify_plain_proof_for_cert_version(cv, header, pp, nbits_override=share_nbits)`, or its FFI equivalent `verify_plain_proof_ffi(header, pp, len, cert_version, nbits_override, …)`. Sources:
- `vendor/pearl/zk-pow/bindings/go/src/plain.rs:16-55`;
- `vendor/pearl/py-pearl-mining/examples/v1_v2_gateway_example.py:84-96`;
- `vendor/pearl/docs/moe-fork-upgrade-guide.md:105-107`;
- `vendor/pearl/zk-pow/src/api/verify.rs:99-131`. Its comment calls nbits_override "e.g. pool share difficulty from `mining.set_difficulty`".

### 1.2 AlphaPool / Hashmonkeys dialect (not usable with K3-SG as-is)

**Handshake** (HMK `doc/PEARL_HASHMONKEYS_INTEGRATION.md:68-144, 241-300`; P3090 `docs/alphapool-stratum.md:27-110`):
1. `pearl.challenge {seed, difficulty:32}`, answered with `pearl.challenge_response {seed, nonce}`. This is a BLAKE3 leading-zero-bits PoW; Hashmonkeys documents it.
2. `mining.configure ["pearl/v1"]`.
3. `mining.subscribe`.
4. `mining.authorize ["addr.worker","x;d=<static diff>"]`.

**Pool-dictated config** via `pearl.set_mining_params`:
- `{m:131072, n:131072, k:4096, rank:128, rows_pattern:[0,32], cols_pattern:[0..63]}`, so h·w = 128.
- Hashmonkeys: "Your miner must use these dimensions".

**Messages.**
- Notify is a 7-element array: `[jobId, prevHash, headerHex, shareNbits, ntime, poolCompactHex, cleanJobs]`.
- Submit is `[addr.worker, job_id, plain_proof]`.
- Vardiff starts at 10,000; static difficulty is set via `d=` (HMK:264-268).

**Note:** P3090 declined to implement the AlphaPool challenge without a public spec (P3090:13-17). Hashmonkeys publishes it (HMK:140-161).

### 1.3 NUSHY "Pearl Stratum Protocol V1" (a proposed standard, from NPMiner/NushyPool)

- **Notify:** an object `{job_id, header(152 hex), target(64 hex, BE), height?, difficulty?}`, with `target = diff1_target / floor(difficulty)` (NUSHY:24-98).
- **Submit:** `{job_id, plain_proof, hs?, lpm_shape?}` (NUSHY:100-197).
- **Optional compact submit:** `pearl.compact.v1` "zero_signal" (NUSHY:199-263). pmk ignores it.
- **Conflict with live pools:**
  - NUSHY's README describes `target` as the share target the proof hash is compared against directly.
  - SOAT's live HeroMiners evidence shows pools instead **scale the target by tile·k** (§1.4), which is the consensus `nbits_override` path.
  - SOAT first coded the literal reading and saw zero shares, because that reading is 2^19 too hard (`COORDINATION.md:181-187, 202-203`).

### 1.4 Share difficulty → nbits_override → bound → work (unit derivation)

**1. Difficulty to target.** Pool difficulty D gives target T = floor(0xFFFF·2^208 / D), which is Bitcoin pdiff.
- SOAT `COORDINATION.md:585-586`.
- ASCEND `src/pools/stratum.c:161-174` uses diff1 = 0xFFFF·2^208.
- Kryptex states `"hashes_per_diff": 4294967296` (2^32) in KRYPTEX-API.

**2. Target to consensus bound.** The bound per tile attempt is `T × tile_size × dot_len`, with tile_size = h·w and dot_len = k − k mod r (`vendor/pearl/zk-pow/src/api/sanity_checks.rs:147-158, 183-196, 229-231`).
- The rank-penalized factor `h·w·(dot_len/r)·128` equals this when r = 128 (`sanity_checks.rs:194-196`). SOAT notes the two agree only at rank 128 (`COORDINATION.md:261-263`).
- **Pools apply the same scaling empirically.**
  - SOAT's 4090 logged 12 accepted shares in 571 s, one per 47.6 s, against 50.5 s predicted by the scaled bound (`COORDINATION.md:184-187`).
  - The unscaled reading would predict ~0 shares; double scaling would predict 626,526 (`COORDINATION.md:276-278`).
  - ASCEND scales `target × tile_elems × rounded_k` (`kryptex.c:112-114`, `k1.c:96-98`).

**3. Work per share.**
- One tile attempt costs h·w·dot_len MACs, and P(hit) = T·h·w·dot_len/2^256.
- So the expected MACs per share is 2^256/T = D·2^32·(65536/65535), **independent of the config**. The bound normalizes the work.
- Patterns partition the output (SPEC C7), so a full GEMM performs exactly m·n·dot_len MACs, the sum of the tile MACs.

**4. Pool hashrate units.** Pool hashrate is D·2^32 per share over time.
- HeroMiners `networkHashps` 4.626e19 ≈ network difficulty 2.199e12 × 2^32 / 202 s observed block time. The computed value is 4.67e19 (HERO-API).
- CPP reports pool `hs` in MAC/s: `tiles × h·w·K / s` (CPP `docs/hashrate_calculation.md:3-20, 66-79`).

**Conclusion:**
- **Pool H/s = MAC/s = m·n·dot_len per second = (our 2·m·n·k ops/s)/2.**
- "9.01 PH" share difficulty = 2^21 × 2^32 = 2^53 ≈ 9.007e15 MACs ≈ 1.8e16 ops per share.
- This matches the SPEC's E[ops/block] = 2^257/target (§1.2) = 2·2^256/T.

**Hardware sanity check (UNVERIFIED).** A high-end consumer NVIDIA GPU does ~450 dense INT8 TOPS, about 225 T MAC/s. A reported ≈210–220 TH/s pool hashrate for one such GPU is physically possible only if H = MAC.

**Corrected numbers for the M3 Ultra.**
- 16.6 TOPS = 8.3 TH/s in pool units.
- At D = 2^21: 9.007e15 / 8.3e12 ≈ **1,085 s ≈ 18 min per share** (≈ 3.3 shares/h).
- Earnings: 8.3e12 / 4.626e19 × (86400/202 × 2,288.8 PRL) ≈ **0.18 PRL/day (~$0.21 at $1.18)**. The block reward is from HERO-API `lastblock.reward` = 228876278125/1e8.

**Share nbits for the verifier gate.**
- HeroMiners' T = 0xFFFF·2^187 is exactly representable as compact **`0x1a07fff8`**.
- General rule: convert the 32-byte target to compact, rounding **down** (harder). Assert `nbits_to_difficulty(nbits) ≤ T`, and log whenever the conversion isn't exact.
- Use `extract_difficulty_bound(nbits, cfg)` (`sanity_checks.rs:229`) for both the kernel's share bound and the gate.

## 2. Who chooses the config; acceptance; ZK

**Object-dialect pools: the miner chooses.** The MiningConfiguration is serialized inside the PlainProof, and the pool reads and grades it (ASCEND `kryptex.c:9-11`, `k1.c:8-9`).

Configs observed as accepted:

| Pool | rank | k | Tile pattern | Source |
|---|---|---|---|---|
| HeroMiners | 128 | 2048 (default) | contiguous 16×16, h·w = 256 | SOAT `job.h:303-321`; shapes 4096×65536 and 2048×65536 seen in `COORDINATION.md:590-593` |
| Kryptex | 128 / 256 | 4096 | rows [0,32] × cols [0..63], h·w = 128 | ASCEND |
| k1pool | 512 | 8192 | — | ASCEND |

- ASCEND README:90, 102 states that a real pool accepts only r = 128 or 256, and that different pools accept different M, N, K. This is anecdotal and UNVERIFIED per pool.
- OPM calls LuckyPool's config "Mandated": m = n = 131072, k = 4096, r = 256, 16×16 (`python/luckypool_miner.py:8`). Whether LuckyPool enforces it is UNVERIFIED.

**AlphaPool / Hashmonkeys: the pool dictates the config** (`pearl.set_mining_params`, HMK:241-260). It is incompatible with our patterns.

**Our config is consensus-valid but untested on any pool.**
- Config: r = 128, k = 4096, rows [0,8,16,24] × cols [0,1,8,9,16,17,24,25], h·w = 32.
- No evidence was found of any pool accepting h·w = 32 or a 2-dimension cols pattern (UNVERIFIED).
- A pool that just calls `verify_plain_proof` accepts it. A pool could still add policy on k, h·w or m/n, since those drive its block-proving cost (e.g. ASCEND "MDIM 16384 keeps the pool's block-proof fast", README:91) (UNVERIFIED).

**The pool does the ZK proving.** For block-qualified shares it runs `prove_plain_proof_ffi` / `generate_proof_for_cert_version` (`plain.rs:2-3, 70-80`; `moe-fork-upgrade-guide.md:49-52`; `salted-seed-fork-upgrade-guide.md:35-55`). The miner never proves in pool mode. SG's smaller h+w than NA means lower degree_bits, so proving is cheaper for the pool.

## 3. Difficulty floors and time-to-share

Rates assumed: M3 Ultra (K3-SG) at 16.6 TOPS = 8.3 T MAC/s; M5 Air at ~8 TOPS = 4 T MAC/s.

| Pool / port | Share diff (min) | Vardiff / fixed | M3 Ultra | M5 Air | Source |
|---|---|---|---|---|---|
| HeroMiners :1200 | 2,097,152 (2^21) | fixed; `fixedDiffEnabled:false`; ignores `+d`/`d=`/`.diff` | **1,085 s (18 min)** | 2,252 s | HERO-API `ports[0].difficulty`; SOAT `COORDINATION.md:581-586` |
| Kryptex :7048 / TLS :8048 | min 2^21, max 2^32 | `vardiff:false`, `custom:true` | 1,085 s | 2,252 s | KRYPTEX-API |
| LuckyPool :3360/3361/3362 | 2.0M / 4.0M / 8.0M | `varDiff:true` | 1,035 s (at 2.0M) | 2,148 s | LUCKY-API; CPU port 3370 has an unknown lower diff (OPM README:172-183) |
| Hashmonkeys / AlphaPool | start 10,000; static `d=50000` seen | vardiff | 5.2 s / 25.9 s | 10.7 / 53.7 s | HMK:264-268; SOAT `COORDINATION.md:464-467` |

Formula: t = D·2^32·(65536/65535) / (MAC/s). For 1 share/min at 16.6 TOPS, D must be ≈ 116,000. Only AlphaPool-family pools go that low.

## 4. Payouts, fees, endpoints, custom-miner policy

### HeroMiners (HERO-API)
- **Fee:** `fee:0`.
- **Reward scheme:** UNVERIFIED; sources conflict. Treat it as proportional per the API.
  - API: `rewardScheme:"prop"`.
  - Site banner: "PPS+ and PROPX" (https://pearl.herominers.com/).
  - A third-party source says PPLNS.
- **Payout:** `minPaymentThreshold` 1e8 atomic = **1 PRL** (`coinUnits` 1e8). Payment interval 3,600 s.
- **Solo mining:** use a `solo:` prefix.
- **Stale-share penalty** (applies after 1,000 shares):

  | Stale rate | Penalty |
  |---|---|
  | ≤ 2% | none |
  | 5% | 50% |
  | 30% | 100% + warning |
  | > 30% | ban for 3,600 s |

- **Regions** for port 1200 (`<region>.pearl.herominers.com`): de, fr, es, fi, ru, ca, us, us2, us3, mx, br, kz, hk, kr, in, sg, tr, au.
- **TLS:** the API lists only port 1200 (plain), and SOAT found :1200 "the only open port" (`COORDINATION.md:582`). TLS is not available (UNVERIFIED for every region).

### Kryptex (KRYPTEX-API; https://pool.kryptex.com/articles/how-to-mine-pearl-en)
- **Fees:** PPS+ 2%, SOLO 1%.
- **Payout:** minimum 1 PRL; Kryptex pays the transaction fee; payouts are hourly.
- **Endpoints:** TLS at `prl[-eu|-us|-br|-sg|-hk|-ru|-ae].kryptex.network:8048`; plain at `:7048`.
- **Listed miners:** SRBMiner, ForgeMiner, ARCMiner and others.
- **Custom miners:** no restriction found. ASCEND, a custom miner, is tested there.

### Other pools
- **LuckyPool** (LUCKY-API): fee 1%, `paymentMode:"prop"`, minimum 1 PRL (default threshold 5), interval 7,200 s, TLS on all ports.
- **AlphaPool:** fee 1%, minimum 1 PRL (ALPHA-API).
- **2Miners:** PPLNS, minimum 1 PRL (2M-API `nodes[0].name:"prl_pplns"`).

### Custom-miner policy
- No miner-name whitelist was found on object-dialect pools.
- SOAT, a custom miner, is accepted at HeroMiners (`COORDINATION.md:527-528, 566-572`).
- `agent` is free text. pmk sends an honest `pmk/<ver>`.
- AlphaPool-family pools gate on `pearl.challenge` instead.

### Per-wallet stats
The likely cryptonote-pool endpoint is `https://pearl.herominers.com/api/stats_address?address=<addr>` (UNVERIFIED).

## 5. Security: what a malicious or compromised pool can do, and what the miner must check

### Threats
- **Hashrate hijack.** A pool, or a MITM on plaintext :1200, can send headers for its own solo work or another chain. A MITM can also rewrite the `wallet` in authorize. HeroMiners has no TLS, so the only defense is detection: the per-wallet dashboard must show our worker names and accepted-share counts consistent with what we submitted.
- **Bad targets.** An absurdly easy target floods submissions, CPU proof-building and bandwidth. A zero or impossible target wastes work.
- **Wrong cert_version.** cert_version ≠ 3 means wrong seed derivation and invalid shares. Treat it as R-A6 and alert.
- **Oversized or malformed frames.** ASCEND reads into a 1 MiB buffer and parses with naive `strstr` (`stratum.c:58-77, 108-145`); don't copy that pattern.
- **Notify floods.** HeroMiners legitimately sends one every ~15 s. A flood would force constant A-rehash churn.
- **Reply mix-ups.** Inline notifies during a submit can steal the submit's reply (SOAT `pearl_pool.h:179-182`).
- **Pool-side size cap.** `verify_plain_proof_ffi` has no size cap, and pools should cap shares at ~8 MiB (`plain.rs:22-26, 79-81`). pmk's shares are ≤ ~100 KB.

### Miner validation (MUST)
- **Framing:** UTF-8 JSON objects only, newline-framed. **Max line length 64 KiB** (a notify is ~400 B). Drop the connection on overflow.
- **Notify fields:**
  - `header` is exactly 152 hex characters;
  - `target` is exactly 64 hex characters and nonzero;
  - `job_id` is ≤ 64 printable ASCII characters;
  - `cert_version == 3`, or reject the job (R-A6).
- **Target sanity:** require D_implied = diff1/T ≥ **D_floor** (config; default 10,000) to bound the submit rate. Require T ≥ the header's block target, or log it.
- **Header sanity:**
  - `nbits` is in the plausible network range;
  - `prev_block` is stable across jobs at the same height;
  - optional: cross-check `prev_block` against a public explorer or the operator node's tip (read-only, no node credentials in pool mode).
- **Rate limits:**
  - Coalesce notifies: build only the newest job; drop notifies more frequent than every 1 s.
  - Cap outstanding submits at 4 and submits per minute at 30.
  - Reply timeout 30 s; correlate replies by id.
- **Never modify the header.** Never accept a config from the pool in the object dialect.
- **Reconnects:** jobs are session-scoped, so drop all jobs on reconnect. Use exponential backoff with jitter, 1–60 s.
- **Secrets:** log the wallet only as prefix…suffix. Pool mode involves no node credentials.

## 6. Recommended pool-mode design for pmk (fills SPEC §7.6)

**Protocol (v1).** Implement only the object dialect: HeroMiners first, with Kryptex TLS as the fallback.
- Authorize with `{"wallet": W, "worker": "<mac-name>", "pass":"x", "agent":"pmk/<ver>"}`. HeroMiners requires the `wallet` key and object params.
- Don't request gzip.
- The AlphaPool dialect is out of scope: it dictates an incompatible config and requires the challenge gate.

**Templates and jobs.**
- Each notify is a template; its identity is the header bytes.
- A is fixed per header; B is fresh per job (SPEC §3.2).
- Each job record keeps (job_id, header, target, share_nbits, cfg).
- **Submit with the job_id the job was built from, never the newest one.** That mismatch was SOAT's root-cause bug (`COORDINATION.md:502-513`).

**Share target.**
- share_nbits = compact(T), rounded down. For HeroMiners this is 0x1a07fff8.
- Kernel bound = `extract_difficulty_bound(share_nbits, cfg)`, the same function the gate uses.
- Classify each find:
  - **block-candidate**, if it also meets `extract_difficulty_bound(header.nbits, cfg)` (log and count these);
  - otherwise **share**.

**Verifier gate (R-A3, unchanged).** `verify_plain_proof_for_cert_version(3, header, proof, nbits_override=share_nbits)` runs before every submit. A gate failure is P0 for that device.

**Submit payload.** `{"job_id": J, "plain_proof": base64(bincode(PlainProof))}`, using the same serialization pearl-gateway accepts. Expect ≈60–100 KB.

**Outcome classification.** accepted / stale / duplicate / low-difficulty / invalid / transport / timeout.

| Event | Action |
|---|---|
| A share that passed the local gate is rejected (invalid / "Jackpot condition not satisfied") on the **first** share | **Config-compatibility failure:** stop the device; try the next pool |
| Same rejection later in a run | **P0** |
| Stale > 5% | Stop (HeroMiners bans above 30%) |

**Checks that replace the solo checks.**
- **Payout:** the configured wallet is in an operator allowlist.
- **Hourly cross-check** of the pool's per-wallet API: the worker is present and the pool's accepted count is ≥ 90% of ours (UNVERIFIED endpoint).
- **Lost-find monitor** (SPEC §7.5) with E = completed MACs × T / 2^256.

**Transport.**
- Use TLS where the pool offers it (Kryptex :8048, LuckyPool).
- HeroMiners is plaintext only: accept that risk explicitly and rely on dashboard detection.
- Apply the §5 limits.

**Proving.** None on the miner; the pool does it (SPEC §13.2 unaffected).

**Vardiff floor.** None below 2^21 at HeroMiners, so the M3 Ultra gets one share per ~18 min. Pools that go lower (AlphaPool / Hashmonkeys) dictate an incompatible config.

## 7. G5b test plan

**T0, offline (CI).**
- **Parser tests** on fixtures from SOAT's documented HeroMiners frames and redacted frames captured from other miners. Cover:
  - valid notify;
  - missing `cert_version`, and `cert_version` 4;
  - header of 150/151/153/154 hex;
  - zero target;
  - array params;
  - a 70 KiB line;
  - non-UTF8 input;
  - reply-id interleaving.
- **Target→nbits vectors:** 2^21 → 0x1a07fff8 (exact); 2,000,000; 50,000; 10,000 (round-down).
- **Bound check:** the kernel bound equals `extract_difficulty_bound` for each.

**T1, local mock pool** (loopback, pearl_mining verifier).
- The mock rotates headers every 15 s at a low difficulty (e.g. D = 1000) and validates with `verify_plain_proof_for_cert_version(3, …, nbits_override)`. It rejects a share submitted under the wrong job_id.
- **Pass:** ≥ 50 shares, 100% accepted, 0 gate failures, and the share count inside the Poisson 99.9% interval.
- **Negative cases:**
  - a corrupted proof is rejected by the pool (the local gate catches it first);
  - a mismatched job_id is rejected;
  - the miner refuses a malicious target or cert_version.

**T2, compatibility probe at HeroMiners** (M3 Ultra, one worker).
- Run until the first pool verdict, or 2 h. P(no share in 1 h) ≈ e^-3.3 ≈ 4%.
- **Pass:** the first share is accepted.
- **Fail** (rejected after the local gate passed): the SG config is not accepted. Repeat at Kryptex TLS (same difficulty).

**T3, G5b proper.** Proposed SPEC change: **≥ 20 submitted shares** (≈ 6 h on the M3 Ultra; fits an overnight run, §13.7) instead of "≥ 1 h".
- **Pass:**
  - ≥ 95% accepted (≥ 19 of 20);
  - 0 gate failures;
  - stale < 2%;
  - share count within the Poisson 99.9% interval for the measured MAC/s;
  - the pool dashboard shows the worker under the operator wallet.
- Record payout credit when it lands. With a prop/PPLNS-style payout, a 1 PRL minimum and ~0.18 PRL/day, that takes ~6 days.

**Risks and infeasibility flags.**
- **The original 1 h criterion is meaningless** at ~3.3 shares/h.
- **HeroMiners' difficulty can't be lowered** (SOAT `COORDINATION.md:581-586`).
- **SG-pattern acceptance is UNVERIFIED.** If both HeroMiners and Kryptex reject it, K3-SG cannot pool-mine on the known object pools. The fallback is a pool-standard pattern kernel (e.g. [0,32]×[0..63] or contiguous 16×16), which is a kernel change.
- **Plaintext HeroMiners** exposes the authorize wallet to a MITM.
- **G5b depends on G5 and G7** (unchanged).

## 8. Open items (UNVERIFIED)
- **HeroMiners:** retention window for old job_ids, per-wallet API path, actual payout scheme, and TLS anywhere.
- **Pool policy:** whether HeroMiners or Kryptex enforce limits on h·w, patterns, k or m/n.
- **A GPU reference miner:** the TH/s it reports (to confirm H = MAC) and the observed job-rotation interval.
- **LuckyPool:** whether the "mandated" config is enforced, and the CPU-port (3370) difficulty.
