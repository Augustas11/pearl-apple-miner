# Pre-PIP discussion: should Pearl consider an exact INT8 profile after v4?

I am exploring whether a later fork could admit a non-FP8 quantizer and an
exact integer matmul profile. This would change Pearl’s quantizer-specific
security premise, so I would like feedback on scope before advancing a PIP.

The current FP implementation admits H100 and B200 arithmetic profiles.
Our Apple M5 implementation reproduces the B200 reference bit-exactly in
software (fp8-scheme @ 2569546; the matmul code is unchanged at f696760 after
#355). On ordinary synthetic operands it reaches only 0.2–0.6 TOPS-equivalent
(TOPS = 2·MACs/s). The same 10-core M5 GPU peaks at about 19–21 TOPS on a
native INT8 GEMM microbenchmark (a fanless machine; sustained rates are
lower). These are component results. The logs are not yet published, and
proposed-profile throughput is unmeasured.

The candidate would add:

- `Int8RowPrequant`: deterministic per-row scaling and signed INT8 rounding
  after noising, with outputs in `[-127, 127]`.
- `INT8_EXACT`: exact INT8 dot products accumulated into INT32. At
  `k <= 65536`, the maximum absolute result is below `2^30`.
- A separately specified jackpot policy. A zero-product census is one
  candidate check, but I do not claim it replaces the current
  unpredictable-summands protection sufficiently.

The main unresolved issue is security. Exact accumulation removes device
truncation, but it does not establish resistance to low-rank, periodic-rounding,
residual-product, or grinding shortcuts.

The other unresolved issue is useful work. Ordinary INT8 inference results
would not validate this particular noising, quantization, and correction
pipeline. Separately, Pearl’s own `Llama-3.1-8B-Instruct-pearl` checkpoint
runs on Apple Silicon through MLX: its int7 layers load bit-exactly, and
perplexity is within 0.4% of the bf16 base model (M3 Ultra). That shows Macs
can serve Pearl’s models. It does not validate the proposed profile.

This would admit conformant INT8 hardware generally, including non-Apple
accelerators. Its effect on miner composition is part of the tradeoff.
Any activation would be considered separately after v4.

Questions for maintainers:

1. Is a non-FP8 quantizer a research direction you would consider for a later
   fork?
2. Which adversarial analyses and acceptance criteria would you require for
   its hardness premise and jackpot policy?
3. Which model-quality and useful-throughput baselines would justify adding
   the profile?

The next artifacts would be a plaintext reference implementation with test
vectors, a security evaluation, and model-quality/full-pipeline measurements.
A working draft exists, but none of those results is established yet.

— Augustas (Malibu), @Augustas11
