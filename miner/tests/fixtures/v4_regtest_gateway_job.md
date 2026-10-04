<!-- SPDX-License-Identifier: Apache-2.0 -->

# V4 solo regression fixture

`v4_regtest_gateway_job.json` is the unmodified `getMiningInfo` result captured
from the pinned B9 regtest gateway during B9-fix. The source node uses Pearl
`f696760b259500ecb608469ea3953aeabbe78948`, with the permanent v4 gateway adapter.
It contains the height-1 incomplete header, target, version 4, complete 108-byte
parent header, and coinbase authorization data; it contains no RPC credentials.

Capture: the unmodified `bash scripts/pmk_regtest_e2e_v4.sh`, with a temporary
read-only Python trace hook to save the gateway response and the otherwise
suppressed `ValueError` traceback. Evidence:
`bench/evidence/b9_fix_v4_diagnostic.txt` and
`bench/evidence/b9_fix_valueerror_traceback.txt`.

`test_v4_solo_entry.py` replays this response over loopback into the real
`pmk_miner.__main__.main` solo path. It uses the actual native libraries and
GPU, checks admission and coinbase authorization, and gracefully stops on GPU
dispatch. The full node/certificate/coinbase acceptance proof is the separate
unmodified v4 E2E run. No mining, admission, or verifier implementation is
replaced in this regression. The test requires the built libraries and a fresh
hardware-matching admission record; set `PMK_V4_G3_ADMISSION_FILE` to a renewed
record when the recorded B9 admission expires. The test deliberately has no skip.
