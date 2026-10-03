"""pearl-gateway launcher that records every submitblock call (timestamp + node verdict).

Negative control: when RTAP_CORRUPT_FIRST=1, before the first genuine submission it
submits two corrupted copies of the same block (one flipped byte in the ZK proof bytes,
one in the public data) and logs pearld's verbatim verdict. The genuine block is then
submitted unchanged. Rejected blocks are not indexed, so the genuine one still lands.
"""
import os
import sys
import time

from pearl_gateway import pearl_client as pc

_LOG = os.environ["RTAP_LOG"]
_CORRUPT_FIRST = os.environ.get("RTAP_CORRUPT_FIRST") == "1"
_orig = pc.PearlNodeClient.submit_block
_state = {"n": 0}


def _log(msg: str) -> None:
    with open(_LOG, "a") as f:
        f.write(f"{time.time():.3f} {msg}\n")


def _flip(block_hex: str, where: str) -> str:
    b = bytearray.fromhex(block_hex)
    publen = int.from_bytes(b[36:40], "little")
    pub_off = 40
    proflen = int.from_bytes(b[pub_off + publen : pub_off + publen + 4], "little")
    proof_off = pub_off + publen + 4
    off = proof_off + proflen // 2 if where == "proof" else pub_off + publen // 2
    b[off] ^= 0x01
    return b.hex()


async def submit_block(self, block_hex: str) -> str:
    _state["n"] += 1
    n = _state["n"]
    if _CORRUPT_FIRST and n == 1:
        for where in ("proof", "public_data"):
            try:
                r = await _orig(self, _flip(block_hex, where))
            except Exception as e:  # RPC error path
                r = "RPC-ERROR redacted"
            _log(f"NEGATIVE corrupt_{where} verdict={r}")
    _log(f"SUBMIT n={n} start")
    r = await _orig(self, block_hex)
    _log(f"SUBMIT n={n} verdict={r}")
    return r


pc.PearlNodeClient.submit_block = submit_block

from pearl_gateway.submission_service import SubmissionService  # noqa: E402

_orig_sub = SubmissionService.submit_plain_proof


async def submit_plain_proof(self, plain_proof, template, identity=None):
    # find -> accept clock starts here: the miner has just handed the proof to the gateway.
    _log(f"FOUND template_time={template.header.timestamp}")
    return await _orig_sub(self, plain_proof, template, identity)


SubmissionService.submit_plain_proof = submit_plain_proof

from pearl_gateway.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = ["pearl-gateway", "start", "--debug"]
    main()
