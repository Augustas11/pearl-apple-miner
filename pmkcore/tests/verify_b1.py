#!/usr/bin/env python3
"""Subprocess verifier for pmkcore B1 integration tests.

Input is one whitespace-separated case per line:

    <name> <expect:ok|err> <header_hex> <plain_proof_hex> [nbits_override_hex|-]
    mine_cmp <name> <m> <n> <k> <header_hex> <config_hex> <rows_csv> <cols_csv> <root_a_hex> <root_b_hex>
    malformed <name> <kind> <header_hex> <proof_hex> <nbits_hex> <expect>
    raw65 <name> <m> <n> <k> <header_hex> <config_hex>

The proof bytes are pmkcore's bincode PlainProof bytes.  The installed
pearl_mining binding exposes PlainProof.from_base64(), so this harness keeps the
Rust side dependency-free by accepting hex and converting locally.
"""

from __future__ import annotations

import base64
import sys

import pearl_mining


def run_case(line: str) -> tuple[bool, str]:
    parts = line.split()
    if parts and parts[0] == "mine_cmp":
        if len(parts) != 11:
            return False, f"mine_cmp: expected 11 fields, got {len(parts)}"
        _, name, m, n, k, header_hex, config_hex, rows_csv, cols_csv, root_a_hex, root_b_hex = parts
        header = pearl_mining.IncompleteBlockHeader.from_bytes(bytes.fromhex(header_hex))
        config = pearl_mining.MiningConfiguration.from_bytes(bytes.fromhex(config_hex))
        proof = pearl_mining.mine(
            int(m),
            int(n),
            int(k),
            header,
            config,
            signal_range=(0, 0),
            wrong_jackpot_hash=False,
            cert_version=3,
        )
        verified, message = pearl_mining.verify_plain_proof_for_cert_version(3, header, proof)
        if not verified:
            return False, f"{name}: mined proof rejected: {message}"
        rows = [int(x) for x in rows_csv.split(",") if x]
        cols = [int(x) for x in cols_csv.split(",") if x]
        checks = [
            (list(proof.a.row_indices) == rows, "A rows"),
            (list(proof.bt.row_indices) == cols, "B cols"),
            (bytes(proof.a.root).hex() == root_a_hex, "A root"),
            (bytes(proof.bt.root).hex() == root_b_hex, "B root"),
        ]
        bad = [label for ok, label in checks if not ok]
        if bad:
            return False, f"{name}: pearl_mining.mine mismatch: {', '.join(bad)}"
        return True, f"{name}: pearl_mining.mine matches oracle"

    if parts and parts[0] == "malformed":
        if len(parts) != 7:
            return False, f"malformed: expected 7 fields, got {len(parts)}"
        _, name, kind, header_hex, proof_hex, nbits_hex, expect = parts
        header = pearl_mining.IncompleteBlockHeader.from_bytes(bytes.fromhex(header_hex))
        proof = pearl_mining.PlainProof.from_base64(
            base64.b64encode(bytes.fromhex(proof_hex)).decode("ascii")
        )
        if kind == "noise_rank64":
            proof = pearl_mining.PlainProof(proof.m, proof.n, proof.k, 64, proof.a, proof.bt, proof.moe)
        elif kind == "noise_rank256":
            proof = pearl_mining.PlainProof(proof.m, proof.n, proof.k, 256, proof.a, proof.bt, proof.moe)
        else:
            return False, f"{name}: unknown malformed kind {kind}"
        return verify_expect(name, expect, header, proof, nbits_hex)

    if parts and parts[0] == "raw65":
        if len(parts) != 7:
            return False, f"raw65: expected 7 fields, got {len(parts)}"
        _, name, m, n, k, header_hex, config_hex = parts
        header = pearl_mining.IncompleteBlockHeader.from_bytes(bytes.fromhex(header_hex))
        config = pearl_mining.MiningConfiguration.from_bytes(bytes.fromhex(config_hex))
        proof = pearl_mining.mine(
            int(m),
            int(n),
            int(k),
            header,
            config,
            signal_range=(65, 65),
            wrong_jackpot_hash=False,
            cert_version=3,
        )
        return verify_expect(name, "err", header, proof, "-")

    if len(parts) != 5:
        return False, f"bad input line: expected 5 fields, got {len(parts)}"
    name, expect, header_hex, proof_hex, nbits_hex = parts
    if expect not in {"ok", "err"}:
        return False, f"{name}: bad expectation {expect!r}"

    header = pearl_mining.IncompleteBlockHeader.from_bytes(bytes.fromhex(header_hex))
    proof_bytes = bytes.fromhex(proof_hex)
    proof = pearl_mining.PlainProof.from_base64(base64.b64encode(proof_bytes).decode("ascii"))
    return verify_expect(name, expect, header, proof, nbits_hex)


def verify_expect(name, expect, header, proof, nbits_hex) -> tuple[bool, str]:
    nbits_override = None if nbits_hex == "-" else int(nbits_hex, 16)
    try:
        ok, detail = pearl_mining.verify_plain_proof_for_cert_version(3, header, proof, nbits_override)
    except Exception as exc:
        ok = False
        detail = str(exc)

    if ok == (expect == "ok"):
        return True, f"{name}: {detail}"
    return False, f"{name}: expected {expect}, got {'ok' if ok else 'err'}: {detail}"


def main() -> int:
    failures: list[str] = []
    count = 0
    for raw in sys.stdin:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        count += 1
        ok, msg = run_case(line)
        print(msg)
        if not ok:
            failures.append(msg)
    if failures:
        print(f"{len(failures)}/{count} verifier cases failed", file=sys.stderr)
        return 1
    print(f"{count} verifier cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
