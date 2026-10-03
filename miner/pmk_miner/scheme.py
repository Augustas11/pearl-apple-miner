"""Mining scheme boundary.

Only Pearl cert-v3 is implemented.  The interface is intentionally small so a
future v4 implementation has a single place to change job construction,
kernel selection metadata, proof decoding and verification.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import pearl_mining as pm


Invoke = Callable[..., object]
MAX_U256 = (1 << 256) - 1
VERIFIER_FUNCTION = "pearl_mining.verify_plain_proof_for_cert_version"


class VerifierRejection(ValueError):
    function = VERIFIER_FUNCTION


def compact_to_target(bits: int) -> int:
    if type(bits) is not int:
        raise ValueError("compact bits must be a strict integer")
    exponent = (bits >> 24) & 0xFF
    mantissa = bits & 0xFFFFFF
    if exponent == 0 or mantissa & 0x800000:
        return 0
    if exponent <= 3:
        target = mantissa >> (8 * (3 - exponent))
    else:
        shift = 8 * (exponent - 3)
        if mantissa.bit_length() + shift > 256:
            raise ValueError("compact target overflows U256")
        target = mantissa << shift
    if target > MAX_U256:
        raise ValueError("compact target overflows U256")
    return target


@dataclass(frozen=True)
class Scheme:
    name: str
    cert_version: int
    rank: int
    pattern_id: int
    kernel_symbol: str
    min_k: int = 2048
    max_k: int = 8192

    def ensure_k(self, k: int) -> None:
        if not (self.min_k <= k <= self.max_k) or k % self.rank:
            raise ValueError(
                f"{self.name} production k must be in [{self.min_k},{self.max_k}] "
                f"and divisible by {self.rank}"
            )

    def build_config(self, native, m: int, n: int, k: int) -> bytes:
        self.ensure_k(k)
        return native.config(self.pattern_id, m, n, k)

    def parse_config(self, config: bytes):
        return pm.MiningConfiguration.from_bytes(config)

    def validate_config(self, config: bytes) -> object:
        cfg = self.parse_config(config)
        if (
            cfg.rank != self.rank
            or cfg.common_dim % self.rank
            or bytes(cfg.to_bytes()) != config
        ):
            raise ValueError(f"{self.name} configuration policy/roundtrip failed")
        return cfg

    def target_bound(self, target: int, config: bytes) -> int:
        cfg = self.validate_config(config)
        bound = target * 32 * cfg.common_dim
        penalized = pm.penalized_target_bound(target, cfg)
        if penalized is None or penalized != bound or bound >= 2**256:
            raise ValueError(f"{self.name} rank-penalized bound differs or overflows")
        return bound

    def nbits_bound(self, nbits: int, config: bytes, target: int | None = None) -> int:
        nbits_to_difficulty = getattr(pm, "nbits_to_difficulty", None)
        extract_difficulty_bound = getattr(pm, "extract_difficulty_bound", None)
        if not callable(nbits_to_difficulty) or not callable(extract_difficulty_bound):
            raise ValueError("py-pearl-mining missing pool difficulty-bound helpers")
        native_target = int(nbits_to_difficulty(nbits))
        if native_target != compact_to_target(nbits):
            raise ValueError(f"{self.name} compact target helper mismatch")
        if target is not None and native_target > target:
            raise ValueError(f"{self.name} compact share target is easier than pool target")
        bound = int(extract_difficulty_bound(nbits, self.validate_config(config)))
        safe = self.target_bound(native_target, config)
        if bound != safe:
            raise ValueError(f"{self.name} native difficulty bound mismatch")
        return bound

    def build_template(self, native, source, config: bytes, shape, a):
        return native.template(source.header, config, shape.m, shape.n, a, shape.k)

    def build_job(self, native, template, bt, shape):
        return native.build(template, bt, shape.n, shape.k)

    @contextmanager
    def oracle(self, native, source, config: bytes, shape, a, bt):
        with native.oracle(source.header, config, shape.m, shape.n, shape.k, a, bt) as handle:
            yield handle

    def build_proof(self, native, oracle, row: int, col: int) -> str:
        return native.proof(oracle, row, col)

    def kernel_entry(self, native):
        return getattr(native.metal, self.kernel_symbol)

    def dispatch_kernel(self, native, desc, callback, user, handle_out) -> None:
        native.call(self.kernel_entry(native), native.context, desc, callback, user, handle_out)

    def kernel_cert_version(self) -> int:
        return self.cert_version

    def validate_cert_version(self, cert_version: int) -> None:
        if type(cert_version) is not int or cert_version != self.cert_version:
            raise ValueError(
                f"{self.name} requires cert_version={self.cert_version}; got {cert_version}"
            )

    def decode_proof(self, encoded: str, invoke: Invoke | None = None):
        call = invoke or (lambda function, *args, **kwargs: function(*args, **kwargs))
        return call(pm.PlainProof.from_base64, encoded)

    def verify(
        self,
        header: bytes,
        proof,
        share_nbits: int | None = None,
        invoke: Invoke | None = None,
    ) -> bool:
        ok, _message = self.verify_with_message(header, proof, share_nbits, invoke=invoke)
        return ok

    def verify_with_message(
        self,
        header: bytes,
        proof,
        share_nbits: int | None = None,
        invoke: Invoke | None = None,
    ) -> tuple[bool, str]:
        call = invoke or (lambda function, *args, **kwargs: function(*args, **kwargs))
        kwargs = {} if share_nbits is None else {"nbits_override": share_nbits}
        ok, message = call(
            pm.verify_plain_proof_for_cert_version,
            self.cert_version,
            pm.IncompleteBlockHeader.from_bytes(header),
            proof,
            **kwargs,
        )
        return bool(ok), str(message)

    def validate_proof_config(self, proof, config: bytes) -> None:
        if proof.noise_rank != self.rank or proof.k % self.rank:
            raise ValueError(f"{self.name} rank policy failed before verifier")
        reconstructed = pm.MiningConfiguration(
            proof.k,
            self.rank,
            pm.MMAType.Int7xInt7ToInt32,
            pm.PeriodicPattern.from_list(
                [int(x) - int(proof.a.row_indices[0]) for x in proof.a.row_indices]
            ),
            pm.PeriodicPattern.from_list(
                [int(x) - int(proof.bt.row_indices[0]) for x in proof.bt.row_indices]
            ),
            None,
        )
        if bytes(reconstructed.to_bytes()) != config:
            raise ValueError(f"{self.name} proof/config mismatch")

    def validate_proof(
        self,
        header: bytes,
        proof,
        config: bytes,
        share_nbits: int | None = None,
        invoke: Invoke | None = None,
    ) -> None:
        self.validate_proof_config(proof, config)
        ok, message = self.verify_with_message(header, proof, share_nbits, invoke=invoke)
        if not ok:
            detail = message.strip() or "rejected without verifier detail"
            raise VerifierRejection(f"{self.name} verifier gate failed: {detail}")


V3_SCHEME = Scheme(name="v3", cert_version=3, rank=128, pattern_id=1, kernel_symbol="pmk_run_job")


def scheme_for_cert_version(cert_version: int) -> Scheme:
    if type(cert_version) is int and cert_version == V3_SCHEME.cert_version:
        return V3_SCHEME
    raise ValueError(f"unsupported Pearl mining cert_version={cert_version}")
