# SPDX-License-Identifier: Apache-2.0
"""Mining scheme boundary for Pearl cert-version specific mining paths."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import base64
from typing import Callable

import pearl_mining as pm

from .native import Desc, V4Desc, V4OperandDesc, words
from .v4_admission import validate_v4_g3_admission_file


Invoke = Callable[..., object]
MAX_U256 = (1 << 256) - 1
VERIFIER_FUNCTION = "pearl_mining.verify_plain_proof_for_cert_version"
V4_VERIFIER_FUNCTION = "pmkcore_v4_verify_plain_proof"


class VerifierRejection(ValueError):
    function = VERIFIER_FUNCTION


class V4VerifierRejection(ValueError):
    function = V4_VERIFIER_FUNCTION


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

    def validate_shape(self, shape, ram: int | None = None) -> int:
        return 0

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

    def job_descriptor(self, native, source, job, shape, a, bt, block_bound, share_bound, job_id):
        return Desc(abi_version=1,m=shape.m,n=shape.n,k=shape.k,a=a,bt=bt,
            a_bytes=shape.m*shape.k,bt_bytes=shape.n*shape.k,
            a_seed=words(bytes(job.a_noise_seed)),b_seed=words(bytes(job.b_noise_seed)),
            block_bound=words(block_bound),share_bound=words(share_bound),
            block_capacity=shape.result_capacity,share_capacity=shape.result_capacity,
            cert_version=self.kernel_cert_version(),rank=self.rank,job_id=job_id)

    def oracle_operand(self, job, bt):
        return bt

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

    def poll_kernel(self, native, handle):
        return native.poll(handle)

    def release_kernel(self, native, handle):
        native.release(handle)

    def kernel_cert_version(self) -> int:
        return self.cert_version

    def validate_cert_version(self, cert_version: int) -> None:
        if type(cert_version) is not int or cert_version != self.cert_version:
            raise ValueError(
                f"{self.name} requires cert_version={self.cert_version}; got {cert_version}"
            )

    def admit_source(self, source, native=None) -> None:
        self.validate_cert_version(source.cert_version)

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

    def scan_oracle(self, native, oracle, share_bound: int, block_bound: int):
        return native.scan(oracle, share_bound, block_bound)

    def release_job(self, native, job) -> None:
        return None


V3_SCHEME = Scheme(name="v3", cert_version=3, rank=128, pattern_id=1, kernel_symbol="pmk_run_job")


@dataclass(frozen=True)
class V4Config:
    m: int
    n: int
    k: int

    def to_bytes(self) -> bytes:
        return b"pmk-v4-grid-b200\0" + self.m.to_bytes(4,"little") + self.n.to_bytes(4,"little") + self.k.to_bytes(4,"little")


@dataclass
class V4Job:
    handle: object
    desc: object


class V4Scheme(Scheme):
    def __init__(self):
        super().__init__(name="v4", cert_version=4, rank=32, pattern_id=0,
                         kernel_symbol="pmk_v4_run_job", min_k=1024, max_k=65536)

    def ensure_k(self, k: int) -> None:
        if type(k) is not int or k not in (1024,4096,16384):
            raise ValueError("v4 production k must be one of {1024,4096,16384}")

    def build_config(self, native, m: int, n: int, k: int) -> bytes:
        self.ensure_k(k)
        return V4Config(m,n,k).to_bytes()

    def validate_config(self, config: bytes) -> V4Config:
        prefix = b"pmk-v4-grid-b200\0"
        if not isinstance(config, bytes) or len(config) != len(prefix) + 12 or not config.startswith(prefix):
            raise ValueError("v4 configuration policy/roundtrip failed")
        m = int.from_bytes(config[len(prefix):len(prefix)+4],"little")
        n = int.from_bytes(config[len(prefix)+4:len(prefix)+8],"little")
        k = int.from_bytes(config[len(prefix)+8:len(prefix)+12],"little")
        self.ensure_k(k)
        if m <= 0 or n <= 0 or m % 32 or n % 32:
            raise ValueError("v4 B200 GPU dimensions must be positive multiples of 32")
        if m > 8192 or n > 8192:
            raise ValueError("v4 B200 GPU dimensions must be <= 8192")
        return V4Config(m,n,k)


    def validate_shape(self, shape, ram: int | None = None) -> int:
        cfg = self.validate_config(self.build_config(None,shape.m,shape.n,shape.k))
        slots = int(getattr(shape,"slots",2))
        if slots not in (2,3):
            raise ValueError("v4 requires 2–3 retained slots")
        raw = (cfg.m + cfg.n) * cfg.k
        full_c = cfg.m * cfg.n * 4
        merkle = (cfg.m + cfg.n) * 64
        tiles = ((cfg.m * cfg.n + 255) // 256) * 128
        # Conservative total-pmk envelope: clean/raw int8, scale grids, noised/codes,
        # float A/Bᵀ staging, full FP32 C output, tile result buffers and Merkle state
        # retained across active slots, plus fixed allocator/command slack.
        per_job = 12 * raw + 2 * full_c + merkle + tiles + 64 * 1024**2
        estimate = slots * per_job + 4 * raw + 64 * 1024**2
        if ram is None:
            import subprocess
            ram = int(subprocess.check_output(['sysctl','-n','hw.memsize']))
        if estimate > ram // 4:
            raise ValueError("v4 combined pmk memory estimate exceeds 25% RAM")
        return estimate

    def target_bound(self, target: int, config: bytes) -> int:
        cfg = self.validate_config(config)
        if type(target) is not int or target <= 0 or target > MAX_U256:
            raise ValueError("v4 target must be in 1..2^256-1")
        return min(target * 256 * cfg.k, MAX_U256)

    def nbits_bound(self, nbits: int, config: bytes, target: int | None = None) -> int:
        native_target = compact_to_target(nbits)
        if native_target <= 0:
            raise ValueError("v4 compact target is invalid")
        if target is not None and native_target > target:
            raise ValueError("v4 compact share target is easier than pool target")
        return self.target_bound(native_target, config)

    def admit_source(self, source, native=None) -> None:
        super().admit_source(source,native=native)
        admission = validate_v4_g3_admission_file()
        ancestor_headers = getattr(source, "ancestor_headers", ())
        if not ancestor_headers:
            raise ValueError("v4 job missing ancestor_headers")
        if any(len(header) != 108 for header in ancestor_headers):
            raise ValueError("v4 ancestor_headers must contain 108-byte complete headers")
        if len(ancestor_headers) > 4:
            raise ValueError("v4 ancestor_headers exceeds four-header state window")
        if native is None or not hasattr(native,"ensure_v4"):
            raise ValueError("v4 G3 admission requires native v4 init")
        native.ensure_v4(admission)

    def build_template(self, native, source, config: bytes, shape, a):
        return source

    def build_job(self, native, template, bt, shape):
        ancestors = tuple(template.ancestor_headers)
        job = native.v4_create_job(template.header, ancestors[-1], b"".join(ancestors[:-1]),
                                   shape.m, shape.n, shape.k)
        try:
            desc = native.v4_gpu_descriptor(job)
        except BaseException:
            native.v4_free_job(job)
            raise
        return V4Job(job, desc)

    def job_descriptor(self, native, source, job, shape, a, bt, block_bound, share_bound, job_id):
        d = job.desc
        return V4Desc(abi_version=1,m=d.m,n=d.n,k=d.k,r=d.rank,
            a=V4OperandDesc(d.a_values, d.m*d.k, d.a_noise_e, d.m*d.rank,
                            d.a_noise_f, d.k*d.rank, d.a_alpha, d.a_beta, d.m),
            bt=V4OperandDesc(d.bt_values, d.n*d.k, d.bt_noise_e, d.n*d.rank,
                             d.bt_noise_f, d.k*d.rank, d.bt_alpha, d.bt_beta, d.n),
            jackpot_key=words(bytes(d.jackpot_key)),block_bound=words(block_bound),share_bound=words(share_bound),
            block_capacity=shape.result_capacity,share_capacity=shape.result_capacity,job_id=job_id)

    def dispatch_kernel(self, native, desc, callback, user, handle_out) -> None:
        native.v4_dispatch_desc(desc, callback, user, handle_out)

    def poll_kernel(self, native, handle):
        return native.v4_poll(handle)

    def release_kernel(self, native, handle):
        native.v4_release(handle)

    def oracle_operand(self, job, bt):
        return job

    @contextmanager
    def oracle(self, native, source, config: bytes, shape, a, bt):
        if not isinstance(bt, V4Job) or not bt.handle:
            raise ValueError("v4 oracle requires a live v4 job handle")
        yield bt.handle

    def scan_oracle(self, native, oracle, share_bound: int, block_bound: int):
        native.v4_prepare_oracle(oracle)
        return native.v4_scan(oracle, share_bound, block_bound)

    def build_proof(self, native, oracle, row: int, col: int) -> str:
        return native.v4_proof(oracle, row, col)

    def decode_proof(self, encoded: str, invoke: Invoke | None = None):
        try:
            return base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ValueError("malformed v4 proof") from exc

    def validate_proof_native(self, native, header: bytes, proof: bytes, config: bytes, share_nbits: int | None = None) -> None:
        self.validate_config(config)
        if not native.v4_verify_plain_proof(header, proof, share_nbits):
            raise V4VerifierRejection("v4 verifier gate failed")

    def release_job(self, native, job) -> None:
        if isinstance(job, V4Job) and job.handle:
            native.v4_free_job(job.handle)
            job.handle = None


V4_SCHEME = V4Scheme()


def scheme_for_cert_version(cert_version: int) -> Scheme:
    if type(cert_version) is int and cert_version == V3_SCHEME.cert_version:
        return V3_SCHEME
    if type(cert_version) is int and cert_version == V4_SCHEME.cert_version:
        return V4_SCHEME
    raise ValueError(f"unsupported Pearl mining cert_version={cert_version}")
