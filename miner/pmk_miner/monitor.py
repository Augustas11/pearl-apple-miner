"""Health monitors and payout checks for pmk solo mode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


POW_DENOMINATOR = 1 << 257
DIFF1_TARGET = 0xFFFF << 208
BECH32M_CONST = 0x2BC830A3
_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


class MonitorAlarm(RuntimeError):
    """Raised when a monitor detects a hard health failure."""


def bits_to_target(bits: int) -> int:
    exponent = (bits >> 24) & 0xFF
    mantissa = bits & 0xFFFFFF
    if exponent <= 3:
        return mantissa >> (8 * (3 - exponent))
    return mantissa << (8 * (exponent - 3))


def target_to_bits_floor(target: int) -> int:
    """Convert target to compact bits, rounding down to avoid easier targets."""

    if target <= 0:
        raise ValueError("target must be positive")
    raw = target.to_bytes(max(1, (target.bit_length() + 7) // 8), "big")
    exponent = len(raw)
    if raw[0] & 0x80:
        mantissa = int.from_bytes(b"\x00" + raw[:2], "big")
        exponent += 1
    else:
        mantissa = int.from_bytes(raw[:3].ljust(3, b"\x00"), "big")
    bits = (exponent << 24) | mantissa
    while bits_to_target(bits) > target:
        mantissa -= 1
        if mantissa <= 0:
            exponent -= 1
            mantissa = 0xFFFF
        bits = (exponent << 24) | mantissa
    return bits


def choose_share_nbits(ops_per_second: float, *, target_shares_per_minute: float = 1.0) -> int:
    if ops_per_second <= 0:
        raise ValueError("ops_per_second must be positive")
    if target_shares_per_minute <= 0:
        raise ValueError("target_shares_per_minute must be positive")
    target = int(POW_DENOMINATOR * target_shares_per_minute / (60.0 * ops_per_second))
    target = max(1, min(target, DIFF1_TARGET))
    return target_to_bits_floor(target)


def expected_shares(completed_ops: int, share_nbits: int) -> float:
    if completed_ops < 0:
        raise ValueError("completed_ops must be non-negative")
    return completed_ops * bits_to_target(share_nbits) / POW_DENOMINATOR


def expected_blocks_per_day(ops_per_second: float, block_target: int) -> float:
    if ops_per_second < 0:
        raise ValueError("ops_per_second must be non-negative")
    if block_target < 0:
        raise ValueError("block_target must be non-negative")
    return ops_per_second * 86400.0 * block_target / POW_DENOMINATOR


def poisson_interval(lam: float, *, alpha: float = 0.001) -> tuple[int, int]:
    """Inclusive exact equal-tail Poisson interval.

    The summation starts at the mode and expands outward. That avoids the
    ``exp(-lambda)`` underflow that appears in 24h G6 windows around E=1440.
    """

    if lam < 0:
        raise ValueError("lambda must be non-negative")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if lam == 0:
        return (0, 0)
    tail = alpha / 2.0

    mode = max(0, int(math.floor(lam)))
    mode_pmf = math.exp(mode * math.log(lam) - lam - math.lgamma(mode + 1))

    lower_probs: list[tuple[int, float]] = [(mode, mode_pmf)]
    mass = mode_pmf
    k = mode
    while k > 0:
        mass *= k / lam
        k -= 1
        lower_probs.append((k, mass))

    cdf = 0.0
    lower = 0
    for k, mass in reversed(lower_probs):
        cdf += mass
        if cdf > tail:
            lower = k
            break

    cdf = sum(mass for _, mass in lower_probs)
    mass = mode_pmf
    k = mode
    upper = mode
    hard_limit = int(math.ceil(lam + 20 * math.sqrt(lam) + 100))
    while k < hard_limit:
        mass *= lam / (k + 1)
        k += 1
        cdf += mass
        if cdf >= 1.0 - tail:
            upper = k
            break
        upper = k
    return (lower, upper)


@dataclass(slots=True)
class ShareWindow:
    completed_ops: int
    observed_shares: int
    expected: float
    lower: int
    upper: int
    alarm: bool


@dataclass(slots=True)
class ShareMonitor:
    share_nbits: int
    alpha: float = 0.001
    max_completed_ops: int | None = None
    completed_ops: int = 0
    observed_shares: int = 0

    def record(self, *, completed_ops: int, shares: int) -> ShareWindow:
        if completed_ops < 0 or shares < 0:
            raise ValueError("completed_ops and shares must be non-negative")
        self.completed_ops += completed_ops
        self.observed_shares += shares
        if self.max_completed_ops is not None and self.completed_ops > self.max_completed_ops:
            raise MonitorAlarm("share monitor window exceeded max_completed_ops; reset fixed window")
        expected = expected_shares(self.completed_ops, self.share_nbits)
        lower, upper = poisson_interval(expected, alpha=self.alpha)
        alarm = self.observed_shares < lower or self.observed_shares > upper
        return ShareWindow(
            completed_ops=self.completed_ops,
            observed_shares=self.observed_shares,
            expected=expected,
            lower=lower,
            upper=upper,
            alarm=alarm,
        )

    def reset(self) -> None:
        self.completed_ops = 0
        self.observed_shares = 0


def validate_payout_startup(
    address: str,
    expected_hrp: str,
    *,
    approved_script_hex: str | None = None,
) -> str:
    script = script_pubkey_from_p2tr_address(address, expected_hrp)
    script_hex = script.hex()
    if approved_script_hex is not None and script_hex.lower() != approved_script_hex.lower():
        raise MonitorAlarm("configured mining address does not match approved payout script")
    return script_hex


def validate_coinbase_payout(
    block: dict[str, Any],
    *,
    approved_script_hex: str,
    expected_hrp: str | None = None,
) -> bool:
    txs = block.get("rawtx") or block.get("tx") or block.get("transactions")
    if not txs:
        raise MonitorAlarm("accepted block has no coinbase transaction")
    coinbase = txs[0]
    if isinstance(coinbase, str):
        raise MonitorAlarm("coinbase payout check needs verbose transaction data")
    for output in coinbase.get("vout", []):
        script = output.get("scriptPubKey", {})
        if expected_hrp is not None:
            address = _first_address(script)
            if address is not None:
                validate_payout_startup(address, expected_hrp)
        if str(script.get("hex", "")).lower() == approved_script_hex.lower():
            return True
    raise MonitorAlarm("accepted block coinbase does not pay the approved script")


def _first_address(script: dict[str, Any]) -> str | None:
    if "address" in script:
        return str(script["address"])
    addresses = script.get("addresses")
    if isinstance(addresses, list) and addresses:
        return str(addresses[0])
    return None


def script_pubkey_from_p2tr_address(address: str, expected_hrp: str) -> bytes:
    hrp, data = _bech32_decode(address)
    if hrp != expected_hrp:
        raise MonitorAlarm(f"payout HRP {hrp!r} does not match expected {expected_hrp!r}")
    if not data or data[0] != 1:
        raise MonitorAlarm("payout address is not witness version 1")
    decoded = _convertbits(data[1:], 5, 8, False)
    program = bytes(decoded)
    if len(program) != 32:
        raise MonitorAlarm("taproot payout program must be 32 bytes")
    return b"\x51\x20" + program


def _bech32_decode(address: str) -> tuple[str, list[int]]:
    if address.lower() != address and address.upper() != address:
        raise MonitorAlarm("mixed-case bech32 address")
    text = address.lower()
    pos = text.rfind("1")
    if pos < 1 or pos + 7 > len(text):
        raise MonitorAlarm("invalid bech32 address separator")
    hrp = text[:pos]
    try:
        data = [_CHARSET.index(ch) for ch in text[pos + 1 :]]
    except ValueError as exc:
        raise MonitorAlarm("invalid bech32 character") from exc
    if _bech32_polymod(_hrp_expand(hrp) + data) != BECH32M_CONST:
        raise MonitorAlarm("taproot address must have a valid bech32m checksum")
    return hrp, data[:-6]


def _hrp_expand(hrp: str) -> list[int]:
    return [ord(ch) >> 5 for ch in hrp] + [0] + [ord(ch) & 31 for ch in hrp]


def _bech32_polymod(values: list[int]) -> int:
    generators = [0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3]
    chk = 1
    for value in values:
        top = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ value
        for i, generator in enumerate(generators):
            if (top >> i) & 1:
                chk ^= generator
    return chk


def _convertbits(data: list[int], from_bits: int, to_bits: int, pad: bool) -> list[int]:
    acc = 0
    bits = 0
    ret: list[int] = []
    maxv = (1 << to_bits) - 1
    max_acc = (1 << (from_bits + to_bits - 1)) - 1
    for value in data:
        if value < 0 or value >> from_bits:
            raise MonitorAlarm("invalid bech32 data value")
        acc = ((acc << from_bits) | value) & max_acc
        bits += from_bits
        while bits >= to_bits:
            bits -= to_bits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (to_bits - bits)) & maxv)
    elif bits >= from_bits or ((acc << (to_bits - bits)) & maxv):
        raise MonitorAlarm("invalid bech32 padding")
    return ret
