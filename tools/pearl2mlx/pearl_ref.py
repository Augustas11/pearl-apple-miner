#!/usr/bin/env python3
"""Numpy reference forward pass for ORIGINAL Pearl Llama checkpoints.

Reads the Pearl safetensors directly (no torch, no MLX) and computes logits:
  * int7 layers  : W = int7 * weight_scale[n,1]                     (exact in fp32)
  * FP8 layers   : W = e4m3(weight) * weight_scale[i//128, j//128]   (exact in fp32)
  * smooth scale : y = (x * smooth_quant_scale) @ W^T
  * --a7         : int7 layers emulate Pearl quant_7bit per-token activations:
                   scale = amax(x*s)/63, q = clamp(rint(x*s*63/amax), +-63),
                   y = (q @ int7^T) * scale * weight_scale   (exact division, not rcp_approx)
FP8 activation quantization (per-group-128 dynamic) is NOT emulated.
Everything else (RMSNorm, llama3 RoPE, GQA attention, SwiGLU) runs in fp32/fp64.

Also the safetensors / E4M3 helpers shared with pearl2mlx.py and the tests.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# safetensors reading (memmap; supports I8 / BF16 / F8_E4M3 which numpy's
# safetensors backend cannot represent)
# ---------------------------------------------------------------------------
_ST_DTYPES = {
    "BOOL": np.bool_, "U8": np.uint8, "I8": np.int8, "I16": np.int16,
    "I32": np.int32, "I64": np.int64, "F16": np.float16, "F32": np.float32,
    "F64": np.float64, "BF16": np.uint16, "F8_E4M3": np.uint8,
}


def read_st_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def bf16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even fp32 -> bf16 bit pattern (NaN preserved)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    nan = np.isnan(x)
    return np.where(nan, (u >> 16) | 0x40, rounded).astype(np.uint16)


def e4m3_lut() -> np.ndarray:
    """Independent OCP E4M3FN decoder: 256-entry fp32 table (0x7F/0xFF = NaN)."""
    out = np.empty(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0xF and m == 0x7:
            out[b] = np.nan
        elif e == 0:
            out[b] = sign * (m / 8.0) * 2.0 ** -6
        else:
            out[b] = sign * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return out


E4M3_LUT = e4m3_lut()


def e4m3_is_nan(codes: np.ndarray) -> np.ndarray:
    return (np.asarray(codes) & 0x7F) == 0x7F


def block_expand(scale: np.ndarray, n: int, k: int, block: int = 128) -> np.ndarray:
    """scale[ceil(n/b), ceil(k/b)] -> [n, k] (ragged edges cropped)."""
    return np.repeat(np.repeat(scale, block, axis=0), block, axis=1)[:n, :k]


class PearlCheckpoint:
    """Lazy, memmapped view over a Pearl HF snapshot directory."""

    def __init__(self, src: str | Path):
        self.src = Path(src)
        self.config = json.loads((self.src / "config.json").read_text())
        idx = self.src / "model.safetensors.index.json"
        if idx.exists():
            files = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
        else:
            files = ["model.safetensors"]
        self.files = files
        self.meta: dict[str, tuple[str, dict, int]] = {}
        for fn in files:
            hdr, base = read_st_header(self.src / fn)
            for name, info in hdr.items():
                self.meta[name] = (fn, info, base)
        self._mm: dict[str, np.memmap] = {}

    def names_in(self, fn: str) -> list[str]:
        return [n for n, (f, _, _) in self.meta.items() if f == fn]

    def has(self, name: str) -> bool:
        return name in self.meta

    def dtype(self, name: str) -> str:
        return self.meta[name][1]["dtype"]

    def shape(self, name: str) -> list[int]:
        return list(self.meta[name][1]["shape"])

    def raw(self, name: str) -> np.ndarray:
        """Raw storage view (BF16 -> uint16 bits, F8_E4M3 -> uint8 codes)."""
        fn, info, base = self.meta[name]
        mm = self._mm.get(fn)
        if mm is None:
            mm = self._mm[fn] = np.memmap(self.src / fn, dtype=np.uint8, mode="r")
        s, e = info["data_offsets"]
        dt = _ST_DTYPES[info["dtype"]]
        return mm[base + s: base + e].view(dt).reshape(info["shape"])

    def f32(self, name: str) -> np.ndarray:
        a = self.raw(name)
        d = self.dtype(name)
        if d == "BF16":
            return bf16_bits_to_f32(a)
        if d == "F8_E4M3":
            return E4M3_LUT[a]
        return a.astype(np.float32)

    # -- linear layers ------------------------------------------------------
    def linear_kind(self, prefix: str) -> str:
        d = self.dtype(prefix + ".weight")
        if d == "I8":
            return "int7"
        if d == "F8_E4M3":
            return "fp8"
        return "dense"

    def smooth(self, prefix: str) -> np.ndarray | None:
        n = prefix + ".smooth_quant_scale"
        return self.f32(n) if self.has(n) else None

    def dequant(self, prefix: str) -> np.ndarray:
        """Dequantized fp32 weight [n, k] (smooth NOT applied)."""
        kind = self.linear_kind(prefix)
        if kind == "int7":
            q = self.raw(prefix + ".weight").astype(np.float32)
            return q * self.f32(prefix + ".weight_scale").reshape(-1, 1)
        if kind == "fp8":
            codes = self.raw(prefix + ".weight")
            if e4m3_is_nan(codes).any():
                raise ValueError(f"{prefix}.weight contains E4M3 NaN codes")
            n, k = codes.shape
            return E4M3_LUT[codes] * block_expand(self.f32(prefix + ".weight_scale"), n, k)
        return self.f32(prefix + ".weight")


# ---------------------------------------------------------------------------
# reference forward
# ---------------------------------------------------------------------------
def _rope_inv_freq(cfg: dict, head_dim: int) -> np.ndarray:
    base = float(cfg.get("rope_theta", 10000.0))
    inv = 1.0 / base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    rs = cfg.get("rope_scaling") or {}
    rtype = rs.get("rope_type", rs.get("type"))
    if rtype == "llama3":
        factor = rs["factor"]
        lo = rs.get("low_freq_factor", 1.0)
        hi = rs.get("high_freq_factor", 4.0)
        old = rs.get("original_max_position_embeddings", 8192)
        wavelen = 2 * np.pi / inv
        lo_wl, hi_wl = old / lo, old / hi
        scaled = np.where(wavelen > lo_wl, inv / factor, inv)
        smooth = (old / wavelen - lo) / (hi - lo)
        smoothed = (1 - smooth) * scaled / factor + smooth * scaled
        medium = (wavelen >= hi_wl) & (wavelen <= lo_wl)
        inv = np.where(medium, smoothed, scaled)
    elif rtype not in (None, "default"):
        raise NotImplementedError(f"rope_type {rtype}")
    return inv


def _apply_rope(x: np.ndarray, inv: np.ndarray) -> np.ndarray:
    # x: [heads, L, d], non-traditional (rotate_half)
    L = x.shape[1]
    ang = np.arange(L, dtype=np.float64)[:, None] * inv[None, :]
    cos = np.cos(ang).astype(np.float32)
    sin = np.sin(ang).astype(np.float32)
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _rmsnorm(x: np.ndarray, g: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * g


def _linear(ck: PearlCheckpoint, prefix: str, x: np.ndarray, a7: bool) -> np.ndarray:
    s = ck.smooth(prefix)
    if a7 and ck.linear_kind(prefix) == "int7":
        xs = x * s if s is not None else x
        amax = np.max(np.abs(xs), axis=-1, keepdims=True)
        q = np.clip(np.rint(xs * (63.0 / (amax + 1e-30))), -63, 63)
        w = ck.raw(prefix + ".weight").astype(np.float64)
        acc = q.astype(np.float64) @ w.T  # integer-exact in fp64
        ws = ck.f32(prefix + ".weight_scale").reshape(1, -1)
        return (acc * (amax / 63.0) * ws).astype(np.float32)
    if s is not None:
        x = x * s
    return x @ ck.dequant(prefix).T


def forward(ck: PearlCheckpoint, prompts: list[list[int]], n_layers: int | None = None,
            a7: bool = False, log=None) -> list[np.ndarray]:
    """Logits [L_i, vocab] (fp32) for each token list; all prompts batched per layer."""
    cfg = ck.config
    if cfg.get("model_type") != "llama":
        raise NotImplementedError("pearl_ref supports model_type=llama only")
    H = cfg["num_attention_heads"]
    KV = cfg.get("num_key_value_heads", H)
    hd = cfg.get("head_dim") or cfg["hidden_size"] // H
    eps = cfg["rms_norm_eps"]
    nl = cfg["num_hidden_layers"] if n_layers is None else n_layers
    inv = _rope_inv_freq(cfg, hd)
    lens = [len(p) for p in prompts]
    offs = np.cumsum([0] + lens)
    ids = np.concatenate([np.asarray(p, dtype=np.int64) for p in prompts])
    x = bf16_bits_to_f32(ck.raw("model.embed_tokens.weight")[ids])
    for li in range(nl):
        p = f"model.layers.{li}"
        h = _rmsnorm(x, ck.f32(f"{p}.input_layernorm.weight"), eps)
        q = _linear(ck, f"{p}.self_attn.q_proj", h, a7)
        k = _linear(ck, f"{p}.self_attn.k_proj", h, a7)
        v = _linear(ck, f"{p}.self_attn.v_proj", h, a7)
        att = np.empty((x.shape[0], H * hd), dtype=np.float32)
        for i in range(len(prompts)):
            a, b = offs[i], offs[i + 1]
            L = b - a
            qi = _apply_rope(q[a:b].reshape(L, H, hd).transpose(1, 0, 2), inv)
            ki = _apply_rope(k[a:b].reshape(L, KV, hd).transpose(1, 0, 2), inv)
            vi = v[a:b].reshape(L, KV, hd).transpose(1, 0, 2)
            rep = H // KV
            ki = np.repeat(ki, rep, axis=0)
            vi = np.repeat(vi, rep, axis=0)
            sc = (qi @ ki.transpose(0, 2, 1)).astype(np.float64) * hd ** -0.5
            sc = sc + np.triu(np.full((L, L), -np.inf), 1)
            sc = np.exp(sc - sc.max(-1, keepdims=True))
            sc = (sc / sc.sum(-1, keepdims=True)).astype(np.float32)
            att[a:b] = (sc @ vi).transpose(1, 0, 2).reshape(L, H * hd)
        x = x + _linear(ck, f"{p}.self_attn.o_proj", att, a7)
        h = _rmsnorm(x, ck.f32(f"{p}.post_attention_layernorm.weight"), eps)
        g = _linear(ck, f"{p}.mlp.gate_proj", h, a7)
        u = _linear(ck, f"{p}.mlp.up_proj", h, a7)
        x = x + _linear(ck, f"{p}.mlp.down_proj", g / (1 + np.exp(-g)) * u, a7)
        if log:
            log(f"layer {li} done")
    x = _rmsnorm(x, ck.f32("model.norm.weight"), eps)
    head = "model.embed_tokens.weight" if cfg.get("tie_word_embeddings") else "lm_head.weight"
    logits = x @ ck.f32(head).T
    return [logits[offs[i]:offs[i + 1]] for i in range(len(prompts))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Pearl HF snapshot dir")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="text; tokenized with the snapshot tokenizer (adds BOS)")
    g.add_argument("--tokens", help="comma-separated token ids")
    ap.add_argument("--layers", type=int, default=None, help="run only the first N layers")
    ap.add_argument("--a7", action="store_true", help="emulate int7 per-token activation quant")
    ap.add_argument("--out", help="save logits to this .npy")
    args = ap.parse_args(argv)
    ck = PearlCheckpoint(args.src)
    if args.tokens:
        toks = [int(t) for t in args.tokens.split(",")]
    else:
        from transformers import AutoTokenizer
        toks = AutoTokenizer.from_pretrained(args.src).encode(args.prompt)
    (logits,) = forward(ck, [toks], args.layers, args.a7,
                        log=lambda m: print(m, file=sys.stderr))
    if args.out:
        np.save(args.out, logits)
    last = logits[-1]
    top = np.argsort(-last)[:5]
    print(json.dumps({"tokens": len(toks), "top5": top.tolist(),
                      "top5_logits": [float(last[t]) for t in top]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
