#!/usr/bin/env python3
"""Convert a Pearl (quant_method "pearl") Llama checkpoint to MLX format.

Modes
  exact : A2. int7 layers packed losslessly as MLX affine 8-bit, group 128:
          q = int7 + 64 (uint32, 4 per word, low byte first), scales = s_row,
          biases = -64 * s_row, so mx.dequantize == int7 * s bit-exactly.
          FP8 layers: dequant -> bf16 -> mx.quantize affine 8-bit g64
          (--fp8 bf16: keep them as dense bf16 instead; larger, near-lossless).
          SmoothQuant: folded into the preceding RMSNorm when all of its
          consumers are int7 and share one identical scale; otherwise kept as a
          runtime ``<layer>.smooth`` param (PearlQuantizedLinear, pearl_layers.py);
          for FP8 layers folded into W columns. embed / norms / lm_head: bf16.
  std8  : A1. dequant everything, fold smooth into W columns, affine 8-bit g64
          for every Linear and the embedding (like mlx_lm.convert). Stock mlx_lm.
  std4  : as std8 with affine 4-bit g64.

Runs entirely on the CPU and processes one source shard at a time.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pearl_ref import PearlCheckpoint, e4m3_is_nan  # noqa: E402

TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "chat_template.jinja", "generation_config.json", "tokenizer.model",
]
NORM_CONSUMERS = {
    "input_layernorm": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "post_attention_layernorm": ["mlp.gate_proj", "mlp.up_proj"],
}
EXACT_GROUP = 128
STD_GROUP = 64


def log(msg: str) -> None:
    print(f"[pearl2mlx {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# tensor conversions (all on mx.cpu)
# ---------------------------------------------------------------------------
def bf16(u16: np.ndarray) -> mx.array:
    return mx.view(mx.array(np.ascontiguousarray(u16, dtype=np.uint16)), mx.bfloat16)


def pack_int7_exact(q: np.ndarray, scale_bits: np.ndarray, group: int = EXACT_GROUP):
    """int8 [n,k] in [-64,64] + bf16 bits [n,1] -> (uint32 [n,k/4], scales, biases)."""
    n, k = q.shape
    if k % group:
        raise ValueError(f"k={k} not divisible by group {group}")
    lo, hi = int(q.min()), int(q.max())
    if lo < -64 or hi > 64:
        raise ValueError(f"int7 weight out of [-64,64]: [{lo},{hi}]")
    u8 = np.ascontiguousarray((q.astype(np.int16) + 64).astype(np.uint8))
    packed = mx.array(u8.view("<u4"))  # byte j of a word = element 4w+j (low bits first)
    s = bf16(scale_bits.reshape(n, 1))
    scales = mx.broadcast_to(s, (n, k // group))
    biases = scales * -64.0  # power-of-two multiply: exact in bf16
    return packed, mx.contiguous(scales), mx.contiguous(biases)


def fp8_dequant(codes: np.ndarray, scale_bits: np.ndarray) -> mx.array:
    """F8_E4M3 codes [n,k] + bf16 block scales -> fp32 [n,k] via mx.from_fp8."""
    if e4m3_is_nan(codes).any():
        raise ValueError("E4M3 NaN code (0x7F/0xFF) in weight")
    n, k = codes.shape
    w = mx.from_fp8(mx.array(np.ascontiguousarray(codes)), dtype=mx.float32)
    sb = scale_bits.shape
    if sb != (-(-n // 128), -(-k // 128)):
        raise ValueError(f"block scale shape {sb} does not match weight [{n},{k}]")
    s = mx.view(mx.array(np.ascontiguousarray(scale_bits)), mx.bfloat16).astype(mx.float32)
    s = mx.repeat(mx.repeat(s, 128, axis=0), 128, axis=1)[:n, :k]
    return w * s


def quantize(w: mx.array, group: int, bits: int):
    return mx.quantize(w.astype(mx.bfloat16), group_size=group, bits=bits, mode="affine")


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
def check_format(ck: PearlCheckpoint, linears: dict[str, str]) -> None:
    qc = ck.config.get("quantization_config") or {}
    if qc.get("quant_method") != "pearl":
        raise ValueError(f"quant_method {qc.get('quant_method')!r} != 'pearl'")
    if qc.get("transform_config"):
        raise ValueError("transform_config (Hadamard) not supported")
    want = {"int-quantized": "int7", "float-quantized": "fp8"}
    for prefix, kind in linears.items():
        hits = []
        for gname, grp in qc["config_groups"].items():
            for t in grp["targets"]:
                if t.startswith("re:") and re.fullmatch(t[3:], prefix) or t == prefix:
                    hits.append((gname, grp))
        if len(hits) != 1:
            raise ValueError(f"{prefix}: matches {len(hits)} quantization groups")
        gname, grp = hits[0]
        if want.get(grp["format"]) != kind:
            raise ValueError(f"{prefix}: dtype says {kind}, {gname} says {grp['format']}")
        w = grp["weights"]
        if kind == "int7" and (w["num_bits"] != 7 or w["strategy"] != "channel" or not w["symmetric"]):
            raise ValueError(f"{prefix}: unexpected int weights spec {w}")
        if kind == "fp8" and (w["strategy"] != "block" or w["block_structure"] != [128, 128]):
            raise ValueError(f"{prefix}: unexpected fp8 weights spec {w}")


def plan(ck: PearlCheckpoint, mode: str):
    linears = {}
    for name in ck.meta:
        if name.endswith(".weight_scale"):
            p = name[: -len(".weight_scale")]
            linears[p] = ck.linear_kind(p)
    for name in ck.meta:
        if name.endswith(".smooth_quant_scale") and name[: -len(".smooth_quant_scale")] not in linears:
            raise ValueError(f"{name} has no quantized layer")
    check_format(ck, linears)

    norm_fold: dict[str, str] = {}      # norm weight name -> smooth tensor name
    runtime_smooth: set[str] = set()    # exact mode PearlQuantizedLinear paths
    col_fold: set[str] = set()          # fold smooth into W columns
    folded_into_norm: set[str] = set()
    smooth_layers = {p for p in linears if ck.has(p + ".smooth_quant_scale")}
    if mode == "exact":
        for li in range(ck.config["num_hidden_layers"]):
            for norm, cons in NORM_CONSUMERS.items():
                paths = [f"model.layers.{li}.{c}" for c in cons]
                if not any(p in smooth_layers for p in paths):
                    continue
                ok = all(p in smooth_layers and linears.get(p) == "int7" for p in paths)
                if ok:
                    ref = ck.raw(paths[0] + ".smooth_quant_scale")
                    ok = all(np.array_equal(ref, ck.raw(p + ".smooth_quant_scale")) for p in paths[1:])
                if ok:
                    norm_fold[f"model.layers.{li}.{norm}.weight"] = paths[0] + ".smooth_quant_scale"
                    folded_into_norm.update(paths)
        for p in smooth_layers - folded_into_norm:
            (runtime_smooth if linears[p] == "int7" else col_fold).add(p)
    else:
        col_fold = set(smooth_layers)
    return linears, norm_fold, runtime_smooth, col_fold


# ---------------------------------------------------------------------------
def convert(src: str, out: str, mode: str, verify: bool = True, fp8: str = "q8") -> dict:
    with mx.stream(mx.cpu):  # all conversion work on the CPU
        return _convert(src, out, mode, verify, fp8)


def _convert(src: str, out: str, mode: str, verify: bool, fp8: str) -> dict:
    t0 = time.time()
    ck = PearlCheckpoint(src)
    cfg = ck.config
    if cfg.get("model_type") != "llama":
        raise NotImplementedError("only model_type=llama is supported")
    bits = 4 if mode == "std4" else 8
    linears, norm_fold, runtime_smooth, col_fold = plan(ck, mode)
    log(f"mode={mode} linears={len(linears)} (int7={sum(k == 'int7' for k in linears.values())}, "
        f"fp8={sum(k == 'fp8' for k in linears.values())}) norm_fold={len(norm_fold)} "
        f"runtime_smooth={len(runtime_smooth)} col_fold={len(col_fold)}")
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    for stale in outdir.glob("model*.safetensors"):  # mlx_lm globs model*.safetensors
        stale.unlink()

    overrides: dict[str, dict] = {}
    weight_map: dict[str, str] = {}
    nverified = 0
    consumed_suffixes = (".weight_scale", ".smooth_quant_scale")
    nfiles = len(ck.files)
    for fi, fn in enumerate(ck.files):
        outname = f"model-{fi + 1:05d}-of-{nfiles:05d}.safetensors"
        tensors: dict[str, mx.array] = {}
        for name in sorted(ck.names_in(fn)):
            if name.endswith(consumed_suffixes):
                continue
            if not name.endswith(".weight"):
                raise ValueError(f"unexpected tensor {name}")
            prefix = name[: -len(".weight")]
            if prefix in linears:
                kind = linears[prefix]
                ws = ck.raw(prefix + ".weight_scale")
                if mode == "exact" and kind == "int7":
                    q = ck.raw(name)
                    if list(ws.shape) != [q.shape[0], 1]:
                        raise ValueError(f"{prefix}: int7 scale shape {ws.shape}")
                    wq, s, b = pack_int7_exact(q, ws)
                    if verify:
                        deq = mx.dequantize(wq, s.astype(mx.float32), b.astype(mx.float32),
                                            group_size=EXACT_GROUP, bits=8)
                        ref = mx.array(q.astype(np.float32)) * bf16(ws).astype(mx.float32)
                        if not mx.array_equal(deq, ref).item():
                            raise AssertionError(f"{prefix}: exact pack not bit-exact")
                        nverified += 1
                    overrides[prefix] = {"group_size": EXACT_GROUP, "bits": 8, "mode": "affine"}
                    tensors[prefix + ".weight"], tensors[prefix + ".scales"], tensors[prefix + ".biases"] = wq, s, b
                    if prefix in runtime_smooth:
                        tensors[prefix + ".smooth"] = bf16(ck.raw(prefix + ".smooth_quant_scale"))
                else:
                    if kind == "int7":
                        w = mx.array(ck.raw(name).astype(np.float32)) * bf16(ws).astype(mx.float32)
                    elif kind == "fp8":
                        w = fp8_dequant(ck.raw(name), ws)
                    else:
                        raise ValueError(f"{prefix}: unsupported quantized kind {kind}")
                    if prefix in col_fold:
                        w = w * bf16(ck.raw(prefix + ".smooth_quant_scale")).astype(mx.float32)[None, :]
                    if mode == "exact" and fp8 == "bf16":
                        tensors[prefix + ".weight"] = w.astype(mx.bfloat16)
                    else:
                        (tensors[prefix + ".weight"], tensors[prefix + ".scales"],
                         tensors[prefix + ".biases"]) = quantize(w, STD_GROUP, bits)
            else:
                if ck.dtype(name) != "BF16":
                    raise ValueError(f"{name}: unexpected dtype {ck.dtype(name)}")
                w = bf16(ck.raw(name))
                if name in norm_fold:
                    s = bf16(ck.raw(norm_fold[name]))
                    w = (w.astype(mx.float32) * s.astype(mx.float32)).astype(mx.bfloat16)
                is_matrix = len(ck.shape(name)) == 2
                if mode != "exact" and is_matrix:
                    (tensors[prefix + ".weight"], tensors[prefix + ".scales"],
                     tensors[prefix + ".biases"]) = quantize(w, STD_GROUP, bits)
                else:
                    tensors[name] = w
            mx.eval(*[tensors[k] for k in tensors if k.startswith(prefix + ".")])
        mx.save_safetensors(str(outdir / outname), tensors, metadata={"format": "mlx"})
        for k in tensors:
            weight_map[k] = outname
        log(f"wrote {outname}: {len(tensors)} tensors ({time.time() - t0:.0f}s)")
        del tensors

    total = sum((outdir / f).stat().st_size for f in set(weight_map.values()))
    (outdir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=2))

    new_cfg = {k: v for k, v in cfg.items() if k != "quantization_config"}
    quant = {"group_size": STD_GROUP, "bits": bits, "mode": "affine"}
    quant.update(dict(sorted(overrides.items())))
    new_cfg["quantization"] = quant
    new_cfg["quantization_config"] = quant
    if mode == "exact" and runtime_smooth:
        new_cfg["model_file"] = "pearl_layers.py"
        new_cfg["pearl_smooth"] = sorted(runtime_smooth)
        shutil.copy(Path(__file__).resolve().parent / "pearl_layers.py", outdir / "pearl_layers.py")
    new_cfg["pearl2mlx"] = {"mode": mode, "fp8": fp8 if mode == "exact" else "q8", "source": str(Path(src).resolve()),
                            "source_quantization_config": cfg.get("quantization_config")}
    (outdir / "config.json").write_text(json.dumps(new_cfg, indent=2))
    copied = []
    for f in TOKENIZER_FILES:
        if (Path(src) / f).exists():
            shutil.copy(Path(src) / f, outdir / f)
            copied.append(f)
    summary = {"mode": mode, "fp8": fp8 if mode == "exact" else "q8", "out": str(outdir), "seconds": round(time.time() - t0, 1),
               "bytes": total, "int7_verified_bit_exact": nverified,
               "norm_fold": len(norm_fold), "runtime_smooth": len(runtime_smooth),
               "col_fold": len(col_fold), "tokenizer_files": copied}
    log("done " + json.dumps(summary))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="Pearl HF snapshot dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", required=True, choices=["exact", "std8", "std4"])
    ap.add_argument("--no-verify", action="store_true",
                    help="exact mode: skip the per-layer dequant == int7*s check")
    ap.add_argument("--fp8", choices=["q8", "bf16"], default="q8",
                    help="exact mode: FP8 layers as affine 8-bit g64 (default) or dense bf16")
    args = ap.parse_args(argv)
    mx.set_default_device(mx.cpu)
    convert(args.src, args.out, args.mode, verify=not args.no_verify, fp8=args.fp8)
    return 0


if __name__ == "__main__":
    sys.exit(main())
