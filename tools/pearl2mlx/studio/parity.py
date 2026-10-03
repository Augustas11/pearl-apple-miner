#!/usr/bin/env python3
"""Logit parity: converted MLX model(s) vs pearl_ref on the ORIGINAL Pearl checkpoint.

For each --mlx model and each reference (plain W7A16 and --a7 W7A7 emulation):
top-1 agreement over all prompt positions, mean KL(ref || mlx) per position,
max |dlogit|. MLX runs on the default device (GPU) in the checkpoint dtype;
pearl_ref runs in numpy fp32 on the CPU.

Kernel check (first model, exact mode only): for a few int7 layers, one-hot
inputs through the real quantized_matmul on the GPU must read back exactly
bf16(int7 * s) from the Pearl checkpoint ("exact-mode weights bit-identical").

Exit 1 if the first model's top-1 vs the plain reference < --min-top1 or the
kernel check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pearl_ref import PearlCheckpoint, bf16_bits_to_f32, f32_to_bf16_bits, forward  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "In 1969, the first humans to walk on the Moon were",
    "Photosynthesis is the process by which plants",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "The three primary colors of light are",
    "Water boils at 100 degrees Celsius at sea level because",
    "A haiku about autumn leaves:\n",
    "The French Revolution began in the year",
    "To make a cup of green tea, first",
    "SELECT name, email FROM users WHERE",
    "The theory of general relativity, proposed by Albert Einstein,",
    "The largest planet in our solar system is",
    "Machine learning models are trained by minimizing",
    "Once upon a time, in a small village by the sea,",
    "The derivative of sin(x) with respect to x is",
    "The Great Wall of China was built to",
    "In economics, inflation refers to",
    "The human heart has four chambers: the",
    "Translate to Spanish: 'Good morning, how are you?'\n",
    "The mitochondria is often called the powerhouse of the cell because",
    "import numpy as np\n\nx = np.linspace(0, 1, 100)\n",
    "Shakespeare wrote the play Hamlet, which tells the story of",
    "The speed of light in a vacuum is approximately",
    "A balanced diet should include",
    "The Pythagorean theorem states that",
    "Bitcoin is a decentralized digital currency that",
    "The main causes of World War I included",
    "To solve the equation 2x + 3 = 11, we",
    "The Amazon rainforest is important for the planet because",
    "Q: What is the boiling point of nitrogen?\nA:",
    "The quick brown fox jumps over the lazy dog. This sentence is famous because",
    "Apple Silicon chips use a unified memory architecture, which means",
]


def log(msg: str) -> None:
    print(f"[parity {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def kl_top1(ref: np.ndarray, got: np.ndarray):
    r = ref.astype(np.float64)
    g = got.astype(np.float64)
    lr = r - r.max(-1, keepdims=True)
    lr -= np.log(np.exp(lr).sum(-1, keepdims=True))
    lg = g - g.max(-1, keepdims=True)
    lg -= np.log(np.exp(lg).sum(-1, keepdims=True))
    kl = (np.exp(lr) * (lr - lg)).sum(-1)
    return kl, ref.argmax(-1) == got.argmax(-1), np.abs(ref - got).max()


def kernel_check(model, ck: PearlCheckpoint, cfg: dict, n_layers: int, ncols: int = 256) -> dict:
    from mlx.utils import tree_flatten
    import mlx.nn as nn
    mods = dict(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))
    int7 = [p for p, v in cfg["quantization"].items() if isinstance(v, dict) and p in mods]
    picks = sorted(set(int7[:: max(1, len(int7) // n_layers)][:n_layers]))
    rng = np.random.default_rng(0)
    res = {}
    for p in picks:
        m = mods[p]
        q = ck.raw(p + ".weight")
        s_bits = ck.raw(p + ".weight_scale")
        n, k = q.shape
        cols = np.sort(rng.choice(k, size=min(ncols, k), replace=False))
        want = bf16_bits_to_f32(f32_to_bf16_bits(q[:, cols].astype(np.float32) * bf16_bits_to_f32(s_bits)))
        x = np.zeros((len(cols), k), np.float32)
        x[np.arange(len(cols)), cols] = 1
        y = mx.quantized_matmul(mx.array(x).astype(m.scales.dtype), m.weight, m.scales, m.biases,
                                transpose=True, group_size=m.group_size, bits=m.bits)
        got = np.array(y.astype(mx.float32)).T
        res[p] = bool(np.array_equal(got, want))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="original Pearl snapshot dir")
    ap.add_argument("--mlx", required=True, nargs="+", help="converted model dir(s); first is primary")
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--layers", type=int, default=None, help="compare an N-layer prefix only")
    ap.add_argument("--a7", choices=["off", "on", "both"], default="both")
    ap.add_argument("--min-top1", type=float, default=0.99)
    ap.add_argument("--kernel-layers", type=int, default=4)
    ap.add_argument("--json-out")
    args = ap.parse_args(argv)

    from mlx_lm import load

    mx.set_default_device(mx.gpu)  # kernel check needs the GPU kernels (CPU bf16 qmm is not exact)
    ck = PearlCheckpoint(args.src)
    texts = PROMPTS[: args.prompts]
    mlx_logits: dict[str, list[np.ndarray]] = {}
    kernel = {}
    toks = None
    for i, path in enumerate(args.mlx):
        t0 = time.time()
        model, tok = load(path)
        cfg = json.loads((Path(path) / "config.json").read_text())
        if toks is None:
            toks = [tok.encode(t) for t in texts]
            log(f"{len(toks)} prompts, {sum(map(len, toks))} tokens, bos={toks[0][0]}")
        if args.layers is not None:
            model.model.layers = model.model.layers[: args.layers]
        mlx_logits[path] = []
        for t in toks:
            out = model(mx.array([t])).astype(mx.float32)
            mlx_logits[path].append(np.array(out)[0])
        if i == 0 and cfg.get("pearl2mlx", {}).get("mode") == "exact" and args.kernel_layers:
            kernel = kernel_check(model, ck, cfg, args.kernel_layers)
            log(f"kernel check: {kernel}")
        log(f"{path}: logits in {time.time() - t0:.1f}s")
        del model
        mx.clear_cache()

    refs = {}
    for a7 in ([False, True] if args.a7 == "both" else [args.a7 == "on"]):
        t0 = time.time()
        refs["a7" if a7 else "w7a16"] = forward(ck, toks, args.layers, a7)
        log(f"pearl_ref a7={a7}: {time.time() - t0:.1f}s")

    results = {"prompts": len(toks), "tokens": int(sum(map(len, toks))), "layers": args.layers,
               "kernel_check": kernel, "pairs": []}
    for path, got in mlx_logits.items():
        for rname, ref in refs.items():
            kls, agree, dmax = [], [], 0.0
            for r, g in zip(ref, got):
                kl, a, d = kl_top1(r, g)
                kls.append(kl)
                agree.append(a)
                dmax = max(dmax, float(d))
            kls = np.concatenate(kls)
            agree = np.concatenate(agree)
            row = {"mlx": path, "ref": rname, "top1": round(float(agree.mean()), 5),
                   "mean_kl": float(kls.mean()), "p99_kl": float(np.quantile(kls, 0.99)),
                   "max_abs_dlogit": round(dmax, 4)}
            results["pairs"].append(row)
            print(json.dumps(row), flush=True)

    primary = next(r for r in results["pairs"] if r["mlx"] == args.mlx[0] and r["ref"] in ("w7a16", "a7"))
    ok_top1 = primary["top1"] >= args.min_top1
    ok_kernel = all(kernel.values()) if kernel else True
    results["pass"] = {"top1_primary": ok_top1, "kernel_exact": ok_kernel}
    print("PARITY " + json.dumps(results["pass"]), flush=True)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
    return 0 if ok_top1 and ok_kernel else 1


if __name__ == "__main__":
    sys.exit(main())
