"""External reference for k3sg perf: MLX mx.matmul fp32 (M x K @ K x N), driven by k3sg.swift in the same paired rounds.

Protocol (stdin/stdout, line based): prints "ready <mlx version> <device>" after allocation + warm-up; for every "run"
line it evaluates one fp32 matmul and prints the wall-clock milliseconds (mx.eval blocks until the GPU is done, so this
includes dispatch overhead; k3sg variants use GPU timestamps). "quit" exits.
Operands: integers in [-127,127] stored as fp32 (the same value range as the K3-SG staging).
"""
import sys
import time

import mlx.core as mx

M, N, K = (int(x) for x in sys.argv[1:4])
mx.random.seed(20261002)
a = mx.random.randint(-127, 128, (M, K)).astype(mx.float32)
b = mx.random.randint(-127, 128, (K, N)).astype(mx.float32)
mx.eval(a, b)
mx.eval(a @ b)
print(f"ready {mx.__version__} {mx.default_device()}", flush=True)
for line in sys.stdin:
    cmd = line.strip()
    if cmd == "quit":
        break
    if cmd == "run":
        t = time.perf_counter()
        c = a @ b
        mx.eval(c)
        print(f"{(time.perf_counter() - t) * 1e3:.6f}", flush=True)
        del c
