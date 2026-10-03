# Does the Apple GPU (via MLX) reproduce Pearl's jackpot math bit-for-bit?
import numpy as np, mlx.core as mx, time
rng = np.random.default_rng(0)
RANK, JACK, LROT = 128, 16, 13

def noised(shape):
    # signal in [-64,64] (Pearl SIGNAL_MIN/MAX) + worst-case int8 noise [-128,127]
    return rng.integers(-64, 65, shape) + rng.integers(-128, 128, shape)

def rotl(x, r): return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF

def jackpot_from_running(run):  # run: (chunks, h, w) int32 running sums for one tile
    jp = [0]*JACK
    for c in range(run.shape[0]):
        x = int(np.bitwise_xor.reduce(run[c].astype(np.uint32).ravel()))
        t = c % JACK
        jp[t] = rotl(jp[t], LROT) ^ x
    return jp

def cpu_reference(A, B):  # port of zk-pow try_mine_one inner loop, int64 exact
    k = A.shape[1]; acc = np.zeros((A.shape[0], B.shape[1]), np.int64); out=[]
    for ll in range(RANK, k+1, RANK):
        acc += A[:, ll-RANK:ll].astype(np.int64) @ B[ll-RANK:ll].astype(np.int64)
        out.append(acc.astype(np.int32))  # wraps like i32
    return np.stack(out)

def gpu_mlx(A, B):  # fp32 matmul per 128-chunk on GPU, int32 running sum
    m, k = A.shape; n = B.shape[1]; C = k // RANK
    a = mx.array(A, mx.float32).reshape(m, C, RANK).transpose(1, 0, 2)
    b = mx.array(B, mx.float32).reshape(C, RANK, n)
    part = (a @ b).astype(mx.int32)           # (C, m, n), each exact (< 2^24)
    return mx.cumsum(part, axis=0)

# 1) bit-exactness on a realistic tile, worst-case values
A, B = noised((64, 4096)), noised((4096, 64))
ref, got = cpu_reference(A, B), np.array(gpu_mlx(A, B))
print("max |chunk sum| =", int(np.abs(np.diff(ref.astype(np.int64), axis=0, prepend=0)).max()), "(fp32 exact limit 16777216)")
print("running sums identical:", np.array_equal(ref, got))
print("jackpot identical:     ", jackpot_from_running(ref) == jackpot_from_running(got))

# 2) adversarial: all entries at max magnitude -> largest possible chunk sums
A2 = np.full((16, 1024), -192); B2 = np.full((1024, 16), -192)
print("worst-case all -192 identical:", np.array_equal(cpu_reference(A2, B2), np.array(gpu_mlx(A2, B2))))

# 3) throughput of the exact fp32 path on this GPU
for (m, n, k) in [(4096, 4096, 4096), (8192, 8192, 8192)]:
    a = mx.random.randint(-192, 193, (m, k)).astype(mx.float32)
    b = mx.random.randint(-192, 193, (k, n)).astype(mx.float32)
    mx.eval(a @ b)
    t = time.perf_counter(); N = 5
    for _ in range(N): mx.eval(a @ b)
    dt = (time.perf_counter() - t) / N
    print(f"fp32 GEMM {m}x{n}x{k}: {2*m*n*k/dt/1e12:.2f} TOPS-equivalent")
    a16, b16 = a.astype(mx.float16), b.astype(mx.float16); mx.eval(a16 @ b16)
    t = time.perf_counter()
    for _ in range(N): mx.eval(a16 @ b16)
    print(f"fp16 GEMM {m}x{n}x{k}: {2*m*n*k/((time.perf_counter()-t)/N)/1e12:.2f} TFLOPS (upper bound, not exact as-is)")
