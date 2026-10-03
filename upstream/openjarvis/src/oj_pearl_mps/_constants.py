# Derived from OpenJarvis src/openjarvis/mining/_constants.py (Apache-2.0).
# Modified for pearl-metal-miner: trimmed to the constants the miner loops use
# (the original also imports openjarvis.core.paths for app-level paths), and the
# CPU default shape raised to satisfy current Pearl consensus checks
# (rank >= PENALTY_BASE_RANK = 128, k >= 16 * rank, k >= 1024).
"""Constants for the standalone OpenJarvis Pearl miner loops."""

from __future__ import annotations

# Default port as Pearl's gateway exposes it (MINER_RPC_PORT).
DEFAULT_GATEWAY_RPC_PORT = 8337

# CPU subprocess provider defaults (originally m=256, n=128, k=1024, rank=32).
CPU_PEARL_DEFAULT_M = 256
CPU_PEARL_DEFAULT_N = 128
CPU_PEARL_DEFAULT_K = 2048
CPU_PEARL_DEFAULT_RANK = 128
CPU_PEARL_DEFAULT_ROWS_PATTERN = (0, 8, 64, 72)
CPU_PEARL_DEFAULT_COLS_PATTERN = (0, 1, 8, 9, 32, 33, 40, 41)
