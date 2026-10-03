"""Standalone copy of OpenJarvis's Pearl miner loops, ported to current Pearl.

Modules:
- ``_mps_miner_loop_main``: Apple-GPU (PyTorch MPS) NoisyGEMM miner loop
  (OpenJarvis provider id ``apple-mps-pearl``).
- ``_miner_loop_main``: CPU miner loop and the shared gateway JSON-RPC helpers.

Run a loop with ``python -m oj_pearl_mps._mps_miner_loop_main --help``.
See NOTICE for provenance and docs/OPENJARVIS_UPGRADE.md for the changes.
"""
