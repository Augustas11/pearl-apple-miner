# SPDX-License-Identifier: Apache-2.0
# Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.
"""Admission evidence must name the exact binary being exercised."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("b9_g3_evidence", ROOT / "scripts/pmk_v4_g3.py")
G3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G3)


def evidence(tmp_path):
    library = tmp_path / "libpmk.dylib"
    core = tmp_path / "libpmkcore_v4.dylib"
    library.write_bytes(b"tested host binary")
    core.write_bytes(b"tested core binary")
    metadata = {"cache_key": "key", "library_sha256": G3.digest(library.read_bytes())}
    record = {
        "schema": "pmk-v4-integration-v1", "upstream_pin": G3.PIN,
        "passed": True, "mismatches": 0, **metadata,
        "library": str(library), "core_library": str(core),
        "core_sha256": G3.digest(core.read_bytes()),
        "quantized_cells_checked": 1, "fold_tiles_checked": 1,
        "proofs_accepted": 1, "corrupt_proofs_rejected": 1,
        "cases": [{"m": m, "n": n, "k": k}
                  for m, n in ((32, 64), (64, 32)) for k in (1024, 4096, 16384)],
    }
    path = tmp_path / "integration.json"
    path.write_text(json.dumps(record))
    return path, metadata, library


def test_g3_requires_loaded_binary_to_match_integration(tmp_path):
    path, metadata, library = evidence(tmp_path)
    assert G3.integration_passed(path, metadata, library)
    metadata["library_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        G3.integration_passed(path, metadata, library)


def test_g3_rejects_rebuild_after_integration(tmp_path):
    path, metadata, library = evidence(tmp_path)
    library.write_bytes(b"rebuilt host binary")
    with pytest.raises(ValueError, match="changed after"):
        G3.integration_passed(path, metadata, library)
