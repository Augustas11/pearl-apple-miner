#!/usr/bin/env python3
"""Print the local HF-cache snapshot dir for <repo> [revision]; never downloads.

A full 40-hex revision goes through snapshot_download(local_files_only=True);
a short sha prefix is matched against the cached revisions; no revision means
the cached "main" ref. Exits 1 (with the reason on stderr) if not cached.
"""
from __future__ import annotations

import sys

from huggingface_hub import scan_cache_dir, snapshot_download


def resolve(repo: str, rev: str | None) -> str:
    if rev is None or len(rev) == 40:
        return snapshot_download(repo, revision=rev, local_files_only=True)
    for r in scan_cache_dir().repos:
        if r.repo_id == repo and r.repo_type == "model":
            hits = [x for x in r.revisions if x.commit_hash.startswith(rev)]
            if len(hits) == 1:
                return str(hits[0].snapshot_path)
            raise SystemExit(f"{repo}@{rev}: {len(hits)} cached revisions match "
                             f"(cached: {[x.commit_hash[:12] for x in r.revisions]})")
    raise SystemExit(f"{repo}: not in the HF cache")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: resolve.py <repo> [revision]")
    try:
        print(resolve(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None))
    except SystemExit:
        raise
    except Exception as e:  # LocalEntryNotFoundError etc.
        raise SystemExit(f"{sys.argv[1]}: {type(e).__name__}: {e}")
