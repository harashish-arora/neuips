#!/usr/bin/env python
"""
Pre-build (and persist) the graph caches used by the data-scaling ablation:

  - feature_cache/molmerger_skeletons.pt  (used by molmerger trainer)
  - feature_cache/gcn_graphs.pt           (used by GCN/GAT/GIN trainers)

Both are dumped to the existing `vansh/feature_cache/` directory so they
sit alongside the rdkit / morgan / etc. .npz caches.  Re-run is a no-op
when nothing has changed.

Usage:
    python cache_graphs.py                  # build both
    python cache_graphs.py --molmerger      # only the molmerger cache
    python cache_graphs.py --gcn            # only the gcn cache
    python cache_graphs.py --force          # rebuild from scratch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VANSH_ROOT = HERE.parent.parent
sys.path.insert(0, str(VANSH_ROOT))

from sc3_bench.data import load_all_splits  # noqa: E402

from scaling_trainers import (  # noqa: E402
    MOLMERGER_CACHE_FILE, GCN_CACHE_FILE,
    build_molmerger_cache, build_gcn_graph_cache,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--molmerger", action="store_true",
                   help="Only (re)build the molmerger skeleton cache.")
    p.add_argument("--gcn", action="store_true",
                   help="Only (re)build the GCN graph cache.")
    p.add_argument("--force", action="store_true",
                   help="Delete existing cache files first.")
    args = p.parse_args()

    do_mm = args.molmerger or not (args.molmerger or args.gcn)
    do_gcn = args.gcn or not (args.molmerger or args.gcn)

    if args.force:
        if do_mm and MOLMERGER_CACHE_FILE.exists():
            print(f"Removing {MOLMERGER_CACHE_FILE}")
            MOLMERGER_CACHE_FILE.unlink()
        if do_gcn and GCN_CACHE_FILE.exists():
            print(f"Removing {GCN_CACHE_FILE}")
            GCN_CACHE_FILE.unlink()

    print("Loading splits...")
    splits = load_all_splits(verbose=True)

    if do_mm:
        print("\n[molmerger]")
        build_molmerger_cache(splits, verbose=True)

    if do_gcn:
        print("\n[gcn]")
        build_gcn_graph_cache(splits, verbose=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
