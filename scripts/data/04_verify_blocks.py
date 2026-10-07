#!/usr/bin/env python
"""Tier 2/3 check: verify the distance blocks on disk and record them in the blocks manifest.

For every metric and class pair the script checks that the block exists, has shape
``(10000, 10000)`` and dtype float64, that diagonal blocks are symmetric with a zero
diagonal (sampled rows), that values are finite and non-negative, and, with ``--hash``,
records the SHA-256 of each file. ``--convention`` recomputes a few distances from the jets
(W2 with POT, SW2 with a direct 1-D optimal transport, PS from the subjets) and confirms
the ``stored_power`` convention of the manifest. ``--write`` updates
``data/manifests/blocks_manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from m3r import load_config
from m3r.blocks import BlockStore, MANIFEST_NAME, default_manifest, load_manifest, write_manifest
from m3r.classes import CLASS_NAMES, KIN_METRICS, N_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", nargs="+", default=list(KIN_METRICS))
    parser.add_argument("--hash", action="store_true", help="Compute SHA-256 of every block (reads 132 GB for all metrics).")
    parser.add_argument("--convention", action="store_true", help="Recompute a few distances from the jets.")
    parser.add_argument("--write", action="store_true", help="Update the manifest with the findings.")
    parser.add_argument("--sample", type=int, default=16, help="Rows sampled for the symmetry/finiteness checks.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def check_convention(cfg, store: BlockStore, metric: str, n_pairs: int = 5) -> list[str]:
    msgs = []
    power = store.stored_power(metric)
    cls = 7
    name = CLASS_NAMES[cls]
    if not store.exists(metric, cls, cls):
        return [f"{metric}: block {cls}_{cls} missing, convention not checked"]
    block = store.load(metric, cls, cls, as_distance=False, mmap=True)
    pairs = [(0, 1), (0, 2), (1, 2), (3, 7), (10, 11)][:n_pairs]
    if metric == "w2":
        from m3r.metrics.w2 import get_w2
        x = np.load(cfg.jets_path(name))["x"]
        ref = [get_w2(x[a], x[b]) for a, b in pairs]
    elif metric == "sw":
        from m3r.metrics.sw import sw2_exact
        x = np.load(cfg.jets_path(name))["x"]
        ref = [sw2_exact(x[a], x[b]) for a, b in pairs]
    else:
        from m3r.metrics.ps import ps_distance
        x = np.load(cfg.ps_input_path(name))["X_transformed_epxpypz"]
        ref = [float(ps_distance(x[a], x[b], Sphere2Simplex=1, l_has_chi=True, return_only_l=True)) for a, b in pairs]
    stored = [float(block[a, b]) for a, b in pairs]
    expected = [r ** power for r in ref]
    ok = np.allclose(stored, expected, rtol=1e-4, atol=1e-9)
    msgs.append(f"{metric}: stored_power={power} {'CONFIRMED' if ok else 'MISMATCH'}; stored={np.round(stored, 6).tolist()} "
                f"expected(d^{power})={np.round(expected, 6).tolist()}")
    return msgs


def main() -> None:
    args = parse_args()
    cfg = load_config()
    store = BlockStore.from_config(cfg)
    manifest_path = cfg.manifest_path(MANIFEST_NAME)
    manifest = load_manifest(manifest_path) if manifest_path.exists() else default_manifest()
    n = store.n_per_class
    rng = np.random.default_rng(0)
    problems = 0
    for metric in args.metrics:
        present = 0
        for i in range(N_CLASSES):
            for j in range(i, N_CLASSES):
                path = store.path(metric, i, j)
                key = f"{metric}/{path.name}"
                if not path.exists():
                    print(f"  missing {key}"); problems += 1; continue
                block = np.load(path, mmap_mode="r")
                entry = {"shape": list(block.shape), "dtype": str(block.dtype), "bytes": path.stat().st_size,
                         "stored_power": store.stored_power(metric)}
                issues = []
                if block.shape != (n, n):
                    issues.append(f"shape {block.shape}")
                if block.dtype != np.float64:
                    issues.append(f"dtype {block.dtype}")
                rows = np.sort(rng.choice(n, size=min(args.sample, n), replace=False))
                sample = np.asarray(block[rows, :])
                if not np.all(np.isfinite(sample)):
                    issues.append("non-finite values")
                if np.any(sample < -1e-9):
                    issues.append("negative values")
                if i == j:
                    if not np.allclose(np.asarray(block[:, rows]).T, sample, rtol=0, atol=1e-9):
                        issues.append("not symmetric")
                    if np.any(np.abs(sample[np.arange(len(rows)), rows]) > 1e-9):
                        issues.append("nonzero diagonal")
                if args.hash:
                    entry["sha256"] = sha256(path)
                if issues:
                    problems += 1
                    print(f"  {key}: " + ", ".join(issues))
                manifest.setdefault("files", {})[key] = entry
                present += 1
        print(f"{metric}: {present}/55 blocks present")
        if args.convention:
            for msg in check_convention(cfg, store, metric):
                print("  " + msg)
    if args.write:
        write_manifest(manifest_path, manifest)
        print(f"updated {manifest_path}")
    print("OK" if problems == 0 else f"{problems} problem(s)")


if __name__ == "__main__":
    main()
