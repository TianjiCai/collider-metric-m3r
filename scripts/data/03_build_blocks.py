#!/usr/bin/env python
"""Tier 3, step 3: compute the pairwise distance blocks for one metric.

Each block ``{metric}_block_{i}_{j}.npy`` (``i <= j``) holds the ``(10000, 10000)`` matrix of
distances between every jet of class ``i`` and every jet of class ``j``:

* ``w2`` - exact energy-flow 2-Wasserstein (POT) on the preprocessed constituents;
* ``sw`` - spectral 2-Wasserstein through SPECTER (needs ``pip install "m3r[sw]"``);
* ``ps`` - phase-space distance on the 10 collinear-frame subjets of step 2.

New files follow the convention recorded for the metric in
``data/manifests/blocks_manifest.json`` (``stored_power``: 1 = distances, 2 = squared;
the published SW2 files are squared), so a partially regenerated directory stays consistent.

Examples
    python scripts/data/03_build_blocks.py --metric w2 --pairs missing           # fill gaps
    python scripts/data/03_build_blocks.py --metric ps --pairs 0,0 --rows 200 --check   # spot check
Runtime per 10k x 10k block on 8 cores: W2 and PS about 3 h (diagonal) / 6 h (off-diagonal);
SW2 on CPU about 50 h / 95 h, so use a GPU JAX backend for SW2 as in the published run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from m3r import load_config
from m3r.blocks import BlockStore, MANIFEST_NAME, default_manifest, load_manifest, write_manifest
from m3r.classes import CLASS_NAMES, N_CLASSES, pairwise_class_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric", choices=["w2", "sw", "ps"], required=True)
    parser.add_argument("--pairs", nargs="+", default=["missing"],
                        help="'all' (55 blocks), 'diag', 'missing' (default) or explicit 'i,j' pairs.")
    parser.add_argument("--rows", type=int, default=None, help="Compute only the first N jets of each class (checks).")
    parser.add_argument("--check", action="store_true", help="Compare with the existing block instead of writing.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=None, help="Rows per worker process (W2/PS) or SW batch size.")
    return parser.parse_args()


def resolve_pairs(spec: list[str], store: BlockStore, metric: str) -> list[tuple[int, int]]:
    if spec == ["all"]:
        return [(i, j) for i in range(N_CLASSES) for j in range(i, N_CLASSES)]
    if spec == ["diag"]:
        return [(i, i) for i in range(N_CLASSES)]
    if spec == ["missing"]:
        return [(i, j) for i in range(N_CLASSES) for j in range(i, N_CLASSES) if not store.exists(metric, i, j)]
    pairs = []
    for s in spec:
        a, b = s.split(",")
        pairs.append(tuple(sorted((int(a), int(b)))))
    return pairs


def load_events(cfg, metric: str, cls: int, rows: int | None) -> np.ndarray:
    name = CLASS_NAMES[cls]
    if metric == "ps":
        x = np.load(cfg.ps_input_path(name))["X_transformed_epxpypz"].astype(np.float64, copy=False)
    else:
        x = np.load(cfg.jets_path(name))["x"].astype(np.float32, copy=False)
        if metric == "sw":
            from m3r.metrics.sw import build_sw_events
            x = build_sw_events(x)
    return x[:rows] if rows else x


def compute_block(cfg, metric: str, ev_i: np.ndarray, ev_j: np.ndarray | None, workers: int | None, squared: bool) -> np.ndarray:
    mcfg = cfg["metrics"][metric]
    if metric == "w2":
        from m3r.metrics.pairwise import parallel_OT_non_square, parallel_OT_square
        n = workers or int(mcfg.get("events_per_thread", 20))
        return parallel_OT_square(ev_i, events_per_thread=n) if ev_j is None else parallel_OT_non_square(ev_i, ev_j, events_per_thread=n)
    if metric == "ps":
        from m3r.metrics.pairwise import parallel_dist_mat_non_square, parallel_dist_mat_square
        from m3r.metrics.ps import ps_distance
        n = workers or int(mcfg.get("events_per_process", 8192))
        kwargs = dict(Sphere2Simplex=float(mcfg.get("sphere2simplex", 1.0)), l_has_chi=bool(mcfg.get("l_has_chi", True)), return_only_l=True)
        if ev_j is None:
            return parallel_dist_mat_square(ev_i, ps_distance, events_per_process=n, **kwargs)
        return parallel_dist_mat_non_square(ev_i, ev_j, ps_distance, events_per_process=n, **kwargs)
    from m3r.metrics.sw import make_specter, sw2_matrix
    specter = make_specter(force_cpu=bool(mcfg.get("force_cpu", True)))
    return sw2_matrix(specter, ev_i, ev_j, batch_size=workers or int(mcfg.get("batch_size", 4096)), squared=squared, progress=True)


def main() -> None:
    args = parse_args()
    cfg = load_config()
    store = BlockStore.from_config(cfg)
    manifest_path = cfg.manifest_path(MANIFEST_NAME)
    manifest = load_manifest(manifest_path) if manifest_path.exists() else default_manifest()
    power = int(manifest.get("metrics", {}).get(args.metric, {}).get("stored_power", 1))
    pairs = resolve_pairs(args.pairs, store, args.metric)
    print(f"{args.metric}: {len(pairs)} block(s), stored_power={power}" + (f", rows={args.rows}" if args.rows else ""))
    for i, j in pairs:
        out = store.path(args.metric, i, j)
        if out.exists() and not (args.overwrite or args.check or args.rows):
            print(f"  skip existing {out.name}"); continue
        t0 = time.time()
        ev_i = load_events(cfg, args.metric, i, args.rows)
        ev_j = None if i == j else load_events(cfg, args.metric, j, args.rows)
        block = compute_block(cfg, args.metric, ev_i, ev_j, args.workers, squared=(power == 2))
        if args.metric != "sw" and power == 2:
            block = block ** 2
        if args.check or args.rows:
            if not out.exists():
                print(f"  {out.name}: computed {block.shape} in {time.time() - t0:.0f}s (no existing file to compare)"); continue
            ref = np.load(out, mmap_mode="r")[: block.shape[0], : block.shape[1]]
            diff = np.abs(np.asarray(ref) - block)
            print(f"  {out.name}[:{block.shape[0]},:{block.shape[1]}]: array_equal={np.array_equal(ref, block)} "
                  f"allclose(rtol=1e-6)={np.allclose(ref, block, rtol=1e-6, atol=1e-9)} max|diff|={diff.max():.3e} ({time.time() - t0:.0f}s)")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, block.astype(np.float64, copy=False))
        manifest.setdefault("files", {})[f"{args.metric}/{out.name}"] = {"shape": list(block.shape), "dtype": "float64",
                                                                       "bytes": out.stat().st_size, "stored_power": power}
        write_manifest(manifest_path, manifest)
        print(f"  wrote {out.name} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
