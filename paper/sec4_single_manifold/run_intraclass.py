#!/usr/bin/env python
"""Section 4.1 analysis: JAUC and MADO versus number of references (inputs of Fig. 3).

For every metric and jet class, loads the intraclass distance block, builds M3R
representations at increasing reference counts and measures how well the
original geometry is reconstructed:

* JAUC - area under the mean k-nearest-neighbour overlap curve;
* MADO - sampled modified average decision ordering.

Writes one CSV per metric and measure under results/paper/sec4/ (merging with
existing rows so classes can be run incrementally). Needs the diagonal blocks
data/blocks/{metric}/{metric}_block_c_c.npy. Runtime: JAUC is the expensive
part (about 1 h per class and metric on a laptop; --jobs N runs classes in parallel, ~1 GB RAM each); MADO takes a minute.

Examples
    python paper/sec4_single_manifold/run_intraclass.py                    # all metrics, 8 classes
    python paper/sec4_single_manifold/run_intraclass.py --metrics sw        # one metric
    python paper/sec4_single_manifold/run_intraclass.py --classes QCD --num-jets 2000 --anchors 1 5 20  # smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import add_common_args, describe, load_cfg  # noqa: E402

from m3r import load_config  # noqa: E402
from m3r.blocks import BlockStore  # noqa: E402
from m3r.classes import class_index, class_name  # noqa: E402
from m3r.intraclass import jauc_sweep, mado_sweep  # noqa: E402
from m3r.results_io import read_csv, write_csv, write_provenance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--metrics", nargs="+", default=None, help="Metrics (default: paper.yaml metrics.kinematic).")
    parser.add_argument("--classes", nargs="+", default=None, help="Classes (default: paper.yaml intraclass.classes).")
    parser.add_argument("--anchors", type=int, nargs="+", default=None, help="Reference counts (default: paper.yaml).")
    parser.add_argument("--num-jets", type=int, default=None, help="Use only the first N jets of each class (smoke tests).")
    parser.add_argument("--measures", nargs="+", choices=["jauc", "mado"], default=["jauc", "mado"])
    parser.add_argument("--exact-mado", action="store_true", help="Exact MADO over all pairs/probes (tiny --num-jets only).")
    parser.add_argument("--sw-coordinates", choices=["distance", "stored"], default=None,
                        help="M3R coordinates for SW2: sqrt of the stored squared distances (paper) or the stored values "
                             "(pre-refactor behaviour). Default from paper.yaml metrics.sw_as_distance.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1, help="Classes processed in parallel (each needs ~4 GB of RAM).")
    return parser.parse_args()


def merge_rows(path: Path, new_rows: list[dict], key: str = "class_name") -> list[dict]:
    rows = {row[key]: row for row in read_csv(path)} if path.exists() else {}
    for row in new_rows:
        rows[row[key]] = row
    order = {name: idx for idx, name in enumerate(sorted(rows, key=lambda n: class_index(n)))}
    return [rows[name] for name in sorted(rows, key=lambda n: order[n])]


def _process_class(job):
    """Run the JAUC/MADO sweeps for one (metric, class); returns (jauc_row, mado_row)."""
    (root, config_path, metric, cls, as_distance, anchors, seed, mado_base, num_jets, measures, exact_mado,
     jauc_cfg, mado_cfg) = job
    cfg = load_config(config_path, root=root)
    store = BlockStore.from_config(cfg)
    c = class_index(cls)
    jauc_row = mado_row = None
    t0 = time.time()
    print(f"[{metric}] {cls} (block {c}): loading", flush=True)
    if "jauc" in measures:
        D = store.load(metric, c, c, as_distance=as_distance, n_jets=num_jets)
        res = jauc_sweep(D, anchors, seed, neighbors_step=jauc_cfg["neighbors_step"],
                         chunk_size=jauc_cfg["rank_chunk_size"], progress=False)
        del D
        jauc_row = {"metric": metric, "class_name": cls, "class_index": c, "num_jets": res.n_events,
                    "neighbors_step": jauc_cfg["neighbors_step"], "seed": seed,
                    "sw_coordinates": "distance" if as_distance else "stored"}
        jauc_row.update({f"mean_jaccard_a{a}": f"{v:.8f}" for a, v in zip(anchors, res.mean_jaccard)})
        jauc_row.update({f"auc_a{a}": f"{v:.8f}" for a, v in zip(anchors, res.aucs)})
        print(f"[{metric}] {cls}: JAUC {np.round(res.aucs, 4)} ({time.time() - t0:.0f}s)", flush=True)
    if "mado" in measures:
        D32 = store.load(metric, c, c, as_distance=as_distance, n_jets=num_jets, dtype=np.float32)
        res = mado_sweep(D32, anchors, mado_base + c, sample_pairs_n=mado_cfg["sample_pairs"],
                         sample_references_n=mado_cfg["sample_references"],
                         pair_chunk_size=mado_cfg["pair_chunk_size"], ref_chunk_size=mado_cfg["ref_chunk_size"],
                         exact=exact_mado, progress=False)
        del D32
        mado_row = {"metric": metric, "class_name": cls, "class_index": c, "num_jets": res.n_events,
                    "sample_pairs": res.sample_pairs, "sample_references": res.sample_references,
                    "seed": res.seed, "exact": res.exact, "sw_coordinates": "distance" if as_distance else "stored"}
        mado_row.update({f"decision_ordering_a{a}": f"{v:.8f}" for a, v in zip(anchors, res.scores)})
        print(f"[{metric}] {cls}: MADO {np.round(res.scores, 4)} ({time.time() - t0:.0f}s)", flush=True)
    return jauc_row, mado_row


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)
    store = BlockStore.from_config(cfg)
    sec = cfg["intraclass"]
    metrics = args.metrics or list(cfg["metrics"]["kinematic"])
    classes = [class_name(c) for c in (args.classes or sec["classes"])]
    anchors = sorted(set(args.anchors or sec["anchors"]))
    seed = cfg["seeds"]["anchors"] if args.seed is None else args.seed
    mado_base = cfg["seeds"]["mado_base"] if args.seed is None else args.seed
    sw_as_distance = cfg["metrics"].get("sw_as_distance", True) if args.sw_coordinates is None else args.sw_coordinates == "distance"
    jauc_cfg, mado_cfg = sec["jauc"], sec["mado"]

    for metric in metrics:
        as_distance = sw_as_distance if metric == "sw" else True
        jobs = [(cfg.root, args.config, metric, cls, as_distance, anchors, seed, mado_base, args.num_jets, list(args.measures),
                 args.exact_mado, dict(jauc_cfg), dict(mado_cfg)) for cls in classes]
        if args.jobs > 1 and len(jobs) > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                outputs = list(pool.map(_process_class, jobs))
        else:
            outputs = [_process_class(job) for job in jobs]
        jauc_rows = [o[0] for o in outputs if o[0] is not None]
        mado_rows = [o[1] for o in outputs if o[1] is not None]

        for key, rows in (("intraclass_jauc", jauc_rows), ("intraclass_mado", mado_rows)):
            if not rows:
                continue
            out = cfg.result_path(key, metric=metric)
            merged = merge_rows(out, rows)
            fieldnames = list(rows[0].keys())
            write_csv(out, fieldnames, merged)
            write_provenance(out, config_hash=cfg.hash(), stage="paper/sec4_single_manifold/run_intraclass.py",
                             metric=metric, classes=[r["class_name"] for r in merged], anchors=anchors)
            print(f"wrote {describe(out, cfg)}")


if __name__ == "__main__":
    main()
