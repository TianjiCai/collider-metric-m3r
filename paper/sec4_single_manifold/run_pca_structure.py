#!/usr/bin/env python
"""Section 4.2 analysis: PCA alignment and PC1 variance transfer between metric-induced M3R manifolds.

For QCD, W->qq and t->bqq (paper.yaml structure.classes) and a shared set of 20
references (seed 42), builds the intraclass M3R coordinates for W2, SW2 and PS,
performs raw-centred PCA and writes

* results/paper/sec4/pca_alignment.csv  - |cos| between same-rank PCs of every metric pair (Fig. 4)
* results/paper/sec4/ev_transfer.csv    - variance of the target manifold explained by the source PC1 (Fig. 5)
* results/paper/sec4/pca_spectra.csv and pca_alignment_bootstrap.csv - eigenvalue spectra and bootstrap
  stability backing the footnote on the PC4/PC5 variance (with --bootstrap N, default from paper.yaml)

Needs the diagonal blocks of the three classes. Runtime: seconds (plus a minute for the bootstrap).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import add_common_args, describe, load_cfg  # noqa: E402

from m3r.blocks import BlockStore  # noqa: E402
from m3r.classes import class_index, class_name  # noqa: E402
from m3r.results_io import write_csv, write_provenance  # noqa: E402
from m3r.structure import (  # noqa: E402
    METRIC_PAIRS,
    alignment_table,
    bootstrap_pca,
    class_relreps,
    structure_anchors,
    transfer_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--anchors", type=int, default=None, help="Number of references (default: paper.yaml structure.anchors).")
    parser.add_argument("--bootstrap", type=int, default=None, help="Bootstrap resamples for the spectra (0 to skip).")
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                        help="Precision of the loaded coordinates (float32 reproduces the pre-refactor Fig. 5 numbers to ~1e-6).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)
    store = BlockStore.from_config(cfg)
    sec = cfg["structure"]
    classes = [class_name(c) for c in (args.classes or sec["classes"])]
    n_anchors = args.anchors or int(sec["anchors"])
    n_pcs = int(sec["n_alignment_pcs"])
    n_boot = int(sec.get("bootstrap", 200)) if args.bootstrap is None else args.bootstrap
    metrics = list(cfg["metrics"]["kinematic"])
    dtype = np.float64 if args.dtype == "float64" else np.float32
    anchors = structure_anchors(store.n_per_class, n_anchors, int(cfg["seeds"]["anchors"]))

    align_rows, transfer_rows, spectra_rows, boot_rows = [], [], [], []
    rng = np.random.default_rng(int(cfg["seeds"]["bootstrap"]))
    for cls in classes:
        c = class_index(cls)
        print(f"{cls} (class {c}): loading M3R coordinates", flush=True)
        relreps = class_relreps(store, c, anchors, metrics, dtype=dtype)
        for (a, b), values in alignment_table(relreps, n_pcs).items():
            for pc, v in enumerate(values, start=1):
                align_rows.append({"class_name": cls, "class_index": c, "metric_a": a, "metric_b": b, "pc": pc,
                                   "abs_cos_sim": f"{v:.8f}"})
        for (s, t), v in transfer_table(relreps).items():
            transfer_rows.append({"class_name": cls, "class_index": c, "source": s, "target": t, "ev_transfer": f"{v:.8f}"})
        if n_boot > 0:
            print(f"{cls}: bootstrap ({n_boot} resamples)", flush=True)
            res = bootstrap_pca(relreps, n_boot, rng, n_pcs)
            for m in metrics:
                ratio = res["ratio"][m]; cumulative = np.cumsum(ratio)
                for pc in range(ratio.size):
                    spectra_rows.append({
                        "class_name": cls, "metric": m, "pc": pc + 1,
                        "explained_ratio": f"{ratio[pc]:.6f}", "cumulative_explained": f"{cumulative[pc]:.6f}",
                        "ratio_boot_std": f"{res['boot_ratio'][m][:, pc].std(ddof=1):.6f}",
                        "direction_stability_median": f"{np.median(res['boot_self_cos'][m][:, pc]):.4f}",
                        "direction_stability_p05": f"{np.percentile(res['boot_self_cos'][m][:, pc], 5):.4f}",
                    })
            for pair in METRIC_PAIRS:
                if pair not in res["boot_pair_cos"]:
                    continue
                boot = res["boot_pair_cos"][pair]
                for pc in range(n_pcs):
                    boot_rows.append({
                        "class_name": cls, "metric_a": pair[0], "metric_b": pair[1], "pc": pc + 1,
                        "abs_cos_sim": f"{res['pair_cos'][pair][pc]:.4f}",
                        "boot_p16": f"{np.percentile(boot[:, pc], 16):.4f}",
                        "boot_p84": f"{np.percentile(boot[:, pc], 84):.4f}",
                        "boot_std": f"{boot[:, pc].std(ddof=1):.4f}",
                    })
        print(f"{cls}: alignment " + "; ".join(
            f"{a}-{b}: " + ", ".join(f"{v:.3f}" for v in vals) for (a, b), vals in alignment_table(relreps, n_pcs).items()))

    extra = dict(config_hash=cfg.hash(), stage="paper/sec4_single_manifold/run_pca_structure.py",
                 classes=classes, anchors=n_anchors, dtype=args.dtype)
    out = cfg.result_path("pca_alignment"); write_csv(out, list(align_rows[0]), align_rows); write_provenance(out, **extra)
    print(f"wrote {describe(out, cfg)}")
    out = cfg.result_path("ev_transfer"); write_csv(out, list(transfer_rows[0]), transfer_rows); write_provenance(out, **extra)
    print(f"wrote {describe(out, cfg)}")
    if spectra_rows:
        out = cfg.result_path("pca_spectra"); write_csv(out, list(spectra_rows[0]), spectra_rows); write_provenance(out, bootstrap=n_boot, **extra)
        print(f"wrote {describe(out, cfg)}")
        out = cfg.result_path("pca_alignment_bootstrap"); write_csv(out, list(boot_rows[0]), boot_rows); write_provenance(out, bootstrap=n_boot, **extra)
        print(f"wrote {describe(out, cfg)}")


if __name__ == "__main__":
    main()
