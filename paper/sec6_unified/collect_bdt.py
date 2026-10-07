#!/usr/bin/env python
"""Collect the Section 6 BDT runs into results/paper/sec6/feature_progression.npz (Fig. 8, Table 1).

Reads results/experiments/rel_rep_pairwise_bdt/{feature_set}/a{N}/results/bdt_rel_rep_large_{N}.npz
for every feature set in configs/paper.yaml and stacks the test AUCs and background rejections.
Entries of feature sets or class pairs that have not been run are NaN (merged with any existing file).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import add_common_args, describe, load_cfg  # noqa: E402

from m3r.results_io import load_npz, save_npz, write_provenance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args()
    cfg = load_cfg(args)
    sets = list(cfg["bdt"]["feature_sets"].keys())
    a = int(cfg["bdt"]["anchors_per_class"])
    tprs = [float(t) for t in cfg["bdt"]["target_tprs"]]
    exp_root = cfg.resolve(cfg["results"]["experiments"])
    out = cfg.result_path("bdt_feature_progression")
    n = 10
    aucs = np.full((len(sets), n, n), np.nan)
    rej = np.full((len(sets), len(tprs), n, n), np.nan)
    mask = np.zeros((len(sets), n, n), dtype=bool)
    if out.exists():
        old = load_npz(out)
        old_sets = [str(s) for s in old["feature_sets"]]
        for k, s in enumerate(sets):
            if s in old_sets:
                i = old_sets.index(s)
                aucs[k] = old["test_aucs"][i]; rej[k] = old["background_rejections"][i]; mask[k] = old["computed_mask"][i]
    sources = {}
    for k, s in enumerate(sets):
        f = exp_root / "rel_rep_pairwise_bdt" / s / f"a{a}" / "results" / f"bdt_rel_rep_large_{a}.npz"
        if not f.exists():
            continue
        p = load_npz(f)
        m = p["computed_mask"].astype(bool)
        aucs[k][m] = p["test_aucs"][m]
        for t in range(len(tprs)):
            rej[k, t][m] = p["background_rejections"][t][m]
        mask[k] |= m
        sources[s] = describe(f, cfg)
    save_npz(out, test_aucs=aucs, background_rejections=rej, target_tprs=np.array(tprs), feature_sets=np.array(sets),
             computed_mask=mask, anchors_per_class=a, seed=int(cfg["seeds"]["bdt"]))
    write_provenance(out, config_hash=cfg.hash(), stage="paper/sec6_unified/collect_bdt.py", sources=sources)
    print(f"wrote {describe(out, cfg)} ({len(sources)}/{len(sets)} feature sets present)")


if __name__ == "__main__":
    main()
