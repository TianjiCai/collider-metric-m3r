#!/usr/bin/env python
"""Collect the Section 5 Hydra outputs into the compact, tracked result files used by Figs. 6-7.

* results/paper/sec5/relrep_sweep_{metric}.npz - test AUC (anchor sweep x 10 x 10) from
  results/experiments/rel_rep_pairwise_svm/fixed/{metric}/a*/
* results/paper/sec5/abs_baseline_{metric}.npz - from results/experiments/abs_pairwise_svm/{metric}/
* results/paper/sec5/sv_study_{metric}.npz     - from results/experiments/rel_rep_pairwise_svm/study/{metric}/a{N}/

Existing entries are kept where a run did not cover a class pair (merging by computed_mask),
so partial re-runs (for example only the six paper tasks) update just those pairs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import add_common_args, describe, load_cfg  # noqa: E402

from m3r.results_io import load_npz, save_npz, write_provenance  # noqa: E402


def merge_matrix(old: np.ndarray | None, new: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if old is None:
        out = np.full_like(new, np.nan, dtype=float)
    else:
        out = np.array(old, dtype=float, copy=True)
    out[mask] = new[mask]
    return out


def collect_rel(cfg, metric: str, exp_root: Path) -> None:
    anchors = [int(a) for a in cfg["svm_relrep"]["anchors_per_class"]]
    out = cfg.result_path("svm_relrep_sweep", metric=metric)
    old = load_npz(out) if out.exists() else None
    n = 10
    aucs = np.full((len(anchors), n, n), np.nan)
    mask_all = np.zeros((n, n), dtype=bool)
    sources = {}
    if old is not None and "test_aucs" in old and list(old["anchors_per_class"]) == anchors:
        aucs = old["test_aucs"].astype(float)
        mask_all = old["computed_mask"].astype(bool)
    found = 0
    for k, a in enumerate(anchors):
        f = exp_root / "rel_rep_pairwise_svm" / "fixed" / metric / f"a{a}" / "results" / f"svm_rel_rep_large_{metric}_{a}.npz"
        if not f.exists():
            continue
        p = load_npz(f)
        mask = p["computed_mask"].astype(bool)
        aucs[k] = merge_matrix(aucs[k], p["test_aucs"], mask)
        mask_all |= mask
        sources[str(a)] = describe(f, cfg)
        found += 1
    if found == 0:
        print(f"  [{metric}] no relative-space runs found; keeping {describe(out, cfg) if old else 'nothing'}")
        return
    save_npz(out, test_aucs=aucs, anchors_per_class=np.array(anchors), computed_mask=mask_all,
             C=float(cfg["svm_relrep"]["svc"]["C"]), gamma=str(cfg["svm_relrep"]["svc"]["gamma"]), kernel="rbf",
             sw_coordinates="distance" if cfg["metrics"].get("sw_as_distance", True) or metric != "sw" else "stored")
    write_provenance(out, config_hash=cfg.hash(), stage="paper/sec5_decision_boundary/collect_svm.py", sources=sources)
    print(f"  [{metric}] wrote {describe(out, cfg)} ({found}/{len(anchors)} anchor counts)")


def collect_abs(cfg, metric: str, exp_root: Path) -> None:
    f = exp_root / "abs_pairwise_svm" / metric / "results" / f"svm_abs_large_{metric}.npz"
    if not f.exists():
        return
    out = cfg.result_path("svm_abs_baseline", metric=metric)
    old = load_npz(out) if out.exists() else None
    p = load_npz(f)
    mask = p["computed_mask"].astype(bool)
    payload = {}
    for key in ("test_aucs", "val_aucs", "best_Cs", "best_gammas", "sv_counts", "sv_fractions"):
        payload[key] = merge_matrix(old.get(key) if old else None, p[key].astype(float), mask)
    payload["computed_mask"] = (old["computed_mask"].astype(bool) if old else np.zeros_like(mask)) | mask
    for key in ("C_grid", "gamma_grid", "test_fraction", "validation_fraction"):
        payload[key] = p[key]
    save_npz(out, **payload)
    write_provenance(out, config_hash=cfg.hash(), stage="paper/sec5_decision_boundary/collect_svm.py", source=describe(f, cfg))
    print(f"  [{metric}] wrote {describe(out, cfg)}")


def collect_sv(cfg, metric: str, exp_root: Path) -> None:
    a = int(cfg["svm_sv"]["anchors_per_class"])
    f = exp_root / "rel_rep_pairwise_svm" / "study" / metric / f"a{a}" / "results" / f"svm_rel_rep_large_{metric}_{a}.npz"
    if not f.exists():
        return
    out = cfg.result_path("svm_sv_study", metric=metric)
    old = load_npz(out) if out.exists() else None
    p = load_npz(f)
    mask = p["computed_mask"].astype(bool)
    payload = {"anchors_per_class": a}
    for key in ("test_aucs", "val_aucs", "best_C", "best_gamma", "sv_fractions", "sv_counts"):
        payload[f"rel_{key}"] = merge_matrix(old.get(f"rel_{key}") if old else None, p[key].astype(float), mask)
    payload["computed_mask"] = (old["computed_mask"].astype(bool) if old else np.zeros_like(mask)) | mask
    payload["rel_C_grid"] = p["C_grid"]; payload["rel_gamma_grid"] = p["gamma_grid"]
    payload["test_fraction"] = p["test_fraction"]; payload["validation_fraction"] = p["validation_fraction"]
    # optional absolute-space companion from the abs baseline run (same stratified split)
    abs_out = cfg.result_path("svm_abs_baseline", metric=metric)
    if abs_out.exists():
        q = load_npz(abs_out)
        payload["abs_test_aucs"] = q["test_aucs"]; payload["abs_sv_fractions"] = q["sv_fractions"]
        payload["abs_sv_counts"] = q["sv_counts"]; payload["abs_best_C"] = q["best_Cs"]; payload["abs_best_gamma"] = q["best_gammas"]
    save_npz(out, **payload)
    write_provenance(out, config_hash=cfg.hash(), stage="paper/sec5_decision_boundary/collect_svm.py", source=describe(f, cfg))
    print(f"  [{metric}] wrote {describe(out, cfg)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--parts", nargs="+", choices=["rel", "abs", "sv"], default=["rel", "abs", "sv"])
    args = parser.parse_args()
    cfg = load_cfg(args)
    exp_root = cfg.resolve(cfg["results"]["experiments"])
    for metric in args.metrics or list(cfg["metrics"]["kinematic"]):
        if "rel" in args.parts:
            collect_rel(cfg, metric, exp_root)
        if "abs" in args.parts:
            collect_abs(cfg, metric, exp_root)
        if "sv" in args.parts:
            collect_sv(cfg, metric, exp_root)


if __name__ == "__main__":
    main()
