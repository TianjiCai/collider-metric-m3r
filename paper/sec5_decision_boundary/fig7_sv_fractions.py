#!/usr/bin/env python
"""Fig. 7: support-vector fractions of the relative-space SVMs for the six paper tasks,
with the tagging AUC printed above each bar.

Reads results/paper/sec5/sv_study_{metric}.npz (from run_svm.sh / collect_svm.py) and writes
figures/sv_fractions_rel.pdf plus a CSV of the plotted values. ``--space abs`` plots the
absolute-space companion (from the Abs baseline run).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import add_common_args, describe, load_cfg  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from m3r.classes import METRIC_LABELS, resolve_tasks  # noqa: E402
from m3r.plotting.style import savefig, use_paper_style  # noqa: E402
from m3r.results_io import load_npz, write_csv  # noqa: E402

METRICS = ("w2", "sw", "ps")
BAR_COLORS = {"w2": "#b2182b", "sw": "#133c8b", "ps": "#43aedf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--space", choices=["rel", "abs"], default="rel")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--file-pattern", default="sv_study_{metric}.npz")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_values(path: Path, space: str) -> tuple[np.ndarray, np.ndarray]:
    p = load_npz(path)
    aucs = p.get(f"{space}_test_aucs", p.get(f"{space}_aucs"))
    frac = p[f"{space}_sv_fractions"]
    if aucs is None:
        raise KeyError(f"{path} has no {space} AUC array")
    return aucs, frac


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)
    tasks = resolve_tasks([t["slug"] for t in cfg["tasks"]])
    results_dir = args.results_dir or cfg.result_path("svm_sv_study", metric="w2").parent
    data = np.full((len(tasks), len(METRICS)), np.nan)
    auc_data = np.full_like(data, np.nan)
    rows = []
    for k, metric in enumerate(METRICS):
        aucs, frac = load_values(results_dir / args.file_pattern.format(metric=metric), args.space)
        for row, t in enumerate(tasks):
            data[row, k] = frac[t.i, t.j]; auc_data[row, k] = aucs[t.i, t.j]
            rows.append({"space": args.space, "task_slug": t.slug, "class_i": t.i, "class_j": t.j, "metric": metric,
                         "support_vector_fraction": f"{data[row, k]:.6f}", "auc": f"{auc_data[row, k]:.6f}"})

    use_paper_style(base=13, title=16, label=15, tick=12, legend=12.5, extra={
        "figure.titlesize": 16, "legend.title_fontsize": 10.5, "xtick.labelsize": 10.5, "xtick.major.size": 4,
        "ytick.major.size": 4, "xtick.minor.size": 2, "ytick.minor.size": 2, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6, "axes.linewidth": 0.9,
        "xtick.top": False, "ytick.right": False})
    x = np.arange(len(tasks), dtype=float) + 0.5
    bar_width = 0.26
    offsets = (np.arange(len(METRICS)) - (len(METRICS) - 1) / 2) * bar_width
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for k, metric in enumerate(METRICS):
        bars = ax.bar(x + offsets[k], data[:, k], width=bar_width, color=BAR_COLORS[metric], label=METRIC_LABELS[metric],
                      edgecolor="none")
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in auc_data[:, k]], padding=4, fontsize=9)
    ax.set_xlabel("Jet-Tagging Task", labelpad=7, loc="center")
    ax.set_ylabel("Support Vector Fraction", labelpad=8, loc="center")
    ax.set_xlim(0, len(tasks)); ax.set_ylim(0.0, 1.12)
    ax.set_xticks(x, [t.label for t in tasks])
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=True, labeltop=False, labelrotation=0,
                   labelsize=10.5, length=0, pad=13)
    for label in ax.get_xticklabels():
        label.set_verticalalignment("baseline")
    ax.tick_params(axis="y", which="major", labelsize=12, pad=4)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.legend(frameon=True, ncols=3, fontsize=12.5, loc="upper left")
    fig.tight_layout(pad=0.9)
    out = args.out or (cfg.figure_path(cfg["figures"]["fig7"]) if args.space == "rel" else cfg.figure_path("sv_fractions_abs.pdf"))
    savefig(fig, out); plt.close(fig)
    csv_out = out.with_suffix(".csv")
    write_csv(csv_out, list(rows[0].keys()), rows)
    for t, frac_row, auc_row in zip(tasks, data, auc_data):
        print(f"  {t.slug:16s} SV " + " ".join(f"{v:.3f}" for v in frac_row) + "  AUC " + " ".join(f"{v:.4f}" for v in auc_row))
    print(f"wrote {describe(out, cfg)} and {describe(csv_out, cfg)}")


if __name__ == "__main__":
    main()
