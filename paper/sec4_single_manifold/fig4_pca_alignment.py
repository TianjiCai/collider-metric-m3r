#!/usr/bin/env python
"""Fig. 4: absolute cosine similarity between the leading principal directions of the
W2, SW2 and PS M3R manifolds, for QCD, W->qq and t->bqq.

Reads results/paper/sec4/pca_alignment.csv (from run_pca_structure.py) and writes
figures/pca_alignment.pdf.
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

from m3r.classes import DISPLAY_NAMES, METRIC_LABELS, class_name  # noqa: E402
from m3r.plotting.style import PAIR_COLORS, savefig, use_paper_style  # noqa: E402
from m3r.results_io import read_csv  # noqa: E402
from m3r.structure import METRIC_PAIRS  # noqa: E402

PC_GROUP_STEP = 0.64
BAR_WIDTH = 0.13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)
    classes = [class_name(c) for c in cfg["structure"]["classes"]]
    n_pcs = int(cfg["structure"]["n_alignment_pcs"])
    rows = read_csv(args.results or cfg.result_path("pca_alignment"))
    values: dict[str, dict[tuple[str, str], np.ndarray]] = {c: {} for c in classes}
    for pair in METRIC_PAIRS:
        for c in classes:
            vals = sorted((int(r["pc"]), float(r["abs_cos_sim"])) for r in rows
                          if class_name(r["class_name"]) == c and (r["metric_a"], r["metric_b"]) == pair)
            values[c][pair] = np.array([v for _, v in vals][:n_pcs])

    use_paper_style(base=16, title=22, label=20, tick=16, legend=19, extra={"axes.linewidth": 0.9, "xtick.top": False, "ytick.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.2), sharey=True)
    x = np.arange(n_pcs, dtype=float) * PC_GROUP_STEP
    offsets = (np.arange(len(METRIC_PAIRS)) - (len(METRIC_PAIRS) - 1) / 2) * BAR_WIDTH
    legend_handles = None
    legend_labels = [f"{METRIC_LABELS[a]} vs {METRIC_LABELS[b]}" for a, b in METRIC_PAIRS]
    for k, (ax, c) in enumerate(zip(axes, classes)):
        handles = []
        for p, pair in enumerate(METRIC_PAIRS):
            bars = ax.bar(x + offsets[p], values[c][pair], width=BAR_WIDTH, color=PAIR_COLORS[pair], zorder=3)
            handles.append(bars[0])
        legend_handles = legend_handles or handles
        title = "QCD" if c == "QCD" else DISPLAY_NAMES[c]
        ax.set_title(f"{title} Manifold", pad=7)
        ax.set_xticks(x, [f"PC{r}" for r in range(1, n_pcs + 1)])
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=True, labeltop=False, length=0)
        ax.yaxis.set_ticks_position("left")
        ax.tick_params(axis="y", which="major", left=True, labelleft=k == 0, right=False, labelright=False, length=4.2, width=0.8)
        ax.tick_params(axis="y", which="minor", left=True, labelleft=k == 0, right=False, labelright=False, length=2.3, width=0.7)
        pad = 0.5 * PC_GROUP_STEP
        ax.set_xlim(x[0] - pad, x[-1] + pad); ax.set_ylim(0.0, 1.02)
        ax.grid(axis="y", alpha=0.2, zorder=0)
        if k < len(axes) - 1:
            ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("AbsCosSim", loc="center", labelpad=10)
    axes[0].yaxis.set_label_coords(-0.085, 0.5)
    fig.legend(legend_handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncols=3, frameon=True,
               columnspacing=1.35, handlelength=1.55)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.25, top=0.91, wspace=0.0)
    out = args.out or cfg.figure_path(cfg["figures"]["fig4"])
    savefig(fig, out); plt.close(fig)
    for c in classes:
        print(c, {f"{a}-{b}": np.round(v, 4).tolist() for (a, b), v in values[c].items()})
    print(f"wrote {describe(out, cfg)}")


if __name__ == "__main__":
    main()
