#!/usr/bin/env python
"""Fig. 5: fraction of target-manifold variance explained by the source manifold's PC1 direction.

Reads results/paper/sec4/ev_transfer.csv (from run_pca_structure.py) and writes figures/ev_transfer.pdf.
Bars are grouped by target metric and coloured by source metric; self-transfers are grey.
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

from m3r.classes import DISPLAY_NAMES, METRIC_SYMBOLS, class_name  # noqa: E402
from m3r.plotting.style import SELF_TRANSFER_COLORS, TRANSFER_COLORS, savefig, use_paper_style  # noqa: E402
from m3r.results_io import read_csv  # noqa: E402

BAR_COMPARISONS = (("sw", "w2"), ("ps", "w2"), ("w2", "w2"), ("w2", "sw"), ("ps", "sw"), ("sw", "sw"),
                   ("w2", "ps"), ("sw", "ps"), ("ps", "ps"))
LEGEND_ROWS = ((("sw", "w2"), ("sw", "ps"), ("sw", "sw")), (("w2", "sw"), ("w2", "ps"), ("w2", "w2")),
               (("ps", "w2"), ("ps", "sw"), ("ps", "ps")))
BAR_WIDTH, TARGET_BLOCK_GAP, GROUP_GAP = 0.36, 0.10, 1.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def bar_color(pair):
    s, t = pair
    return SELF_TRANSFER_COLORS[s] if s == t else TRANSFER_COLORS[pair]


def legend_label(pair):
    s, t = pair
    return rf"${METRIC_SYMBOLS[s]}^{{(1)}} \to {METRIC_SYMBOLS[t]}$"


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)
    classes = [class_name(c) for c in cfg["structure"]["classes"]]
    rows = read_csv(args.results or cfg.result_path("ev_transfer"))
    transfers = {c: {(r["source"], r["target"]): float(r["ev_transfer"]) for r in rows if class_name(r["class_name"]) == c}
                 for c in classes}

    use_paper_style(base=11, title=12, label=11, tick=9.5, legend=10, extra={
        "figure.titlesize": 13, "legend.title_fontsize": 10, "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.minor.size": 2, "ytick.minor.size": 2, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6, "axes.linewidth": 0.9, "xtick.top": False, "ytick.right": False})
    n_bars = len(BAR_COMPARISONS)
    pos = np.arange(n_bars, dtype=float) * BAR_WIDTH
    pos[3:] += TARGET_BLOCK_GAP; pos[6:] += TARGET_BLOCK_GAP
    offsets = pos - pos.mean()
    centers = np.arange(len(classes)) * (n_bars * BAR_WIDTH + GROUP_GAP)
    colors = [bar_color(p) for p in BAR_COMPARISONS]
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    all_values, handles = [], {}
    for k, c in enumerate(classes):
        values = [transfers[c][p] for p in BAR_COMPARISONS]
        all_values.extend(values)
        bars = ax.bar(centers[k] + offsets, values, width=BAR_WIDTH, color=colors, edgecolor="none", linewidth=0.0)
        if k == 0:
            handles = dict(zip(BAR_COMPARISONS, bars))
    ax.set_ylabel("PC1 Explained Variance Transfer", fontsize=15, loc="center")
    ax.yaxis.set_label_coords(-0.055, 0.5)
    ax.set_xticks(centers, [f"{'QCD' if c == 'QCD' else DISPLAY_NAMES[c]} Manifold" for c in classes])
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelsize=12.5)
    ax.tick_params(axis="y", which="both", labelsize=12)
    ax.set_xlim(centers[0] + offsets[0] - 0.48, centers[-1] + offsets[-1] + 0.48)
    ax.set_ylim(0.0, min(1.16, max(0.2, max(all_values) * 1.22)))
    ax.grid(axis="y", alpha=0.2)
    legend_pairs = [p for column in zip(*LEGEND_ROWS) for p in column]
    ax.legend([handles[p] for p in legend_pairs], [legend_label(p) for p in legend_pairs], frameon=True, ncols=3,
              fontsize=10.0, loc="upper right", columnspacing=0.9, handlelength=1.7)
    fig.tight_layout()
    out = args.out or cfg.figure_path(cfg["figures"]["fig5"])
    savefig(fig, out); plt.close(fig)
    for c in classes:
        print(c, {f"{s}->{t}": round(v, 4) for (s, t), v in transfers[c].items()})
    print(f"wrote {describe(out, cfg)}")


if __name__ == "__main__":
    main()
