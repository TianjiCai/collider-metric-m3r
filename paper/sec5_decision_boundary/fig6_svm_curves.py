#!/usr/bin/env python
"""Fig. 6: SVM tagging AUC versus number of references in the relative representation,
with the original-metric ("Abs") SVM as a benchmark, for the six paper tasks.

Reads results/paper/sec5/relrep_sweep_{metric}.npz and abs_baseline_{metric}.npz (from
run_svm.sh / collect_svm.py) and writes figures/svm_curves_all.pdf. ``--variant heatmap``
writes the per-metric AUC tables instead.
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

from m3r.classes import resolve_tasks  # noqa: E402
from m3r.plotting.style import (  # noqa: E402
    ABS_MARKER_GAIN,
    TASK_COLORS,
    TASK_MARKERS,
    savefig,
    scaled_marker_size,
    use_plain_serif_style,
)
from m3r.results_io import load_npz  # noqa: E402

VMIN, VMAX = 0.50, 1.00
PANEL_TITLES = {"ps": r"PS", "sw": r"SW$_2$", "w2": r"$\mathrm{W}_2$"}
TABLE_TITLES = {m: rf"AUC with SVM as the Classifier ({t})" for m, t in
                {"ps": "PS", "sw": r"$\mathrm{SW}_2$", "w2": r"$\mathrm{W}_2$"}.items()}
CURVE_AXIS_LABEL_FONT_SIZE = 26
CURVE_TICK_LABEL_FONT_SIZE = 23.5
CURVE_LEGEND_FONT_SIZE = 24
CURVE_TITLE_FONT_SIZE = 31
CURVE_MARKER_SIZE = 5.4
METRIC_TITLE_PAD = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--variant", choices=["curves", "heatmap"], default="curves")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dump-data", type=Path, default=None)
    return parser.parse_args()


def build_matrix(rel: dict, abs_: dict, tasks) -> tuple[list[int], np.ndarray]:
    anchors = [int(a) for a in rel["anchors_per_class"]]
    z = np.full((len(tasks), len(anchors) + 1), np.nan)
    for row, t in enumerate(tasks):
        z[row, :-1] = rel["test_aucs"][:, t.i, t.j]
        z[row, -1] = abs_["test_aucs"][t.i, t.j]
    return anchors, z


def curve_figure(payload: dict[str, dict], tasks, metric_order) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 8.0), sharey=True)
    handles = labels = None
    for k, (ax, metric) in enumerate(zip(axes, metric_order)):
        anchors, data = payload[metric]["anchors"], payload[metric]["data"]
        x = np.arange(len(anchors) + 1)
        x_labels = [str(2 * a) for a in anchors] + ["Abs"]
        for row, t in enumerate(tasks):
            color, marker = TASK_COLORS[t.slug], TASK_MARKERS[t.slug]
            ms = scaled_marker_size(marker, CURVE_MARKER_SIZE)
            ax.plot(x[:-1], data[row, :-1], marker=marker, linewidth=2.0, markersize=ms, color=color, label=t.label)
            ax.plot(x[-2:], data[row, -2:], linestyle="--", linewidth=1.45, color=color, alpha=0.9)
            ax.scatter(x[-1], data[row, -1], marker=marker, s=(ms * ABS_MARKER_GAIN) ** 2, facecolor=color,
                       edgecolor="black", linewidth=0.95, zorder=4)
        ax.axvline(len(anchors) - 0.5, color="0.35", linestyle="--", linewidth=1.0)
        ax.set_title(PANEL_TITLES[metric], fontsize=CURVE_TITLE_FONT_SIZE, pad=METRIC_TITLE_PAD)
        ax.set_xlabel("Number of Reference Events", fontsize=CURVE_AXIS_LABEL_FONT_SIZE)
        ax.set_xticks(x); ax.set_xticklabels(x_labels)
        ax.set_ylim(VMIN, VMAX + 0.015)
        ax.grid(True, axis="both", linestyle="-", linewidth=0.7, alpha=0.25)
        ax.tick_params(axis="both", labelsize=CURVE_TICK_LABEL_FONT_SIZE)
        ax.tick_params(axis="y", left=True, labelleft=k == 0, right=False)
        if k < len(axes) - 1:
            ax.spines["right"].set_visible(False)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("AUC", fontsize=CURVE_AXIS_LABEL_FONT_SIZE)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, frameon=False,
               fontsize=CURVE_LEGEND_FONT_SIZE, columnspacing=1.3, handlelength=2.0)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.29, top=0.93, wspace=0.0)
    return fig


def heatmap_figure(payload: dict[str, dict], tasks, metric_order) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), constrained_layout=True)
    im = None
    for ax, metric in zip(axes, metric_order):
        anchors, data = payload[metric]["anchors"], payload[metric]["data"]
        im = ax.imshow(data, aspect="auto", vmin=VMIN, vmax=VMAX, cmap="inferno")
        x_labels = [str(a) for a in anchors] + ["Abs"]
        ax.set_title(TABLE_TITLES[metric], fontsize=16, pad=METRIC_TITLE_PAD)
        ax.set_xticks(np.arange(len(x_labels))); ax.set_xticklabels(x_labels)
        ax.set_yticks(np.arange(len(tasks))); ax.set_yticklabels([t.label for t in tasks])
        ax.set_xlabel("Reference events per class", fontsize=18); ax.set_ylabel("Jet tagging task", fontsize=18)
        ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True); ax.set_yticks(np.arange(-0.5, len(tasks), 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=0.8, color="white", alpha=0.85)
        ax.tick_params(which="minor", bottom=False, left=False, length=0)
        ax.tick_params(axis="both", which="both", labelsize=13, length=0, top=False, right=False)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isfinite(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=16,
                            color="black" if data[i, j] >= 0.86 else "white")
    cbar = fig.colorbar(im, ax=axes, pad=0.015, shrink=0.95); cbar.set_label("AUC", fontsize=16)
    fig.suptitle("Relative-to-absolute pairwise SVM AUC progression across metrics", fontsize=24)
    return fig


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)
    tasks = resolve_tasks([t["slug"] for t in cfg["tasks"]])
    metric_order = list(cfg["display"]["metric_panel_order"])
    results_dir = args.results_dir or cfg.result_path("svm_relrep_sweep", metric="w2").parent
    payload = {}
    for metric in metric_order:
        rel = load_npz(results_dir / f"relrep_sweep_{metric}.npz")
        abs_ = load_npz(results_dir / f"abs_baseline_{metric}.npz")
        anchors, data = build_matrix(rel, abs_, tasks)
        payload[metric] = {"anchors": anchors, "data": data}
        if np.isnan(data).any():
            print(f"warning: {metric} has missing entries (NaN) for some tasks/anchors")
    if args.dump_data:
        np.savez(args.dump_data, **{f"{m}_data": v["data"] for m, v in payload.items()},
                 **{f"{m}_anchors": v["anchors"] for m, v in payload.items()})
    use_plain_serif_style(**{"font.size": 14, "axes.labelsize": 18, "axes.titlesize": 16, "xtick.labelsize": 13,
                             "ytick.labelsize": 13, "axes.linewidth": 1.1, "xtick.direction": "in",
                             "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
    if args.variant == "curves":
        fig = curve_figure(payload, tasks, metric_order)
        out = args.out or cfg.figure_path(cfg["figures"]["fig6"])
    else:
        fig = heatmap_figure(payload, tasks, metric_order)
        out = args.out or cfg.figure_path("svm_tables_all.pdf")
    savefig(fig, out); plt.close(fig)
    for metric in metric_order:
        print(metric, "anchors", payload[metric]["anchors"])
        for t, row in zip(tasks, payload[metric]["data"]):
            print(f"  {t.slug:16s} " + " ".join(f"{v:.4f}" for v in row))
    print(f"wrote {describe(out, cfg)}")


if __name__ == "__main__":
    main()
