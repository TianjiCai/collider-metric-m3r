#!/usr/bin/env python
"""Fig. 3: JAUC and MADO versus number of references, by metric.

Reads results/paper/sec4/{jauc,mado}_{metric}.csv (from run_intraclass.py) and writes

* figures/jm_raw_rep.pdf   - absolute JAUC (top) and MADO (bottom), six representative classes
* figures/jm_ratio_rep.pdf - the same curves normalised to their value at N_ref = 100

Variants: --variant all (eight classes), avg (class-averaged ratio with spread), grid (one
panel per class, metrics overlaid); --scale linear for a linear x axis (files get a _ln suffix).
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
from matplotlib.ticker import FormatStrFormatter, MultipleLocator  # noqa: E402

from m3r.classes import DISPLAY_NAMES_COMPACT, class_name  # noqa: E402
from m3r.plotting.style import (  # noqa: E402
    CLASS_MARKERS,
    JET_COLORS,
    METRIC_STYLE,
    REPRESENTATIVE_CLASS_COLORS,
    savefig,
    use_paper_style,
)
from m3r.results_io import read_csv  # noqa: E402

METRICS = ("ps", "sw", "w2")   # panel order
METRIC_TITLES = {"ps": "PS", "sw": r"$\mathrm{SW}_2$", "w2": r"$W_2$"}
RATIO_METRIC_TITLES = {m: f"{t} (ratio)" for m, t in METRIC_TITLES.items()}
PLOT_BASE_FONT_SIZE = 15
PLOT_TITLE_FONT_SIZE = 24
PLOT_AXIS_LABEL_FONT_SIZE = 19
PLOT_TICK_LABEL_FONT_SIZE = 15.5
PLOT_LEGEND_FONT_SIZE = 17.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--variant", choices=["rep", "all", "avg", "grid"], default="rep")
    parser.add_argument("--scale", choices=["log", "linear"], default="log")
    parser.add_argument("--results-dir", type=Path, default=None, help="Directory holding jauc_*.csv / mado_*.csv.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: figures/).")
    parser.add_argument("--dump-data", type=Path, default=None, help="Write the plotted arrays to this .npz.")
    return parser.parse_args()


def load_values(path: Path, columns: list[str], classes: list[str]) -> np.ndarray:
    rows = {class_name(r["class_name"]): r for r in read_csv(path)}
    missing = [c for c in classes if c not in rows]
    if missing:
        raise ValueError(f"{path} is missing classes {missing}")
    return np.array([[float(rows[c][col]) for col in columns] for c in classes])


def normalize_to_asymptote(values: np.ndarray) -> np.ndarray:
    asym = values[:, [-1]]
    if np.any(~np.isfinite(asym)) or np.any(asym == 0):
        raise ValueError(f"invalid asymptotic values {asym.ravel()}")
    return values / asym


def class_style(name: str, representative: bool) -> dict:
    style = dict(color=JET_COLORS.get(name, "0.3"), marker=CLASS_MARKERS.get(name, "o"), label=DISPLAY_NAMES_COMPACT[name])
    if name == "QCD":
        style["zorder"] = 4
    if representative:
        style["color"] = REPRESENTATIVE_CLASS_COLORS[name]
    return style


def legend_row_major(ax, ncols):
    handles, labels = ax.get_legend_handles_labels()
    if ncols <= 1:
        return handles, labels
    nrows = (len(handles) + ncols - 1) // ncols
    order = [r * ncols + c for c in range(ncols) for r in range(nrows) if r * ncols + c < len(handles)]
    return [handles[i] for i in order], [labels[i] for i in order]


def align_log_endpoint_labels(axes) -> None:
    axes = np.asarray(axes)
    axes.flat[0].figure.canvas.draw()
    for ax in axes.flat:
        for tick in ax.xaxis.get_major_ticks():
            loc = tick.get_loc()
            if np.isclose(loc, 1.0):
                tick.label1.set_horizontalalignment("left"); tick.label2.set_horizontalalignment("left")
            elif np.isclose(loc, 100.0):
                tick.label1.set_horizontalalignment("right"); tick.label2.set_horizontalalignment("right")


def add_centered_row_ylabels(fig, axes, row_labels, fontsize) -> None:
    axes = np.asarray(axes)
    fig.canvas.draw()
    left = min(ax.get_position().x0 for ax in axes.flat)
    for r, label in enumerate(row_labels):
        bottom = min(ax.get_position().y0 for ax in axes[r]); top = max(ax.get_position().y1 for ax in axes[r])
        fig.text(left - 0.035, 0.5 * (bottom + top), label, rotation=90, ha="center", va="center", fontsize=fontsize)


def limits(values_by_metric: dict[str, np.ndarray], idx: list[int]) -> tuple[float, float]:
    y = np.concatenate([values_by_metric[m][idx].ravel() for m in METRICS])
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    span = max(ymax - ymin, 0.02)
    return max(0.0, ymin - 0.08 * span), max(1.02, ymax + 0.08 * span)


def column_plot(anchors, jauc, mado, classes, all_classes, scale, xlim, ratio: bool, representative: bool):
    x = np.asarray(anchors)
    idx = [all_classes.index(c) for c in classes]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.4), sharex=True, sharey="row", gridspec_kw={"wspace": 0.0, "hspace": 0.0})
    asym = anchors[-1]
    if ratio:
        rows = [(rf"JAUC / JAUC$_{{{asym}}}$", {m: normalize_to_asymptote(v) for m, v in jauc.items()}),
                (rf"MADO / MADO$_{{{asym}}}$", {m: normalize_to_asymptote(v) for m, v in mado.items()})]
        titles = RATIO_METRIC_TITLES
    else:
        rows = [("JAUC", jauc), ("MADO", mado)]
        titles = METRIC_TITLES
    for r, (ylabel, data) in enumerate(rows):
        ymin, ymax = limits(data, idx)
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            if r == 0:
                ax.set_title(titles[metric], fontsize=PLOT_TITLE_FONT_SIZE, pad=18)
            for name, i in zip(classes, idx):
                ax.plot(x, data[metric][i], lw=2.1, markersize=4.5, **class_style(name, representative))
            if ratio:
                ax.axhline(1.0, color="0.25", lw=0.9, ls="--", alpha=0.55, zorder=0)
            ax.set_xscale(scale); ax.set_xlim(*xlim); ax.set_ylim(ymin, ymax)
            if ratio:
                ax.yaxis.set_major_locator(MultipleLocator(0.1)); ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.grid(True, which="both", alpha=0.16)
            ax.tick_params(top=False, labeltop=False, labelbottom=r == 1, pad=6)
            if c != 0:
                ax.tick_params(labelleft=False)
            if r == 0 and c == 0:
                ncols = 2 if len(classes) > 3 else 1
                handles, labels = legend_row_major(ax, ncols)
                ax.legend(handles, labels, frameon=True, facecolor="white", framealpha=0.82, edgecolor="none",
                          loc="lower right", fontsize=PLOT_LEGEND_FONT_SIZE if len(classes) <= 6 else 14.5,
                          ncol=ncols, handlelength=1.3, columnspacing=0.75, labelspacing=0.25, borderpad=0.25)
    align_log_endpoint_labels(axes)
    fig.supxlabel("Number of Reference Events", fontsize=PLOT_AXIS_LABEL_FONT_SIZE, y=0.025)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.12, top=0.86, wspace=0.0, hspace=0.0)
    add_centered_row_ylabels(fig, axes, [l for l, _ in rows], PLOT_AXIS_LABEL_FONT_SIZE)
    return fig


def mean_ratio_plot(anchors, jauc, mado, scale, xlim):
    x = np.asarray(anchors); asym = anchors[-1]
    specs = []
    for label, data in ((rf"Mean JAUC / JAUC$_{{{asym}}}$", jauc), (rf"Mean MADO / MADO$_{{{asym}}}$", mado)):
        ratios = {m: normalize_to_asymptote(v) for m, v in data.items()}
        specs.append((label, {m: r.mean(axis=0) for m, r in ratios.items()}, {m: r.std(axis=0) for m, r in ratios.items()}))
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.4), sharex=True, sharey="row", gridspec_kw={"wspace": 0.0, "hspace": 0.0})
    for r, (ylabel, mean, std) in enumerate(specs):
        y = np.concatenate([(mean[m] - std[m]).ravel() for m in METRICS] + [(mean[m] + std[m]).ravel() for m in METRICS])
        span = max(float(np.nanmax(y) - np.nanmin(y)), 0.02)
        ymin, ymax = max(0.0, float(np.nanmin(y)) - 0.08 * span), max(1.02, float(np.nanmax(y)) + 0.08 * span)
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            if r == 0:
                ax.set_title(METRIC_TITLES[metric], fontsize=PLOT_TITLE_FONT_SIZE, pad=18)
            color = METRIC_STYLE[metric]["color"]
            ax.fill_between(x, mean[metric] - std[metric], mean[metric] + std[metric], color=color, alpha=0.18, linewidth=0.0, zorder=1)
            ax.plot(x, mean[metric], lw=2.4, markersize=5.0, zorder=2, **METRIC_STYLE[metric])
            ax.axhline(1.0, color="0.25", lw=0.9, ls="--", alpha=0.55, zorder=0)
            ax.set_xscale(scale); ax.set_xlim(*xlim); ax.set_ylim(ymin, ymax)
            ax.grid(True, which="both", alpha=0.16)
            ax.tick_params(top=False, labeltop=False, labelbottom=r == 1, pad=6)
            if c != 0:
                ax.tick_params(labelleft=False)
    align_log_endpoint_labels(axes)
    fig.supxlabel("Number of Reference Events", fontsize=22, y=0.025)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.12, top=0.86, wspace=0.0, hspace=0.0)
    add_centered_row_ylabels(fig, axes, [s[0] for s in specs], 22)
    return fig


def class_grid_plot(anchors, data, all_classes, ylabel, scale, xlim):
    x = np.asarray(anchors)
    ratios = {m: normalize_to_asymptote(v) for m, v in data.items()}
    idx = list(range(len(all_classes)))
    ymin, ymax = limits(ratios, idx)
    fig, axes = plt.subplots(2, 4, figsize=(17.0, 7.8), sharex=True, sharey=True, gridspec_kw={"wspace": 0.0, "hspace": 0.0})
    for i, ax in enumerate(axes.flat):
        if i >= len(all_classes):
            ax.axis("off"); continue
        for metric, style in METRIC_STYLE.items():
            ax.plot(x, ratios[metric][i], lw=2.1, markersize=4.5, **style)
        ax.axhline(1.0, color="0.25", lw=0.9, ls="--", alpha=0.55, zorder=0)
        ax.text(0.5, 0.12, DISPLAY_NAMES_COMPACT[all_classes[i]], transform=ax.transAxes, ha="center", va="bottom", fontsize=19,
                zorder=20, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.0})
        ax.set_xscale(scale); ax.set_xlim(*xlim); ax.set_ylim(ymin, ymax)
        ax.grid(True, which="both", alpha=0.16)
        ax.tick_params(top=False, labeltop=False, labelbottom=i >= 4, pad=6)
        if i % 4 != 0:
            ax.tick_params(labelleft=False)
        if i == 0:
            ax.legend(frameon=True, facecolor="white", framealpha=0.82, edgecolor="none", loc="lower right",
                      fontsize=15, handlelength=1.4, borderpad=0.25)
    align_log_endpoint_labels(axes)
    fig.supxlabel("Number of Reference Events", fontsize=22, y=0.025)
    fig.supylabel(ylabel, fontsize=22, x=0.04)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.12, top=0.94, wspace=0.0, hspace=0.0)
    return fig


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)
    sec = cfg["intraclass"]
    anchors = list(sec["anchors"])
    all_classes = [class_name(c) for c in sec["classes"]]
    rep_classes = [class_name(c) for c in sec["representative_classes"]]
    results_dir = args.results_dir or cfg.result_path("intraclass_jauc", metric="w2").parent
    out_dir = args.out_dir or cfg.path("figures")
    jauc = {m: load_values(results_dir / f"jauc_{m}.csv", [f"auc_a{a}" for a in anchors], all_classes) for m in METRICS}
    mado = {m: load_values(results_dir / f"mado_{m}.csv", [f"decision_ordering_a{a}" for a in anchors], all_classes) for m in METRICS}
    if args.dump_data:
        np.savez(args.dump_data, anchors=anchors, classes=all_classes,
                 **{f"jauc_{m}": jauc[m] for m in METRICS}, **{f"mado_{m}": mado[m] for m in METRICS})

    use_paper_style(base=PLOT_BASE_FONT_SIZE, title=PLOT_TITLE_FONT_SIZE, label=PLOT_AXIS_LABEL_FONT_SIZE,
                    tick=PLOT_TICK_LABEL_FONT_SIZE, legend=PLOT_LEGEND_FONT_SIZE)
    scale = args.scale
    xlim = (0.8, 120) if scale == "log" else (0, 105)
    sfx = "" if scale == "log" else "_ln"
    fig_names = cfg["figures"]["fig3"]
    written = []
    if args.variant == "rep":
        fig = column_plot(anchors, jauc, mado, rep_classes, all_classes, scale, xlim, ratio=False, representative=True)
        written.append(savefig(fig, out_dir / (Path(fig_names[0]).stem + sfx + ".pdf"), dpi=300)); plt.close(fig)
        fig = column_plot(anchors, jauc, mado, rep_classes, all_classes, scale, xlim, ratio=True, representative=True)
        written.append(savefig(fig, out_dir / (Path(fig_names[1]).stem + sfx + ".pdf"), dpi=300)); plt.close(fig)
    elif args.variant == "all":
        fig = column_plot(anchors, jauc, mado, all_classes, all_classes, scale, xlim, ratio=False, representative=False)
        written.append(savefig(fig, out_dir / f"jm_raw_all{sfx}.pdf", dpi=300)); plt.close(fig)
        fig = column_plot(anchors, jauc, mado, all_classes, all_classes, scale, xlim, ratio=True, representative=False)
        written.append(savefig(fig, out_dir / f"jm_ratio_all{sfx}.pdf", dpi=300)); plt.close(fig)
    elif args.variant == "avg":
        fig = mean_ratio_plot(anchors, jauc, mado, scale, xlim)
        written.append(savefig(fig, out_dir / f"jm_ratio_avg{sfx}.pdf", dpi=300)); plt.close(fig)
    else:
        asym = anchors[-1]
        fig = class_grid_plot(anchors, jauc, all_classes, rf"JAUC / JAUC$_{{{asym}}}$", scale, xlim)
        written.append(savefig(fig, out_dir / f"jauc_ratio_grid{sfx}.pdf", dpi=300)); plt.close(fig)
        fig = class_grid_plot(anchors, mado, all_classes, rf"MADO / MADO$_{{{asym}}}$", scale, xlim)
        written.append(savefig(fig, out_dir / f"mado_ratio_grid{sfx}.pdf", dpi=300)); plt.close(fig)
    for p in written:
        print(f"wrote {describe(p, cfg)}")


if __name__ == "__main__":
    main()
