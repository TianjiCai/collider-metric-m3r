#!/usr/bin/env python
"""Fig. 8 and Table 1: BDT AUC as M3R feature blocks are combined.

Reads results/paper/sec6/feature_progression.npz (from run_bdt.sh / collect_bdt.py) and writes

* figures/unified_repr.pdf               - the PS -> SW2 -> W2 -> PID progression (Fig. 8)
* figures/table1_feature_progression.tex - Table 1 (orderings a and d side by side)

``--variant all`` writes the four-ordering panel figure and the cumulative-gain figure.
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
from matplotlib.ticker import FormatStrFormatter  # noqa: E402

from m3r.classes import resolve_tasks  # noqa: E402
from m3r.plotting.style import TASK_COLORS, TASK_MARKERS, savefig, use_plain_serif_style  # noqa: E402
from m3r.results_io import load_npz  # noqa: E402

AXIS_LABEL_FONT_SIZE = 24
X_TICK_LABEL_FONT_SIZE = 19
Y_TICK_LABEL_FONT_SIZE = 19
LEGEND_FONT_SIZE = 20
VMIN, VMAX = 0.50, 1.00
BLOCK_LABELS = {"ps": "PS", "sw": r"SW$_2$", "w2": r"W$_2$", "pid": "PID"}
TEX_BLOCK = {"ps": r"\mathrm{PS}", "sw": r"\mathrm{SW}_2", "w2": r"\mathrm{W}_2", "pid": r"\mathrm{PID}"}
TEX_TASK_LABELS = {
    "qcd_vs_ttbar": r"\(\mathrm{QCD}\) vs\ \(t\to bqq\)",
    "qcd_vs_hww2q1l": r"\(\mathrm{QCD}\) vs\ \(H \to \ell\nu q q\)",
    "hcc_vs_hgg": r"\(H \to c\bar{c}\) vs\ \(H \to gg\)",
    "hbb_vs_hww4q": r"\(H \to b\bar{b}\) vs\ \(H \to 4q\)",
    "hbb_vs_hww2q1l": r"\(H \to b\bar{b}\) vs\ \(H \to \ell \nu q q\)",
    "wqq_vs_zqq": r"\(W \to qq\) vs\ \(Z \to qq\)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--variant", choices=["paper", "all"], default="paper")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def stage_labels(stages: list[str]) -> list[str]:
    labels = []
    for s in stages:
        labels.append("+".join(BLOCK_LABELS[b] for b in s.split("+")))
    return labels


def progression_data(payload: dict, stages: list[str], tasks) -> np.ndarray:
    sets = [str(s) for s in payload["feature_sets"]]
    data = np.full((len(tasks), len(stages)), np.nan)
    for col, s in enumerate(stages):
        if s not in sets:
            raise KeyError(f"feature set {s!r} not in results ({sets})")
        k = sets.index(s)
        for row, t in enumerate(tasks):
            data[row, col] = payload["test_aucs"][k, t.i, t.j]
    return data


def draw_task_curves(ax, labels, data, tasks, include_mean=False) -> None:
    x = np.arange(len(labels))
    for row, t in enumerate(tasks):
        ax.plot(x, data[row], marker=TASK_MARKERS[t.slug], linewidth=2.0, markersize=9.0, markeredgecolor="0.25",
                markeredgewidth=0.65, color=TASK_COLORS[t.slug], label=t.label, alpha=0.82 if include_mean else 1.0)
    if include_mean:
        ax.plot(x, np.nanmean(data, axis=0), color="black", marker="*", linewidth=2.8, markersize=13.0,
                markeredgecolor="white", markeredgewidth=0.65, label="Mean across shown tasks", zorder=10)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlim(-0.12, len(labels) - 0.88)
    ax.grid(True, axis="both", linestyle="-", linewidth=0.7, alpha=0.25)
    ax.tick_params(axis="x", labelsize=X_TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=Y_TICK_LABEL_FONT_SIZE, left=True, right=False)


def unified_figure(labels, data, tasks) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    draw_task_curves(ax, labels, data, tasks)
    ax.set_ylabel("AUC", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylim(VMIN, VMAX + 0.015); ax.set_yticks(np.arange(VMIN, VMAX + 0.001, 0.1))
    handles, leg = ax.get_legend_handles_labels()
    fig.legend(handles, leg, loc="lower center", bbox_to_anchor=(0.525, 0.0), ncol=3, frameon=False,
               fontsize=LEGEND_FONT_SIZE, columnspacing=1.3, handlelength=2.0)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.225, top=0.985)
    return fig


def panel_figure(progressions, tasks, *, gain: bool = False) -> plt.Figure:
    fig, axes = plt.subplots(nrows=len(progressions), ncols=1, figsize=(13.5, 25.5), sharey=True)
    if gain:
        deltas = np.concatenate([(d - d[:, [0]]).ravel() for _, _, d in progressions])
        step = 0.05
        lo = step * np.floor(np.nanmin(deltas) / step) if np.nanmin(deltas) < -0.01 else 0.0
        hi = max(step, step * np.ceil(np.nanmax(deltas) / step))
    for ax, (title, labels, data) in zip(axes, progressions):
        plot_data = data - data[:, [0]] if gain else data
        draw_task_curves(ax, labels, plot_data, tasks, include_mean=gain)
        if gain:
            ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1.1, alpha=0.75, zorder=0)
            ax.set_ylim(min(-0.015, lo - 0.0075), hi + 0.0075); ax.set_yticks(np.arange(lo, hi + 0.0001, step))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.set_ylabel(r"$\Delta$AUC from First", fontsize=AXIS_LABEL_FONT_SIZE)
        else:
            ax.set_ylim(VMIN, VMAX + 0.015); ax.set_yticks(np.arange(VMIN, VMAX + 0.001, 0.1))
            ax.set_ylabel("AUC", fontsize=AXIS_LABEL_FONT_SIZE)
        ax.set_title(title, fontsize=AXIS_LABEL_FONT_SIZE, loc="left", pad=10)
    handles, leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg, loc="lower center", bbox_to_anchor=(0.525, 0.025), ncol=4 if gain else 3, frameon=False,
               fontsize=LEGEND_FONT_SIZE, columnspacing=1.3, handlelength=2.0)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.105, top=0.985, hspace=0.34)
    return fig


def latex_table(cfg, blocks: list[tuple[list[str], np.ndarray]], tasks) -> str:
    bold, under = float(cfg["bdt"]["table_bold_threshold"]), float(cfg["bdt"]["table_underline_threshold"])
    headers = []
    for stages, _ in blocks:
        first = stages[0].split("+")
        headers.append([rf"\({TEX_BLOCK[first[-1]] if len(first) == 1 else '+'.join(TEX_BLOCK[b] for b in first)}\)"])
        prev = set(first)
        for s in stages[1:]:
            added = [b for b in s.split("+") if b not in prev]
            headers[-1].append(rf"\(+{TEX_BLOCK[added[0]]}\)")
            prev |= set(added)
    cols = "|".join("c" * len(b[0]) for b in blocks)
    lines = [
        r"\begin{table}[h]", r"  \centering", r"  \small", r"  \renewcommand{\arraystretch}{1.15}",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{@{}|l|" + "|".join("|".join(["c"] * len(b[0])) + "|" for b in blocks).replace("||", "|||", 1) + "@{}}",
        r"    \hline",
        "    & " + " & ".join(h for hs in headers for h in hs) + r" \\",
        r"    \hline",
    ]
    for row, t in enumerate(tasks):
        cells = []
        for stages, data in blocks:
            values = data[row]
            for col, v in enumerate(values):
                text = f"{v:.4f}"
                if col > 0 and np.isfinite(values[col - 1]) and values[col - 1] > 0:
                    gain = (v - values[col - 1]) / values[col - 1]
                    if gain > under:
                        text = r"\textbf{\underline{" + text + "}}"
                    elif gain > bold:
                        text = r"\textbf{" + text + "}"
                cells.append(text)
        lines.append(f"    {TEX_TASK_LABELS.get(t.slug, t.slug)} & " + " & ".join(cells) + r" \\")
    lines += [r"    \hline", r"  \end{tabular}",
              r"  \caption{BDT classification AUCs for two orderings in which the M3R feature blocks are successively "
              r"combined. Within each ordering, a column labeled \(+X\) reports the AUC after adding feature block \(X\) "
              r"to those preceding it. Bold entries indicate an incremental AUC improvement greater than "
              + f"{bold:.0%}".replace("%", r"\%") + r", while bold and underlined entries indicate an improvement greater than "
              + f"{under:.0%}".replace("%", r"\%") + r".}",
              r"  \label{tab:feature_progression}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args)
    tasks = resolve_tasks([t["slug"] for t in cfg["tasks"]])
    payload = load_npz(args.results or cfg.result_path("bdt_feature_progression"))
    progs = cfg["bdt"]["progressions"]
    out_dir = args.out_dir or cfg.path("figures")
    use_plain_serif_style(**{"axes.linewidth": 1.1, "xtick.direction": "in", "ytick.direction": "in",
                             "xtick.top": True, "ytick.right": True})
    written = []
    key = cfg["bdt"]["figure_progression"]
    stages = list(progs[key]["stages"])
    data = progression_data(payload, stages, tasks)
    fig = unified_figure(stage_labels(stages), data, tasks)
    written.append(savefig(fig, out_dir / cfg["figures"]["fig8"])); plt.close(fig)
    blocks = [(list(progs[k]["stages"]), progression_data(payload, list(progs[k]["stages"]), tasks))
              for k in cfg["bdt"]["table_progressions"]]
    table_path = out_dir / cfg["figures"]["table1"]
    table_path.write_text(latex_table(cfg, blocks, tasks), encoding="utf-8"); written.append(table_path)
    if args.variant == "all":
        letters = "abcd"
        progressions = []
        for n, (k, p) in enumerate(progs.items()):
            st = list(p["stages"])
            title = f"({letters[n]}) " + r" $\rightarrow$ ".join(BLOCK_LABELS[b] for b in
                                                                  [st[0]] + [[x for x in s.split('+') if x not in st[m].split('+')][0] for m, s in zip(range(len(st) - 1), st[1:])])
            progressions.append((title, stage_labels(st), progression_data(payload, st, tasks)))
        fig = panel_figure(progressions, tasks); written.append(savefig(fig, out_dir / "bdt_feature_progression_all.pdf")); plt.close(fig)
        fig = panel_figure(progressions, tasks, gain=True); written.append(savefig(fig, out_dir / "bdt_feature_progression_gain.pdf")); plt.close(fig)
    for stages_k, d in blocks:
        print(stages_k)
        for t, row in zip(tasks, d):
            print(f"  {t.slug:16s} " + " ".join(f"{v:.4f}" for v in row))
    for p in written:
        print(f"wrote {describe(p, cfg)}")


if __name__ == "__main__":
    main()
