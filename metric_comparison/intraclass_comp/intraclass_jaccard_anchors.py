"""
Intraclass Jaccard anchor progression for JetClass metrics.

This is the metric-aware replacement for the separate
metric_comparison/{w2,ps,sw}/intraclass_comp/intraclass_w2_anchors.py scripts.
For each requested metric and jet class, it builds relative representations at
increasing anchor counts and measures neighborhood preservation with strict
Jaccard similarity.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import simpson
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "collider-metric-comparison-mpl"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils.comparison_utils


CLASS_CONFIGS = {
    "qcd": 0,
    "HToBB": 1,
    "HToCC": 2,
    "HToGG": 3,
    "HToWW4Q": 4,
    "TTbar": 5,
    "WToQQ": 6,
    "ZToQQ": 7,
}

METRIC_LABELS = {
    "w2": "W2",
    "ps": "PS",
    "sw": "SW",
}

DEFAULT_ANCHORS = [1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 100]


@dataclass
class ClassResult:
    metric: str
    class_name: str
    block_idx: int
    num_jets: int
    neighbors_step: int
    seed: int
    mean_jaccard: np.ndarray
    aucs: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intraclass Jaccard anchor analysis for JetClass metrics.",
    )
    parser.add_argument(
        "--project-path",
        type=Path,
        default=REPO_ROOT,
        help=f"Repository root containing data/dist_mat (default: {REPO_ROOT}).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["w2", "ps", "sw"],
        choices=list(METRIC_LABELS.keys()),
        help="Metrics to process (default: w2 ps sw).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(CLASS_CONFIGS.keys()),
        choices=list(CLASS_CONFIGS.keys()),
        help="Jet classes to process (default: all 8 classes used by the notebooks).",
    )
    parser.add_argument(
        "--num-jets",
        type=int,
        default=10000,
        help="Number of jets per class to use from the top-left block (default: 10000).",
    )
    parser.add_argument(
        "--anchors",
        type=int,
        nargs="+",
        default=DEFAULT_ANCHORS,
        help="Anchor counts to evaluate (default: 1 2 5 8 10 15 20 30 40 50 100).",
    )
    parser.add_argument(
        "--neighbors-step",
        type=int,
        default=100,
        help="Step size for neighbor counts in Jaccard computation (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for anchor selection (default: 42).",
    )
    parser.add_argument(
        "--rank-chunk-size",
        type=int,
        default=1024,
        help="Row chunk size for vectorized Jaccard computation (default: 1024).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "metric_comparison",
        help=(
            "Root used for per-metric outputs. Summaries go to "
            "<output-root>/<metric>/intraclass_comp/output/jaccard/."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write summary CSVs without generating PNG plots.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, anchor_values: list[int]) -> None:
    if args.num_jets < 2:
        raise ValueError("--num-jets must be at least 2.")
    if not anchor_values:
        raise ValueError("At least one anchor count is required.")
    if anchor_values[0] <= 0:
        raise ValueError("Anchor counts must be positive.")
    if anchor_values[-1] > args.num_jets:
        raise ValueError(
            f"Maximum anchor count {anchor_values[-1]} exceeds --num-jets={args.num_jets}."
        )
    if args.neighbors_step <= 0:
        raise ValueError("--neighbors-step must be positive.")
    if args.rank_chunk_size <= 0:
        raise ValueError("--rank-chunk-size must be positive.")


def distance_matrix_path(project_path: Path, metric: str, block_idx: int) -> Path:
    return (
        project_path
        / "data"
        / "dist_mat"
        / metric
        / f"large_{metric}"
        / f"{metric}_block_{block_idx}_{block_idx}.npy"
    )


def load_distance_matrix(project_path: Path, metric: str, block_idx: int, num_jets: int) -> np.ndarray:
    matrix_path = distance_matrix_path(project_path, metric, block_idx)
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing distance matrix: {matrix_path}")

    matrix_full = np.load(matrix_path, mmap_mode="r")
    n = min(num_jets, matrix_full.shape[0], matrix_full.shape[1])
    return np.asarray(matrix_full[:n, :n])


def create_hist_figure(
    n_panels: int,
    metric: str,
    class_name: str,
    ncols: int = 4,
) -> tuple[plt.Figure, np.ndarray]:
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4.2, nrows * 3.2),
        constrained_layout=True,
    )
    axes = np.ravel(axes)
    for ax in axes[n_panels:]:
        ax.axis("off")
    fig.suptitle(
        f"{METRIC_LABELS[metric]} / {class_name}: Pairwise Distance Distributions",
        fontsize=13,
    )
    return fig, axes


def fill_hist_panel(ax: plt.Axes, anchor_count: int, distances: np.ndarray) -> None:
    ax.hist(distances, bins="fd", density=True)
    ax.set_title(f"anchors = {anchor_count}  (N={distances.size})", fontsize=10)
    ax.set_xlabel("Pairwise distance")
    ax.set_ylabel("Density")
    lo, hi = float(distances.min()), float(distances.max())
    if hi > lo:
        pad = 0.02 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)


def plot_jaccard_curves(
    jaccard_sims: np.ndarray,
    neighbors_list: np.ndarray,
    anchor_values: list[int],
    num_jets: int,
    metric: str,
    class_name: str,
    ncols: int = 4,
) -> plt.Figure:
    neighbor_pct = neighbors_list / num_jets
    theoretical = neighbors_list / (num_jets - 1)
    n_plots = len(anchor_values)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = np.ravel(axes)

    for i, (ax, anchor_count) in enumerate(zip(axes, anchor_values)):
        ax.plot(neighbor_pct, jaccard_sims[i], label="Avg Jaccard")
        ax.plot(neighbor_pct, theoretical, label="Theoretical", linestyle="--")
        ax.set_title(f"anchors = {anchor_count}")
        ax.set_ylabel("Similarity")
        ax.legend(fontsize=8)

    for ax in axes[n_plots:]:
        ax.axis("off")

    axes[-1].set_xlabel("Neighbor fraction")
    fig.suptitle(
        f"{METRIC_LABELS[metric]} / {class_name}: Jaccard Similarity Curves",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def compute_aucs(
    jaccard_sims: np.ndarray,
    neighbors_list: np.ndarray,
    num_jets: int,
) -> np.ndarray:
    neighbor_pct = neighbors_list / num_jets
    return np.array(
        [
            simpson(y=jaccard_sims[i], x=neighbor_pct)
            for i in range(jaccard_sims.shape[0])
        ]
    )


def jaccard_for_one_anchor(
    original_rank: np.ndarray,
    anchor_rank: np.ndarray,
    neighbors_list: np.ndarray,
    anchor_label: int,
    chunk_size: int,
) -> np.ndarray:
    n = original_rank.shape[0]
    sims = np.empty(len(neighbors_list))

    for j, k in tqdm(
        enumerate(neighbors_list),
        total=len(neighbors_list),
        desc=f"  anchors={anchor_label}",
    ):
        total = 0.0
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            buf = np.empty((end - start, 2 * k), dtype=original_rank.dtype)
            buf[:, :k] = original_rank[start:end, :k]
            buf[:, k:] = anchor_rank[start:end, :k]
            buf.sort(axis=1)
            total += np.sum(buf[:, :-1] == buf[:, 1:])
        sims[j] = total / (n * k)

    return sims


def process_class(
    metric: str,
    class_name: str,
    block_idx: int,
    project_path: Path,
    anchor_values: list[int],
    num_jets: int,
    neighbors_step: int,
    output_dir: Path,
    seed: int,
    rank_chunk_size: int,
    save_plots: bool,
) -> ClassResult:
    print(f"\n{METRIC_LABELS[metric]} / {class_name}: loading block {block_idx}")
    original_matrix = load_distance_matrix(project_path, metric, block_idx, num_jets)
    n_jets = original_matrix.shape[0]
    if anchor_values[-1] > n_jets:
        raise ValueError(
            f"Maximum anchor count {anchor_values[-1]} exceeds available jets ({n_jets}) "
            f"for {metric}/{class_name}."
        )

    labels = np.zeros(n_jets)
    neighbors_list = np.arange(1, n_jets - 1, neighbors_step)
    if neighbors_list.size == 0:
        raise ValueError(
            f"No neighbor counts produced for n_jets={n_jets}, neighbors_step={neighbors_step}."
        )

    original_rank = np.argsort(original_matrix, axis=1)[:, 1:]
    jaccard_sims = np.zeros((len(anchor_values), len(neighbors_list)))
    hist_fig = None
    hist_axes = None
    if save_plots:
        hist_fig, hist_axes = create_hist_figure(len(anchor_values), metric, class_name)

    print("  Computing relative representations and Jaccard curves...")
    for i, anchor_count in enumerate(anchor_values):
        rel_rep = utils.comparison_utils.relative_rep(
            original_matrix,
            labels=labels,
            num_classes=1,
            anchors_per_class=anchor_count,
            seed=seed,
        )

        condensed = pdist(rel_rep, metric="euclidean")
        if save_plots and hist_axes is not None:
            fill_hist_panel(hist_axes[i], anchor_count, condensed)

        relrep_distance_matrix = squareform(condensed)
        del condensed
        anchor_rank = np.argsort(relrep_distance_matrix, axis=1)[:, 1:]
        del relrep_distance_matrix

        jaccard_sims[i] = jaccard_for_one_anchor(
            original_rank=original_rank,
            anchor_rank=anchor_rank,
            neighbors_list=neighbors_list,
            anchor_label=anchor_count,
            chunk_size=rank_chunk_size,
        )
        del anchor_rank

    del original_rank

    if save_plots and hist_fig is not None:
        hist_fig.savefig(
            output_dir / f"{class_name}_pairwise_hist.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(hist_fig)

        fig_jaccard = plot_jaccard_curves(
            jaccard_sims,
            neighbors_list,
            anchor_values,
            n_jets,
            metric,
            class_name,
        )
        fig_jaccard.savefig(
            output_dir / f"{class_name}_jaccard_curves.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig_jaccard)

    aucs = compute_aucs(jaccard_sims, neighbors_list, n_jets)
    mean_jaccard = np.mean(jaccard_sims, axis=1)
    print(f"  Mean Jaccard per anchor: {np.round(mean_jaccard, 4)}")
    print(f"  AUCs: {np.round(aucs, 4)}")

    return ClassResult(
        metric=metric,
        class_name=class_name,
        block_idx=block_idx,
        num_jets=n_jets,
        neighbors_step=neighbors_step,
        seed=seed,
        mean_jaccard=mean_jaccard,
        aucs=aucs,
    )


def output_dir_for_metric(output_root: Path, metric: str) -> Path:
    return output_root / metric / "intraclass_comp" / "output" / "jaccard"


def save_summary_csv(
    results: list[ClassResult],
    anchor_values: list[int],
    output_path: Path,
) -> None:
    fieldnames = [
        "metric",
        "class_name",
        "block_idx",
        "num_jets",
        "neighbors_step",
        "seed",
    ]
    fieldnames += [f"mean_jaccard_a{a}" for a in anchor_values]
    fieldnames += [f"auc_a{a}" for a in anchor_values]

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "metric": result.metric,
                "class_name": result.class_name,
                "block_idx": result.block_idx,
                "num_jets": result.num_jets,
                "neighbors_step": result.neighbors_step,
                "seed": result.seed,
            }
            row.update(
                {
                    f"mean_jaccard_a{a}": f"{value:.8f}"
                    for a, value in zip(anchor_values, result.mean_jaccard)
                }
            )
            row.update(
                {
                    f"auc_a{a}": f"{value:.8f}"
                    for a, value in zip(anchor_values, result.aucs)
                }
            )
            writer.writerow(row)

    print(f"Saved summary CSV: {output_path}")


def plot_jaccard_aucs_summary(
    results: list[ClassResult],
    anchor_values: list[int],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for result in results:
        ax.plot(anchor_values, result.aucs, marker="o", label=result.class_name)

    metric_label = METRIC_LABELS[results[0].metric]
    ax.set_title(f"{metric_label}: Jaccard AUC vs. Anchors")
    ax.set_xlabel("Anchors")
    ax.set_ylabel("Jaccard AUC")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary plot: {output_path}")


def main() -> None:
    args = parse_args()
    anchor_values = sorted(set(args.anchors))
    validate_args(args, anchor_values)
    save_plots = not args.no_plots

    for metric in args.metrics:
        metric_output_dir = output_dir_for_metric(args.output_root, metric)
        metric_output_dir.mkdir(parents=True, exist_ok=True)

        metric_results: list[ClassResult] = []
        for class_name in args.classes:
            metric_results.append(
                process_class(
                    metric=metric,
                    class_name=class_name,
                    block_idx=CLASS_CONFIGS[class_name],
                    project_path=args.project_path,
                    anchor_values=anchor_values,
                    num_jets=args.num_jets,
                    neighbors_step=args.neighbors_step,
                    output_dir=metric_output_dir,
                    seed=args.seed,
                    rank_chunk_size=args.rank_chunk_size,
                    save_plots=save_plots,
                )
            )

        save_summary_csv(
            results=metric_results,
            anchor_values=anchor_values,
            output_path=metric_output_dir / "intraclass_jaccard_summary.csv",
        )
        if save_plots and len(metric_results) > 1:
            plot_jaccard_aucs_summary(
                results=metric_results,
                anchor_values=anchor_values,
                output_path=metric_output_dir / "jaccard_aucs_summary.png",
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
