"""
Intraclass W2 anchor analysis for all JetClass particle types.

Computes relative representations at varying anchor counts, evaluates
neighborhood preservation via strict Jaccard similarity, and reports
AUC scores. Saves all figures and a summary CSV.

Refactored from per-class notebooks ({class_type}_w2_anchors.ipynb).
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

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


def load_distance_matrix(project_path: str, block_idx: int) -> np.ndarray:
    fpath = os.path.join(project_path, f"data/dist_mat/w2/large_w2/w2_block_{block_idx}_{block_idx}.npy")
    return np.load(fpath)


def _create_hist_figure(
    n_panels: int,
    class_name: str,
    ncols: int = 4,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a histogram figure with empty subplots, to be filled one at a time."""
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.2, nrows * 3.2),
        constrained_layout=True,
    )
    axes = np.ravel(axes)
    for ax in axes[n_panels:]:
        ax.axis("off")
    fig.suptitle(f"{class_name}: Pairwise Distance Distributions", fontsize=13)
    return fig, axes


def _fill_hist_panel(ax: plt.Axes, anchor_count: int, uniq_d: np.ndarray):
    """Plot a single histogram panel and let the caller free uniq_d."""
    ax.hist(uniq_d, bins="fd", density=True)
    ax.set_title(f"anchors = {anchor_count}  (N={uniq_d.size})", fontsize=10)
    ax.set_xlabel("Pairwise distance")
    ax.set_ylabel("Density")
    lo, hi = uniq_d.min(), uniq_d.max()
    if hi > lo:
        pad = 0.02 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)


def plot_jaccard_curves(
    jaccard_sims: np.ndarray,
    neighbors_list: np.ndarray,
    anchor_values: list[int],
    num_jets: int,
    class_name: str,
    ncols: int = 4,
) -> plt.Figure:
    neighbor_pct = neighbors_list / num_jets
    theoretical = neighbors_list / (num_jets - 1)

    n_plots = len(anchor_values)
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 4 * nrows),
        sharex=True, sharey=True,
    )
    axes = np.ravel(axes)

    for i, (ax, a) in enumerate(zip(axes, anchor_values)):
        ax.plot(neighbor_pct, jaccard_sims[i], label="Avg Jaccard")
        ax.plot(neighbor_pct, theoretical, label="Theoretical", linestyle="--")
        ax.set_title(f"anchors = {a}")
        ax.set_ylabel("Similarity")
        ax.legend(fontsize=8)

    for ax in axes[n_plots:]:
        ax.axis("off")

    axes[-1].set_xlabel("Neighbor fraction")
    fig.suptitle(f"{class_name}: Jaccard Similarity Curves", fontsize=13)
    fig.tight_layout()
    return fig


def compute_aucs(
    jaccard_sims: np.ndarray,
    neighbors_list: np.ndarray,
    num_jets: int,
) -> np.ndarray:
    neighbor_pct = neighbors_list / num_jets
    return np.array([
        simpson(y=jaccard_sims[i], x=neighbor_pct)
        for i in range(jaccard_sims.shape[0])
    ])


def _jaccard_for_one_anchor(
    original_rank: np.ndarray,
    anchor_rank: np.ndarray,
    neighbors_list: np.ndarray,
    anchor_label: int,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Vectorised Jaccard similarity curve for a single anchor count.

    Processes rows in chunks to keep temporary memory bounded at
    ~chunk_size * 2k * 8 bytes instead of N * 2k * 8 bytes.
    """
    n = original_rank.shape[0]
    sims = np.empty(len(neighbors_list))

    for j, k in tqdm(
        enumerate(neighbors_list), total=len(neighbors_list),
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
    class_name: str,
    block_idx: int,
    project_path: str,
    anchor_values: list[int],
    num_jets: int,
    neighbors_step: int,
    output_dir: str,
    seed: int,
    save_figs: bool,
) -> dict:
    """Run the full analysis pipeline for one class type.

    Processes each anchor value one at a time, freeing its N×N distance
    matrix and argsort before moving to the next, so peak memory stays at
    roughly 4×N² regardless of how many anchor values are requested.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {class_name} (block {block_idx})")
    print(f"{'='*60}")

    original_matrix = load_distance_matrix(project_path, block_idx)
    labels = np.zeros(num_jets)
    neighbors_list = np.arange(1, num_jets - 1, neighbors_step)
    n_anchors = len(anchor_values)

    # Sort original matrix once — kept for all anchors
    original_rank = np.argsort(original_matrix, axis=1)[:, 1:]

    fig_hist, hist_axes = _create_hist_figure(n_anchors, class_name)
    jaccard_sims = np.zeros((n_anchors, len(neighbors_list)))

    print("  Computing relative representations & Jaccard curves...")
    for i, a in enumerate(anchor_values):
        rel_rep = utils.comparison_utils.relative_rep(
            original_matrix, labels=labels, num_classes=1,
            anchors_per_class=a, seed=seed,
        )

        # pdist returns upper-triangle distances directly; plot then discard
        condensed = pdist(rel_rep, metric="euclidean")
        _fill_hist_panel(hist_axes[i], a, condensed)

        # Build full square matrix only for argsort, then free it
        dmat = squareform(condensed)
        del condensed
        anchor_rank = np.argsort(dmat, axis=1)[:, 1:]
        del dmat

        jaccard_sims[i] = _jaccard_for_one_anchor(
            original_rank, anchor_rank, neighbors_list, anchor_label=a,
        )
        del anchor_rank

    del original_rank

    if save_figs:
        fig_hist.savefig(
            os.path.join(output_dir, f"{class_name}_pairwise_hist.png"),
            dpi=150, bbox_inches="tight",
        )
    plt.close(fig_hist)

    fig_jaccard = plot_jaccard_curves(
        jaccard_sims, neighbors_list, anchor_values, num_jets, class_name,
    )
    if save_figs:
        fig_jaccard.savefig(
            os.path.join(output_dir, f"{class_name}_jaccard_curves.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig_jaccard)

    aucs = compute_aucs(jaccard_sims, neighbors_list, num_jets)

    print(f"  Mean Jaccard per anchor: {np.round(np.mean(jaccard_sims, axis=1), 4)}")
    print(f"  AUCs: {np.round(aucs, 4)}")

    return {
        "class_name": class_name,
        "block_idx": block_idx,
        "anchor_values": anchor_values,
        "mean_jaccard": np.mean(jaccard_sims, axis=1),
        "aucs": aucs,
    }


def plot_jaccard_aucs_summary(
    all_results: list[dict],
    anchor_values: list[int],
) -> plt.Figure:
    """Single figure with one line per class: Jaccard AUC vs. anchor count."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for r in all_results:
        ax.plot(anchor_values, r["aucs"], marker="o", label=r["class_name"])

    ax.set_title("Jaccard AUC vs. Anchors")
    ax.set_xlabel("Anchors")
    ax.set_ylabel("AUC")
    ax.legend()
    ax.grid(True)

    return fig


def save_summary_csv(all_results: list[dict], anchor_values: list[int], output_dir: str):
    header_parts = ["class_name", "block_idx"]
    header_parts += [f"mean_jaccard_a{a}" for a in anchor_values]
    header_parts += [f"auc_a{a}" for a in anchor_values]

    rows = []
    for r in all_results:
        row = [r["class_name"], str(r["block_idx"])]
        row += [f"{v:.6f}" for v in r["mean_jaccard"]]
        row += [f"{v:.6f}" for v in r["aucs"]]
        rows.append(",".join(row))

    csv_path = os.path.join(output_dir, "intraclass_w2_summary.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(header_parts) + "\n")
        for row in rows:
            f.write(row + "\n")
    print(f"\nSummary saved to {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Intraclass W2 anchor analysis for JetClass data",
    )
    parser.add_argument(
        "--project-path", type=str, default=None,
        help="Path to the project data directory. Defaults to reading project_path.txt.",
    )
    parser.add_argument(
        "--classes", nargs="+", default=list(CLASS_CONFIGS.keys()),
        choices=list(CLASS_CONFIGS.keys()),
        help="Class types to process (default: all).",
    )
    parser.add_argument(
        "--num-jets", type=int, default=10000,
        help="Number of jets per class (default: 10000).",
    )
    parser.add_argument(
        "--anchors", type=int, nargs="+",
        default=[1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 100],
        help="Anchor counts to evaluate (default: 1 2 5 8 10 15 20 30 40 50 100).",
    )
    parser.add_argument(
        "--neighbors-step", type=int, default=100,
        help="Step size for neighbor counts in Jaccard computation (default: 100).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Directory for saving figures and CSV (default: output).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Show plots interactively instead of saving to files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.project_path is None:
        script_dir = Path(__file__).resolve().parent
        path_file = script_dir / "project_path.txt"
        with open(path_file) as f:
            project_path = f.read().strip()
    else:
        project_path = args.project_path

    anchor_values = sorted(args.anchors)
    save_figs = not args.no_save

    if save_figs:
        os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for class_name in args.classes:
        block_idx = CLASS_CONFIGS[class_name]
        result = process_class(
            class_name=class_name,
            block_idx=block_idx,
            project_path=project_path,
            anchor_values=anchor_values,
            num_jets=args.num_jets,
            neighbors_step=args.neighbors_step,
            output_dir=args.output_dir,
            seed=args.seed,
            save_figs=save_figs,
        )
        all_results.append(result)

    if len(all_results) > 1:
        fig_summary = plot_jaccard_aucs_summary(all_results, anchor_values)
        if save_figs:
            fig_summary.savefig(
                os.path.join(args.output_dir, "jaccard_aucs_summary.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig_summary)
            print(f"\nSummary plot saved to {args.output_dir}/jaccard_aucs_summary.png")

    if save_figs:
        save_summary_csv(all_results, anchor_values, args.output_dir)

    if not save_figs:
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()
