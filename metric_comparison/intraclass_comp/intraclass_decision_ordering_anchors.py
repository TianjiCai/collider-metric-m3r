"""
Intraclass decision-ordering anchor progression for JetClass metrics.

For each requested metric and jet class, this script compares the original
intraclass distance matrix with the Euclidean distance matrix induced by a
relative representation at increasing anchor counts. The reported score is a
sampled estimate of:

    decision_ordering_sim(original_distance_matrix, relrep_distance_matrix)

Use --exact for small smoke tests where evaluating all jet pairs and reference
columns is practical.
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

from utils.comparison_utils import decision_ordering_sim


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
    sample_pairs: int
    sample_references: int
    seed: int
    exact: bool
    scores: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decision-ordering progression for intraclass relative "
            "representations as anchor count increases."
        ),
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
        help="Jet classes to process (default: all 8 classes used by the intraclass notebooks).",
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
        "--sample-pairs",
        type=int,
        default=20000,
        help=(
            "Number of unordered jet pairs used to estimate decision_ordering_sim "
            "(default: 20000). Ignored by --exact."
        ),
    )
    parser.add_argument(
        "--sample-references",
        type=int,
        default=128,
        help=(
            "Number of reference columns used to estimate decision_ordering_sim "
            "(default: 128). Ignored by --exact."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for nested anchor, pair, and reference sampling (default: 42).",
    )
    parser.add_argument(
        "--pair-chunk-size",
        type=int,
        default=4096,
        help="Pair chunk size for sampled decision-ordering evaluation (default: 4096).",
    )
    parser.add_argument(
        "--ref-chunk-size",
        type=int,
        default=128,
        help="Reference chunk size for incremental relative-distance updates (default: 128).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "metric_comparison",
        help=(
            "Root used for per-metric outputs. Summaries go to "
            "<output-root>/<metric>/intraclass_comp/output/decision_ordering/."
        ),
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help=(
            "Call utils.comparison_utils.decision_ordering_sim exactly. "
            "This is intended for small --num-jets smoke tests."
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write summary CSVs without per-metric summary plots.",
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
    if args.sample_pairs <= 0:
        raise ValueError("--sample-pairs must be positive.")
    if args.sample_references <= 0:
        raise ValueError("--sample-references must be positive.")
    if args.pair_chunk_size <= 0:
        raise ValueError("--pair-chunk-size must be positive.")
    if args.ref_chunk_size <= 0:
        raise ValueError("--ref-chunk-size must be positive.")


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
    return np.asarray(matrix_full[:n, :n], dtype=np.float32)


def condensed_to_square_indices(condensed_indices: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    condensed_indices = np.asarray(condensed_indices, dtype=np.int64)
    i = n - 2 - np.floor(
        np.sqrt(-8 * condensed_indices + 4 * n * (n - 1) - 7) / 2.0 - 0.5
    ).astype(np.int64)
    j = (
        condensed_indices
        + i
        + 1
        - n * (n - 1) // 2
        + (n - i) * (n - i - 1) // 2
    )
    return i, j


def sample_upper_triangle_pairs(
    n: int,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    total_pairs = n * (n - 1) // 2
    sample_size = min(sample_size, total_pairs)
    condensed_indices = rng.choice(total_pairs, size=sample_size, replace=False)
    return condensed_to_square_indices(condensed_indices, n)


def sample_reference_indices(
    n: int,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if sample_size >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=sample_size, replace=False))


def sampled_decision_ordering_sim(
    original_ref_distances: np.ndarray,
    approx_ref_distances: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    chunk_size: int,
) -> float:
    """Sampled equivalent of decision_ordering_sim over chosen pairs/references."""
    total = 0.0
    count = 0

    for start in range(0, pair_i.size, chunk_size):
        stop = min(start + chunk_size, pair_i.size)
        i_chunk = pair_i[start:stop]
        j_chunk = pair_j[start:stop]

        original_diff = original_ref_distances[i_chunk] - original_ref_distances[j_chunk]
        approx_diff = approx_ref_distances[i_chunk] - approx_ref_distances[j_chunk]
        product = original_diff * approx_diff

        total += float(np.count_nonzero(product > 0.0))
        total += 0.5 * float(np.count_nonzero(product == 0.0))
        count += int(product.size)

    return total / count


def sampled_scores_for_anchor_sweep(
    original_matrix: np.ndarray,
    anchor_indices: np.ndarray,
    reference_indices: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    anchor_values: list[int],
    pair_chunk_size: int,
    ref_chunk_size: int,
    progress_label: str,
) -> np.ndarray:
    n_jets = original_matrix.shape[0]
    n_refs = reference_indices.size
    original_ref_distances = np.asarray(original_matrix[:, reference_indices], dtype=np.float32)

    approx_sq = np.zeros((n_jets, n_refs), dtype=np.float32)
    approx_ref_distances = np.empty_like(approx_sq)
    scores = np.empty(len(anchor_values), dtype=np.float64)

    next_anchor_pos = 0
    for i, anchor_count in enumerate(tqdm(anchor_values, desc=progress_label, leave=False)):
        while next_anchor_pos < anchor_count:
            anchor_idx = anchor_indices[next_anchor_pos]
            anchor_distances = original_matrix[:, anchor_idx]
            reference_anchor_distances = original_matrix[reference_indices, anchor_idx]

            for ref_start in range(0, n_refs, ref_chunk_size):
                ref_stop = min(ref_start + ref_chunk_size, n_refs)
                diff = (
                    anchor_distances[:, None]
                    - reference_anchor_distances[ref_start:ref_stop][None, :]
                )
                approx_sq[:, ref_start:ref_stop] += diff * diff

            next_anchor_pos += 1

        np.sqrt(approx_sq, out=approx_ref_distances)
        scores[i] = sampled_decision_ordering_sim(
            original_ref_distances=original_ref_distances,
            approx_ref_distances=approx_ref_distances,
            pair_i=pair_i,
            pair_j=pair_j,
            chunk_size=pair_chunk_size,
        )

    return scores


def exact_score_for_anchor_count(original_matrix: np.ndarray, anchor_indices: np.ndarray) -> float:
    rel_rep = np.asarray(original_matrix[:, anchor_indices], dtype=np.float64)
    relrep_distance_matrix = squareform(pdist(rel_rep, metric="euclidean"))
    return float(decision_ordering_sim(original_matrix, relrep_distance_matrix))


def exact_scores_for_anchor_sweep(
    original_matrix: np.ndarray,
    anchor_indices: np.ndarray,
    anchor_values: list[int],
    progress_label: str,
) -> np.ndarray:
    scores = np.empty(len(anchor_values), dtype=np.float64)
    for i, anchor_count in enumerate(tqdm(anchor_values, desc=progress_label, leave=False)):
        scores[i] = exact_score_for_anchor_count(
            original_matrix=original_matrix,
            anchor_indices=anchor_indices[:anchor_count],
        )
    return scores


def process_class(
    metric: str,
    class_name: str,
    block_idx: int,
    project_path: Path,
    anchor_values: list[int],
    num_jets: int,
    sample_pairs: int,
    sample_references: int,
    seed: int,
    pair_chunk_size: int,
    ref_chunk_size: int,
    exact: bool,
) -> ClassResult:
    print(f"\n{METRIC_LABELS[metric]} / {class_name}: loading block {block_idx}")
    original_matrix = load_distance_matrix(project_path, metric, block_idx, num_jets)
    n_jets = original_matrix.shape[0]

    if anchor_values[-1] > n_jets:
        raise ValueError(
            f"Maximum anchor count {anchor_values[-1]} exceeds available jets ({n_jets}) "
            f"for {metric}/{class_name}."
        )

    class_seed = seed + block_idx
    anchor_rng = np.random.default_rng(class_seed)
    anchor_indices = anchor_rng.choice(n_jets, size=anchor_values[-1], replace=False)

    progress_label = f"{metric}:{class_name}"
    if exact:
        scores = exact_scores_for_anchor_sweep(
            original_matrix=original_matrix,
            anchor_indices=anchor_indices,
            anchor_values=anchor_values,
            progress_label=progress_label,
        )
        used_pairs = n_jets * (n_jets - 1) // 2
        used_references = n_jets
    else:
        pair_rng = np.random.default_rng(class_seed + 10_000)
        ref_rng = np.random.default_rng(class_seed + 20_000)
        pair_i, pair_j = sample_upper_triangle_pairs(n_jets, sample_pairs, pair_rng)
        reference_indices = sample_reference_indices(n_jets, sample_references, ref_rng)
        scores = sampled_scores_for_anchor_sweep(
            original_matrix=original_matrix,
            anchor_indices=anchor_indices,
            reference_indices=reference_indices,
            pair_i=pair_i,
            pair_j=pair_j,
            anchor_values=anchor_values,
            pair_chunk_size=pair_chunk_size,
            ref_chunk_size=ref_chunk_size,
            progress_label=progress_label,
        )
        used_pairs = pair_i.size
        used_references = reference_indices.size

    print(f"  decision-ordering: {np.round(scores, 4)}")
    return ClassResult(
        metric=metric,
        class_name=class_name,
        block_idx=block_idx,
        num_jets=n_jets,
        sample_pairs=used_pairs,
        sample_references=used_references,
        seed=class_seed,
        exact=exact,
        scores=scores,
    )


def output_dir_for_metric(output_root: Path, metric: str) -> Path:
    return output_root / metric / "intraclass_comp" / "output" / "decision_ordering"


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
        "sample_pairs",
        "sample_references",
        "seed",
        "exact",
    ]
    fieldnames += [f"decision_ordering_a{anchor_count}" for anchor_count in anchor_values]

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "metric": result.metric,
                "class_name": result.class_name,
                "block_idx": result.block_idx,
                "num_jets": result.num_jets,
                "sample_pairs": result.sample_pairs,
                "sample_references": result.sample_references,
                "seed": result.seed,
                "exact": result.exact,
            }
            row.update(
                {
                    f"decision_ordering_a{anchor_count}": f"{score:.8f}"
                    for anchor_count, score in zip(anchor_values, result.scores)
                }
            )
            writer.writerow(row)

    print(f"Saved summary CSV: {output_path}")


def plot_metric_summary(
    results: list[ClassResult],
    anchor_values: list[int],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for result in results:
        ax.plot(anchor_values, result.scores, marker="o", linewidth=1.8, label=result.class_name)

    metric_label = METRIC_LABELS[results[0].metric]
    ax.set_title(f"{metric_label}: Decision-Ordering Similarity vs. Anchors")
    ax.set_xlabel("Anchors")
    ax.set_ylabel("Decision-ordering similarity")
    ax.set_ylim(0.45, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary plot: {output_path}")


def main() -> None:
    args = parse_args()
    anchor_values = sorted(set(args.anchors))
    validate_args(args, anchor_values)

    for metric in args.metrics:
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
                    sample_pairs=args.sample_pairs,
                    sample_references=args.sample_references,
                    seed=args.seed,
                    pair_chunk_size=args.pair_chunk_size,
                    ref_chunk_size=args.ref_chunk_size,
                    exact=args.exact,
                )
            )

        metric_output_dir = output_dir_for_metric(args.output_root, metric)
        metric_output_dir.mkdir(parents=True, exist_ok=True)
        save_summary_csv(
            results=metric_results,
            anchor_values=anchor_values,
            output_path=metric_output_dir / "intraclass_decision_ordering_summary.csv",
        )
        if not args.no_plots:
            plot_metric_summary(
                results=metric_results,
                anchor_values=anchor_values,
                output_path=metric_output_dir / "decision_ordering_summary.png",
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
