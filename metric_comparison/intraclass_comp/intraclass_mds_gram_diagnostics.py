"""
Intraclass MDS Gram-matrix diagnostics for JetClass distance matrices.

For each requested metric and jet class, this script loads the intraclass
distance block D, forms the classical MDS Gram matrix

    B = -0.5 * J D^2 J,

and eigendecomposes B. The output answers whether the finite distance matrix is
exactly Euclidean up to numerical tolerance, and summarizes the positive and
negative spectrum so the matrix complexity can be compared across metrics.

The full large-dataset run uses dense 10000 x 10000 eigendecompositions. Use
--sample-jets for quick exploratory runs.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "collider-metric-comparison-mpl"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.jetclass_utils import (
    canonical_class_names,
    class_labels_path,
    dataset_size_to_count,
    metric_block_path,
    ordered_class_names_from_labels,
)


METRIC_LABELS = {
    "w2": "W2",
    "ps": "PS",
    "sw": "SW",
}

VARIANCE_LEVELS = (0.50, 0.90, 0.95, 0.99)


@dataclass
class GramResult:
    metric: str
    class_name: str
    block_idx: int
    dataset_size: str
    matrix_path: Path
    requested_num_jets: int
    source_num_jets: int
    num_jets: int
    sample_jets: int | None
    sample_seed: int
    eigenvalue_tolerance: float
    max_abs_diagonal: float
    max_abs_asymmetry: float
    min_eigenvalue: float
    max_eigenvalue: float
    trace: float
    positive_eigenvalue_count: int
    negative_eigenvalue_count: int
    zero_eigenvalue_count: int
    is_euclidean_by_gram_psd: bool
    exact_embedding_dimension: int | None
    positive_sum: float
    negative_sum_abs: float
    negative_sum_abs_fraction: float
    negative_frobenius_fraction: float
    participation_effective_rank: float
    entropy_effective_rank: float
    dim_50: int
    dim_90: int
    dim_95: int
    dim_99: int
    top_1_fraction: float
    top_5_fraction: float
    top_10_fraction: float
    top_50_fraction: float
    top_100_fraction: float
    eigenvalues_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute classical MDS Gram-matrix eigenvalue diagnostics for "
            "intraclass JetClass distance matrices."
        ),
    )
    parser.add_argument(
        "--project-path",
        type=Path,
        default=REPO_ROOT,
        help=f"Repository root containing data/dist_mat (default: {REPO_ROOT}).",
    )
    parser.add_argument(
        "--dataset-size",
        default="large",
        choices=["mini", "medium", "large", "full"],
        help="Dataset size tag used to find data/dist_mat/<metric>/<size>_<metric>.",
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
        default=["all"],
        help=(
            "Jet classes to process. Use class names, label indices, or 'all' "
            "(default: all classes from class_labels.npz)."
        ),
    )
    parser.add_argument(
        "--num-jets",
        type=int,
        default=None,
        help=(
            "Number of leading jets per class to consider before optional "
            "sampling (default: dataset-size count, e.g. 10000 for large)."
        ),
    )
    parser.add_argument(
        "--sample-jets",
        type=int,
        default=None,
        help=(
            "Randomly sample this many jets from the requested leading jets. "
            "Useful for fast exploratory runs; omitted means use all requested jets."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for --sample-jets (default: 42).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-10,
        help="Absolute eigenvalue tolerance floor for PSD classification.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-8,
        help="Relative eigenvalue tolerance, scaled by max(abs(eigenvalue)).",
    )
    parser.add_argument(
        "--symmetry-chunk-size",
        type=int,
        default=512,
        help="Chunk size for exact asymmetry checks and optional symmetrization.",
    )
    parser.add_argument(
        "--skip-symmetry-check",
        action="store_true",
        help="Skip the exact max |D - D.T| diagnostic.",
    )
    parser.add_argument(
        "--symmetrize",
        action="store_true",
        help=(
            "Replace D with 0.5 * (D + D.T) before forming B. This is normally "
            "unneeded for the generated intraclass blocks."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "metric_comparison",
        help=(
            "Root used for outputs. Per-metric CSVs go to "
            "<output-root>/<metric>/intraclass_comp/output/mds_gram/."
        ),
    )
    parser.add_argument(
        "--save-eigenvalues",
        action="store_true",
        help="Save each full eigenvalue spectrum as a .npy file.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write CSV summaries without PNG summary plots.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, requested_num_jets: int) -> None:
    if requested_num_jets < 2:
        raise ValueError("--num-jets must be at least 2.")
    if args.sample_jets is not None:
        if args.sample_jets < 2:
            raise ValueError("--sample-jets must be at least 2.")
        if args.sample_jets > requested_num_jets:
            raise ValueError(
                f"--sample-jets={args.sample_jets} exceeds --num-jets={requested_num_jets}."
            )
    if args.atol < 0:
        raise ValueError("--atol must be nonnegative.")
    if args.rtol < 0:
        raise ValueError("--rtol must be nonnegative.")
    if args.symmetry_chunk_size <= 0:
        raise ValueError("--symmetry-chunk-size must be positive.")


def available_class_names(project_path: Path, dataset_size: str) -> list[str]:
    labels_path = class_labels_path(project_path, dataset_size)
    if labels_path.exists():
        return ordered_class_names_from_labels(labels_path)
    return canonical_class_names()


def resolve_requested_classes(
    requested_classes: list[str],
    ordered_names: list[str],
) -> list[tuple[str, int]]:
    if len(requested_classes) == 1 and requested_classes[0].lower() == "all":
        return [(class_name, idx) for idx, class_name in enumerate(ordered_names)]

    lower_to_name = {class_name.lower(): class_name for class_name in ordered_names}
    aliases = {
        "qcd": "QCD",
        "ttbar": "TTBar",
        "ttbarlep": "TTBarLep",
    }

    resolved: list[tuple[str, int]] = []
    seen: set[int] = set()
    for token in requested_classes:
        if token.lower() == "all":
            for idx, class_name in enumerate(ordered_names):
                if idx not in seen:
                    resolved.append((class_name, idx))
                    seen.add(idx)
            continue

        if token.isdigit():
            idx = int(token)
            if not 0 <= idx < len(ordered_names):
                raise ValueError(
                    f"Class index {idx} is out of range for labels 0..{len(ordered_names) - 1}."
                )
            class_name = ordered_names[idx]
        else:
            canonical_token = aliases.get(token.lower(), token)
            class_name = lower_to_name.get(canonical_token.lower())
            if class_name is None:
                raise ValueError(
                    f"Unknown class {token!r}. Available classes: {', '.join(ordered_names)}."
                )
            idx = ordered_names.index(class_name)

        if idx not in seen:
            resolved.append((class_name, idx))
            seen.add(idx)

    return resolved


def selected_indices(
    source_num_jets: int,
    requested_num_jets: int,
    sample_jets: int | None,
    seed: int,
) -> tuple[np.ndarray, int]:
    n_available = min(source_num_jets, requested_num_jets)
    if sample_jets is None:
        return np.arange(n_available, dtype=np.int64), n_available

    if sample_jets > n_available:
        raise ValueError(
            f"--sample-jets={sample_jets} exceeds available jets ({n_available})."
        )

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_available, size=sample_jets, replace=False))
    return indices.astype(np.int64), n_available


def load_intraclass_distance_matrix(
    matrix_path: Path,
    requested_num_jets: int,
    sample_jets: int | None,
    seed: int,
) -> tuple[np.ndarray, int, int]:
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing distance matrix: {matrix_path}")

    matrix_full = np.load(matrix_path, mmap_mode="r")
    if matrix_full.ndim != 2 or matrix_full.shape[0] != matrix_full.shape[1]:
        raise ValueError(f"Expected a square matrix at {matrix_path}, got {matrix_full.shape}.")

    indices, source_n = selected_indices(
        source_num_jets=matrix_full.shape[0],
        requested_num_jets=requested_num_jets,
        sample_jets=sample_jets,
        seed=seed,
    )

    if indices.size == source_n and np.array_equal(indices, np.arange(source_n)):
        matrix = np.array(matrix_full[:source_n, :source_n], dtype=np.float64, copy=True)
    else:
        matrix = np.array(matrix_full[np.ix_(indices, indices)], dtype=np.float64, copy=True)

    return matrix, matrix_full.shape[0], source_n


def max_abs_asymmetry(matrix: np.ndarray, chunk_size: int) -> float:
    max_value = 0.0
    n = matrix.shape[0]
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        diff = matrix[start:stop, :] - matrix[:, start:stop].T
        max_value = max(max_value, float(np.max(np.abs(diff))))
    return max_value


def symmetrize_in_place(matrix: np.ndarray, block_size: int) -> None:
    n = matrix.shape[0]
    for i_start in range(0, n, block_size):
        i_stop = min(i_start + block_size, n)

        diag_block = matrix[i_start:i_stop, i_start:i_stop]
        diag_block[:] = 0.5 * (diag_block + diag_block.T)

        for j_start in range(i_stop, n, block_size):
            j_stop = min(j_start + block_size, n)
            upper = matrix[i_start:i_stop, j_start:j_stop]
            lower_t = matrix[j_start:j_stop, i_start:i_stop].T
            averaged = 0.5 * (upper.copy() + lower_t.copy())
            upper[:] = averaged
            matrix[j_start:j_stop, i_start:i_stop] = averaged.T


def distance_matrix_to_mds_gram_in_place(matrix: np.ndarray) -> np.ndarray:
    np.fill_diagonal(matrix, 0.0)
    np.square(matrix, out=matrix)

    row_means = matrix.mean(axis=1)
    grand_mean = float(row_means.mean())

    matrix -= row_means[:, None]
    matrix -= row_means[None, :]
    matrix += grand_mean
    matrix *= -0.5
    return matrix


def eigvalsh(gram_matrix: np.ndarray) -> np.ndarray:
    try:
        import scipy.linalg

        return scipy.linalg.eigvalsh(
            gram_matrix,
            overwrite_a=True,
            check_finite=False,
            driver="evr",
        )
    except Exception:
        return np.linalg.eigvalsh(gram_matrix)


def dimension_for_fraction(positive_eigenvalues_desc: np.ndarray, fraction: float) -> int:
    if positive_eigenvalues_desc.size == 0:
        return 0

    total = float(np.sum(positive_eigenvalues_desc))
    if total <= 0.0:
        return 0

    cumulative = np.cumsum(positive_eigenvalues_desc)
    return int(np.searchsorted(cumulative, fraction * total, side="left") + 1)


def top_fraction(positive_eigenvalues_desc: np.ndarray, k: int) -> float:
    if positive_eigenvalues_desc.size == 0:
        return 0.0
    total = float(np.sum(positive_eigenvalues_desc))
    if total <= 0.0:
        return 0.0
    return float(np.sum(positive_eigenvalues_desc[:k]) / total)


def effective_ranks(positive_eigenvalues_desc: np.ndarray) -> tuple[float, float]:
    if positive_eigenvalues_desc.size == 0:
        return 0.0, 0.0

    positive_sum = float(np.sum(positive_eigenvalues_desc))
    if positive_sum <= 0.0:
        return 0.0, 0.0

    participation = positive_sum**2 / float(np.sum(positive_eigenvalues_desc**2))
    probabilities = positive_eigenvalues_desc / positive_sum
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(participation), float(math.exp(entropy))


def summarize_eigenvalues(
    eigenvalues: np.ndarray,
    atol: float,
    rtol: float,
) -> tuple[float, dict[str, object]]:
    max_abs_eigenvalue = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    tolerance = max(atol, rtol * max_abs_eigenvalue)

    positive = eigenvalues[eigenvalues > tolerance]
    negative = eigenvalues[eigenvalues < -tolerance]
    positive_desc = positive[::-1]

    positive_sum = float(np.sum(positive))
    negative_sum_abs = float(-np.sum(negative))
    total_signed_mass = positive_sum + negative_sum_abs
    negative_sum_abs_fraction = (
        negative_sum_abs / total_signed_mass if total_signed_mass > 0.0 else 0.0
    )

    total_fro_sq = float(np.sum(eigenvalues**2))
    negative_fro_sq = float(np.sum(negative**2))
    negative_frobenius_fraction = (
        math.sqrt(negative_fro_sq / total_fro_sq) if total_fro_sq > 0.0 else 0.0
    )

    participation_rank, entropy_rank = effective_ranks(positive_desc)
    dims = {
        int(round(level * 100)): dimension_for_fraction(positive_desc, level)
        for level in VARIANCE_LEVELS
    }

    positive_count = int(positive.size)
    negative_count = int(negative.size)
    summary: dict[str, object] = {
        "min_eigenvalue": float(eigenvalues[0]),
        "max_eigenvalue": float(eigenvalues[-1]),
        "trace": float(np.sum(eigenvalues)),
        "positive_eigenvalue_count": positive_count,
        "negative_eigenvalue_count": negative_count,
        "zero_eigenvalue_count": int(eigenvalues.size - positive_count - negative_count),
        "is_euclidean_by_gram_psd": negative_count == 0,
        "exact_embedding_dimension": positive_count if negative_count == 0 else None,
        "positive_sum": positive_sum,
        "negative_sum_abs": negative_sum_abs,
        "negative_sum_abs_fraction": float(negative_sum_abs_fraction),
        "negative_frobenius_fraction": float(negative_frobenius_fraction),
        "participation_effective_rank": float(participation_rank),
        "entropy_effective_rank": float(entropy_rank),
        "dim_50": dims[50],
        "dim_90": dims[90],
        "dim_95": dims[95],
        "dim_99": dims[99],
        "top_1_fraction": top_fraction(positive_desc, 1),
        "top_5_fraction": top_fraction(positive_desc, 5),
        "top_10_fraction": top_fraction(positive_desc, 10),
        "top_50_fraction": top_fraction(positive_desc, 50),
        "top_100_fraction": top_fraction(positive_desc, 100),
    }
    return tolerance, summary


def output_dir_for_metric(output_root: Path, metric: str) -> Path:
    return output_root / metric / "intraclass_comp" / "output" / "mds_gram"


def combined_output_dir(output_root: Path) -> Path:
    return output_root / "intraclass_comp" / "output" / "mds_gram"


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def process_class(
    metric: str,
    class_name: str,
    block_idx: int,
    dataset_size: str,
    project_path: Path,
    requested_num_jets: int,
    sample_jets: int | None,
    sample_seed: int,
    atol: float,
    rtol: float,
    symmetry_chunk_size: int,
    skip_symmetry_check: bool,
    symmetrize: bool,
    eigenvalue_dir: Path | None,
) -> GramResult:
    matrix_path = metric_block_path(
        project_root=project_path,
        metric=metric,
        dataset_size=dataset_size,
        class_i=block_idx,
        class_j=block_idx,
    )
    sample_label = f", sample={sample_jets}" if sample_jets is not None else ""
    print(f"\n{METRIC_LABELS[metric]} / {class_name}: loading block {block_idx}{sample_label}")

    distance_matrix, source_num_jets, _used_source_n = load_intraclass_distance_matrix(
        matrix_path=matrix_path,
        requested_num_jets=requested_num_jets,
        sample_jets=sample_jets,
        seed=sample_seed,
    )

    max_diagonal = float(np.max(np.abs(np.diag(distance_matrix))))
    if skip_symmetry_check:
        asymmetry = float("nan")
    else:
        asymmetry = max_abs_asymmetry(distance_matrix, symmetry_chunk_size)

    if symmetrize:
        symmetrize_in_place(distance_matrix, symmetry_chunk_size)

    gram_matrix = distance_matrix_to_mds_gram_in_place(distance_matrix)
    print(f"  eigendecomposing B ({gram_matrix.shape[0]} x {gram_matrix.shape[1]})")
    eigenvalues = eigvalsh(gram_matrix)

    eigenvalues_path: Path | None = None
    if eigenvalue_dir is not None:
        eigenvalue_dir.mkdir(parents=True, exist_ok=True)
        eigenvalues_path = eigenvalue_dir / f"{safe_name(class_name)}_eigenvalues.npy"
        np.save(eigenvalues_path, eigenvalues)

    tolerance, summary = summarize_eigenvalues(
        eigenvalues=eigenvalues,
        atol=atol,
        rtol=rtol,
    )
    print(
        "  min eig = "
        f"{summary['min_eigenvalue']:.6g}, "
        f"neg count = {summary['negative_eigenvalue_count']}, "
        f"d95 = {summary['dim_95']}"
    )

    return GramResult(
        metric=metric,
        class_name=class_name,
        block_idx=block_idx,
        dataset_size=dataset_size,
        matrix_path=matrix_path,
        requested_num_jets=requested_num_jets,
        source_num_jets=source_num_jets,
        num_jets=gram_matrix.shape[0],
        sample_jets=sample_jets,
        sample_seed=sample_seed,
        eigenvalue_tolerance=tolerance,
        max_abs_diagonal=max_diagonal,
        max_abs_asymmetry=asymmetry,
        eigenvalues_path=eigenvalues_path,
        **summary,
    )


def result_to_row(result: GramResult) -> dict[str, object]:
    row = {
        "metric": result.metric,
        "class_name": result.class_name,
        "block_idx": result.block_idx,
        "dataset_size": result.dataset_size,
        "matrix_path": str(result.matrix_path),
        "requested_num_jets": result.requested_num_jets,
        "source_num_jets": result.source_num_jets,
        "num_jets": result.num_jets,
        "sample_jets": "" if result.sample_jets is None else result.sample_jets,
        "sample_seed": result.sample_seed,
        "eigenvalue_tolerance": result.eigenvalue_tolerance,
        "max_abs_diagonal": result.max_abs_diagonal,
        "max_abs_asymmetry": result.max_abs_asymmetry,
        "min_eigenvalue": result.min_eigenvalue,
        "max_eigenvalue": result.max_eigenvalue,
        "trace": result.trace,
        "positive_eigenvalue_count": result.positive_eigenvalue_count,
        "negative_eigenvalue_count": result.negative_eigenvalue_count,
        "zero_eigenvalue_count": result.zero_eigenvalue_count,
        "is_euclidean_by_gram_psd": result.is_euclidean_by_gram_psd,
        "exact_embedding_dimension": (
            "" if result.exact_embedding_dimension is None else result.exact_embedding_dimension
        ),
        "positive_sum": result.positive_sum,
        "negative_sum_abs": result.negative_sum_abs,
        "negative_sum_abs_fraction": result.negative_sum_abs_fraction,
        "negative_frobenius_fraction": result.negative_frobenius_fraction,
        "participation_effective_rank": result.participation_effective_rank,
        "entropy_effective_rank": result.entropy_effective_rank,
        "dim_50": result.dim_50,
        "dim_90": result.dim_90,
        "dim_95": result.dim_95,
        "dim_99": result.dim_99,
        "top_1_fraction": result.top_1_fraction,
        "top_5_fraction": result.top_5_fraction,
        "top_10_fraction": result.top_10_fraction,
        "top_50_fraction": result.top_50_fraction,
        "top_100_fraction": result.top_100_fraction,
        "eigenvalues_path": "" if result.eigenvalues_path is None else str(result.eigenvalues_path),
    }

    for key, value in list(row.items()):
        if isinstance(value, float):
            row[key] = f"{value:.12g}"
    return row


def save_summary_csv(results: list[GramResult], output_path: Path) -> None:
    if not results:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(result_to_row(results[0]).keys())
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result_to_row(result))

    print(f"Saved summary CSV: {output_path}")


def plot_metric_summary(results: list[GramResult], output_path: Path) -> None:
    if not results:
        return

    class_names = [result.class_name for result in results]
    x = np.arange(len(results))
    colors = [
        "#c43c39" if result.negative_eigenvalue_count > 0 else "#3478b8"
        for result in results
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), constrained_layout=True)

    axes[0].bar(x, [result.min_eigenvalue for result in results], color=colors)
    axes[0].axhline(0.0, color="black", linewidth=0.9)
    axes[0].set_ylabel("Minimum eigenvalue")

    axes[1].bar(
        x,
        [result.negative_frobenius_fraction for result in results],
        color=colors,
    )
    axes[1].set_ylabel("Negative Frobenius fraction")
    axes[1].set_ylim(bottom=0.0)

    axes[2].plot(
        x,
        [result.dim_95 for result in results],
        color="#2a7f62",
        marker="o",
        linewidth=1.8,
    )
    axes[2].set_ylabel("Positive dims for 95%")
    axes[2].set_xlabel("Class")

    for ax in axes:
        ax.set_xticks(x, class_names, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.25)

    metric_label = METRIC_LABELS[results[0].metric]
    sample_suffix = (
        f" (sample {results[0].sample_jets})"
        if results[0].sample_jets is not None
        else ""
    )
    fig.suptitle(f"{metric_label}: MDS Gram Diagnostics{sample_suffix}", fontsize=14)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary plot: {output_path}")


def main() -> None:
    args = parse_args()
    requested_num_jets = (
        dataset_size_to_count(args.dataset_size) if args.num_jets is None else args.num_jets
    )
    validate_args(args, requested_num_jets)

    ordered_names = available_class_names(args.project_path, args.dataset_size)
    selected_classes = resolve_requested_classes(args.classes, ordered_names)
    print(
        "Processing classes: "
        + ", ".join(f"{name}[{idx}]" for name, idx in selected_classes)
    )

    all_results: list[GramResult] = []
    for metric_offset, metric in enumerate(args.metrics):
        metric_output_dir = output_dir_for_metric(args.output_root, metric)
        metric_output_dir.mkdir(parents=True, exist_ok=True)
        eigenvalue_dir = metric_output_dir / "eigenvalues" if args.save_eigenvalues else None

        metric_results: list[GramResult] = []
        for class_name, block_idx in selected_classes:
            sample_seed = args.seed + 10_000 * metric_offset + block_idx
            result = process_class(
                metric=metric,
                class_name=class_name,
                block_idx=block_idx,
                dataset_size=args.dataset_size,
                project_path=args.project_path,
                requested_num_jets=requested_num_jets,
                sample_jets=args.sample_jets,
                sample_seed=sample_seed,
                atol=args.atol,
                rtol=args.rtol,
                symmetry_chunk_size=args.symmetry_chunk_size,
                skip_symmetry_check=args.skip_symmetry_check,
                symmetrize=args.symmetrize,
                eigenvalue_dir=eigenvalue_dir,
            )
            metric_results.append(result)
            all_results.append(result)

        save_summary_csv(
            metric_results,
            metric_output_dir / "intraclass_mds_gram_summary.csv",
        )
        if not args.no_plots:
            plot_metric_summary(
                metric_results,
                metric_output_dir / "mds_gram_summary.png",
            )

    combined_dir = combined_output_dir(args.output_root)
    save_summary_csv(
        all_results,
        combined_dir / "intraclass_mds_gram_summary_all_metrics.csv",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
