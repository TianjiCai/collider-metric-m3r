from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLOT_CACHE_DIR = SCRIPT_DIR / ".mpl_cache"
PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE_DIR))

import matplotlib.pyplot as plt

from utils import jetclass_utils


METRICS = ("w2", "sw", "ps")
METRIC_LABELS = {
    "w2": r"$W_2$",
    "sw": r"SW$_2$",
    "ps": "PS",
}
METRIC_PAIRS = (("w2", "sw"), ("w2", "ps"), ("sw", "ps"))
SUMMARY_K_VALUES = (1, 2, 5, 10, 20)
ALIGNMENT_TOP_K = 10
TARGET_CLASSES = (
    ("qcd", "QCD", 0),
    ("ttbar", "TTBar", 7),
)


@dataclass
class PCAResult:
    z: np.ndarray
    cov: np.ndarray
    components: np.ndarray
    scores: np.ndarray
    eigenvalues: np.ndarray
    explained_ratio: np.ndarray
    cumulative_explained: np.ndarray
    effective_rank: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze intraclass QCD and TTBar large-dataset relative "
            "representations for W2, SW, and PS."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "output" / "qcd_ttbar_separate_large_20anchors",
        help="Directory for rel-rep arrays, plots, and summaries.",
    )
    parser.add_argument(
        "--anchor-count",
        type=int,
        default=20,
        help="Number of same-class anchor events per jet type.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Anchor sampling seed. The same local anchor indices are reused for QCD and TTBar.",
    )
    parser.add_argument(
        "--pca-sample",
        type=int,
        default=4000,
        help="Maximum events to draw in each PCA score scatter plot.",
    )
    return parser.parse_args()


def sample_anchor_indices(events_per_class: int, anchor_count: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.choice(events_per_class, size=anchor_count, replace=False).astype(int)


def metric_block_path(metric: str, class_idx: int) -> Path:
    return jetclass_utils.metric_block_path(REPO_ROOT, metric, "large", class_idx, class_idx)


def build_intraclass_rel_rep(
    metric: str,
    class_idx: int,
    anchor_indices: np.ndarray,
    events_per_class: int,
) -> np.ndarray:
    block = np.load(metric_block_path(metric, class_idx), mmap_mode="r")
    expected_shape = (events_per_class, events_per_class)
    if block.shape != expected_shape:
        raise ValueError(
            f"{metric} class {class_idx} self block has shape {block.shape}, "
            f"expected {expected_shape}."
        )
    return np.asarray(block[:, anchor_indices]).astype(np.float32, copy=False)


def zscore_columns(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, ddof=1, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return (x - mean) / std


def safe_explained_fractions(eigenvalues: np.ndarray) -> np.ndarray:
    vals = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    total = vals.sum()
    if total <= 0.0:
        return np.zeros_like(vals)
    return vals / total


def effective_rank(eigenvalues: np.ndarray) -> float:
    fractions = safe_explained_fractions(eigenvalues)
    positive = fractions[fractions > 0.0]
    if positive.size == 0:
        return 0.0
    return float(np.exp(-np.sum(positive * np.log(positive))))


def covariance_and_spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = x.astype(np.float64, copy=False) - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / (centered.shape[0] - 1)
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]
    return cov, eigenvalues


def pca_from_standardized(x: np.ndarray) -> PCAResult:
    z = zscore_columns(x.astype(np.float64, copy=False))
    _, singular_values, components = np.linalg.svd(z, full_matrices=False)
    scores = z @ components.T
    explained_variance = singular_values**2 / (z.shape[0] - 1)
    explained_ratio = safe_explained_fractions(explained_variance)
    cumulative_explained = np.cumsum(explained_ratio)
    cov = z.T @ z / (z.shape[0] - 1)
    return PCAResult(
        z=z,
        cov=cov,
        components=components,
        scores=scores,
        eigenvalues=explained_variance,
        explained_ratio=explained_ratio,
        cumulative_explained=cumulative_explained,
        effective_rank=effective_rank(explained_variance),
    )


def align_component_signs(
    components_by_metric: dict[str, np.ndarray],
    reference_metric: str = "w2",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    aligned = {metric: comps.copy() for metric, comps in components_by_metric.items()}
    signs = {metric: np.ones(comps.shape[0], dtype=float) for metric, comps in aligned.items()}
    reference = aligned[reference_metric]
    for metric, comps in aligned.items():
        if metric == reference_metric:
            continue
        for pc_idx in range(reference.shape[0]):
            if np.dot(reference[pc_idx], comps[pc_idx]) < 0:
                comps[pc_idx] *= -1.0
                signs[metric][pc_idx] = -1.0
    return aligned, signs


def make_anchor_labels(class_name: str, anchor_count: int) -> list[str]:
    return [f"{class_name} {idx + 1}" for idx in range(anchor_count)]


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def two_sample_ks(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a = np.sort(values_a)
    b = np.sort(values_b)
    merged = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, merged, side="right") / a.size
    cdf_b = np.searchsorted(b, merged, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def ks_conclusion(value: float) -> str:
    if value < 0.1:
        return "similar"
    if value < 0.2:
        return "modest difference"
    return "visibly different"


def save_coordinate_summaries(
    output_dir: Path,
    rel_reps: dict[str, np.ndarray],
    anchor_labels: list[str],
) -> list[dict[str, object]]:
    summary_path = output_dir / "coordinate_summary.csv"
    ks_path = output_dir / "coordinate_standardized_ks.csv"
    distribution_summary_path = output_dir / "coordinate_distribution_summary.csv"

    with summary_path.open("w") as handle:
        handle.write(
            "metric,coordinate,anchor_label,mean,std,p05,p25,p50,p75,p95\n"
        )
        for metric, rel_rep in rel_reps.items():
            quantiles = np.percentile(rel_rep, [5, 25, 50, 75, 95], axis=0)
            means = rel_rep.mean(axis=0)
            stds = rel_rep.std(axis=0, ddof=1)
            for coord, label in enumerate(anchor_labels):
                row = [
                    metric,
                    str(coord),
                    label,
                    f"{means[coord]:.10g}",
                    f"{stds[coord]:.10g}",
                    *[f"{quantiles[row_idx, coord]:.10g}" for row_idx in range(5)],
                ]
                handle.write(",".join(row) + "\n")

    standardized = {metric: zscore_columns(rel_rep) for metric, rel_rep in rel_reps.items()}
    ks_by_pair: dict[tuple[str, str], list[float]] = {
        pair: [] for pair in METRIC_PAIRS
    }
    with ks_path.open("w") as handle:
        handle.write("coordinate,anchor_label,metric_a,metric_b,ks_statistic\n")
        for coord, label in enumerate(anchor_labels):
            for metric_a, metric_b in METRIC_PAIRS:
                ks_stat = two_sample_ks(
                    standardized[metric_a][:, coord],
                    standardized[metric_b][:, coord],
                )
                ks_by_pair[(metric_a, metric_b)].append(ks_stat)
                handle.write(
                    f"{coord},{label},{metric_a},{metric_b},{ks_stat:.10g}\n"
                )

    summaries: list[dict[str, object]] = []
    with distribution_summary_path.open("w") as handle:
        handle.write(
            "metric_a,metric_b,mean_ks,median_ks,max_ks,mean_conclusion,max_conclusion\n"
        )
        for metric_a, metric_b in METRIC_PAIRS:
            values = np.asarray(ks_by_pair[(metric_a, metric_b)], dtype=float)
            mean_ks = float(np.mean(values))
            median_ks = float(np.median(values))
            max_ks = float(np.max(values))
            mean_label = ks_conclusion(mean_ks)
            max_label = ks_conclusion(max_ks)
            summaries.append(
                {
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "mean_ks": mean_ks,
                    "median_ks": median_ks,
                    "max_ks": max_ks,
                    "mean_conclusion": mean_label,
                    "max_conclusion": max_label,
                }
            )
            handle.write(
                f"{metric_a},{metric_b},{mean_ks:.10g},{median_ks:.10g},"
                f"{max_ks:.10g},{mean_label},{max_label}\n"
            )

    return summaries


def plot_coordinate_quantiles(
    output_dir: Path,
    class_name: str,
    rel_reps: dict[str, np.ndarray],
    anchor_count: int,
) -> None:
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(11, 9), sharex=True)
    x = np.arange(anchor_count)

    for ax, metric in zip(axes, METRICS):
        rel_rep = rel_reps[metric]
        q05, q25, q50, q75, q95 = np.percentile(rel_rep, [5, 25, 50, 75, 95], axis=0)
        ax.fill_between(x, q05, q95, color="tab:blue", alpha=0.12, label="5-95%")
        ax.fill_between(x, q25, q75, color="tab:blue", alpha=0.28, label="25-75%")
        ax.plot(x, q50, color="tab:blue", linewidth=1.4, marker="o", markersize=3, label="Median")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(alpha=0.2)

    axes[0].legend(loc="upper right", frameon=False)
    axes[-1].set_xlabel(f"{class_name} anchor coordinate")
    axes[-1].set_xticks([0, anchor_count - 1], ["1", str(anchor_count)])
    fig.suptitle(f"Coordinate-wise Rel-Rep Distributions, {class_name}")
    fig.tight_layout()
    fig.savefig(output_dir / "coordinate_distribution_quantiles.png", dpi=300)
    plt.close(fig)


def plot_standardized_distribution_overlay(
    output_dir: Path,
    class_name: str,
    rel_reps: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-4, 4, 120)
    for metric in METRICS:
        z = zscore_columns(rel_reps[metric]).ravel()
        z = z[np.isfinite(z)]
        ax.hist(
            z,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=METRIC_LABELS[metric],
        )

    ax.set_xlabel("Coordinate value after per-coordinate standardization")
    ax.set_ylabel("Density")
    ax.set_title(f"Pooled Coordinate Distribution Shapes, {class_name}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "coordinate_distribution_standardized_overlay.png", dpi=300)
    plt.close(fig)


def plot_covariances_and_spectra(
    output_dir: Path,
    class_name: str,
    covariances: dict[str, np.ndarray],
    eigenvalues: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4.3), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        vmax = np.percentile(np.abs(covariances[metric]), 99)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = np.max(np.abs(covariances[metric]))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        im = ax.imshow(covariances[metric], cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Anchor coordinate")
        ax.set_ylabel("Anchor coordinate")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Covariance")

    fig.suptitle(f"Rel-Rep Covariance Matrices, {class_name}")
    fig.savefig(output_dir / "covariance_matrices.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4.3), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        diag = np.sqrt(np.diag(covariances[metric]))
        denom = diag[:, None] * diag[None, :]
        corr = np.divide(
            covariances[metric],
            denom,
            out=np.zeros_like(covariances[metric]),
            where=denom > 0,
        )
        np.save(output_dir / f"{metric}_correlation.npy", corr)
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Anchor coordinate")
        ax.set_ylabel("Anchor coordinate")

    fig.colorbar(im, ax=axes, shrink=0.85, label="Correlation")
    fig.suptitle(f"Rel-Rep Correlation Matrices, {class_name}")
    fig.savefig(output_dir / "correlation_matrices.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in METRICS:
        vals = np.clip(eigenvalues[metric], 0.0, None)
        vals = np.where(vals > 0.0, vals, np.nan)
        ax.plot(
            np.arange(1, len(vals) + 1),
            vals,
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=METRIC_LABELS[metric],
        )
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Covariance eigenvalue")
    ax.set_yscale("log")
    ax.set_title(f"Covariance Spectrum, {class_name}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "covariance_spectrum.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in METRICS:
        vals = eigenvalues[metric]
        explained = safe_explained_fractions(vals)
        explained = np.where(explained > 0.0, explained, np.nan)
        ax.plot(
            np.arange(1, len(vals) + 1),
            explained,
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=METRIC_LABELS[metric],
        )
    ax.set_xlabel("Principal direction rank")
    ax.set_ylabel("Explained covariance fraction")
    ax.set_yscale("log")
    ax.set_title(f"Normalized Covariance Spectrum, {class_name}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "covariance_spectrum_normalized.png", dpi=300)
    plt.close(fig)


def save_covariance_spectrum_csv(
    output_dir: Path,
    eigenvalues: dict[str, np.ndarray],
) -> None:
    path = output_dir / "covariance_spectrum.csv"
    with path.open("w") as handle:
        handle.write(
            "metric,rank,eigenvalue,explained_fraction,"
            "cumulative_explained_fraction,effective_rank\n"
        )
        for metric, vals in eigenvalues.items():
            fractions = safe_explained_fractions(vals)
            cumulative = np.cumsum(fractions)
            eff_rank = effective_rank(vals)
            for rank, (value, frac, cum) in enumerate(zip(vals, fractions, cumulative), start=1):
                handle.write(
                    f"{metric},{rank},{value:.10g},{frac:.10g},"
                    f"{cum:.10g},{eff_rank:.10g}\n"
                )


def correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    diag = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = diag[:, None] * diag[None, :]
    return np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)


def average_abs_offdiag(matrix: np.ndarray) -> float:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(np.mean(np.abs(matrix[mask])))


def frobenius_cosine(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    flat_a = matrix_a.ravel()
    flat_b = matrix_b.ravel()
    denom = np.linalg.norm(flat_a) * np.linalg.norm(flat_b)
    if denom <= 0.0:
        return 0.0
    return float(np.dot(flat_a, flat_b) / denom)


def save_covariance_structure_summary(
    output_dir: Path,
    eigenvalues: dict[str, np.ndarray],
    correlations: dict[str, np.ndarray],
) -> dict[str, object]:
    path = output_dir / "covariance_structure_summary.csv"
    metric_summaries: dict[str, dict[str, float]] = {}
    pair_summaries: dict[str, dict[str, float]] = {}

    with path.open("w") as handle:
        handle.write(
            "row_type,metric,metric_a,metric_b,top1_cumulative,top2_cumulative,"
            "top5_cumulative,effective_rank,avg_abs_offdiag_corr,"
            "frobenius_cosine\n"
        )
        for metric in METRICS:
            fractions = safe_explained_fractions(eigenvalues[metric])
            cumulative = np.cumsum(fractions)
            summary = {
                "top1_cumulative": float(cumulative[min(0, len(cumulative) - 1)]),
                "top2_cumulative": float(cumulative[min(1, len(cumulative) - 1)]),
                "top5_cumulative": float(cumulative[min(4, len(cumulative) - 1)]),
                "effective_rank": effective_rank(eigenvalues[metric]),
                "avg_abs_offdiag_corr": average_abs_offdiag(correlations[metric]),
            }
            metric_summaries[metric] = summary
            handle.write(
                "metric,"
                f"{metric},,,{summary['top1_cumulative']:.10g},"
                f"{summary['top2_cumulative']:.10g},"
                f"{summary['top5_cumulative']:.10g},"
                f"{summary['effective_rank']:.10g},"
                f"{summary['avg_abs_offdiag_corr']:.10g},\n"
            )

        for metric_a, metric_b in METRIC_PAIRS:
            pair_key = f"{metric_a}-{metric_b}"
            cosine = frobenius_cosine(correlations[metric_a], correlations[metric_b])
            pair_summaries[pair_key] = {"frobenius_cosine": cosine}
            handle.write(
                "pair,,"
                f"{metric_a},{metric_b},,,,,,{cosine:.10g}\n"
            )

    return {"metrics": metric_summaries, "pairs": pair_summaries}


def plot_pca_loadings(
    output_dir: Path,
    class_name: str,
    components: dict[str, np.ndarray],
    explained_ratios: dict[str, np.ndarray],
    anchor_count: int,
) -> None:
    x = np.arange(anchor_count)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for pc_idx, ax in enumerate(axes):
        for metric in METRICS:
            label = f"{METRIC_LABELS[metric]} ({explained_ratios[metric][pc_idx]:.1%})"
            ax.plot(
                x,
                components[metric][pc_idx],
                marker="o",
                markersize=3,
                linewidth=1.4,
                label=label,
            )
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.set_ylabel(f"PC{pc_idx + 1} loading")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, ncols=3, fontsize=9)

    axes[-1].set_xlabel(f"{class_name} anchor coordinate")
    axes[-1].set_xticks([0, anchor_count - 1], ["1", str(anchor_count)])
    fig.suptitle(f"First Two PCA Components as {class_name} Anchor Linear Combinations")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_pc1_pc2_loadings.png", dpi=300)
    plt.close(fig)


def plot_pca_scores(
    output_dir: Path,
    class_name: str,
    scores: dict[str, np.ndarray],
    events_per_class: int,
    sample_size: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    sample = rng.choice(
        np.arange(events_per_class),
        size=min(sample_size, events_per_class),
        replace=False,
    )

    fig, axes = plt.subplots(1, len(METRICS), figsize=(15, 4.5), sharex=False, sharey=False)
    for ax, metric in zip(axes, METRICS):
        metric_scores = scores[metric]
        ax.scatter(
            metric_scores[sample, 0],
            metric_scores[sample, 1],
            s=7,
            alpha=0.35,
            color="tab:blue",
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("PC1 score")
        ax.set_ylabel("PC2 score")
        ax.grid(alpha=0.2)

    fig.suptitle(f"PCA Scores on Standardized {class_name} Relative Representations")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_pc1_pc2_scores.png", dpi=300)
    plt.close(fig)


def save_pca_csvs(
    output_dir: Path,
    components: dict[str, np.ndarray],
    explained_ratios: dict[str, np.ndarray],
    anchor_labels: list[str],
) -> None:
    loading_path = output_dir / "pca_loadings.csv"
    with loading_path.open("w") as handle:
        handle.write("metric,pc,coordinate,anchor_label,loading,explained_ratio\n")
        for metric in METRICS:
            for pc_idx in range(2):
                for coord, label in enumerate(anchor_labels):
                    handle.write(
                        f"{metric},PC{pc_idx + 1},{coord},{label},"
                        f"{components[metric][pc_idx, coord]:.10g},"
                        f"{explained_ratios[metric][pc_idx]:.10g}\n"
                    )

    corr_path = output_dir / "pca_loading_correlations.csv"
    similarity_path = output_dir / "pca_loading_similarities.csv"
    pairs = [("w2", "sw"), ("w2", "ps"), ("sw", "ps")]
    with corr_path.open("w") as handle:
        handle.write("pc,metric_a,metric_b,loading_correlation\n")
        for pc_idx in range(2):
            for metric_a, metric_b in pairs:
                corr = np.corrcoef(
                    components[metric_a][pc_idx],
                    components[metric_b][pc_idx],
                )[0, 1]
                handle.write(f"PC{pc_idx + 1},{metric_a},{metric_b},{corr:.10g}\n")

    with similarity_path.open("w") as handle:
        handle.write("pc,metric_a,metric_b,signed_cosine_similarity,loading_correlation\n")
        for pc_idx in range(2):
            for metric_a, metric_b in pairs:
                cosine = float(
                    np.dot(components[metric_a][pc_idx], components[metric_b][pc_idx])
                )
                corr = np.corrcoef(
                    components[metric_a][pc_idx],
                    components[metric_b][pc_idx],
                )[0, 1]
                handle.write(
                    f"PC{pc_idx + 1},{metric_a},{metric_b},{cosine:.10g},{corr:.10g}\n"
                )

    top_path = output_dir / "pca_top_loadings.csv"
    with top_path.open("w") as handle:
        handle.write("metric,pc,rank,coordinate,anchor_label,loading,abs_loading\n")
        for metric in METRICS:
            for pc_idx in range(2):
                order = np.argsort(np.abs(components[metric][pc_idx]))[::-1]
                for rank, coord in enumerate(order[:10], start=1):
                    loading = components[metric][pc_idx, coord]
                    handle.write(
                        f"{metric},PC{pc_idx + 1},{rank},{coord},"
                        f"{anchor_labels[coord]},{loading:.10g},{abs(loading):.10g}\n"
                    )


def save_pca_spectrum_outputs(
    output_dir: Path,
    class_name: str,
    pca_by_metric: dict[str, PCAResult],
) -> dict[str, dict[str, float]]:
    path = output_dir / "pca_spectrum.csv"
    summaries: dict[str, dict[str, float]] = {}

    with path.open("w") as handle:
        handle.write(
            "metric,rank,eigenvalue,explained_fraction,"
            "cumulative_explained_fraction,effective_rank\n"
        )
        for metric in METRICS:
            pca = pca_by_metric[metric]
            summaries[metric] = {
                "top1_cumulative": float(pca.cumulative_explained[0]),
                "top2_cumulative": float(pca.cumulative_explained[1]),
                "top5_cumulative": float(pca.cumulative_explained[min(4, len(pca.cumulative_explained) - 1)]),
                "effective_rank": pca.effective_rank,
            }
            for rank, (eig, frac, cum) in enumerate(
                zip(pca.eigenvalues, pca.explained_ratio, pca.cumulative_explained),
                start=1,
            ):
                handle.write(
                    f"{metric},{rank},{eig:.10g},{frac:.10g},"
                    f"{cum:.10g},{pca.effective_rank:.10g}\n"
                )

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in METRICS:
        fractions = np.where(
            pca_by_metric[metric].explained_ratio > 0.0,
            pca_by_metric[metric].explained_ratio,
            np.nan,
        )
        ax.plot(
            np.arange(1, len(fractions) + 1),
            fractions,
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=METRIC_LABELS[metric],
        )
    ax.set_xlabel("PC rank")
    ax.set_ylabel("Explained variance fraction")
    ax.set_yscale("log")
    ax.set_title(f"PCA Spectrum, {class_name}")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_spectrum.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in METRICS:
        ax.plot(
            np.arange(1, len(pca_by_metric[metric].cumulative_explained) + 1),
            pca_by_metric[metric].cumulative_explained,
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=METRIC_LABELS[metric],
        )
    ax.set_xlabel("PC rank")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"Cumulative PCA Spectrum, {class_name}")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_cumulative_spectrum.png", dpi=300)
    plt.close(fig)

    return summaries


def subspace_overlap(components_a: np.ndarray, components_b: np.ndarray, k: int) -> float:
    basis_overlap = components_a[:k] @ components_b[:k].T
    return float(np.linalg.norm(basis_overlap, ord="fro") ** 2 / k)


def save_alignment_matrix_csv(
    path: Path,
    signed_matrix: np.ndarray,
    abs_matrix: np.ndarray,
) -> None:
    with path.open("w") as handle:
        handle.write("pc_a,pc_b,signed_dot,abs_dot\n")
        for i in range(signed_matrix.shape[0]):
            for j in range(signed_matrix.shape[1]):
                handle.write(
                    f"PC{i + 1},PC{j + 1},"
                    f"{signed_matrix[i, j]:.10g},{abs_matrix[i, j]:.10g}\n"
                )


def save_pca_alignment_outputs(
    output_dir: Path,
    class_name: str,
    components: dict[str, np.ndarray],
) -> dict[str, object]:
    top_k = min(ALIGNMENT_TOP_K, next(iter(components.values())).shape[0])
    summary: dict[str, dict[str, float]] = {}

    fig, axes = plt.subplots(
        2,
        len(METRIC_PAIRS),
        figsize=(4.7 * len(METRIC_PAIRS), 8),
        constrained_layout=True,
    )
    for col, (metric_a, metric_b) in enumerate(METRIC_PAIRS):
        signed = components[metric_a][:top_k] @ components[metric_b][:top_k].T
        abs_matrix = np.abs(signed)
        pair_key = f"{metric_a}-{metric_b}"
        np.save(output_dir / f"pca_alignment_signed_{pair_key}.npy", signed)
        np.save(output_dir / f"pca_alignment_abs_{pair_key}.npy", abs_matrix)
        save_alignment_matrix_csv(
            output_dir / f"pca_alignment_{pair_key}.csv",
            signed,
            abs_matrix,
        )

        im_signed = axes[0, col].imshow(signed, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        axes[0, col].set_title(f"{METRIC_LABELS[metric_a]} vs {METRIC_LABELS[metric_b]} signed")
        axes[0, col].set_xlabel(f"{metric_b.upper()} PC")
        axes[0, col].set_ylabel(f"{metric_a.upper()} PC")

        im_abs = axes[1, col].imshow(abs_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[1, col].set_title(f"{METRIC_LABELS[metric_a]} vs {METRIC_LABELS[metric_b]} absolute")
        axes[1, col].set_xlabel(f"{metric_b.upper()} PC")
        axes[1, col].set_ylabel(f"{metric_a.upper()} PC")

        summary[pair_key] = {
            "pc1_signed": float(signed[0, 0]),
            "pc1_abs": float(abs_matrix[0, 0]),
            "pc2_signed": float(signed[1, 1]),
            "pc2_abs": float(abs_matrix[1, 1]),
            "subspace_overlap_k1": subspace_overlap(components[metric_a], components[metric_b], 1),
            "subspace_overlap_k2": subspace_overlap(components[metric_a], components[metric_b], 2),
            "subspace_overlap_k5": subspace_overlap(components[metric_a], components[metric_b], min(5, top_k)),
            "subspace_overlap_k10": subspace_overlap(components[metric_a], components[metric_b], top_k),
        }

    fig.colorbar(im_signed, ax=axes[0, :], shrink=0.85, label="Signed PC dot product")
    fig.colorbar(im_abs, ax=axes[1, :], shrink=0.85, label="Absolute PC dot product")
    fig.suptitle(f"PCA Alignment Matrices, {class_name}")
    fig.savefig(output_dir / "pca_alignment_matrices.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, top_k + 1)
    for metric_a, metric_b in METRIC_PAIRS:
        signed = components[metric_a][:top_k] @ components[metric_b][:top_k].T
        ax.plot(
            x,
            np.abs(np.diag(signed)),
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=f"{metric_a}-{metric_b}",
        )
    ax.set_xlabel("Matched PC rank")
    ax.set_ylabel("Absolute same-rank PC dot product")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"Same-rank PCA Alignment, {class_name}")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_alignment_diagonal.png", dpi=300)
    plt.close(fig)

    with (output_dir / "pca_alignment_summary.csv").open("w") as handle:
        handle.write(
            "metric_a,metric_b,pc1_signed,pc1_abs,pc2_signed,pc2_abs,"
            "subspace_overlap_k1,subspace_overlap_k2,subspace_overlap_k5,"
            "subspace_overlap_k10\n"
        )
        for metric_a, metric_b in METRIC_PAIRS:
            values = summary[f"{metric_a}-{metric_b}"]
            handle.write(
                f"{metric_a},{metric_b},{values['pc1_signed']:.10g},"
                f"{values['pc1_abs']:.10g},{values['pc2_signed']:.10g},"
                f"{values['pc2_abs']:.10g},{values['subspace_overlap_k1']:.10g},"
                f"{values['subspace_overlap_k2']:.10g},"
                f"{values['subspace_overlap_k5']:.10g},"
                f"{values['subspace_overlap_k10']:.10g}\n"
            )

    return {"pairs": summary}


def transfer_curve(source_components: np.ndarray, target_cov: np.ndarray) -> np.ndarray:
    component_variances = np.einsum(
        "ij,jk,ik->i",
        source_components,
        target_cov,
        source_components,
    )
    total_variance = float(np.trace(target_cov))
    if total_variance <= 0.0:
        return np.zeros(source_components.shape[0], dtype=float)
    return np.cumsum(component_variances) / total_variance


def save_transfer_outputs(
    output_dir: Path,
    class_name: str,
    pca_by_metric: dict[str, PCAResult],
    components: dict[str, np.ndarray],
) -> dict[str, object]:
    transfer: dict[tuple[str, str], np.ndarray] = {}
    residual: dict[tuple[str, str], np.ndarray] = {}
    n_components = next(iter(components.values())).shape[0]
    ordered_pairs = [
        (source, target)
        for source in METRICS
        for target in METRICS
        if source != target
    ]

    for source, target in ordered_pairs:
        curve = transfer_curve(components[source], pca_by_metric[target].cov)
        transfer[(source, target)] = curve
        residual[(source, target)] = 1.0 - curve

    with (output_dir / "explained_variance_transfer.csv").open("w") as handle:
        handle.write("source_metric,target_metric,k,transfer_fraction\n")
        for source, target in ordered_pairs:
            for k, value in enumerate(transfer[(source, target)], start=1):
                handle.write(f"{source},{target},{k},{value:.10g}\n")

    with (output_dir / "projection_residuals.csv").open("w") as handle:
        handle.write("source_metric,target_metric,k,residual_fraction\n")
        for source, target in ordered_pairs:
            for k, value in enumerate(residual[(source, target)], start=1):
                handle.write(f"{source},{target},{k},{value:.10g}\n")

    with (output_dir / "explained_variance_transfer_summary.csv").open("w") as handle:
        handle.write("source_metric,target_metric,k,transfer_fraction,residual_fraction\n")
        for source, target in ordered_pairs:
            for k in SUMMARY_K_VALUES:
                if k <= n_components:
                    value = transfer[(source, target)][k - 1]
                    handle.write(
                        f"{source},{target},{k},{value:.10g},{1.0 - value:.10g}\n"
                    )

    hierarchy: dict[str, dict[int, list[dict[str, object]]]] = {
        "source_power": {},
        "target_explainability": {},
    }
    with (output_dir / "metric_hierarchy_summary.csv").open("w") as handle:
        handle.write(
            "hierarchy_type,k,rank,metric,average_transfer,average_residual\n"
        )
        for k in SUMMARY_K_VALUES:
            if k > n_components:
                continue
            source_scores = []
            target_scores = []
            for source in METRICS:
                values = [
                    transfer[(source, target)][k - 1]
                    for target in METRICS
                    if target != source
                ]
                source_scores.append((source, float(np.mean(values))))
            for target in METRICS:
                values = [
                    transfer[(source, target)][k - 1]
                    for source in METRICS
                    if source != target
                ]
                target_scores.append((target, float(np.mean(values))))

            source_scores.sort(key=lambda item: item[1], reverse=True)
            target_scores.sort(key=lambda item: item[1], reverse=True)
            hierarchy["source_power"][k] = []
            hierarchy["target_explainability"][k] = []

            for rank, (metric, value) in enumerate(source_scores, start=1):
                row = {
                    "rank": rank,
                    "metric": metric,
                    "average_transfer": value,
                    "average_residual": 1.0 - value,
                }
                hierarchy["source_power"][k].append(row)
                handle.write(
                    f"source_power,{k},{rank},{metric},{value:.10g},{1.0 - value:.10g}\n"
                )
            for rank, (metric, value) in enumerate(target_scores, start=1):
                row = {
                    "rank": rank,
                    "metric": metric,
                    "average_transfer": value,
                    "average_residual": 1.0 - value,
                }
                hierarchy["target_explainability"][k].append(row)
                handle.write(
                    "target_explainability,"
                    f"{k},{rank},{metric},{value:.10g},{1.0 - value:.10g}\n"
                )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(1, n_components + 1)
    for source, target in ordered_pairs:
        ax.plot(
            x,
            transfer[(source, target)],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=f"{source}->{target}",
        )
    ax.set_xlabel("Source PCA basis dimension k")
    ax.set_ylabel("Target variance explained")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"Explained-variance Transfer, {class_name}")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "explained_variance_transfer_curves.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for source, target in ordered_pairs:
        ax.plot(
            x,
            residual[(source, target)],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=f"{source}->{target}",
        )
    ax.set_xlabel("Source PCA basis dimension k")
    ax.set_ylabel("Target residual variance")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"Projection Residuals, {class_name}")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "projection_residual_curves.png", dpi=300)
    plt.close(fig)

    heatmap_k_values = [k for k in SUMMARY_K_VALUES if k <= n_components]
    fig, axes = plt.subplots(1, len(heatmap_k_values), figsize=(4.1 * len(heatmap_k_values), 4))
    if len(heatmap_k_values) == 1:
        axes = np.array([axes])
    for ax, k in zip(axes, heatmap_k_values):
        matrix = np.eye(len(METRICS))
        for row, source in enumerate(METRICS):
            for col, target in enumerate(METRICS):
                if source != target:
                    matrix[row, col] = transfer[(source, target)][k - 1]
        im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"k={k}")
        ax.set_xticks(range(len(METRICS)), METRICS)
        ax.set_yticks(range(len(METRICS)), METRICS)
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Transfer fraction")
    fig.suptitle(f"Explained-variance Transfer Heatmaps, {class_name}")
    fig.savefig(output_dir / "explained_variance_transfer_heatmaps.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(heatmap_k_values), figsize=(4.1 * len(heatmap_k_values), 4))
    if len(heatmap_k_values) == 1:
        axes = np.array([axes])
    for ax, k in zip(axes, heatmap_k_values):
        matrix = np.zeros((len(METRICS), len(METRICS)))
        for row, source in enumerate(METRICS):
            for col, target in enumerate(METRICS):
                if source != target:
                    matrix[row, col] = residual[(source, target)][k - 1]
        im = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(f"k={k}")
        ax.set_xticks(range(len(METRICS)), METRICS)
        ax.set_yticks(range(len(METRICS)), METRICS)
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Residual fraction")
    fig.suptitle(f"Projection Residual Heatmaps, {class_name}")
    fig.savefig(output_dir / "projection_residual_heatmaps.png", dpi=300)
    plt.close(fig)

    return {
        "transfer_at_k": {
            f"{source}->{target}": {
                str(k): float(transfer[(source, target)][k - 1])
                for k in SUMMARY_K_VALUES
                if k <= n_components
            }
            for source, target in ordered_pairs
        },
        "residual_at_k": {
            f"{source}->{target}": {
                str(k): float(residual[(source, target)][k - 1])
                for k in SUMMARY_K_VALUES
                if k <= n_components
            }
            for source, target in ordered_pairs
        },
        "hierarchy": hierarchy,
    }


def analyze_class(
    root_output_dir: Path,
    class_slug: str,
    class_name: str,
    class_idx: int,
    anchor_indices: np.ndarray,
    events_per_class: int,
    pca_sample: int,
    seed: int,
) -> dict[str, object]:
    output_dir = root_output_dir / class_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_labels = make_anchor_labels(class_name, len(anchor_indices))

    save_json(
        output_dir / "analysis_metadata.json",
        {
            "dataset_size": "large",
            "events_per_class": events_per_class,
            "class_name": class_name,
            "class_index": class_idx,
            "metrics": list(METRICS),
            "anchor_count": int(len(anchor_indices)),
            "seed": seed,
            "anchor_indices_local": anchor_indices.tolist(),
        },
    )

    rel_reps: dict[str, np.ndarray] = {}
    for metric in METRICS:
        print(f"Building {class_name} {metric} intraclass relative representation...")
        rel_rep = build_intraclass_rel_rep(
            metric=metric,
            class_idx=class_idx,
            anchor_indices=anchor_indices,
            events_per_class=events_per_class,
        )
        rel_reps[metric] = rel_rep
        np.save(output_dir / f"{metric}_{class_slug}_large_rel_rep_{len(anchor_indices)}.npy", rel_rep)

    print(f"Writing {class_name} coordinate distribution summaries and plots...")
    coordinate_summary = save_coordinate_summaries(output_dir, rel_reps, anchor_labels)
    plot_coordinate_quantiles(output_dir, class_name, rel_reps, len(anchor_indices))
    plot_standardized_distribution_overlay(output_dir, class_name, rel_reps)

    print(f"Computing {class_name} covariance matrices and spectra...")
    covariances: dict[str, np.ndarray] = {}
    eigenvalues: dict[str, np.ndarray] = {}
    correlations: dict[str, np.ndarray] = {}
    for metric, rel_rep in rel_reps.items():
        cov, eig = covariance_and_spectrum(rel_rep)
        covariances[metric] = cov
        eigenvalues[metric] = eig
        correlations[metric] = correlation_from_covariance(cov)
        np.save(output_dir / f"{metric}_covariance.npy", cov)
        np.save(output_dir / f"{metric}_covariance_eigenvalues.npy", eig)
        np.save(output_dir / f"{metric}_correlation.npy", correlations[metric])
    save_covariance_spectrum_csv(output_dir, eigenvalues)
    plot_covariances_and_spectra(output_dir, class_name, covariances, eigenvalues)
    covariance_summary = save_covariance_structure_summary(
        output_dir,
        eigenvalues,
        correlations,
    )

    print(f"Running {class_name} PCA on standardized relative representations...")
    pca_by_metric: dict[str, PCAResult] = {}
    components_by_metric: dict[str, np.ndarray] = {}
    scores_by_metric: dict[str, np.ndarray] = {}
    explained_ratios: dict[str, np.ndarray] = {}
    for metric, rel_rep in rel_reps.items():
        pca = pca_from_standardized(rel_rep)
        pca_by_metric[metric] = pca
        components_by_metric[metric] = pca.components
        scores_by_metric[metric] = pca.scores
        explained_ratios[metric] = pca.explained_ratio

    pca_spectrum_summary = save_pca_spectrum_outputs(
        output_dir,
        class_name,
        pca_by_metric,
    )
    aligned_components, component_signs = align_component_signs(components_by_metric)
    for metric in METRICS:
        signed_scores = scores_by_metric[metric] * component_signs[metric][None, :]
        scores_by_metric[metric] = signed_scores
        np.save(output_dir / f"{metric}_pca_components.npy", aligned_components[metric])
        np.save(output_dir / f"{metric}_pca_components_pc1_pc2.npy", aligned_components[metric][:2])
        np.save(output_dir / f"{metric}_pca_scores_pc1_pc2.npy", signed_scores[:, :2])
        np.save(output_dir / f"{metric}_pca_eigenvalues.npy", pca_by_metric[metric].eigenvalues)

    plot_pca_loadings(
        output_dir,
        class_name,
        aligned_components,
        explained_ratios,
        len(anchor_indices),
    )
    plot_pca_scores(
        output_dir,
        class_name,
        scores_by_metric,
        events_per_class,
        pca_sample,
        seed,
    )
    save_pca_csvs(output_dir, aligned_components, explained_ratios, anchor_labels)
    pca_alignment_summary = save_pca_alignment_outputs(
        output_dir,
        class_name,
        aligned_components,
    )
    transfer_summary = save_transfer_outputs(
        output_dir,
        class_name,
        pca_by_metric,
        aligned_components,
    )

    return {
        "class_slug": class_slug,
        "class_name": class_name,
        "output_dir": str(output_dir),
        "coordinate_distribution": coordinate_summary,
        "covariance_structure": covariance_summary,
        "pca_spectrum": pca_spectrum_summary,
        "pca_alignment": pca_alignment_summary,
        "explained_variance_transfer": transfer_summary,
        "pca_explained_ratios": {
            metric: explained_ratios[metric][:2].tolist() for metric in METRICS
        },
        "top_covariance_explained_fractions": {
            metric: safe_explained_fractions(eigenvalues[metric])[:2].tolist()
            for metric in METRICS
        },
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def metric_pair_label(metric_a: str, metric_b: str) -> str:
    return f"{metric_a}-{metric_b}"


def write_summary_metrics_csv(root_output_dir: Path, summaries: list[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []

    for class_summary in summaries:
        class_slug = str(class_summary["class_slug"])
        class_name = str(class_summary["class_name"])

        for item in class_summary["coordinate_distribution"]:
            for value_name in ("mean_ks", "median_ks", "max_ks"):
                rows.append(
                    {
                        "class_slug": class_slug,
                        "class_name": class_name,
                        "analysis": "coordinate_distribution",
                        "metric": "",
                        "metric_a": item["metric_a"],
                        "metric_b": item["metric_b"],
                        "source_metric": "",
                        "target_metric": "",
                        "k": "",
                        "value_name": value_name,
                        "value": item[value_name],
                        "conclusion": item["mean_conclusion"]
                        if value_name != "max_ks"
                        else item["max_conclusion"],
                    }
                )

        cov_summary = class_summary["covariance_structure"]
        for metric, item in cov_summary["metrics"].items():
            for value_name, value in item.items():
                rows.append(
                    {
                        "class_slug": class_slug,
                        "class_name": class_name,
                        "analysis": "covariance_structure",
                        "metric": metric,
                        "metric_a": "",
                        "metric_b": "",
                        "source_metric": "",
                        "target_metric": "",
                        "k": "",
                        "value_name": value_name,
                        "value": value,
                        "conclusion": "",
                    }
                )
        for pair_key, item in cov_summary["pairs"].items():
            metric_a, metric_b = pair_key.split("-")
            rows.append(
                {
                    "class_slug": class_slug,
                    "class_name": class_name,
                    "analysis": "correlation_frobenius_cosine",
                    "metric": "",
                    "metric_a": metric_a,
                    "metric_b": metric_b,
                    "source_metric": "",
                    "target_metric": "",
                    "k": "",
                    "value_name": "frobenius_cosine",
                    "value": item["frobenius_cosine"],
                    "conclusion": "",
                }
            )

        for metric, item in class_summary["pca_spectrum"].items():
            for value_name, value in item.items():
                rows.append(
                    {
                        "class_slug": class_slug,
                        "class_name": class_name,
                        "analysis": "pca_spectrum",
                        "metric": metric,
                        "metric_a": "",
                        "metric_b": "",
                        "source_metric": "",
                        "target_metric": "",
                        "k": "",
                        "value_name": value_name,
                        "value": value,
                        "conclusion": "",
                    }
                )

        for pair_key, item in class_summary["pca_alignment"]["pairs"].items():
            metric_a, metric_b = pair_key.split("-")
            for value_name, value in item.items():
                rows.append(
                    {
                        "class_slug": class_slug,
                        "class_name": class_name,
                        "analysis": "pca_alignment",
                        "metric": "",
                        "metric_a": metric_a,
                        "metric_b": metric_b,
                        "source_metric": "",
                        "target_metric": "",
                        "k": "",
                        "value_name": value_name,
                        "value": value,
                        "conclusion": "",
                    }
                )

        transfer = class_summary["explained_variance_transfer"]
        for ordered_pair, values_by_k in transfer["transfer_at_k"].items():
            source, target = ordered_pair.split("->")
            for k, value in values_by_k.items():
                rows.append(
                    {
                        "class_slug": class_slug,
                        "class_name": class_name,
                        "analysis": "explained_variance_transfer",
                        "metric": "",
                        "metric_a": "",
                        "metric_b": "",
                        "source_metric": source,
                        "target_metric": target,
                        "k": k,
                        "value_name": "transfer_fraction",
                        "value": value,
                        "conclusion": "",
                    }
                )
        for ordered_pair, values_by_k in transfer["residual_at_k"].items():
            source, target = ordered_pair.split("->")
            for k, value in values_by_k.items():
                rows.append(
                    {
                        "class_slug": class_slug,
                        "class_name": class_name,
                        "analysis": "projection_residual",
                        "metric": "",
                        "metric_a": "",
                        "metric_b": "",
                        "source_metric": source,
                        "target_metric": target,
                        "k": k,
                        "value_name": "residual_fraction",
                        "value": value,
                        "conclusion": "",
                    }
                )

    columns = [
        "class_slug",
        "class_name",
        "analysis",
        "metric",
        "metric_a",
        "metric_b",
        "source_metric",
        "target_metric",
        "k",
        "value_name",
        "value",
        "conclusion",
    ]
    with (root_output_dir / "summary_metrics.csv").open("w") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(str(row[col]) for col in columns) + "\n")


def best_coordinate_sentence(summary: dict[str, object]) -> str:
    items = sorted(summary["coordinate_distribution"], key=lambda item: item["mean_ks"])
    best = items[0]
    worst = items[-1]
    return (
        f"The closest standardized coordinate distributions are {best['metric_a']}-"
        f"{best['metric_b']} (mean KS {best['mean_ks']:.3f}, "
        f"{best['mean_conclusion']}); the largest difference is "
        f"{worst['metric_a']}-{worst['metric_b']} (mean KS {worst['mean_ks']:.3f}, "
        f"{worst['mean_conclusion']})."
    )


def covariance_sentence(summary: dict[str, object]) -> str:
    cov = summary["covariance_structure"]
    pieces = []
    for metric in METRICS:
        item = cov["metrics"][metric]
        pieces.append(
            f"{metric}: top-1 {pct(item['top1_cumulative'])}, "
            f"top-2 {pct(item['top2_cumulative'])}, "
            f"effective rank {item['effective_rank']:.2f}"
        )
    pair_items = sorted(
        cov["pairs"].items(),
        key=lambda item: item[1]["frobenius_cosine"],
        reverse=True,
    )
    strongest = pair_items[0]
    weakest = pair_items[-1]
    return (
        "Raw covariance spectra: "
        + "; ".join(pieces)
        + ". Correlation-matrix similarity is strongest for "
        f"{strongest[0]} ({strongest[1]['frobenius_cosine']:.3f}) and weakest for "
        f"{weakest[0]} ({weakest[1]['frobenius_cosine']:.3f})."
    )


def pca_spectrum_sentence(summary: dict[str, object]) -> str:
    pieces = []
    for metric in METRICS:
        item = summary["pca_spectrum"][metric]
        pieces.append(
            f"{metric}: PC1 {pct(item['top1_cumulative'])}, "
            f"PC1-2 {pct(item['top2_cumulative'])}, "
            f"effective rank {item['effective_rank']:.2f}"
        )
    return "Standardized PCA spectra: " + "; ".join(pieces) + "."


def pca_alignment_sentence(summary: dict[str, object]) -> str:
    items = sorted(
        summary["pca_alignment"]["pairs"].items(),
        key=lambda item: item[1]["pc1_abs"],
        reverse=True,
    )
    strongest = items[0]
    weakest = items[-1]
    subspace_items = sorted(
        summary["pca_alignment"]["pairs"].items(),
        key=lambda item: item[1]["subspace_overlap_k2"],
        reverse=True,
    )
    best_subspace = subspace_items[0]
    return (
        f"PC1 alignment is strongest for {strongest[0]} "
        f"({strongest[1]['pc1_abs']:.3f}) and weakest for {weakest[0]} "
        f"({weakest[1]['pc1_abs']:.3f}). The best two-dimensional subspace "
        f"overlap is {best_subspace[0]} ({best_subspace[1]['subspace_overlap_k2']:.3f})."
    )


def transfer_sentence(summary: dict[str, object], k: int = 5) -> str:
    transfer = summary["explained_variance_transfer"]
    pairs = [
        (pair, values[str(k)])
        for pair, values in transfer["transfer_at_k"].items()
        if str(k) in values
    ]
    pairs.sort(key=lambda item: item[1], reverse=True)
    best_pair = pairs[0]
    worst_pair = pairs[-1]
    source_best = transfer["hierarchy"]["source_power"][k][0]
    target_best = transfer["hierarchy"]["target_explainability"][k][0]
    target_worst = transfer["hierarchy"]["target_explainability"][k][-1]
    return (
        f"At k={k}, best ordered transfer is {best_pair[0]} "
        f"({pct(best_pair[1])}); weakest is {worst_pair[0]} ({pct(worst_pair[1])}). "
        f"As a source basis, {source_best['metric']} has the highest average transfer "
        f"({pct(source_best['average_transfer'])}). As a target, "
        f"{target_best['metric']} is easiest to explain on average "
        f"({pct(target_best['average_transfer'])}), while {target_worst['metric']} "
        f"is hardest ({pct(target_worst['average_transfer'])})."
    )


def residual_sentence(summary: dict[str, object], k: int = 5) -> str:
    transfer = summary["explained_variance_transfer"]
    source_best = transfer["hierarchy"]["source_power"][k][0]
    source_worst = transfer["hierarchy"]["source_power"][k][-1]
    return (
        f"Projection residuals are the complement of transfer. At k={k}, "
        f"{source_best['metric']} leaves the smallest average residual "
        f"({pct(source_best['average_residual'])}), while {source_worst['metric']} "
        f"leaves the largest ({pct(source_worst['average_residual'])})."
    )


def write_conclusions(root_output_dir: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# Intraclass Statistical Structure and Shared Backbone Conclusions",
        "",
        "This report uses separate intraclass relative representations: QCD events "
        "to QCD anchors and TTBar events to TTBar anchors. PCA alignment, "
        "explained-variance transfer, and residuals use per-coordinate standardized "
        "representations. Raw covariance spectra use the native metric scales.",
        "",
    ]

    for summary in summaries:
        class_name = summary["class_name"]
        lines.extend(
            [
                f"## {class_name}",
                "",
                "### Coordinate-wise Distribution Comparison",
                best_coordinate_sentence(summary),
                "",
                "### Covariance Matrices and Covariance Spectrum",
                covariance_sentence(summary),
                "",
                "### PCA Spectrum",
                pca_spectrum_sentence(summary),
                "",
                "### PCA Alignment",
                pca_alignment_sentence(summary),
                "",
                "### Explained-variance Transfer",
                transfer_sentence(summary, k=5),
                "",
                "### Projection Residuals",
                residual_sentence(summary, k=5),
                "",
            ]
        )

    (root_output_dir / "conclusions.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    events_per_class = jetclass_utils.dataset_size_to_count("large")
    anchor_indices = sample_anchor_indices(
        events_per_class=events_per_class,
        anchor_count=args.anchor_count,
        seed=args.seed,
    )
    np.save(args.output_dir / "shared_anchor_indices_local.npy", anchor_indices)

    summaries = []
    for class_slug, class_name, class_idx in TARGET_CLASSES:
        summaries.append(
            analyze_class(
                root_output_dir=args.output_dir,
                class_slug=class_slug,
                class_name=class_name,
                class_idx=class_idx,
                anchor_indices=anchor_indices,
                events_per_class=events_per_class,
                pca_sample=args.pca_sample,
                seed=args.seed,
            )
        )

    save_json(args.output_dir / "summary_metrics.json", {"summaries": summaries})
    write_summary_metrics_csv(args.output_dir, summaries)
    write_conclusions(args.output_dir, summaries)

    save_json(
        args.output_dir / "analysis_metadata.json",
        {
            "dataset_size": "large",
            "events_per_class": events_per_class,
            "metrics": list(METRICS),
            "target_classes": [
                {"slug": slug, "name": name, "class_index": class_idx}
                for slug, name, class_idx in TARGET_CLASSES
            ],
            "analysis": "separate intraclass relative representations",
            "anchor_count": args.anchor_count,
            "seed": args.seed,
            "shared_anchor_indices_local": anchor_indices.tolist(),
            "summaries": summaries,
        },
    )
    print(f"Done. Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
