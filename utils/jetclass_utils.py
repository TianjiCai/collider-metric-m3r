from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

JETCLASS_CANONICAL_ORDER = (
    "QCD",
    "HToBB",
    "HToCC",
    "HToGG",
    "HToWW4Q",
    "ZToQQ",
    "WToQQ",
    "TTBar",
    "HToWW2Q1L",
    "TTBarLep",
)
LEGACY_LARGE_CLASS_COUNT = 8

JETCLASS_PARTICLE_ID_FEATURES = (
    "part_charge",
    "part_isElectron",
    "part_isMuon",
    "part_isPhoton",
    "part_isChargedHadron",
    "part_isNeutralHadron",
)

DATASET_SIZE_TO_COUNT = {
    "mini": 1_000,
    "medium": 4_000,
    "large": 10_000,
    "full": 100_000,
}
COUNT_TO_TAG = {count: f"{count // 1000}k" for count in DATASET_SIZE_TO_COUNT.values()}


def canonical_class_names(n_classes: int | None = None) -> list[str]:
    if n_classes is None:
        return list(JETCLASS_CANONICAL_ORDER)
    if not 0 < n_classes <= len(JETCLASS_CANONICAL_ORDER):
        raise ValueError(
            f"n_classes must be between 1 and {len(JETCLASS_CANONICAL_ORDER)}, "
            f"got {n_classes}."
        )
    return list(JETCLASS_CANONICAL_ORDER[:n_classes])


def canonical_label_map(n_classes: int | None = None) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(canonical_class_names(n_classes))}


def dataset_size_to_count(dataset_size: str) -> int:
    if dataset_size not in DATASET_SIZE_TO_COUNT:
        raise ValueError(
            f"Unsupported dataset size {dataset_size!r}. "
            f"Expected one of {sorted(DATASET_SIZE_TO_COUNT)}."
        )
    return DATASET_SIZE_TO_COUNT[dataset_size]


def dataset_size_tag(events_per_class: int) -> str:
    if events_per_class not in COUNT_TO_TAG:
        raise ValueError(
            f"Unsupported events_per_class {events_per_class}. "
            f"Expected one of {sorted(COUNT_TO_TAG)}."
        )
    return COUNT_TO_TAG[events_per_class]


def dataset_dir(project_root: Path, dataset_size: str) -> Path:
    return project_root / "data" / f"{dataset_size}_dataset"


def jetclass_source_dir(project_root: Path) -> Path:
    return project_root / "data" / "jetclass_dataset"


def class_labels_path(project_root: Path, dataset_size: str) -> Path:
    return dataset_dir(project_root, dataset_size) / "class_labels.npz"


def class_dataset_path(
    dataset_root: Path, class_name: str, events_per_class: int
) -> Path:
    return dataset_root / f"{class_name}_{dataset_size_tag(events_per_class)}.npz"


def part_ids_path(dataset_root: Path, class_name: str, events_per_class: int) -> Path:
    return dataset_root / f"{class_name}_part_ids_{dataset_size_tag(events_per_class)}.npz"


def class_failed_index_dir(dataset_root: Path) -> Path:
    return dataset_root / "ps_failed_indices"


def class_failed_index_path(dataset_root: Path, class_name: str) -> Path:
    return class_failed_index_dir(dataset_root) / f"{class_name}_failed_index.npy"


def metric_block_dir(project_root: Path, metric: str, dataset_size: str) -> Path:
    return project_root / "data" / "dist_mat" / metric / f"{dataset_size}_{metric}"


def metric_block_path(
    project_root: Path,
    metric: str,
    dataset_size: str,
    class_i: int,
    class_j: int,
) -> Path:
    i, j = sorted((class_i, class_j))
    return metric_block_dir(project_root, metric, dataset_size) / f"{metric}_block_{i}_{j}.npy"


def legacy_large_ps_archive_path(project_root: Path) -> Path:
    return (
        project_root
        / "data_gen"
        / "dist_mat_gen"
        / "ps"
        / "large_ps"
        / "all_jets_10_subjets_p4s.npz"
    )


def new_large_ps_input_path(project_root: Path, class_name: str) -> Path:
    return (
        project_root
        / "data"
        / "dist_mat"
        / "ps"
        / "large_ps"
        / "class_inputs"
        / f"{class_name}_ps_10k.npz"
    )


def appended_upper_triangle_pairs(
    existing_class_count: int = LEGACY_LARGE_CLASS_COUNT,
    total_class_count: int = len(JETCLASS_CANONICAL_ORDER),
) -> list[tuple[int, int]]:
    if existing_class_count >= total_class_count:
        return []

    new_pairs: list[tuple[int, int]] = []
    for new_idx in range(existing_class_count, total_class_count):
        for old_idx in range(new_idx + 1):
            new_pairs.append((old_idx, new_idx))
    return new_pairs


def pairwise_class_pairs(n_classes: int) -> list[tuple[int, int]]:
    return list(combinations(range(n_classes), 2))


def load_label_map(labels_path: Path) -> dict[str, int]:
    payload = np.load(labels_path)
    return {name: int(np.asarray(payload[name]).item()) for name in payload.files}


def ordered_class_names_from_labels(
    labels_path: Path, expected_n_classes: int | None = None
) -> list[str]:
    label_map = load_label_map(labels_path)
    ordered_names = [
        name for name, _ in sorted(label_map.items(), key=lambda item: item[1])
    ]
    if expected_n_classes is not None:
        if len(ordered_names) < expected_n_classes:
            raise ValueError(
                f"{labels_path} contains {len(ordered_names)} classes, "
                f"expected at least {expected_n_classes}."
            )
        ordered_names = ordered_names[:expected_n_classes]
    return ordered_names
