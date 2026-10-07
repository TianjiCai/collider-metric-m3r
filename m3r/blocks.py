"""Access to the stored pairwise distance blocks.

The large JetClass dataset is stored as 55 upper-triangular blocks per metric,
``{metric}_block_{i}_{j}.npy`` with ``i <= j``, each a ``(10000, 10000)``
float64 array of event-to-event distances between class ``i`` and class ``j``.

Which quantity a block holds is recorded in ``data/manifests/blocks_manifest.json``
as ``stored_power``: ``1`` means the file holds the distance ``d``, ``2`` means
it holds ``d**2``. The published SW2 blocks hold squared distances; W2 and PS hold
plain distances. ``BlockStore`` is the only place in the code base that looks at
this convention, so every consumer gets distances by default (``as_distance=True``)
and can ask for the raw stored values when it needs them (the absolute-space SVM
kernel, for example, is defined on the stored values to stay bit-exact with the
published runs).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from m3r import paths as _paths
from m3r.classes import CLASS_NAMES, KIN_METRICS

MANIFEST_NAME = "blocks_manifest.json"


@dataclass(frozen=True)
class MetricConvention:
    metric: str
    stored_power: int = 1
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.stored_power not in (1, 2):
            raise ValueError(f"stored_power must be 1 or 2, got {self.stored_power}.")


def to_distance(stored: np.ndarray, stored_power: int, *, out: np.ndarray | None = None) -> np.ndarray:
    """Convert stored values to distances in place-able fashion."""
    if stored_power == 1:
        if out is None:
            return stored
        out[...] = stored
        return out
    if out is None:
        out = np.array(stored, dtype=np.result_type(stored.dtype, np.float64), copy=True)
    elif out is not stored:
        out[...] = stored
    np.maximum(out, 0.0, out=out)
    np.sqrt(out, out=out)
    return out


def combine_blocks(blocks: Sequence[np.ndarray], dtype: Any = None) -> np.ndarray:
    """Assemble upper-triangular blocks (row-major, ``i <= j``) into a full symmetric matrix.

    Array-input port of the legacy ``combine_distance_blocks``.
    """
    m = len(blocks)
    n = int((math.sqrt(8 * m + 1) - 1) / 2)
    if n * (n + 1) // 2 != m:
        raise ValueError(f"Expected n(n+1)/2 blocks, got {m}")
    sizes = []
    idx = 0
    for i in range(n):
        sizes.append(blocks[idx].shape[0])
        idx += n - i
    starts = np.cumsum([0] + sizes[:-1])
    total = int(sum(sizes))
    out = np.zeros((total, total), dtype=blocks[0].dtype if dtype is None else dtype)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            block = blocks[idx]
            si, sj, di, dj = starts[i], starts[j], sizes[i], sizes[j]
            out[si:si + di, sj:sj + dj] = block
            if i != j:
                out[sj:sj + dj, si:si + di] = block.T
            idx += 1
    return out


@dataclass
class BlockPair:
    """The three blocks describing one class pair, in the *stored* convention.

    ``d00`` and ``d11`` are the diagonal blocks, ``d01`` the cross block with
    rows in class ``i`` and columns in class ``j``. Arrays may be memmaps.
    """

    d00: np.ndarray
    d01: np.ndarray
    d11: np.ndarray
    stored_power: int = 1

    @property
    def n0(self) -> int:
        return int(self.d00.shape[0])

    @property
    def n1(self) -> int:
        return int(self.d11.shape[0])

    @property
    def n(self) -> int:
        return self.n0 + self.n1

    def labels(self) -> np.ndarray:
        return np.repeat([0, 1], [self.n0, self.n1])

    def slice(self, rows: np.ndarray, cols: np.ndarray, *, as_distance: bool = False) -> np.ndarray:
        """Return the sub-matrix ``D[rows][:, cols]`` of the combined pair matrix.

        Port of ``AbsPairwiseSVM.distance_slice_from_blocks`` (memmap friendly).
        """
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        n0 = self.n0
        out = np.empty((len(rows), len(cols)), dtype=np.float64)
        row_is_0 = rows < n0
        col_is_0 = cols < n0
        row_pos_0 = np.flatnonzero(row_is_0)
        row_pos_1 = np.flatnonzero(~row_is_0)
        col_pos_0 = np.flatnonzero(col_is_0)
        col_pos_1 = np.flatnonzero(~col_is_0)
        rows0, rows1 = rows[row_is_0], rows[~row_is_0] - n0
        cols0, cols1 = cols[col_is_0], cols[~col_is_0] - n0
        if row_pos_0.size and col_pos_0.size:
            out[np.ix_(row_pos_0, col_pos_0)] = self.d00[np.ix_(rows0, cols0)]
        if row_pos_0.size and col_pos_1.size:
            out[np.ix_(row_pos_0, col_pos_1)] = self.d01[np.ix_(rows0, cols1)]
        if row_pos_1.size and col_pos_0.size:
            out[np.ix_(row_pos_1, col_pos_0)] = self.d01[np.ix_(cols0, rows1)].T
        if row_pos_1.size and col_pos_1.size:
            out[np.ix_(row_pos_1, col_pos_1)] = self.d11[np.ix_(rows1, cols1)]
        if as_distance:
            to_distance(out, self.stored_power, out=out)
        return out

    def column(self, index: int, *, as_distance: bool = False) -> np.ndarray:
        """One column of the combined pair matrix (distances of every event to event ``index``)."""
        n0 = self.n0
        if index < n0:
            col = np.concatenate([np.asarray(self.d00[:, index]), np.asarray(self.d01[index, :])])
        else:
            b = index - n0
            col = np.concatenate([np.asarray(self.d01[:, b]), np.asarray(self.d11[:, b])])
        col = col.astype(np.float64, copy=False)
        if as_distance:
            to_distance(col, self.stored_power, out=col)
        return col

    def matrix(self, *, as_distance: bool = True) -> np.ndarray:
        """The full ``(n0+n1, n0+n1)`` combined matrix (3.2 GB for 10k + 10k events)."""
        out = combine_blocks([np.asarray(self.d00), np.asarray(self.d01), np.asarray(self.d11)])
        if as_distance:
            to_distance(out, self.stored_power, out=out)
        return out

    def upper_triangle_values(self) -> np.ndarray:
        """All strictly-upper-triangle stored values of the combined matrix (no 3.2 GB copy)."""
        iu0 = np.triu_indices(self.n0, k=1)
        iu1 = np.triu_indices(self.n1, k=1)
        return np.concatenate([
            np.asarray(self.d00)[iu0],
            np.asarray(self.d01).ravel(),
            np.asarray(self.d11)[iu1],
        ])

    def upper_triangle_median(self) -> float:
        """Median of the stored upper-triangle values (the legacy abs-SVM kernel normalisation)."""
        return float(np.median(self.upper_triangle_values()))


class BlockStore:
    """Metric-aware loader for distance blocks under one root directory."""

    def __init__(
        self,
        block_root: str | Path,
        manifest: Mapping[str, Any] | None = None,
        *,
        pattern: str = _paths.BLOCK_PATTERN,
        n_per_class: int = 10_000,
        size: str = "large",
    ):
        self.block_root = Path(block_root)
        self.pattern = pattern
        self.n_per_class = int(n_per_class)
        self.size = size
        self.manifest: dict[str, Any] = dict(manifest or {})
        metrics = self.manifest.get("metrics", {})
        self._conventions = {
            metric: MetricConvention(
                metric,
                int(metrics.get(metric, {}).get("stored_power", 1)),
                str(metrics.get(metric, {}).get("dtype", "float64")),
            )
            for metric in set(KIN_METRICS) | set(metrics)
        }

    # Construction -----------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], manifest_path: str | Path | None = None, *,
                    size: str = "large", n_per_class: int | None = None) -> "BlockStore":
        """Build a store from a ``Config`` (uses ``paths.blocks`` and the manifest)."""
        root = cfg.root if hasattr(cfg, "root") else _paths.repo_root()
        block_root = Path(cfg["paths"]["blocks"])
        if not block_root.is_absolute():
            block_root = root / block_root
        if manifest_path is None:
            manifest_dir = Path(cfg["paths"].get("manifests", "data/manifests"))
            if not manifest_dir.is_absolute():
                manifest_dir = root / manifest_dir
            manifest_path = manifest_dir / MANIFEST_NAME
        manifest = load_manifest(manifest_path) if Path(manifest_path).exists() else {}
        pattern = str(cfg.get("patterns", {}).get("block", _paths.BLOCK_PATTERN))
        if n_per_class is None:
            n_per_class = int(cfg.get("dataset", {}).get("events_per_class", 10_000))
        return cls(block_root, manifest, pattern=pattern, n_per_class=n_per_class, size=size)

    # Metadata ---------------------------------------------------------------
    def convention(self, metric: str) -> MetricConvention:
        return self._conventions.get(metric, MetricConvention(metric))

    def stored_power(self, metric: str) -> int:
        return self.convention(metric).stored_power

    def path(self, metric: str, i: int, j: int) -> Path:
        i, j = sorted((int(i), int(j)))
        rel = self.pattern.format(blocks=str(self.block_root), metric=metric, i=i, j=j, size=self.size)
        return Path(rel)

    def exists(self, metric: str, i: int, j: int) -> bool:
        return self.path(metric, i, j).exists()

    def missing(self, metric: str, pairs: Sequence[tuple[int, int]]) -> list[Path]:
        return [self.path(metric, i, j) for i, j in pairs if not self.exists(metric, i, j)]

    # Loading ----------------------------------------------------------------
    def load(
        self,
        metric: str,
        i: int,
        j: int,
        *,
        as_distance: bool = True,
        n_jets: int | None = None,
        dtype: Any = np.float64,
        mmap: bool = False,
    ) -> np.ndarray:
        """Load one block.

        ``n_jets`` keeps only the top-left ``n_jets x n_jets`` corner (diagonal
        blocks) or the first ``n_jets`` rows and columns (cross blocks).
        ``mmap=True`` returns the memmap in the stored convention.
        """
        path = self.path(metric, i, j)
        if not path.exists():
            raise FileNotFoundError(f"Missing distance block: {path}")
        block = np.load(path, mmap_mode="r")
        if mmap:
            if as_distance and self.stored_power(metric) != 1:
                raise ValueError("mmap=True returns stored values; pass as_distance=False.")
            return block
        if n_jets is not None:
            block = block[:n_jets, :n_jets]
        arr = np.asarray(block)
        if as_distance and self.stored_power(metric) != 1:
            arr = to_distance(arr, self.stored_power(metric))
        return np.asarray(arr, dtype=dtype)

    def pair(self, metric: str, i: int, j: int) -> BlockPair:
        """Memmapped blocks for class pair ``(i, j)`` with ``i < j`` (stored convention)."""
        i, j = sorted((int(i), int(j)))
        if i == j:
            raise ValueError("pair() needs two distinct classes.")
        return BlockPair(
            d00=np.load(self.path(metric, i, i), mmap_mode="r"),
            d01=np.load(self.path(metric, i, j), mmap_mode="r"),
            d11=np.load(self.path(metric, j, j), mmap_mode="r"),
            stored_power=self.stored_power(metric),
        )

    def pair_matrix(self, metric: str, i: int, j: int, *, as_distance: bool = True) -> np.ndarray:
        return self.pair(metric, i, j).matrix(as_distance=as_distance)

    # Verification -----------------------------------------------------------
    def describe(self, metric: str, i: int, j: int) -> dict[str, Any]:
        path = self.path(metric, i, j)
        block = np.load(path, mmap_mode="r")
        return {
            "path": str(path),
            "shape": list(block.shape),
            "dtype": str(block.dtype),
            "bytes": path.stat().st_size,
            "stored_power": self.stored_power(metric),
        }


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return json.load(handle)


def default_manifest(n_per_class: int = 10_000) -> dict[str, Any]:
    """Manifest describing the published blocks (SW2 stored squared)."""
    return {
        "schema": 1,
        "dataset": "jetclass_large_10k",
        "n_per_class": n_per_class,
        "class_order": list(CLASS_NAMES),
        "metrics": {
            "w2": {"stored_power": 1, "dtype": "float64",
                   "generator": "m3r.metrics.w2.get_w2 (POT emd2, sqrt)", "notes": "plain distances"},
            "sw": {"stored_power": 2, "dtype": "float64",
                   "generator": "pyspecter SPECTER.spectralEMD (ds2), GPU, batch 512",
                   "notes": "files hold SW2**2; BlockStore returns sqrt by default"},
            "ps": {"stored_power": 1, "dtype": "float64",
                   "generator": "m3r.metrics.ps.ps_distance (Sphere2Simplex=1, l_has_chi=True)",
                   "notes": "plain distances; classes 0-7 computed before the arccos clipping fix"},
        },
        "files": {},
    }


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n")
    return path
