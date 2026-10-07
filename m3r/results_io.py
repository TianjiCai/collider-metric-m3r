"""Deterministic result writers with provenance sidecars."""

from __future__ import annotations

import csv
import importlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from m3r import paths as _paths


def git_sha(root: Path | None = None) -> str | None:
    root = root or _paths.repo_root()
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git may be absent in a tarball
        return None


def package_versions(names: Iterable[str] = ("numpy", "scipy", "sklearn", "ot", "matplotlib")) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            continue
    return versions


def write_provenance(path: Path, *, config_hash: str | None = None, **extra: Any) -> Path:
    """Write ``<path>.provenance.json`` next to a result file.

    No timestamps are recorded so that re-running a stage with identical
    inputs produces byte-identical sidecars.
    """
    sidecar = path.with_name(path.name + ".provenance.json")
    payload = {
        "result": path.name,
        "git_sha": git_sha(),
        "config_hash": config_hash,
        "versions": package_versions(),
    }
    payload.update(extra)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_jsonable) + "\n")
    return sidecar


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def save_npz(path: Path, **arrays: Any) -> Path:
    """``np.savez`` with sorted keys (stable file layout)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {key: np.asarray(arrays[key]) for key in sorted(arrays)}
    np.savez(path, **ordered)
    return path


def load_npz(path: Path, allow_pickle: bool = False) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=allow_pickle) as payload:
        return {key: payload[key] for key in payload.files}


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))
