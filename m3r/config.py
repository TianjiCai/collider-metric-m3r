"""Loading of ``configs/paper.yaml`` and typed path helpers.

The paper configuration is a plain YAML mapping. ``Config`` wraps it with the
handful of path helpers that every script needs, resolving relative paths
against the repository root. An optional, git-ignored ``configs/local.yaml``
is deep-merged on top (use it to point ``paths.blocks`` at external storage).
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

import yaml

from m3r import paths as _paths
from m3r.classes import class_name


def _deep_merge(base: dict, override: Mapping) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config(Mapping):
    """Read-only view of the paper configuration with path helpers."""

    def __init__(self, data: Mapping[str, Any], root: Path, source: Path | None = None):
        self._data = dict(data)
        self.root = Path(root)
        self.source = source

    # Mapping protocol -----------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def section(self, *keys: str) -> Any:
        node: Any = self._data
        for key in keys:
            node = node[key]
        return node

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def hash(self) -> str:
        payload = json.dumps(self._data, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    # Paths ------------------------------------------------------------------
    def resolve(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        return path if path.is_absolute() else (self.root / path)

    def path(self, key: str) -> Path:
        return self.resolve(self._data["paths"][key])

    def _pattern(self, key: str, default: str) -> str:
        return str(self._data.get("patterns", {}).get(key, default))

    def jets_path(self, cls: str | int) -> Path:
        return self.resolve(self._pattern("jets", _paths.JETS_PATTERN).format(
            jets=self._data["paths"]["jets"], cls=class_name(cls)))

    def part_ids_path(self, cls: str | int) -> Path:
        return self.resolve(self._pattern("part_ids", _paths.PART_IDS_PATTERN).format(
            jets=self._data["paths"]["jets"], cls=class_name(cls)))

    def ps_input_path(self, cls: str | int) -> Path:
        return self.resolve(self._pattern("ps_inputs", _paths.PS_INPUTS_PATTERN).format(
            ps_inputs=self._data["paths"]["ps_inputs"], cls=class_name(cls)))

    def block_pattern(self) -> str:
        return self._pattern("block", _paths.BLOCK_PATTERN)

    def block_path(self, metric: str, i: int, j: int, size: str = "large") -> Path:
        i, j = sorted((int(i), int(j)))
        return self.resolve(self.block_pattern().format(
            blocks=self._data["paths"]["blocks"], metric=metric, i=i, j=j, size=size))

    def manifest_path(self, name: str) -> Path:
        return self.path("manifests") / name

    def result_path(self, key: str, **fmt: Any) -> Path:
        return self.resolve(str(self._data["results"][key]).format(**fmt))

    def figure_path(self, name: str) -> Path:
        return self.path("figures") / name


def load_config(
    path: str | Path | None = None,
    *,
    root: str | Path | None = None,
    local: bool = True,
) -> Config:
    """Load ``configs/paper.yaml`` (or ``path``) and merge ``configs/local.yaml``."""
    root_path = Path(root).resolve() if root is not None else _paths.repo_root()
    cfg_path = Path(path) if path is not None else root_path / _paths.DEFAULT_CONFIG_RELPATH
    if not cfg_path.is_absolute():
        cfg_path = root_path / cfg_path
    with cfg_path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if local:
        local_path = root_path / _paths.DEFAULT_LOCAL_CONFIG_RELPATH
        if local_path.exists():
            with local_path.open() as handle:
                data = _deep_merge(data, yaml.safe_load(handle) or {})
    return Config(data, root=root_path, source=cfg_path)
