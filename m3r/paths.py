"""Repository-root discovery and canonical on-disk layout helpers."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root.

    Set ``M3R_ROOT`` to point the library at a checkout or data root that is
    not the one the package was imported from.
    """
    env = os.environ.get("M3R_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


DEFAULT_CONFIG_RELPATH = Path("configs") / "paper.yaml"
DEFAULT_LOCAL_CONFIG_RELPATH = Path("configs") / "local.yaml"

JETS_PATTERN = "{jets}/{cls}_10k.npz"
PART_IDS_PATTERN = "{jets}/{cls}_part_ids_10k.npz"
PS_INPUTS_PATTERN = "{ps_inputs}/{cls}_ps_10k.npz"
BLOCK_PATTERN = "{blocks}/{metric}/{metric}_block_{i}_{j}.npy"
