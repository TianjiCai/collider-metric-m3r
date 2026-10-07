"""Small helpers shared by the paper/ scripts (argument parsing, config, output paths)."""

from __future__ import annotations

import argparse
from pathlib import Path

from m3r import load_config
from m3r.config import Config


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=Path, default=None, help="Path to paper.yaml (default: configs/paper.yaml).")
    parser.add_argument("--root", type=Path, default=None, help="Repository root (default: auto-detected).")
    parser.add_argument("--force", action="store_true", help="Recompute even if the result file exists.")
    return parser


def load_cfg(args: argparse.Namespace) -> Config:
    return load_config(args.config, root=args.root)


def describe(path: Path, cfg: Config) -> str:
    try:
        return str(path.relative_to(cfg.root))
    except ValueError:
        return str(path)
