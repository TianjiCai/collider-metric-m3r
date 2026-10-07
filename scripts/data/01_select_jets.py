#!/usr/bin/env python
"""Tier 3, step 1: select and preprocess the 10k jets per class from the raw JetClass files.

For every class the script

1. loads the constituents of the raw JetClass shard (``{Class}_120.root``; QCD from the
   parquet export ``train_file_0.parquet``) with pt, rapidity, phi and mass;
2. drops the jets listed in the class's failed-index file (``data/manifests/selection``):
   seven of the eight original classes share one list (jets whose 20-subjet kt reclustering
   failed in the original study), the two later classes use per-class lists with the
   20-constituent cut (recomputed here if missing), and QCD uses the explicit parquet row
   list ``QCD_rows.npy`` recovered from the published sample (its original selection
   notebook is lost; the rows are the first 10,013 parquet jets minus 13);
3. keeps the first ``events_per_class`` surviving jets, centres them on the jet axis, wraps
   phi and rotates them (``preprocess_kinematic_events``);
4. writes ``data/jets/{Class}_10k.npz`` (``x``: ``(N, 150, 4)`` = pt, y, phi, m; ``y``: label)
   and ``data/jets/{Class}_part_ids_10k.npz`` (six per-particle PID arrays) plus a manifest with
   content digests (SHA-256 over the array bytes, reproducible across runs).

``--verify`` compares the regenerated arrays with the files already present instead of
writing. Needs ``pip install "m3r[gen]"`` and the raw files under data/jetclass/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from m3r import load_config
from m3r.classes import CLASS_NAMES, LABEL_MAP, PID_FEATURES, class_name
from m3r.jetclass import (
    KINEMATIC_FEATURES,
    MAX_PARTICLES,
    build_part_id_payload,
    find_failed_subjet_indices,
    parquet_part_id_payload,
    preprocess_kinematic_events,
    read_jetclass_parquet,
    read_jetclass_root,
    select_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", nargs="+", default=list(CLASS_NAMES))
    parser.add_argument("--selection", choices=["auto", "legacy-shared", "per-class"], default="auto",
                        help="auto: shared list for the original eight classes, per-class otherwise (the published setup).")
    parser.add_argument("--source-dir", type=Path, default=None, help="Raw JetClass directory (default: paths.jetclass).")
    parser.add_argument("--jets-dir", type=Path, default=None, help="Output directory (default: paths.jets).")
    parser.add_argument("--events-per-class", type=int, default=None)
    parser.add_argument("--no-part-ids", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Compare with existing files; write nothing.")
    return parser.parse_args()


def array_digest(path: Path) -> str:
    """SHA-256 of the array contents of an .npz (stable across runs, unlike the zip bytes)."""
    h = hashlib.sha256()
    with np.load(path) as payload:
        for key in sorted(payload.files):
            arr = np.ascontiguousarray(payload[key])
            h.update(f"{key}:{arr.dtype}:{arr.shape}".encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def explicit_rows_for(cfg, cls: str) -> tuple[np.ndarray, str] | None:
    """Explicit row list ``data/manifests/selection/{cls}_rows.npy`` (takes precedence when present)."""
    path = cfg.resolve(str(cfg["dataset"]["selection"].get("explicit_rows", "data/manifests/selection/{cls}_rows.npy")).format(cls=cls))
    if path.exists():
        return np.load(path).astype(np.int64), _rel(cfg, path)
    return None


def _rel(cfg, path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(cfg.root))
    except ValueError:
        return str(path)


def failed_indices_for(cfg, cls: str, selection: str, x_particles, x_jets, source_dir: Path) -> tuple[np.ndarray, str]:
    sel = cfg["dataset"]["selection"]
    shared_classes = [class_name(c) for c in sel["legacy_shared_classes"]]
    use_shared = selection == "legacy-shared" or (selection == "auto" and cls in shared_classes)
    if use_shared:
        path = cfg.resolve(sel["legacy_shared_failed_index"])
        return np.load(path), _rel(cfg, path)
    path = cfg.resolve(str(sel["per_class_failed_index"]).format(cls=cls))
    if path.exists():
        return np.load(path), _rel(cfg, path)
    print(f"  computing per-class failed indices for {cls} (kt reclustering of {len(x_particles)} jets)...", flush=True)
    events = preprocess_kinematic_events(x_particles, x_jets[:, 0], x_jets[:, 1])
    failed = find_failed_subjet_indices(events, n_subjets=int(cfg["dataset"]["ps_inputs"]["n_subjets"]),
                                        radius=float(cfg["dataset"]["ps_inputs"]["radius"]),
                                        min_constituents=int(sel["min_constituents_new_classes"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, failed)
    return failed, _rel(cfg, path)


def main() -> None:
    args = parse_args()
    cfg = load_config()
    source_dir = args.source_dir or cfg.path("jetclass")
    jets_dir = args.jets_dir or cfg.path("jets")
    n_keep = args.events_per_class or int(cfg["dataset"]["events_per_class"])
    shard = int(cfg["dataset"]["source"]["shard"])
    manifest_path = cfg.manifest_path("jets_manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema": 1, "classes": {}}
    for cls in [class_name(c) for c in args.classes]:
        label = LABEL_MAP[cls]
        if cls == "QCD":
            source = source_dir / cfg["dataset"]["source"]["qcd_parquet"]
            x_particles, x_jets, df = read_jetclass_parquet(source, MAX_PARTICLES, KINEMATIC_FEATURES)
        else:
            source = source_dir / str(cfg["dataset"]["source"]["root_pattern"]).format(cls=cls, shard=shard)
            x_particles, x_jets = read_jetclass_root(source, MAX_PARTICLES, KINEMATIC_FEATURES, ("jet_eta", "jet_phi"))
            df = None
        print(f"{cls}: {len(x_particles)} jets in {source.name}", flush=True)
        explicit = explicit_rows_for(cfg, cls)
        if explicit is not None:
            rows, failed_path = explicit
            rows = rows[:n_keep]
            failed = np.setdiff1d(np.arange(rows.max() + 1), rows)
            print(f"  using explicit row list {Path(failed_path).name} ({len(rows)} rows, {len(failed)} skipped)")
        else:
            failed, failed_path = failed_indices_for(cfg, cls, args.selection, x_particles, x_jets, source_dir)
            rows = select_rows(len(x_particles), failed, n_keep)
        x = preprocess_kinematic_events(x_particles[rows], x_jets[rows, 0], x_jets[rows, 1])
        y = np.full(n_keep, label, dtype=np.int64)
        payload = None
        if not args.no_part_ids:
            if df is not None:
                payload = parquet_part_id_payload(df, rows, PID_FEATURES, MAX_PARTICLES)
            else:
                x_pid, _ = read_jetclass_root(source, MAX_PARTICLES, PID_FEATURES, ("jet_eta",))
                payload = build_part_id_payload(x_pid[rows], PID_FEATURES)
        jets_path = cfg.jets_path(cls) if args.jets_dir is None else jets_dir / cfg.jets_path(cls).name
        pid_path = cfg.part_ids_path(cls) if args.jets_dir is None else jets_dir / cfg.part_ids_path(cls).name
        if args.verify:
            if jets_path.exists():
                ref = np.load(jets_path)["x"]
                same = np.array_equal(ref, x)
                print(f"  {jets_path.name}: {'IDENTICAL' if same else 'DIFFERS'} (max |diff| {np.abs(ref.astype(np.float64) - x).max():.3e})")
            if payload is not None and pid_path.exists():
                ref = np.load(pid_path)
                same = all(np.array_equal(ref[k], payload[k]) for k in PID_FEATURES)
                n_kin = (x[:, :, 0] > 0).sum(axis=1)
                n_pid = sum(payload[k] for k in PID_FEATURES[1:]).astype(bool).sum(axis=1)
                print(f"  {pid_path.name}: {'IDENTICAL' if same else 'DIFFERS'}; regenerated PID multiplicity matches kinematics for "
                      f"{int((n_kin == n_pid).sum())}/{n_keep} jets")
            continue
        for path in (jets_path, pid_path if payload is not None else None):
            if path is not None and path.exists() and not args.overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite.")
        jets_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(jets_path, x=x, y=y)
        entry = {"label": label, "source": _rel(cfg, source),
                 "failed_index_file": failed_path, "n_failed": int(len(failed)), "rows_first": int(rows[0]), "rows_last": int(rows[-1]),
                 "jets_file": jets_path.name, "jets_digest": array_digest(jets_path)}
        if payload is not None:
            np.savez(pid_path, **payload)
            entry["part_ids_file"] = pid_path.name
            entry["part_ids_digest"] = array_digest(pid_path)
        manifest["classes"][cls] = entry
        manifest["events_per_class"] = n_keep
        manifest["class_order"] = list(CLASS_NAMES)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"  wrote {jets_path.name}" + (f" and {pid_path.name}" if payload is not None else ""))


if __name__ == "__main__":
    main()
