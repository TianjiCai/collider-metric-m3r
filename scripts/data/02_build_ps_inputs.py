#!/usr/bin/env python
"""Tier 3, step 2: recluster each jet into 10 exclusive kt subjets for the phase-space metric.

Reads data/jets/{Class}_10k.npz, clusters the constituents with the kt algorithm (R = 0.8),
keeps the 10 hardest exclusive subjets as massless four-vectors, boosts them to the
collinear frame and writes data/ps_inputs/{Class}_ps_10k.npz with
``X_ptyphim``, ``X_epxpypz``, ``X_transformed_epxpypz`` (``(N, 10, 4)``) and the per-jet
``E_list``, ``m_list``, ``p_minus_list``, ``p_plus_list``.

``--verify`` compares ``X_transformed_epxpypz`` with an existing file (``allclose``, rtol 1e-5;
the published inputs for the eight original classes passed through a 6-decimal text file, so
they are not byte-identical). Needs ``pip install "m3r[gen]"`` (fastjet, energyflow).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from m3r import load_config
from m3r.classes import CLASS_NAMES, class_name
from m3r.jetclass import build_ps_input_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", nargs="+", default=list(CLASS_NAMES))
    parser.add_argument("--out-dir", type=Path, default=None, help="Default: paths.ps_inputs")
    parser.add_argument("--rows", type=int, default=None, help="Only the first N jets (spot checks).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    ps_cfg = cfg["dataset"]["ps_inputs"]
    manifest_path = cfg.manifest_path("ps_inputs_manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema": 1, "classes": {}}
    for cls in [class_name(c) for c in args.classes]:
        x = np.load(cfg.jets_path(cls))["x"]
        if args.rows:
            x = x[: args.rows]
        counts = (x[:, :, 0] > 0).sum(axis=1)
        print(f"{cls}: clustering {len(x)} jets (min constituents {int(counts.min())})", flush=True)
        payload = build_ps_input_payload(x, n_subjets=int(ps_cfg["n_subjets"]), radius=float(ps_cfg["radius"]))
        out = cfg.ps_input_path(cls) if args.out_dir is None else args.out_dir / cfg.ps_input_path(cls).name
        if args.verify:
            if not out.exists():
                print(f"  {out} missing; nothing to compare"); continue
            ref = np.load(out)["X_transformed_epxpypz"][: len(x)]
            new = payload["X_transformed_epxpypz"]
            diff = np.abs(ref - new); rel = diff / np.maximum(np.abs(ref), 1e-12)
            print(f"  {out.name}: array_equal={np.array_equal(ref, new)} allclose(rtol=1e-5)={np.allclose(ref, new, rtol=1e-5, atol=1e-8)} "
                  f"max|diff|={diff.max():.3e} max rel={rel.max():.3e}")
            continue
        if out.exists() and not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite.")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **payload)
        manifest["classes"][cls] = {"file": out.name, "n_events": int(len(x)), "n_subjets": int(ps_cfg["n_subjets"]),
                                    "radius": float(ps_cfg["radius"]), "algorithm": str(ps_cfg["algorithm"])}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
