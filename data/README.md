# data/

Only the manifests are tracked. The payloads are regenerated from the public JetClass files with the
Tier 3 scripts (`scripts/data/`, see the README for runtimes) or can be requested from the authors.

| Directory | Contents | Size | Source |
|---|---|---|---|
| `manifests/` | `blocks_manifest.json` (block conventions, checksums), `jets_manifest.json` (selection provenance and content digests), `selection/` (failed-index lists and the QCD row list) | < 2 MB | tracked |
| `jets/` | `{Class}_10k.npz` (`x`: `(10000, 150, 4)` pt, y, phi, m after centring and rotation; `y`: label) and `{Class}_part_ids_10k.npz` (six `(10000, 150)` per-particle arrays) | 0.6 GB | `01_select_jets.py` |
| `ps_inputs/` | `{Class}_ps_10k.npz`: 10 kt subjets per jet (`X_transformed_epxpypz` is the PS metric input) | 0.1 GB | `02_build_ps_inputs.py` |
| `blocks/{w2,sw,ps}/` | `{metric}_block_{i}_{j}.npy`, `i <= j`, 55 per metric, `(10000, 10000)` float64 | 44 GB each | `03_build_blocks.py` (or on request) |
| `jetclass/` | JetClass training shard `{Class}_120.root` for the nine ROOT classes and the JetClass QCD parquet file `train_file_0.parquet` | ~2 GB | <https://doi.org/10.5281/zenodo.6619768> |

Conventions: class indices follow `QCD, HToBB, HToCC, HToGG, HToWW4Q, ZToQQ, WToQQ, TTBar, HToWW2Q1L, TTBarLep`;
the SW2 blocks store squared distances (`stored_power: 2` in the manifest), W2 and PS store distances.
`scripts/data/04_verify_blocks.py --convention` checks the convention against freshly computed distances.
