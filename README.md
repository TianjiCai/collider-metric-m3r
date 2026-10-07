# collider-metric-comparison

Code and reproduction package for

> **A Unified Geometric Framework for Understanding Collider Event Manifolds**
> Tianji Cai and Hancheng Li

The paper introduces the **multi-reference relative representation (M3R)**: a jet is
represented by its distances to a finite set of reference jets, which puts different
collider metrics (the energy-flow 2-Wasserstein W2, the spectral 2-Wasserstein SW2 and the
phase-space metric PS) on an equal footing. This repository contains the `m3r` library, the
scripts that produce every figure and table of the paper, the Hydra experiments behind
them, and the pipeline that regenerates the data from the public JetClass dataset.

## Installation

```bash
git clone https://github.com/TianjiCai/collider-metric-comparison.git
cd collider-metric-comparison
conda env create -f environment.yml && conda activate m3r     # or any Python >= 3.10 environment
pip install -e .                                               # core: numpy, scipy, scikit-learn, POT, matplotlib, mplhep, hydra
```

Optional extras: `pip install -e ".[gen]"` (ROOT/parquet readers, fastjet, energyflow for
Tier 3), `".[sw]"` (JAX + SPECTER for SW2 distances), `".[nn]"` (torch, for the multi-tower MLP
check of Section 6).

## Three ways to reproduce

| Tier | What you need | What you get | Time |
|---|---|---|---|
| **1** | this repository | every figure and Table 1 from the tracked result files in `results/paper/` | seconds |
| **2** | the processed jets and distance blocks (`data/jets`, `data/blocks`; regenerated with Tier 3 or available from the authors on request, see `data/README.md`) | every analysis recomputed from the distance blocks | minutes to hours per section |
| **3** | the raw JetClass files (`data/jetclass`) and the `[gen]`/`[sw]` extras | jets, PS inputs and all 165 distance blocks regenerated from scratch | days of CPU |

```bash
make figures                 # Tier 1: figures/*.pdf and figures/table1_feature_progression.tex
make fig6                    # one figure
make sec4                    # Tier 2: recompute Section 4 (needs data/blocks) and redraw Figs. 3-5
```

### Tier 2 commands and runtimes (8-core laptop)

| Section | Command | Needs | Runtime |
|---|---|---|---|
| 2 | `python paper/sec2_metrics/fig1_transport_plans.py --recompute` | `data/jets/QCD_10k.npz` | seconds |
| 4.1 | `python paper/sec4_single_manifold/run_intraclass.py [--metrics sw] [--jobs 2]` | diagonal blocks of the 8 classes | JAUC about 1 h per class and metric; MADO about 1 min |
| 4.2 | `python paper/sec4_single_manifold/run_pca_structure.py` | diagonal blocks of QCD, WToQQ, TTBar | about 1 min |
| 5 | `paper/sec5_decision_boundary/run_svm.sh` (`PART=rel|abs|sv`, `METRICS="w2 sw ps"`, `TASKS=paper|all`) | pair blocks of the tasks | 15 min per metric (relative sweep), 5 min per task (Abs), 10 min per metric (support vectors) |
| 6 | `paper/sec6_unified/run_bdt.sh` (`FEATURE_SETS="ps ps+sw ..."`) | pair blocks and `part_ids` | 5-10 min per feature set |

The shell scripts call the Hydra experiments (`python main.py -cn <name> ...`, configs in
`configs/experiments/`) with the settings of `configs/paper.yaml` and then collect the raw outputs
from `results/experiments/` into `results/paper/`. The multi-tower MLP check mentioned in Section 6
is `scripts/extras/mlp_feature_progression.sh` (needs the `[nn]` extra).

### Tier 3 commands

```bash
pip install -e ".[gen,sw]"
python scripts/data/01_select_jets.py                 # jets and particle IDs (--verify compares with existing files)
python scripts/data/02_build_ps_inputs.py             # 10 kt subjets per jet, collinear frame
python scripts/data/03_build_blocks.py --metric w2    # also --metric ps / sw; --pairs i,j; --rows N --check
python scripts/data/04_verify_blocks.py --convention --write
```

Raw inputs under `data/jetclass/`: the JetClass training shard `{Class}_120.root` for the nine ROOT
classes and the JetClass QCD parquet file `train_file_0.parquet` (the QCD selection is a row list into
that file, so use this file and not a ROOT shard). Seven of the eight original classes keep the first 10,000 jets not in the shared failed-index list of
the original study; HToWW2Q1L and TTBarLep use per-class lists with a 20-constituent cut; QCD uses the
explicit row list `QCD_rows.npy`. All lists are tracked under `data/manifests/selection/`, and
`01_select_jets.py --verify` reproduces the published jets and PID arrays bit-for-bit for every class.

The processed jets, PS inputs and the 165 distance blocks (132 GB) are also available from the
authors on request. Block regeneration is a cluster-scale job. Measured on an 8-core laptop: W2 about 3 h per diagonal
and 6 h per off-diagonal block (about two CPU-weeks for all 55), PS about the same, SW2 on CPU about
50 and 95 h per block (the published SW2 blocks were computed on a GPU; `03_build_blocks.py --metric sw`
runs on whatever JAX backend is available). Regenerated W2 blocks agree with the stored ones to about
1e-8, PS blocks of classes 8-9 to 1e-15; the PS inputs of classes 0-7 passed through a 6-decimal text
file originally, so their blocks agree to about 1e-5; CPU-recomputed SW2 distances agree with the
GPU-computed stored values to about 1e-6, except for near-identical jet pairs with tiny distances
where the stored float32 values carry up to 1e-3 absolute error.

## Paper map

| Section | Item | Script (Tier 1) | Analysis behind it (Tier 2) |
|---|---|---|---|
| 2 | Fig. 1 transport plans | `paper/sec2_metrics/fig1_transport_plans.py` | same script with `--recompute` (needs `data/jets/QCD_10k.npz`) |
| 3 | Fig. 2 M3R schematic | `paper/sec3_m3r/fig2_m3r_geometry.py` | toy model, no data |
| 4.1 | Fig. 3 JAUC and MADO vs N_ref | `paper/sec4_single_manifold/fig3_jauc_mado.py` | `paper/sec4_single_manifold/run_intraclass.py` |
| 4.2 | Fig. 4 PCA alignment, Fig. 5 PC1 variance transfer | `fig4_pca_alignment.py`, `fig5_ev_transfer.py` | `paper/sec4_single_manifold/run_pca_structure.py` |
| 5.1 | Fig. 6 SVM AUC vs N_ref | `paper/sec5_decision_boundary/fig6_svm_curves.py` | `paper/sec5_decision_boundary/run_svm.sh` (Hydra experiments `rel_rep_pairwise_svm`, `abs_pairwise_svm`) |
| 5.2 | Fig. 7 support-vector fractions | `paper/sec5_decision_boundary/fig7_sv_fractions.py` | `run_svm.sh` (`rel_rep_pairwise_svm mode=study`) |
| 6 | Fig. 8 and Table 1 feature progression | `paper/sec6_unified/fig8_unified_repr.py` | `paper/sec6_unified/run_bdt.sh` (`rel_rep_pairwise_bdt`) |

Every number used in the paper (classes, tagging tasks, reference counts, seeds, SVM and
BDT hyperparameters, output paths) lives in **`configs/paper.yaml`**. The figure scripts read
`results/paper/`; the analysis scripts and experiments write there, each result file with a
`*.provenance.json` sidecar recording the code version and configuration hash.

## Layout

```
m3r/            library: classes and tasks, block loader, M3R construction, JAUC/MADO, PCA structure,
                SVM and BDT helpers, metrics (W2, SW2, PS), JetClass preprocessing, plotting style
paper/          one folder per paper section: run_*.py / run_*.sh (analyses) and fig*.py (figures)
experiments/    Hydra experiments behind Sections 5 and 6 (python main.py -cn <name> ...)
configs/        paper.yaml (paper settings) and experiments/ (Hydra configs)
scripts/data/   Tier 3: select jets, build PS inputs, compute distance blocks, verify
scripts/extras/ multi-tower MLP check (Section 6 footnote)
results/paper/  small tracked result files that the figure scripts consume
figures/        the paper PDFs (regenerated by `make figures`)
data/           manifests (tracked) and the data payloads (not tracked; see data/README.md)
```

## Conventions worth knowing

* **Class order** (indices used everywhere): QCD, HToBB, HToCC, HToGG, HToWW4Q, ZToQQ, WToQQ, TTBar,
  HToWW2Q1L, TTBarLep.
* **Distance blocks** are `(10000, 10000)` float64 arrays `data/blocks/{metric}/{metric}_block_{i}_{j}.npy`
  with `i <= j`. The published SW2 blocks store *squared* distances; `data/manifests/blocks_manifest.json`
  records this and `m3r.blocks.BlockStore` converts to distances, so no other code needs to know.
* **References** are drawn with `m3r.relrep.sample_anchors` (seed 42), which reproduces the published draws exactly.
* **Seeds and splits**: MADO uses `default_rng(42 + class_index)` for anchors with 20,000 sampled pairs and 128
  probes; the relative-space SVM sweep uses `SVC(rbf, C=1, gamma='scale')` on an 80/20 permutation split; the
  Abs benchmark selects `gamma` and `C` on a stratified 64/16/20 split; the support-vector study uses 10 references
  per class with `C=10`; the BDT is `HistGradientBoostingClassifier` (lr 0.05, 31 leaves, 50 min samples per leaf,
  L2 1, 255 bins, early stopping after 20 of at most 2000 iterations, `random_state=42`) on a stratified 80/20 split
  with 20 references per class and feature blocks ordered PS, SW2, W2, PID.

## Citing

See `CITATION.cff`. The JetClass dataset is described in Qu, Li and Qian, *Particle Transformer for Jet
Tagging* (ICML 2022) and available at <https://doi.org/10.5281/zenodo.6619768>.
