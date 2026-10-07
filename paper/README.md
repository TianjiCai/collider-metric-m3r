# paper/ — one folder per section

| Folder | Analyses (Tier 2) | Figures (Tier 1) |
|---|---|---|
| `sec2_metrics/` | `fig1_transport_plans.py --recompute` | `fig1_transport_plans.py` -> `w2_plan.pdf`, `sw2_plan.pdf` |
| `sec3_m3r/` | — | `fig2_m3r_geometry.py` -> `m3r_geometry.pdf` |
| `sec4_single_manifold/` | `run_intraclass.py`, `run_pca_structure.py` | `fig3_jauc_mado.py`, `fig4_pca_alignment.py`, `fig5_ev_transfer.py` |
| `sec5_decision_boundary/` | `run_svm.sh` (+ `collect_svm.py`) | `fig6_svm_curves.py`, `fig7_sv_fractions.py` |
| `sec6_unified/` | `run_bdt.sh` (+ `collect_bdt.py`) | `fig8_unified_repr.py` (also writes Table 1) |

All scripts take `--config` (default `configs/paper.yaml`) and `--help`. Results are read from
and written to `results/paper/sec*/`; figures go to `figures/`. See the README for the three reproduction tiers.
