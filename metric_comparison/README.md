# Metric Comparison Workspace

This folder contains exploratory and paper-plot support scripts that sit outside
the Hydra experiment runner in `experiments/`.

## Layout

- `intraclass_comp/`: shared intraclass analysis scripts that can run across
  multiple metrics. New reusable scripts should usually start here.
- `w2/`, `ps/`, `sw/`: metric-specific analysis workspaces.
  - `intraclass_comp/`: historical per-class notebooks, metric-specific
    intraclass scripts, and generated `output/` summaries.
  - `interclass_comp/`: interclass notebooks and pairwise comparison scripts.
- `w2/approximation_test/`: W2-specific relative-representation approximation
  diagnostics.
- `w2/mlp/`: older W2 multiclass MLP experiments and artifacts.

Generated arrays, plots, and CSV summaries are intentionally kept in each
metric's local `output/` directory and are ignored by the repository-wide
`.gitignore`.

## Current Shared Scripts

- `intraclass_comp/intraclass_jaccard_anchors.py` computes intraclass
  neighborhood-preservation Jaccard curves and AUCs for W2, PS, and SW. It
  writes per-metric summaries to
  `<metric>/intraclass_comp/output/jaccard/intraclass_jaccard_summary.csv`.
- `intraclass_comp/intraclass_decision_ordering_anchors.py` computes the
  decision-ordering similarity progression for each class as the number of
  anchors increases. It writes per-metric summaries to
  `<metric>/intraclass_comp/output/decision_ordering/intraclass_decision_ordering_summary.csv`.
- `intraclass_comp/intraclass_mds_gram_diagnostics.py` computes the classical
  MDS Gram matrix `B = -0.5 J D^2 J` for intraclass distance blocks and reports
  PSD/negative-spectrum/effective-dimension diagnostics. It writes per-metric
  summaries to
  `<metric>/intraclass_comp/output/mds_gram/intraclass_mds_gram_summary.csv`.

The corresponding paper plot script lives in:

- `paper_plots/complexity_estimation/decision_ordering_progression_all_metrics.py`
