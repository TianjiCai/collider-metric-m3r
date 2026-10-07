#!/usr/bin/env bash
# Section 6 experiments (inputs of Fig. 8 and Table 1): one BDT run per feature set listed in
# configs/paper.yaml (bdt.feature_sets), then collection into results/paper/sec6/.
#
#   paper/sec6_unified/run_bdt.sh                     # all feature sets, six paper tasks
#   FEATURE_SETS="ps ps+sw" TASKS=all paper/sec6_unified/run_bdt.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
FEATURE_SETS=${FEATURE_SETS:-$($PY paper/_cfg.py bdt.feature_sets)}
TASKS=${TASKS:-paper}
A=$($PY paper/_cfg.py bdt.anchors_per_class)
for fs in $FEATURE_SETS; do
  $PY main.py -cn rel_rep_pairwise_bdt "feature_set=$fs" anchor_per_class=$A tasks=$TASKS
done
$PY paper/sec6_unified/collect_bdt.py
