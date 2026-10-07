#!/usr/bin/env bash
# Robustness check mentioned in Sec. 6: the multi-tower MLP on the same cumulative feature sets
# as the BDT (one tower per M3R block, fused by a joint head). Needs `pip install -e ".[nn]"`.
#   scripts/extras/mlp_feature_progression.sh                      # paper feature sets, six tasks
#   FEATURE_SETS="ps ps+sw" EPOCHS=200 scripts/extras/mlp_feature_progression.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
FEATURE_SETS=${FEATURE_SETS:-$($PY paper/_cfg.py bdt.feature_sets)}
TASKS=${TASKS:-paper}
A=$($PY paper/_cfg.py extras.mlp.anchors_per_class)
EPOCHS=${EPOCHS:-$($PY paper/_cfg.py extras.mlp.training.epochs)}
for fs in $FEATURE_SETS; do
  features="[$(echo "$fs" | tr '+' ',')]"
  $PY main.py -cn rel_rep_pairwise_nn "features=$features" anchor_per_class=$A tasks=$TASKS training.epochs=$EPOCHS
done
echo "results under results/experiments/rel_rep_pairwise_nn/<feature-set>/a$A/results/"
