#!/usr/bin/env bash
# Section 5 experiments (inputs of Figs. 6 and 7). Runs the Hydra experiments with the
# paper settings from configs/paper.yaml and collects the results into results/paper/sec5/.
#
#   paper/sec5_decision_boundary/run_svm.sh            # everything (hours)
#   PART=rel METRICS="sw" paper/sec5_decision_boundary/run_svm.sh   # one part / metric subset
#
# PART: rel (relative-space sweep, Fig. 6) | abs (absolute-space baseline, Fig. 6 "Abs")
#       | sv (support-vector study, Fig. 7) | collect | all
# TASKS: paper (six tasks, default) | all (45 pairs) | task slugs
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
PART=${PART:-all}
METRICS=${METRICS:-$($PY paper/_cfg.py metrics.kinematic)}
TASKS=${TASKS:-paper}

if [[ "$PART" == "rel" || "$PART" == "all" ]]; then
  ANCHORS=$($PY paper/_cfg.py svm_relrep.anchors_per_class)
  C=$($PY paper/_cfg.py svm_relrep.svc.C); GAMMA=$($PY paper/_cfg.py svm_relrep.svc.gamma)
  for m in $METRICS; do for a in $ANCHORS; do
    $PY main.py -cn rel_rep_pairwise_svm mode=fixed metric=$m anchor_per_class=$a tasks=$TASKS svm.C=$C svm.gamma=$GAMMA
  done; done
fi
if [[ "$PART" == "abs" || "$PART" == "all" ]]; then
  for m in $METRICS; do
    $PY main.py -cn abs_pairwise_svm metric=$m tasks=$TASKS
  done
fi
if [[ "$PART" == "sv" || "$PART" == "all" ]]; then
  A=$($PY paper/_cfg.py svm_sv.anchors_per_class); C=$($PY paper/_cfg.py svm_sv.rel.C); GAMMA=$($PY paper/_cfg.py svm_sv.rel.gamma)
  for m in $METRICS; do
    $PY main.py -cn rel_rep_pairwise_svm mode=study metric=$m anchor_per_class=$A tasks=$TASKS "study.C_grid=[$C]" "study.gamma_grid=[$GAMMA]"
  done
fi
$PY paper/sec5_decision_boundary/collect_svm.py
