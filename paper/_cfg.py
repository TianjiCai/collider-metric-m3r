#!/usr/bin/env python
"""Print a value from configs/paper.yaml for use in shell scripts: python paper/_cfg.py svm_relrep.anchors_per_class"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m3r import load_config  # noqa: E402

cfg = load_config()
node = cfg
for key in sys.argv[1].split("."):
    node = node[key]
if isinstance(node, (list, tuple)):
    print(" ".join(str(x) for x in node))
elif isinstance(node, dict):
    print(" ".join(str(k) for k in node))
else:
    print(node)
