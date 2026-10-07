"""m3r: multi-reference relative representations of collider event metrics.

Companion library for *A Unified Geometric Framework for Understanding
Collider Event Manifolds*. Importing ``m3r`` pulls in only NumPy; heavier
dependencies (POT, scikit-learn, JAX/pyspecter, torch) are imported by the
submodules that need them.
"""

from __future__ import annotations

__version__ = "1.0.0"

from m3r.classes import (  # noqa: E402
    CLASS_NAMES,
    DISPLAY_NAMES,
    KIN_METRICS,
    LABEL_MAP,
    METRIC_LABELS,
    PID_FEATURES,
    TASKS,
    Task,
    class_index,
    class_name,
    pairwise_class_pairs,
)
from m3r.config import Config, load_config  # noqa: E402
from m3r.paths import repo_root  # noqa: E402

__all__ = [
    "CLASS_NAMES",
    "DISPLAY_NAMES",
    "KIN_METRICS",
    "LABEL_MAP",
    "METRIC_LABELS",
    "PID_FEATURES",
    "TASKS",
    "Task",
    "Config",
    "class_index",
    "class_name",
    "load_config",
    "pairwise_class_pairs",
    "repo_root",
    "__version__",
]
