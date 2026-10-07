"""Canonical JetClass class metadata, paper tagging tasks and display names.

Everything that used to be scattered over per-script ``CLASS_CONFIGS`` /
``EVENT_NAMES`` / ``TAGGING_TASKS`` tables lives here. The class order is the
one used by every stored dataset and distance block (block index ``7`` is
``TTBar``; see ``data/manifests/blocks_manifest.json``).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

CLASS_NAMES: tuple[str, ...] = (
    "QCD",
    "HToBB",
    "HToCC",
    "HToGG",
    "HToWW4Q",
    "ZToQQ",
    "WToQQ",
    "TTBar",
    "HToWW2Q1L",
    "TTBarLep",
)
N_CLASSES = len(CLASS_NAMES)
LABEL_MAP: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

PID_FEATURES: tuple[str, ...] = (
    "part_charge",
    "part_isElectron",
    "part_isMuon",
    "part_isPhoton",
    "part_isChargedHadron",
    "part_isNeutralHadron",
)

KIN_METRICS: tuple[str, ...] = ("w2", "sw", "ps")
METRIC_LABELS: dict[str, str] = {
    "w2": r"$\mathrm{W}_2$",
    "sw": r"$\mathrm{SW}_2$",
    "ps": r"$\mathrm{PS}$",
}
METRIC_SYMBOLS: dict[str, str] = {
    "w2": r"\mathrm{W}_2",
    "sw": r"\mathrm{SW}_2",
    "ps": r"\mathrm{PS}",
}
#: Panel order used by the multi-metric paper figures (Figs. 3 and 6).
METRIC_PANEL_ORDER: tuple[str, ...] = ("ps", "sw", "w2")

#: LaTeX display names (with antiquark bars) for captions and legends.
DISPLAY_NAMES: dict[str, str] = {
    "QCD": r"$\mathrm{QCD}$",
    "HToBB": r"$H \to b\bar{b}$",
    "HToCC": r"$H \to c\bar{c}$",
    "HToGG": r"$H \to gg$",
    "HToWW4Q": r"$H \to 4q$",
    "ZToQQ": r"$Z \to qq$",
    "WToQQ": r"$W \to qq$",
    "TTBar": r"$t \to bqq$",
    "HToWW2Q1L": r"$H \to \ell\nu qq$",
    "TTBarLep": r"$t \to b\ell\nu$",
}
#: Compact variants without antiquark bars, used where space is tight.
DISPLAY_NAMES_COMPACT: dict[str, str] = {
    **DISPLAY_NAMES,
    "HToBB": r"$H \to bb$",
    "HToCC": r"$H \to cc$",
}

_ALIASES: dict[str, str] = {
    "qcd": "QCD",
    "htobb": "HToBB",
    "hbb": "HToBB",
    "htocc": "HToCC",
    "hcc": "HToCC",
    "htogg": "HToGG",
    "hgg": "HToGG",
    "htoww4q": "HToWW4Q",
    "h4q": "HToWW4Q",
    "ztoqq": "ZToQQ",
    "zqq": "ZToQQ",
    "wtoqq": "WToQQ",
    "wqq": "WToQQ",
    "ttbar": "TTBar",
    "tbqq": "TTBar",
    "htoww2q1l": "HToWW2Q1L",
    "h2q1l": "HToWW2Q1L",
    "ttbarlep": "TTBarLep",
}


def class_name(name_or_index: str | int) -> str:
    """Return the canonical class name for an index, name or common alias."""
    if isinstance(name_or_index, (int,)) and not isinstance(name_or_index, bool):
        return CLASS_NAMES[int(name_or_index)]
    text = str(name_or_index)
    if text in LABEL_MAP:
        return text
    key = text.lower()
    if key in _ALIASES:
        return _ALIASES[key]
    if key.isdigit():
        return CLASS_NAMES[int(key)]
    raise KeyError(f"Unknown JetClass class {name_or_index!r}; expected one of {CLASS_NAMES}.")


def class_index(name_or_index: str | int) -> int:
    """Return the canonical class index for an index, name or alias."""
    return LABEL_MAP[class_name(name_or_index)]


def class_indices(names: Iterable[str | int]) -> list[int]:
    return [class_index(name) for name in names]


def pairwise_class_pairs(n_classes: int = N_CLASSES) -> list[tuple[int, int]]:
    """All unordered class pairs ``(i, j)`` with ``i < j``."""
    return list(combinations(range(n_classes), 2))


@dataclass(frozen=True)
class Task:
    """A binary tagging task between two canonical classes (``i < j``)."""

    slug: str
    i: int
    j: int
    label: str

    @property
    def classes(self) -> tuple[int, int]:
        return (self.i, self.j)

    @property
    def names(self) -> tuple[str, str]:
        return (CLASS_NAMES[self.i], CLASS_NAMES[self.j])

    @staticmethod
    def from_classes(a: str | int, b: str | int, slug: str | None = None, label: str | None = None) -> "Task":
        i, j = sorted((class_index(a), class_index(b)))
        if slug is None:
            slug = f"{CLASS_NAMES[i].lower()}_vs_{CLASS_NAMES[j].lower()}"
        if label is None:
            label = f"{DISPLAY_NAMES[CLASS_NAMES[i]]} vs {DISPLAY_NAMES[CLASS_NAMES[j]]}"
        return Task(slug=slug, i=i, j=j, label=label)


#: The six binary tagging tasks used throughout the paper (Figs. 6-8, Table 1),
#: in the order they appear in the figures.
TASKS: tuple[Task, ...] = (
    Task("qcd_vs_ttbar", 0, 7, r"$\mathrm{QCD}$ vs $t \to bqq$"),
    Task("qcd_vs_hww2q1l", 0, 8, r"$\mathrm{QCD}$ vs $H \to \ell\nu qq$"),
    Task("hcc_vs_hgg", 2, 3, r"$H \to c\bar{c}$ vs $H \to gg$"),
    Task("hbb_vs_hww4q", 1, 4, r"$H \to b\bar{b}$ vs $H \to 4q$"),
    Task("hbb_vs_hww2q1l", 1, 8, r"$H \to b\bar{b}$ vs $H \to \ell\nu qq$"),
    Task("wqq_vs_zqq", 5, 6, r"$W \to qq$ vs $Z \to qq$"),
)
TASKS_BY_SLUG: dict[str, Task] = {task.slug: task for task in TASKS}
TASKS_BY_PAIR: dict[tuple[int, int], Task] = {task.classes: task for task in TASKS}


def task(slug_or_pair: str | tuple[int, int]) -> Task:
    if isinstance(slug_or_pair, str):
        return TASKS_BY_SLUG[slug_or_pair]
    i, j = sorted(slug_or_pair)
    return TASKS_BY_PAIR[(i, j)]


def resolve_tasks(spec: Iterable[str] | str | None) -> list[Task]:
    """Resolve a CLI task specification.

    ``None`` or ``"paper"`` gives the six paper tasks; ``"all"`` gives every
    class pair; otherwise each entry is a task slug or ``"A,B"`` class pair.
    """
    if spec is None or spec == "paper" or (not isinstance(spec, str) and list(spec) == ["paper"]):
        return list(TASKS)
    if spec == "all" or (not isinstance(spec, str) and list(spec) == ["all"]):
        return [Task.from_classes(i, j) for i, j in pairwise_class_pairs()]
    entries = [spec] if isinstance(spec, str) else list(spec)
    tasks: list[Task] = []
    for entry in entries:
        if entry in TASKS_BY_SLUG:
            tasks.append(TASKS_BY_SLUG[entry])
        elif "," in entry:
            a, b = entry.split(",", 1)
            i, j = sorted((class_index(a.strip()), class_index(b.strip())))
            tasks.append(TASKS_BY_PAIR.get((i, j), Task.from_classes(i, j)))
        else:
            raise KeyError(f"Unknown task {entry!r}; use a slug from {sorted(TASKS_BY_SLUG)} or 'A,B'.")
    return tasks
