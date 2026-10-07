"""JetClass I/O and preprocessing for the Tier 3 pipeline (needs ``pip install "m3r[gen]"``).

* ``read_jetclass_root`` / ``read_jetclass_parquet`` - load constituent four-momenta from the
  raw JetClass files (the ROOT reader is adapted from the ParT dataloader, Qu et al. 2022);
* ``preprocess_kinematic_events`` - centre on the jet axis, wrap phi, rotate (ptscheme);
* ``find_failed_subjet_indices`` / ``build_ps_input_payload`` - kt subjets for the PS metric.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from m3r.metrics.ps import transform_ps_to_collinear_frame
from m3r.metrics.w2 import get_clean_phi

KINEMATIC_FEATURES = ("part_pt", "part_eta", "part_phi", "part_mass")
MAX_PARTICLES = 150


def _pad(a, maxlen: int, value: float = 0, dtype: str = "float32"):
    import awkward as ak

    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    if isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.values_astype(a, dtype)
    x = (np.ones((len(a), maxlen)) * value).astype(dtype)
    for idx, s in enumerate(a):
        if not len(s):
            continue
        trunc = s[:maxlen].astype(dtype)
        x[idx, : len(trunc)] = trunc
    return x


def read_jetclass_root(
    filepath: str | Path,
    max_num_particles: int = MAX_PARTICLES,
    particle_features: Sequence[str] = KINEMATIC_FEATURES,
    jet_features: Sequence[str] = ("jet_eta", "jet_phi"),
) -> tuple[np.ndarray, np.ndarray]:
    """Load one JetClass ROOT file.

    Returns ``x_particles`` with shape ``(n_jets, n_features, max_num_particles)`` (zero padded)
    and ``x_jets`` with shape ``(n_jets, n_jet_features)``. ``part_pt/eta/phi/mass`` are derived
    from the stored Cartesian four-momenta.
    """
    import awkward as ak
    import uproot
    import vector

    vector.register_awkward()
    table = uproot.open(str(filepath))["tree"].arrays()
    p4 = vector.zip({"px": table["part_px"], "py": table["part_py"], "pz": table["part_pz"], "energy": table["part_energy"]})
    table["part_pt"], table["part_eta"], table["part_phi"], table["part_mass"] = p4.pt, p4.eta, p4.phi, p4.mass
    x_particles = np.stack([ak.to_numpy(_pad(table[n], maxlen=max_num_particles)) for n in particle_features], axis=1)
    x_jets = np.stack([ak.to_numpy(table[n]).astype("float32") for n in jet_features], axis=1)
    return x_particles, x_jets


def read_jetclass_parquet(
    filepath: str | Path,
    max_num_particles: int = MAX_PARTICLES,
    particle_features: Sequence[str] = KINEMATIC_FEATURES,
) -> tuple[np.ndarray, np.ndarray, "pd.DataFrame"]:
    """Load the QCD jets of the parquet export (same columns as the ROOT files).

    Returns ``x_particles`` ``(n_jets, n_features, max_num_particles)``, ``x_jets`` ``(n_jets, 2)``
    with ``(jet_eta, jet_phi)`` and the DataFrame (for the per-particle PID columns).
    """
    import awkward as ak
    import pandas as pd
    import pyarrow as pa
    import vector

    vector.register_awkward()
    df = pd.read_parquet(filepath)
    jagged = ak.from_arrow(pa.Table.from_pandas(df[["part_px", "part_py", "part_pz", "part_energy"]], preserve_index=False))
    p4 = vector.zip({"px": jagged["part_px"], "py": jagged["part_py"], "pz": jagged["part_pz"], "energy": jagged["part_energy"]})
    columns = {"part_pt": p4.pt, "part_eta": p4.eta, "part_phi": p4.phi, "part_mass": p4.mass}
    x_particles = np.stack([ak.to_numpy(_pad(columns[n], maxlen=max_num_particles)) for n in particle_features], axis=1)
    x_jets = np.stack([df["jet_eta"].values.astype("float32"), df["jet_phi"].values.astype("float32")], axis=1)
    return x_particles, x_jets, df


def parquet_part_id_payload(df, rows: np.ndarray, attr_columns: Sequence[str], max_particles: int = MAX_PARTICLES) -> dict[str, np.ndarray]:
    """Per-particle PID arrays for the selected ``rows`` of the parquet DataFrame."""
    payload: dict[str, np.ndarray] = {}
    for column in attr_columns:
        out = np.zeros((len(rows), max_particles), dtype=np.float32)
        series = df[column]
        for k, row in enumerate(rows):
            values = np.asarray(series.iloc[int(row)], dtype=np.float32)
            out[k, : min(len(values), max_particles)] = values[:max_particles]
        payload[column] = out
    return payload


def select_rows(n_total: int, failed_indices: np.ndarray, n_keep: int) -> np.ndarray:
    """Indices of the first ``n_keep`` jets that are not in ``failed_indices`` (the published selection)."""
    keep = np.ones(n_total, dtype=bool)
    keep[np.asarray(failed_indices, dtype=np.int64)] = False
    rows = np.flatnonzero(keep)
    if rows.size < n_keep:
        raise ValueError(f"Only {rows.size} jets survive the selection, fewer than the requested {n_keep}.")
    return rows[:n_keep]


def preprocess_kinematic_events(
    x_particles: np.ndarray,
    jet_eta: np.ndarray,
    jet_phi: np.ndarray,
) -> np.ndarray:
    import energyflow as ef

    events = np.transpose(x_particles, axes=(0, 2, 1)).astype(np.float32, copy=True)
    events[..., 3] = np.clip(events[..., 3], a_min=0.0, a_max=None)

    for event_idx in range(events.shape[0]):
        active = events[event_idx, :, 0] > 0.0
        if not np.any(active):
            continue

        events[event_idx, active, 1] -= jet_eta[event_idx]
        events[event_idx, active, 2] -= jet_phi[event_idx]
        events[event_idx, active, 2] = get_clean_phi(
            events[event_idx, active, 2], range="unsigned"
        )
        events[event_idx] = ef.rotate_ptyphims(
            events[event_idx], rotate="ptscheme", copy=False
        )
        events[event_idx, active, 2] -= np.pi

    return events


def build_part_id_payload(
    x_particles: np.ndarray, attr_columns: Sequence[str]
) -> dict[str, np.ndarray]:
    events = np.transpose(x_particles, axes=(0, 2, 1)).astype(np.float32, copy=False)
    return {column: events[:, :, idx] for idx, column in enumerate(attr_columns)}


def _cluster_event_subjets(
    event_epxpypz: np.ndarray,
    n_subjets: int,
    radius: float,
) -> tuple[bool, np.ndarray, np.ndarray]:
    import fastjet as fj

    jet_def = fj.JetDefinition(fj.kt_algorithm, radius)
    selector = fj.SelectorPtMin(0.0)

    active = event_epxpypz[event_epxpypz[:, 0] > 0.0]
    pseudojets = [
        fj.PseudoJet(
            float(particle[1]),
            float(particle[2]),
            float(particle[3]),
            float(particle[0]),
        )
        for particle in active
    ]
    cluster = fj.ClusterSequence(pseudojets, jet_def)
    subjets = selector(cluster.exclusive_jets_up_to(n_subjets))
    subjets = fj.sorted_by_pt(subjets)

    if len(subjets) < n_subjets:
        return (
            False,
            np.empty((0, 4), dtype=np.float64),
            np.empty((0, 4), dtype=np.float64),
        )

    subjets_ptyphim = []
    subjets_epxpypz = []
    for subjet in subjets[:n_subjets]:
        subjet_pt = subjet.pt()
        if subjet_pt <= 0.0:
            return (
                False,
                np.empty((0, 4), dtype=np.float64),
                np.empty((0, 4), dtype=np.float64),
            )
        subjets_ptyphim.append([subjet_pt, subjet.rap(), subjet.phi_std(), 0.0])
        energy = math.sqrt(subjet.px() ** 2 + subjet.py() ** 2 + subjet.pz() ** 2)
        subjets_epxpypz.append([energy, subjet.px(), subjet.py(), subjet.pz()])

    subjet_array = np.asarray(subjets_ptyphim, dtype=np.float64)
    if np.any(subjet_array[:, 0] <= 0.0):
        return (
            False,
            np.empty((0, 4), dtype=np.float64),
            np.empty((0, 4), dtype=np.float64),
        )

    return (
        True,
        subjet_array,
        np.asarray(subjets_epxpypz, dtype=np.float64),
    )


def find_failed_subjet_indices(
    events_ptyphim: np.ndarray,
    n_subjets: int = 10,
    radius: float = 0.8,
    min_constituents: int | None = None,
) -> np.ndarray:
    import energyflow as ef

    events_epxpypz = ef.p4s_from_ptyphims(
        np.asarray(events_ptyphim, dtype=np.float64)
    ).astype(np.float64, copy=False)
    failed_indices = []
    for event_idx, event in enumerate(events_epxpypz):
        if min_constituents is not None:
            constituent_count = int(np.sum(event[:, 0] > 0.0))
            if constituent_count < min_constituents:
                failed_indices.append(event_idx)
                continue

        success, _, _ = _cluster_event_subjets(event, n_subjets=n_subjets, radius=radius)
        if not success:
            failed_indices.append(event_idx)
    return np.asarray(failed_indices, dtype=np.int64)


def build_ps_input_payload(
    events_ptyphim: np.ndarray,
    n_subjets: int = 10,
    radius: float = 0.8,
) -> dict[str, np.ndarray]:
    import energyflow as ef

    events_epxpypz = ef.p4s_from_ptyphims(
        np.asarray(events_ptyphim, dtype=np.float64)
    ).astype(np.float64, copy=False)
    subjets_ptyphim = np.zeros(
        (len(events_epxpypz), n_subjets, 4), dtype=np.float64
    )
    subjets_epxpypz = np.zeros(
        (len(events_epxpypz), n_subjets, 4), dtype=np.float64
    )

    for event_idx, event in enumerate(events_epxpypz):
        success, event_ptyphim, event_epxpypz = _cluster_event_subjets(
            event,
            n_subjets=n_subjets,
            radius=radius,
        )
        if not success:
            raise ValueError(
                f"Event {event_idx} failed the {n_subjets}-subjet selection after filtering."
            )
        subjets_ptyphim[event_idx] = event_ptyphim
        subjets_epxpypz[event_idx] = event_epxpypz

    transformed = transform_ps_to_collinear_frame(subjets_epxpypz)
    jet_axes = transformed.sum(axis=1)
    energy = jet_axes[:, 0]
    px = jet_axes[:, 1]
    py = jet_axes[:, 2]
    pz = jet_axes[:, 3]
    mass = np.sqrt(
        np.clip(energy**2 - px**2 - py**2 - pz**2, a_min=0.0, a_max=None)
    )

    return {
        "X_ptyphim": subjets_ptyphim,
        "X_epxpypz": subjets_epxpypz,
        "X_transformed_epxpypz": transformed,
        "E_list": energy.astype(np.float64, copy=False),
        "m_list": mass.astype(np.float64, copy=False),
        "p_minus_list": (energy + pz).astype(np.float64, copy=False),
        "p_plus_list": (energy - pz).astype(np.float64, copy=False),
    }
