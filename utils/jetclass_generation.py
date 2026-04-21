from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import energyflow as ef
import fastjet as fj
import numpy as np
import pandas as pd

from utils.event_transforms import transform_ps_to_collinear_frame
from utils.ot_utils import get_clean_phi


def preprocess_kinematic_events(
    x_particles: np.ndarray,
    jet_eta: np.ndarray,
    jet_phi: np.ndarray,
) -> np.ndarray:
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


def load_qcd_part_id_payload(parquet_path: Path, num_jets: int) -> dict[str, np.ndarray]:
    df = pd.read_parquet(parquet_path)
    attr_columns = (
        "part_charge",
        "part_isElectron",
        "part_isMuon",
        "part_isPhoton",
        "part_isChargedHadron",
        "part_isNeutralHadron",
    )

    max_particles = 150
    payload: dict[str, np.ndarray] = {}
    for column in attr_columns:
        output = np.zeros((num_jets, max_particles), dtype=np.float32)
        for row_idx, jet in enumerate(df[column].iloc[:num_jets]):
            values = np.asarray(jet, dtype=np.float32)
            output[row_idx, : len(values)] = values[:max_particles]
        payload[column] = output
    return payload


def _cluster_event_subjets(
    event_epxpypz: np.ndarray,
    n_subjets: int,
    radius: float,
) -> tuple[bool, np.ndarray, np.ndarray]:
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
        return False, np.empty((0, 4), dtype=np.float32), np.empty((0, 4), dtype=np.float32)

    subjets_ptyphim = []
    subjets_epxpypz = []
    for subjet in subjets[:n_subjets]:
        subjet_pt = subjet.pt()
        if subjet_pt <= 0.0:
            return (
                False,
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 4), dtype=np.float32),
            )
        subjets_ptyphim.append([subjet_pt, subjet.rap(), subjet.phi_std(), 0.0])
        energy = math.sqrt(subjet.px() ** 2 + subjet.py() ** 2 + subjet.pz() ** 2)
        subjets_epxpypz.append([energy, subjet.px(), subjet.py(), subjet.pz()])

    subjet_array = np.asarray(subjets_ptyphim, dtype=np.float32)
    if np.any(subjet_array[:, 0] <= 0.0):
        return False, np.empty((0, 4), dtype=np.float32), np.empty((0, 4), dtype=np.float32)

    return (
        True,
        subjet_array,
        np.asarray(subjets_epxpypz, dtype=np.float32),
    )


def find_failed_subjet_indices(
    events_ptyphim: np.ndarray,
    n_subjets: int = 10,
    radius: float = 0.8,
) -> np.ndarray:
    events_epxpypz = ef.p4s_from_ptyphims(events_ptyphim).astype(np.float32, copy=False)
    failed_indices = []
    for event_idx, event in enumerate(events_epxpypz):
        success, _, _ = _cluster_event_subjets(event, n_subjets=n_subjets, radius=radius)
        if not success:
            failed_indices.append(event_idx)
    return np.asarray(failed_indices, dtype=np.int64)


def build_ps_input_payload(
    events_ptyphim: np.ndarray,
    n_subjets: int = 10,
    radius: float = 0.8,
) -> dict[str, np.ndarray]:
    events_epxpypz = ef.p4s_from_ptyphims(events_ptyphim).astype(np.float32, copy=False)
    subjets_ptyphim = np.zeros((len(events_epxpypz), n_subjets, 4), dtype=np.float32)
    subjets_epxpypz = np.zeros((len(events_epxpypz), n_subjets, 4), dtype=np.float32)

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
    mass = np.sqrt(np.clip(energy**2 - px**2 - py**2 - pz**2, a_min=0.0, a_max=None))

    return {
        "X_ptyphim": subjets_ptyphim,
        "X_epxpypz": subjets_epxpypz,
        "X_transformed_epxpypz": transformed,
        "E_list": energy.astype(np.float32, copy=False),
        "m_list": mass.astype(np.float32, copy=False),
        "p_minus_list": (energy + pz).astype(np.float32, copy=False),
        "p_plus_list": (energy - pz).astype(np.float32, copy=False),
    }
