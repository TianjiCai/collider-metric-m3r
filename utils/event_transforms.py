from __future__ import annotations

import math

import numpy as np


def build_sw_events(kinematic_events: np.ndarray) -> np.ndarray:
    sw_events = np.array(kinematic_events[..., :3], dtype=np.float32, copy=True)
    pt_sums = sw_events[..., 0].sum(axis=1, keepdims=True)
    np.divide(
        sw_events[..., 0],
        pt_sums,
        out=sw_events[..., 0],
        where=pt_sums > 0.0,
    )
    return sw_events


def transform_ps_to_collinear_frame(x_epxpypz: np.ndarray) -> np.ndarray:
    transformed = np.zeros_like(x_epxpypz, dtype=np.float32)

    for event_idx in range(x_epxpypz.shape[0]):
        event = x_epxpypz[event_idx]
        energy = event[:, 0].sum(dtype=np.float64)
        px_jet = event[:, 1].sum(dtype=np.float64)
        py_jet = event[:, 2].sum(dtype=np.float64)
        pz_jet = event[:, 3].sum(dtype=np.float64)

        p_jet = math.sqrt(max(px_jet**2 + py_jet**2 + pz_jet**2, 0.0))
        if p_jet == 0.0:
            raise ValueError(f"Jet {event_idx} has zero total momentum.")

        cos_theta = pz_jet / p_jet
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        sin_theta = math.sqrt(max(1.0 - cos_theta**2, 0.0))

        xy_norm = math.sqrt(px_jet**2 + py_jet**2)
        if xy_norm == 0.0:
            ux, uy = 1.0, 0.0
        else:
            ux = py_jet / xy_norm
            uy = -px_jet / xy_norm

        for constituent_idx in range(event.shape[0]):
            px = float(event[constituent_idx, 1])
            py = float(event[constituent_idx, 2])
            pz = float(event[constituent_idx, 3])

            px_new = (
                (cos_theta + ux**2 * (1.0 - cos_theta)) * px
                + ux * uy * (1.0 - cos_theta) * py
                + uy * sin_theta * pz
            )
            py_new = (
                ux * uy * (1.0 - cos_theta) * px
                + (cos_theta + uy**2 * (1.0 - cos_theta)) * py
                - ux * sin_theta * pz
            )
            pz_new = -uy * sin_theta * px + ux * sin_theta * py + cos_theta * pz

            transformed[event_idx, constituent_idx] = np.array(
                [event[constituent_idx, 0], px_new, py_new, pz_new],
                dtype=np.float32,
            )

    return transformed
