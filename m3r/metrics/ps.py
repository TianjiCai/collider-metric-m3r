"""Relativistic N-body phase-space (PS) distance for boosted jets (collinear formulation)."""

from __future__ import annotations

import math

import numpy as np


def PSdistance(
    pN_evt1,
    pN_evt2,
    E_plus_pz_onSimplex=False,
    Sphere2Simplex=4.0,
    l_has_chi=False,
    Chi2Simplex=16.0 * np.pi**2,
    return_only_l=False,
):
    """
    Compute the custom phase-space distance between two events/jets.

    This routine decomposes each event into three pieces:

    - a simplex coordinate `rho`, built from normalized light-cone energies,
    - a sphere coordinate `v'`, which captures transverse geometry,
    - an optional event-level `chi` coordinate that measures longitudinal
      imbalance / boost information.

    The final distance is a weighted combination of those pieces. Unlike the
    OT-based routines above, this function does not solve a matching problem;
    it assumes the two inputs already have the same number of particles and a
    consistent constituent ordering.

    Parameters
    ----------
    pN_evt1, pN_evt2 : ndarray, shape (N, 4)
        Per-particle four-momenta `(E, px, py, pz)` for the two events. The
        events may be in any frame as long as the total momentum is along `+/-z`.
    E_plus_pz_onSimplex : bool, default=False
        If `True`, define the simplex coordinate from `E + pz`; otherwise use
        `E - pz`.
    Sphere2Simplex : float, default=4.0
        Relative weight of the sphere contribution with respect to the simplex
        contribution in the total distance.
    l_has_chi : bool, default=False
        Whether to include the event-level `chi` contribution in the total
        distance.
    Chi2Simplex : float, default=16*pi^2
        Relative weight of the `chi` term with respect to the simplex term.
        Only used when `l_has_chi=True`.
    return_only_l : bool, default=False
        If `True`, return only the combined phase-space distance.

    Returns
    -------
    float
        Returned when `return_only_l=True`.
    tuple[float, float, float, float]
        `(l, l_simplex, l_sphere, l_chi)` otherwise.
    """

    # Event-level totals set the normalization of the phase-space coordinates.
    N = pN_evt1.shape[0]
    E_evt1, E_evt2 = (
        np.sum(pN_evt1[:, 0]),
        np.sum(pN_evt2[:, 0]),
    )  # event/jet total energy
    pz_evt1, pz_evt2 = (
        np.sum(pN_evt1[:, 3]),
        np.sum(pN_evt2[:, 3]),
    )  # event/jet total pz, should be 0 in CM frame
    chi_evt1, chi_evt2 = (
        (E_evt1 - pz_evt1) / (E_evt1 + pz_evt1),
        (E_evt2 - pz_evt2) / (E_evt2 + pz_evt2),
    )
    # In the collinear limit, E_evt + pz_evt should equal 2 * E and E_evt - pz_evt should be m**2 / (2 * E)

    # Build the phase-space coordinates `(u, v)` from light-cone components.
    # `u` controls the simplex part and `v` is a complex transverse coordinate.
    if E_plus_pz_onSimplex == True:
        uN_evt1 = np.sqrt(1 / (E_evt1 + pz_evt1)) * np.transpose(
            np.array([np.sqrt(p[0] + p[3]) for p in pN_evt1])
        )
        uN_evt2 = np.sqrt(1 / (E_evt2 + pz_evt2)) * np.transpose(
            np.array([np.sqrt(p[0] + p[3]) for p in pN_evt2])
        )
        vN_evt1 = np.sqrt(1 / (E_evt1 - pz_evt1)) * np.transpose(
            np.array([(p[1] + 1j * (p[2])) / np.sqrt(p[0] + p[3]) for p in pN_evt1])
        )
        vN_evt2 = np.sqrt(1 / (E_evt2 - pz_evt2)) * np.transpose(
            np.array([(p[1] + 1j * (p[2])) / np.sqrt(p[0] + p[3]) for p in pN_evt2])
        )
    else:
        uN_evt1 = np.sqrt(1 / (E_evt1 - pz_evt1)) * np.transpose(
            np.array([np.sqrt(p[0] - p[3]) for p in pN_evt1])
        )
        uN_evt2 = np.sqrt(1 / (E_evt2 - pz_evt2)) * np.transpose(
            np.array([np.sqrt(p[0] - p[3]) for p in pN_evt2])
        )
        vN_evt1 = np.sqrt(1 / (E_evt1 + pz_evt1)) * np.transpose(
            np.array([(p[1] + 1j * (p[2])) / np.sqrt(p[0] - p[3]) for p in pN_evt1])
        )
        vN_evt2 = np.sqrt(1 / (E_evt2 + pz_evt2)) * np.transpose(
            np.array([(p[1] + 1j * (p[2])) / np.sqrt(p[0] - p[3]) for p in pN_evt2])
        )

    # `rho = u^2` lies on a simplex and is used for the Euclidean simplex
    # distance. Keep `r = u` as well because the reduced-sphere formulas are
    # written in terms of square roots of the simplex coordinates.
    rhoN_evt1 = np.array([u**2 for u in uN_evt1])
    rhoN_evt2 = np.array([u**2 for u in uN_evt2])
    rN_evt1 = np.array([u for u in uN_evt1])
    rN_evt2 = np.array([u for u in uN_evt2])

    # Map the `v` coordinates to reduced coordinates `v'` that live on the
    # sphere factor of the phase-space decomposition.
    if N == 2:
        # For two-particle events the reduced coordinate has a simple closed
        # form, so we can skip the general nested-loop construction.
        vprimeNminus1_evt1 = 1 / np.sqrt(rhoN_evt1[1]) * vN_evt1[: N - 1]
        vprimeNminus1_evt2 = 1 / np.sqrt(rhoN_evt2[1]) * vN_evt2[: N - 1]

    else:  # N>=3
        # For larger multiplicities, build each reduced component explicitly
        # from the published formula used in this metric.
        vprimeNminus1_evt1 = []
        for i in range(N - 1):
            vprime_i_real_evt1 = 0
            vprime_i_imag_evt1 = 0
            for j in range(N - 1):
                if i == j:
                    vprime_i_real_evt1 += (
                        rN_evt1[i] ** 2
                        + rN_evt1[-1] * (1 - rN_evt1[i] ** 2 - rN_evt1[-1] ** 2)
                    ) * vN_evt1[i].real
                    vprime_i_imag_evt1 += (
                        rN_evt1[i] ** 2
                        + rN_evt1[-1] * (1 - rN_evt1[i] ** 2 - rN_evt1[-1] ** 2)
                    ) * vN_evt1[i].imag
                else:
                    vprime_i_real_evt1 += (
                        rN_evt1[i] * rN_evt1[j] * (1 - rN_evt1[-1]) * vN_evt1[j].real
                    )
                    vprime_i_imag_evt1 += (
                        rN_evt1[i] * rN_evt1[j] * (1 - rN_evt1[-1]) * vN_evt1[j].imag
                    )
            vprime_i_real_evt1 /= rN_evt1[-1] * (1 - rN_evt1[-1] ** 2)
            vprime_i_imag_evt1 /= rN_evt1[-1] * (1 - rN_evt1[-1] ** 2)
            vprime_i_evt1 = vprime_i_real_evt1 + 1j * vprime_i_imag_evt1
            vprimeNminus1_evt1.append(vprime_i_evt1)
        vprimeNminus1_evt1 = np.array(vprimeNminus1_evt1)

        vprimeNminus1_evt2 = []
        for i in range(N - 1):
            vprime_i_real_evt2 = 0
            vprime_i_imag_evt2 = 0
            for j in range(N - 1):
                if i == j:
                    vprime_i_real_evt2 += (
                        rN_evt2[i] ** 2
                        + rN_evt2[-1] * (1 - rN_evt2[i] ** 2 - rN_evt2[-1] ** 2)
                    ) * vN_evt2[i].real
                    vprime_i_imag_evt2 += (
                        rN_evt2[i] ** 2
                        + rN_evt2[-1] * (1 - rN_evt2[i] ** 2 - rN_evt2[-1] ** 2)
                    ) * vN_evt2[i].imag
                else:
                    vprime_i_real_evt2 += (
                        rN_evt2[i] * rN_evt2[j] * (1 - rN_evt2[-1]) * vN_evt2[j].real
                    )
                    vprime_i_imag_evt2 += (
                        rN_evt2[i] * rN_evt2[j] * (1 - rN_evt2[-1]) * vN_evt2[j].imag
                    )
            vprime_i_real_evt2 /= rN_evt2[-1] * (1 - rN_evt2[-1] ** 2)
            vprime_i_imag_evt2 /= rN_evt2[-1] * (1 - rN_evt2[-1] ** 2)
            vprime_i_evt2 = vprime_i_real_evt2 + 1j * vprime_i_imag_evt2
            vprimeNminus1_evt2.append(vprime_i_evt2)
        vprimeNminus1_evt2 = np.array(vprimeNminus1_evt2)

    # Compute the three geometric pieces of the distance separately so callers
    # can inspect how much each sector contributes.

    # Event-level `chi` contribution, written as a distance in log-space.
    l_chi = np.abs(np.log(chi_evt1) - np.log(chi_evt2))

    # Euclidean distance on the simplex coordinates `rho`.
    l_simplex = np.sqrt(np.sum(((rhoN_evt2 - rhoN_evt1) ** 2)))

    # Two candidate inner products are considered for the sphere term. The code
    # ultimately keeps the larger absolute overlap, which corresponds to the
    # smaller spherical angle.
    dot_product1 = np.dot(vprimeNminus1_evt1, np.conjugate(vprimeNminus1_evt2))
    dot_product1_abs = np.absolute(dot_product1)
    dot_product2 = np.dot(vprimeNminus1_evt1, vprimeNminus1_evt2)
    dot_product2_abs = np.absolute(dot_product2)

    # The reduced coordinates are sphere coordinates, so their norms should be
    # one. Normalize defensively to avoid precision-induced arccos NaNs.
    norm_evt1 = np.sqrt(
        np.absolute(np.dot(vprimeNminus1_evt1, np.conjugate(vprimeNminus1_evt1)))
    )
    norm_evt2 = np.sqrt(
        np.absolute(np.dot(vprimeNminus1_evt2, np.conjugate(vprimeNminus1_evt2)))
    )
    norm_product = norm_evt1 * norm_evt2
    if np.isfinite(norm_product) and norm_product > 0.0:
        dot_product1_abs = dot_product1_abs / norm_product
        dot_product2_abs = dot_product2_abs / norm_product

    # Pick the larger dot product argument to obtain the smaller l_sphere
    sphere_arg = np.maximum(dot_product1_abs, dot_product2_abs)
    sphere_arg = np.clip(sphere_arg, 0.0, 1.0)
    l_sphere = np.arccos(sphere_arg)

    # Combine the pieces using the normalization/weighting convention encoded
    # in the metric definition.
    if l_has_chi == False:
        OverallFactor = (1 / (4 * np.pi**2 * Sphere2Simplex)) * (
            Sphere2Simplex / 4
        ) ** ((N - 1) / (3 * N - 4))
        l = np.sqrt(OverallFactor * (l_simplex**2 + Sphere2Simplex * l_sphere**2))

    if l_has_chi == True:
        OverallFactor = (
            (1 / (4 * np.pi**2 * Sphere2Simplex))
            * (Sphere2Simplex / 4) ** (N / (3 * N - 3))
            * (Chi2Simplex / (16 * np.pi**2)) ** (-1 / (3 * N - 3))
        )
        l = np.sqrt(
            OverallFactor
            * (l_simplex**2 + Sphere2Simplex * l_sphere**2 + Chi2Simplex * l_chi**2)
        )

    if return_only_l:
        return l
    else:
        return l, l_simplex, l_sphere, l_chi



def transform_ps_to_collinear_frame(x_epxpypz: np.ndarray) -> np.ndarray:
    output_dtype = np.result_type(x_epxpypz.dtype, np.float64)
    transformed = np.zeros_like(x_epxpypz, dtype=output_dtype)

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

            transformed[event_idx, constituent_idx] = (
                event[constituent_idx, 0],
                px_new,
                py_new,
                pz_new,
            )

    return transformed


# Snake-case aliases used by the refactored code.
ps_distance = PSdistance
transform_to_collinear_frame = transform_ps_to_collinear_frame
