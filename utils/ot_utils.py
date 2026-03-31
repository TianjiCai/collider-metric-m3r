import numpy as np
import ot  # PyOT library: https://pythonot.github.io/index.html
import itertools

from tqdm import tqdm
import concurrent.futures as cf


import math


def get_clean_phi(phi, range="unsigned"):
    """
    Wrap azimuthal angles or angle differences into a standard interval.

    The input is assumed to be a `phi` value or `delta_phi` that may lie outside
    the principal branch because of periodicity. This helper maps it back to:

    - `[-pi, pi]` when `range="signed"`
    - `[0, 2*pi]` when `range="unsigned"`

    Parameters
    ----------
    phi : array-like
        Angular values to wrap.
    range : {"unsigned", "signed"}, default="unsigned"
        Choice of output convention.

    Returns
    -------
    ndarray
        Wrapped angles with the same broadcasted shape as the input.
    """
    assert range in ["unsigned", "signed"]

    # First wrap onto the signed interval [-pi, pi].
    result = np.where(
        phi > np.pi, -(2 * np.pi - phi), np.where(phi < -np.pi, 2 * np.pi + phi, phi)
    )

    if range == "unsigned":
        # Shift the signed interval to the unsigned convention [0, 2*pi].
        result += np.pi

    return result


def get_W2_Kin(jet1, jet2, *, numItermax=100000, return_gamma=False):
    """
    Compute the exact kinematic 2-Wasserstein distance between two jets.

    Each jet is interpreted as a discrete measure supported on constituent
    coordinates `(y, phi)`, with the constituent `pt` values used as transport
    weights. The routine:

    1. removes zero-padded constituents (`pt == 0`),
    2. builds the pairwise squared ground metric
       `Delta y^2 + Delta phi^2`, using the shortest azimuthal separation,
    3. normalizes the `pt` weights so each jet has total mass 1,
    4. calls PyOT's exact EMD solver and returns `sqrt(emd2)`.

    Parameters
    ----------
    jet1, jet2 : ndarray, shape (n_particles, >=3)
        Jet constituent arrays whose first three columns are interpreted as
        `(pt, y, phi)`. Any additional columns are ignored by this function.
    numItermax : int, default=100000
        Maximum number of iterations passed to the PyOT solver.
    return_gamma : bool, default=False
        If `True`, also return the optimal transport plan (coupling matrix).

    Returns
    -------
    float
        The Wasserstein-2 distance between the two jets.
    tuple[ndarray, float]
        Returned instead when `return_gamma=True`; contains the optimal
        transport matrix and the corresponding distance.
    """

    # Split off the weights and the `(y, phi)` coordinates used in the ground
    # metric. Extra per-particle features are intentionally ignored here.
    jet1_pts, jet1_coords = jet1[:, 0], jet1[:, 1:3]
    jet2_pts, jet2_coords = jet2[:, 0], jet2[:, 1:3]

    # Many datasets store jets in a fixed-size array with zero-padded rows.
    # Those rows should not participate in the transport problem.
    mask_1 = jet1_pts > 0
    mask_2 = jet2_pts > 0
    jet1_pts, jet1_coords = jet1_pts[mask_1], jet1_coords[mask_1]
    jet2_pts, jet2_coords = jet2_pts[mask_2], jet2_coords[mask_2]

    # `phi` is periodic, so use the shortest distance on the circle rather than
    # a naive subtraction.
    d_phis = np.pi - np.abs(np.pi - np.abs(jet1_coords[:, 1, None] - jet2_coords[:, 1]))
    squareDist = (jet1_coords[:, 0, None] - jet2_coords[:, 0]) ** 2 + d_phis**2

    # EMD expects two measures with the same total mass, so normalize each jet.
    jet1_pts, jet2_pts = jet1_pts / np.sum(jet1_pts), jet2_pts / np.sum(jet2_pts)

    # `emd2` returns the optimal squared transport cost; take the square root to
    # obtain the W2 distance itself.
    OT = (ot.emd2(jet1_pts, jet2_pts, squareDist, numItermax=numItermax)) ** (1 / 2)
    Gamma = None
    if return_gamma:
        Gamma = ot.emd(jet1_pts, jet2_pts, squareDist, numItermax=numItermax)
        return Gamma, OT
    else:
        return OT


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
    if np.abs(dot_product1_abs) > 1 and np.isclose(
        dot_product1_abs, 1.0, atol=1e-02
    ):  # tolerate numerical error for arccos
        dot_product1_abs = np.round(dot_product1_abs)

    dot_product2 = np.dot(vprimeNminus1_evt1, vprimeNminus1_evt2)
    dot_product2_abs = np.absolute(dot_product2)
    if np.abs(dot_product2_abs) > 1 and np.isclose(
        dot_product2_abs, 1.0, atol=1e-02
    ):  # tolerate numerical error for arccos
        dot_product2_abs = np.round(dot_product2_abs)

    # Pick the larger dot product argument to obtain the smaller l_sphere
    l_sphere = np.arccos(np.maximum(dot_product1_abs, dot_product2_abs))

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


def calcOTDistance(events):
    """
    Build the full symmetric pairwise `get_W2_Kin` matrix for one event set.

    The upper triangle (including the diagonal) is computed explicitly and then
    mirrored to avoid duplicate OT solves.

    Parameters
    ----------
    events : ndarray, shape (N, ...)
        Collection of `N` events/jets compatible with `get_W2_Kin`.

    Returns
    -------
    ndarray, shape (N, N)
        Symmetric pairwise distance matrix.
    """
    N = events.shape[0]
    wXY = np.zeros(shape=(N, N))
    for i, j in tqdm(
        itertools.combinations_with_replacement(range(N), 2), total=N * (N + 1) // 2
    ):
        wXY[i, j] = get_W2_Kin(events[i], events[j])
        wXY[j, i] = wXY[i, j]
    return wXY


def calcOTDistance_non_square(xEvents, yEvents):
    """
    Compute a rectangular pairwise `get_W2_Kin` distance matrix.

    Parameters
    ----------
    xEvents : ndarray, shape (N, ...)
        First event collection.
    yEvents : ndarray, shape (M, ...)
        Second event collection.

    Returns
    -------
    ndarray, shape (N, M)
        Matrix whose `(i, j)` entry is the OT distance between `xEvents[i]`
        and `yEvents[j]`.
    """
    N = xEvents.shape[0]
    M = yEvents.shape[0]
    wXY = np.zeros(shape=(N, M))

    for i, j in tqdm(itertools.product(range(N), range(M)), total=N * M):
        wXY[i, j] = get_W2_Kin(xEvents[i], yEvents[j])

    return wXY


def parallel_OT_non_square_helper(xEvents, yEvents):
    """
    Worker helper for `parallel_OT_non_square`.

    Each worker receives a contiguous chunk of rows from the first event set and
    computes all distances from that chunk to the full second event set.
    """
    N = xEvents.shape[0]
    M = yEvents.shape[0]
    wXY = np.zeros(shape=(N, M))

    for i, j in itertools.product(range(N), range(M)):
        wXY[i, j] = get_W2_Kin(xEvents[i], yEvents[j])

    return wXY


def parallel_OT_non_square(event_set1, event_set2, events_per_thread=100):
    """
    Parallel version of `calcOTDistance_non_square`.

    The first event set is split into row-chunks, each chunk is processed in a
    separate worker process, and the resulting blocks are stacked vertically.

    Parameters
    ----------
    event_set1 : sequence-like, shape (N, ...)
        Event collection providing the rows of the output matrix.
    event_set2 : sequence-like, shape (M, ...)
        Event collection providing the columns of the output matrix.
    events_per_thread : int, default=100
        Number of rows from `event_set1` handled by each worker.

    Returns
    -------
    ndarray, shape (N, M)
        Rectangular distance matrix.
    """
    num_events = len(event_set1)
    # assert num_events % events_per_thread == 0, (
    #     "Number of events in the first set must be divisible by the number of events per thread"
    # )
    if num_events % events_per_thread != 0:
        num_threads = num_events // events_per_thread + 1
    else:
        num_threads = num_events // events_per_thread

    with cf.ProcessPoolExecutor() as executor:
        # Only chunk the row dimension; every worker compares its local rows
        # against the full second set.
        futures = [
            executor.submit(
                parallel_OT_non_square_helper,
                event_set1[events_per_thread * i : events_per_thread * (i + 1)],
                event_set2,
            )
            for i in range(num_threads)
        ]

    results = [f.result() for f in futures]

    distance_matrix = np.vstack(results)

    return distance_matrix


def parallel_OT_helper(events: np.ndarray, starting_index, events_per_thread):
    """
    Worker helper for `parallel_OT_square`.

    The returned block contains only entries below the diagonal for the assigned
    row range; the caller mirrors the matrix afterwards.
    """
    num_events = events.shape[0]

    result = np.zeros(shape=(events_per_thread, num_events))

    for i in range(events_per_thread):
        # Distances are only needed to previously seen rows because the final
        # matrix is symmetric.
        for j in range(starting_index + i):
            result[i, j] = get_W2_Kin(events[starting_index + i], events[j])

    return result


def parallel_OT_square(event_set, events_per_thread=100):
    """
    Compute a full symmetric OT distance matrix in parallel.

    This version assumes `len(event_set)` is exactly divisible by
    `events_per_thread`, because each worker is assigned a fixed-size row block.

    Parameters
    ----------
    event_set : sequence-like, shape (N, ...)
        Events/jets compatible with `get_W2_Kin`.
    events_per_thread : int, default=100
        Number of rows assigned to each worker process.

    Returns
    -------
    ndarray, shape (N, N)
        Symmetric pairwise OT distance matrix.
    """
    num_events = len(event_set)
    assert num_events % events_per_thread == 0, (
        "Number of events must be divisible by the number of events per thread"
    )
    num_threads = num_events // events_per_thread

    with cf.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                parallel_OT_helper, event_set, i * events_per_thread, events_per_thread
            )
            for i in range(num_threads)
        ]

    results = [f.result() for f in futures]

    distance_matrix = np.vstack(results)
    distance_matrix = distance_matrix + distance_matrix.T

    return distance_matrix


def dist_func_helper(dist_func, events_1, events_2, i_batch, j_batch, **kwargs):
    """
    Evaluate a user-supplied distance function on a batch of index pairs.

    This small helper keeps the multiprocessing interface generic: the caller
    specifies which pairs to evaluate, and the helper returns the distances in
    exactly the same order.
    """
    dists = np.zeros(len(i_batch))
    for idx, (i, j) in enumerate(zip(i_batch, j_batch)):
        dists[idx] = dist_func(events_1[i], events_2[j], **kwargs)

    return dists


# def dist_func_helper(dist_func, events_1_batch, events_2_batch, **kwargs):
#     dists = np.zeros(len(events_1_batch) * len(events_2_batch))
#     print("events_1_batch shape:", events_1_batch.shape)
#     print("events_2_batch shape:", events_2_batch.shape)
#     print("dists shape:", dists.shape)
#     for i, (event_1, event_2) in enumerate(zip(events_1_batch, events_2_batch)):
#         dists[i] = dist_func(event_1, event_2, **kwargs)

#     return dists


def parallel_dist_mat_square(event_set, dist_func, events_per_process=100, **kwargs):
    """
    Compute a symmetric pairwise distance matrix for an arbitrary distance function.

    Unlike `parallel_OT_square`, this helper works with any callable of the form
    `dist_func(event_a, event_b, **kwargs) -> scalar`. Only upper-triangular
    pairs are evaluated and the result is mirrored at the end.

    Parameters
    ----------
    event_set : sequence-like, shape (N, ...)
        Events to compare.
    dist_func : callable
        Pairwise distance function.
    events_per_process : int, default=100
        Number of index pairs bundled into each submitted job.
    **kwargs
        Extra keyword arguments forwarded to `dist_func`.

    Returns
    -------
    ndarray, shape (N, N)
        Symmetric distance matrix.
    """
    num_events = len(event_set)
    i_ut, j_ut = np.triu_indices(num_events, k=1)

    distance_matrix = np.zeros((num_events, num_events))

    with cf.ProcessPoolExecutor() as executor:
        futures = []
        for n in range(0, len(i_ut), events_per_process):
            # Batch the upper-triangular index pairs so each submitted job does
            # a reasonable amount of work.
            i_batch = i_ut[n : n + events_per_process]
            j_batch = j_ut[n : n + events_per_process]

            futures.append(
                executor.submit(
                    dist_func_helper,
                    dist_func,
                    event_set,
                    event_set,
                    i_batch,
                    j_batch,
                    **kwargs,
                )
            )

    results = [f.result() for f in futures]

    flattened_results = [x for sub in results for x in sub]

    distance_matrix[i_ut, j_ut] = flattened_results
    distance_matrix = distance_matrix + distance_matrix.T

    return distance_matrix


def parallel_dist_mat_non_square(
    event_set_1, event_set_2, dist_func, events_per_process=100, **kwargs
):
    """
    Compute a rectangular distance matrix for an arbitrary distance function.

    Parameters
    ----------
    event_set_1 : sequence-like, shape (N, ...)
        Events providing the rows of the output matrix.
    event_set_2 : sequence-like, shape (M, ...)
        Events providing the columns of the output matrix.
    dist_func : callable
        Pairwise distance function.
    events_per_process : int, default=100
        Number of `(row, column)` index pairs bundled into each submitted job.
    **kwargs
        Extra keyword arguments forwarded to `dist_func`.

    Returns
    -------
    ndarray, shape (N, M)
        Rectangular distance matrix.
    """
    num_events_1 = len(event_set_1)
    num_events_2 = len(event_set_2)

    # Enumerate every matrix entry once, flatten the pair list, then batch it
    # for multiprocessing.
    i_indices, j_indices = np.indices((num_events_1, num_events_2))
    i_indices = i_indices.ravel()
    j_indices = j_indices.ravel()
    # assert False, "Debugging stop here"

    distance_matrix = np.zeros((num_events_1, num_events_2))

    with cf.ProcessPoolExecutor() as executor:
        futures = []
        for n in range(0, len(i_indices), events_per_process):
            i_batch = i_indices[n : n + events_per_process]
            j_batch = j_indices[n : n + events_per_process]
            # print("Submitting batch:", i_batch, j_batch)

            futures.append(
                executor.submit(
                    dist_func_helper,
                    dist_func,
                    event_set_1,
                    event_set_2,
                    i_batch,
                    j_batch,
                    **kwargs,
                )
            )

    results = [f.result() for f in futures]

    flattened_results = [x for sub in results for x in sub]

    distance_matrix[i_indices, j_indices] = flattened_results

    return distance_matrix


def combine_distance_blocks(block_files, dtype="default"):
    """
    Combine upper-triangular distance matrix blocks into a full symmetric distance matrix.

    Parameters
    ----------
    block_files : list of str
        List of filenames (in .npy format) for the upper-triangular blocks,
        ordered row-by-row: [block_0_0, block_0_1, ..., block_0_{n-1},
                              block_1_1, block_1_2, ..., block_1_{n-1},
                              ...
                              block_{n-1}_{n-1}]
    dtype : str or numpy dtype, default="default"
        Output dtype. When left as `"default"`, the dtype of the first block is
        reused.

    Returns
    -------
    D : ndarray, shape (N, N)
        The reconstructed full distance matrix. Diagonal block sizes are used to
        infer the size of each event chunk, and off-diagonal blocks are mirrored
        with a transpose to fill the lower triangle.
    """

    m = len(block_files)

    # If there are `n` block rows in the upper triangle, there must be exactly
    # `n(n+1)/2` stored blocks.
    n = int((math.sqrt(8 * m + 1) - 1) / 2)
    if n * (n + 1) // 2 != m:
        raise ValueError(f"Expected n(n+1)/2 blocks, got {m}")

    blocks = [np.load(f) for f in block_files]

    # The diagonal blocks tell us how many events belonged to each original
    # chunk, which determines where each block should be placed in the final
    # dense matrix.
    sizes = []
    idx = 0
    for i in range(n):
        diag_block = blocks[idx]
        sizes.append(diag_block.shape[0])
        idx += n - i

    starts = np.cumsum([0] + sizes[:-1])
    total_size = sum(sizes)

    if dtype == "default":
        D = np.zeros((total_size, total_size), dtype=blocks[0].dtype)
    else:
        D = np.zeros((total_size, total_size), dtype=dtype)

    idx = 0
    for i in range(n):
        for j in range(i, n):
            block = blocks[idx]
            si, sj = starts[i], starts[j]
            di, dj = sizes[i], sizes[j]

            D[si : si + di, sj : sj + dj] = block

            if i != j:
                # Upper-triangular blocks are stored on disk; copy their
                # transpose to recover the symmetric lower triangle.
                D[sj : sj + dj, si : si + di] = block.T
            idx += 1

    return D
