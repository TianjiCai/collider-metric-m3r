import numpy as np
import ot  # PyOT library: https://pythonot.github.io/index.html
import itertools

from tqdm import tqdm
import concurrent.futures as cf
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory


import math


def get_clean_phi(phi, range="unsigned"):
    """
    in place of  [- (2*np.pi - delta_phi) if delta_phi > np.pi
                 else 2*np.pi + delta_phi if delta_phi < - np.pi
                 else delta_phi]
    ------
    phi, result: np.array
    """
    assert range in ["unsigned", "signed"]

    result = np.where(
        phi > np.pi, -(2 * np.pi - phi), np.where(phi < -np.pi, 2 * np.pi + phi, phi)
    )

    if range == "unsigned":
        result += np.pi

    return result


def get_W2_Kin(jet1, jet2, *, numItermax=100000, return_gamma=False):
    """
    Compute the exact W2 distance between two jets,
    with the ground space being the kinematic space (y, phi), and pt (normalized) or charge as mass.
    ------------------------------------------------------------------------------------------------------
    INPUTS:
    jet1, jet2: np.array of shape (nparticles, 4), where the 4-dim is (pt, y, phi, chg).
    mass: str; if 'pt', then mass of each particle is its pt;
               if 'chg', then mass is the charge of the particle, scaled up by the following upscale_factor.
    upscale_factor: float or None; only if mass='chg', then increase all charge values by a certain amount
                    to avoid zero and negative mass (recommended: 2; does not matter much); default None.
    numItermax: int; default 100000; hyperparameter in computing W2 distance.
    return_gamma: boolean; default False, i.e., do not return the transport matrix.

    OUTPUTS:
    Gamma: None or np.array of shape (nparticles, nparticles)
    OT: float
    """

    # separate pts, kinematic coordinates, charges
    jet1_pts, jet1_coords = jet1[:, 0], jet1[:, 1:3]
    jet2_pts, jet2_coords = jet2[:, 0], jet2[:, 1:3]

    # remove zero-padding
    mask_1 = jet1_pts > 0
    mask_2 = jet2_pts > 0
    jet1_pts, jet1_coords = jet1_pts[mask_1], jet1_coords[mask_1]
    jet2_pts, jet2_coords = jet2_pts[mask_2], jet2_coords[mask_2]

    # ground distance
    d_phis = np.pi - np.abs(np.pi - np.abs(jet1_coords[:, 1, None] - jet2_coords[:, 1]))
    squareDist = (jet1_coords[:, 0, None] - jet2_coords[:, 0]) ** 2 + d_phis**2

    # normalize total pt to be one
    jet1_pts, jet2_pts = jet1_pts / np.sum(jet1_pts), jet2_pts / np.sum(jet2_pts)

    # ot computation
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
    Compute the phase space distance (and the contributions on separate parts) between two given events/jets
    in some frame.

    INPUTS:
    pN_evt1(2): np.array; 4-momenta of particles in event/jet 1(2) in any frame (CM, non-CM, collinear, as long as total momentum is along +/- z).
    E_plus_pz_onSimplex: boolean; whether u (simplex) is defined by E+pz; default is False, i.e., u defined by E-pz.
    Sphere2Simplex: float; relative weight for the sphere to the simplex; default is 4:1.
    l_has_chi: boolean; whether to include the chi coordinate in the total PS distance l; default is False.
    Chi2Simplex: float; relative weight for the chi to the simplex; default is 16*pi^2 :1; matters only if l_has_chi=True.

    OUTPUTS:
    l: float; total phase space distance between the two given events/jets.
    l_simplex: float; distance on the simplex between the two given events/jets.
    l_sphere: float; distance on the sphere between the two given events/jets.
    l_chi: float; distance for the chi coordinate between the two given events/jets; meaningful only if l_has_chi=True, disregard otherwise.
    """

    # kinematics for evt1 and evt2
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

    # PS coords
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

    rhoN_evt1 = np.array([u**2 for u in uN_evt1])
    rhoN_evt2 = np.array([u**2 for u in uN_evt2])
    rN_evt1 = np.array([u for u in uN_evt1])
    rN_evt2 = np.array([u for u in uN_evt2])

    # v to vprime
    if N == 2:
        vprimeNminus1_evt1 = 1 / np.sqrt(rhoN_evt1[1]) * vN_evt1[: N - 1]
        vprimeNminus1_evt2 = 1 / np.sqrt(rhoN_evt2[1]) * vN_evt2[: N - 1]

    else:  # N>=3
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

    # Compute the distances

    # l_chi:
    l_chi = np.abs(np.log(chi_evt1) - np.log(chi_evt2))

    # l_simplex:
    l_simplex = np.sqrt(np.sum(((rhoN_evt2 - rhoN_evt1) ** 2)))

    # l_sphere dot product with conjugation:
    dot_product1 = np.dot(vprimeNminus1_evt1, np.conjugate(vprimeNminus1_evt2))
    dot_product1_abs = np.absolute(dot_product1)
    if np.abs(dot_product1_abs) > 1 and np.isclose(
        dot_product1_abs, 1.0, atol=1e-02
    ):  # tolerate numerical error for arccos
        dot_product1_abs = np.round(dot_product1_abs)

    # l_sphere dot product without conjugation:
    dot_product2 = np.dot(vprimeNminus1_evt1, vprimeNminus1_evt2)
    dot_product2_abs = np.absolute(dot_product2)
    if np.abs(dot_product2_abs) > 1 and np.isclose(
        dot_product2_abs, 1.0, atol=1e-02
    ):  # tolerate numerical error for arccos
        dot_product2_abs = np.round(dot_product2_abs)

    # Pick the larger dot product argument to obtain the smaller l_sphere
    l_sphere = np.arccos(np.maximum(dot_product1_abs, dot_product2_abs))

    # Only include l_simplex and l_sphere in the total PS distance l
    if l_has_chi == False:
        OverallFactor = (1 / (4 * np.pi**2 * Sphere2Simplex)) * (
            Sphere2Simplex / 4
        ) ** ((N - 1) / (3 * N - 4))
        l = np.sqrt(OverallFactor * (l_simplex**2 + Sphere2Simplex * l_sphere**2))

    # Also include l_chi in the total PS distance l
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
    N = events.shape[0]
    wXY = np.zeros(shape=(N, N))
    for i, j in tqdm(
        itertools.combinations_with_replacement(range(N), 2), total=N * (N + 1) // 2
    ):
        wXY[i, j] = get_W2_Kin(events[i], events[j])
        wXY[j, i] = wXY[i, j]
    return wXY


def calcOTDistance_non_square(xEvents, yEvents):
    N = xEvents.shape[0]
    M = yEvents.shape[0]
    wXY = np.zeros(shape=(N, M))

    for i, j in tqdm(itertools.product(range(N), range(M)), total=N * M):
        wXY[i, j] = get_W2_Kin(xEvents[i], yEvents[j])

    return wXY


def parallel_OT_non_square_helper(xEvents, yEvents):
    N = xEvents.shape[0]
    M = yEvents.shape[0]
    wXY = np.zeros(shape=(N, M))

    for i, j in itertools.product(range(N), range(M)):
        wXY[i, j] = get_W2_Kin(xEvents[i], yEvents[j])

    return wXY


def parallel_OT_helper(events: np.ndarray, starting_index, events_per_thread):
    num_events = events.shape[0]

    result = np.zeros(shape=(events_per_thread, num_events))

    for i in range(events_per_thread):
        for j in range(starting_index + i):
            result[i, j] = get_W2_Kin(events[starting_index + i], events[j])

    return result


def parallel_OT_square(event_set, events_per_thread=100):
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
    num_events = len(event_set)
    i_ut, j_ut = np.triu_indices(num_events, k=1)

    distance_matrix = np.zeros((num_events, num_events))

    with cf.ProcessPoolExecutor() as executor:
        futures = []
        for n in range(0, len(i_ut), events_per_process):
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
    num_events_1 = len(event_set_1)
    num_events_2 = len(event_set_2)

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


def parallel_OT_non_square(event_set1, event_set2, events_per_thread=100):
    num_events = len(event_set1)
    # assert num_events % events_per_thread == 0, (
    #     "Number of events in the first set must be divisible by the number of events per thread"
    # )
    if num_events % events_per_thread != 0:
        num_threads = num_events // events_per_thread + 1
    else:
        num_threads = num_events // events_per_thread

    with cf.ProcessPoolExecutor() as executor:
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


# def _worker_single_pair(args):
#     # Unpack for pickling-safe top-level function
#     i, j, events, dist, kwargs = args
#     return i, j, dist(events[i], events[j], **kwargs)


# def pairwise_parallel(
#     events,
#     dist,
#     *,
#     n_jobs=None,
#     backend="process",  # "process" for CPU-bound; "thread" if dist releases GIL or is I/O-bound
#     chunksize=256,  # how many (i, j) pairs per chunk sent to each worker
#     fill_diagonal=0.0,
#     dtype=float,
#     show_progress=True,
#     **kwargs,  # forwarded to dist(x, y, **kwargs)
# ):
#     """
#     Compute a symmetric pairwise distance matrix using a single-event distance function.

#     Parameters
#     ----------
#     events : sequence/array-like, shape (n_samples, ...)
#         Your event objects; must be indexable.
#     dist : callable
#         Signature: dist(x, y, **kwargs) -> scalar
#     n_jobs : int or None
#         Number of workers. None uses os.cpu_count() for processes; 1 forces serial.
#     backend : {"process","thread"}
#         Execution backend.
#     chunksize : int
#         Chunk size hint for executor.map.
#     fill_diagonal : scalar
#         Value to put on the diagonal.
#     dtype : dtype
#         Output dtype.
#     show_progress : bool
#         Show tqdm progress bar.
#     **kwargs :
#         Extra args passed to dist.
#     """
#     # events = np.asarray(events, dtype=object)  # keep objects intact
#     n = len(events)

#     out = np.empty((n, n), dtype=dtype)
#     if fill_diagonal is not None:
#         np.fill_diagonal(out, fill_diagonal)

#     # Upper-triangular index pairs (excluding diagonal)
#     I, J = np.triu_indices(n, k=1)

#     if n_jobs == 1:
#         # Serial (handy for debugging)
#         iterator = zip(I, J)
#         if show_progress:
#             iterator = tqdm(iterator, total=len(I), desc="pairwise(dist)")

#         for i, j in iterator:
#             val = dist(events[i], events[j], **kwargs)
#             out[i, j] = val
#             out[j, i] = val
#         return out

#     # Parallel
#     Exe = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
#     with Exe(max_workers=n_jobs) as ex:
#         tasks = ((i, j, events, dist, kwargs) for i, j in zip(I, J))
#         # executor.map preserves order; zip back into indices
#         iterator = ex.map(_worker_single_pair, tasks, chunksize=chunksize)
#         if show_progress:
#             iterator = tqdm(iterator, total=len(I), desc=f"pairwise({backend})")

#         for i, j, val in iterator:
#             out[i, j] = val
#             out[j, i] = val

#     return out


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

    Returns
    -------
    D : ndarray, shape (N, N)
        The reconstructed full distance matrix.
    """

    m = len(block_files)

    n = int((math.sqrt(8 * m + 1) - 1) / 2)
    if n * (n + 1) // 2 != m:
        raise ValueError(f"Expected n(n+1)/2 blocks, got {m}")

    blocks = [np.load(f) for f in block_files]

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
                D[sj : sj + dj, si : si + di] = block.T
            idx += 1

    return D
