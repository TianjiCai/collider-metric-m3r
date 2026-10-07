"""Parallel builders for pairwise distance matrices (any metric) plus the W2-specific
helpers that generated the stored W2 blocks."""

from __future__ import annotations

import concurrent.futures as cf
import itertools

import numpy as np
from tqdm import tqdm

from m3r.metrics.w2 import get_W2_Kin


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


