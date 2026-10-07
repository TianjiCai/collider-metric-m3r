"""Energy-flow 2-Wasserstein (W2) distance between jets, via exact OT (POT)."""

from __future__ import annotations

import numpy as np
import ot  # PyOT library: https://pythonot.github.io/index.html


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




# Snake-case alias used by the refactored code; ``get_W2_Kin`` is the historical name.
get_w2 = get_W2_Kin
