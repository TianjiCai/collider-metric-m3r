import jax
import jax.numpy as jnp
import numpy as np
import tqdm

def euclidean_metric(points):
    return jnp.sum(jnp.square(points[:, None, :] - points[None, :, :]), axis=-1)

def spherical_metric(points):

    # Square so the beta = 1 default works
    #
    return jnp.nan_to_num(jnp.square(1 - (jnp.sum(points[:, None, :] * points[None, :, :], axis=-1) / jnp.linalg.norm(points[:, None, :], axis=-1) / jnp.linalg.norm(points[None, :], axis=-1))))



    return jnp.nan_to_num(jnp.square(jnp.arccos(jnp.sum(points[:, None, :] * points[None, :, :], axis=-1) / jnp.linalg.norm(points[:, None, :], axis=-1) / jnp.linalg.norm(points[None, :], axis=-1))))
    return jnp.sqrt(1 - jnp.sum(points[:, None, :] * points[None, :, :], axis=-1) / (jnp.linalg.norm(points[:, None, :], axis=-1) * jnp.linalg.norm(points[None, :, :], axis=-1)))

# ########## Spectral Representation ##########
def compute_spectral_representation(events, omega_max = 2, beta = 1.0, dtype = jnp.float32, euclidean = True):
        """Function to compute the spectral representation of a set of events. Must be compiled before use on batched events -- see SPECTER.compile().
euclidean_distance_squared
        Args:
            events (ndarray): Array of events with shape (batch_size, pad, 3)
            omega_max (float, optional): Maximum omega value. Defaults to 2.
            beta (float, optional): Beta value for the spectral representation. Defaults to 1.
            dtype (jax.numpy.dtype, optional): Data type for the output. Defaults to jax.numpy.float32.

        Returns:
            ndarray: Spectral representation of the events with shape (batch_size, pad*(pad-1)/2, 2)

        FROM EQN 2.5 IN SPECTER 2410.05379
        """

        # Events shape is (pad, 3)
        points, zs = events[:,1:], events[:,0]


        euclidean_distance_squared = jax.lax.cond(euclidean, euclidean_metric, spherical_metric, points)
        

        # Upper Triangle Matrices
        omega_ij = jnp.triu((euclidean_distance_squared), k = 1)
        triangle_indices = jnp.triu_indices(zs.shape[0], k = 1)
        triangle_indices_i = triangle_indices[0]
        triangle_indices_j = triangle_indices[1]

        # Get pairwise products of energies
        ee_ij = jnp.triu(zs[:,None] * zs[None,:])
        ee2 = jnp.trace(ee_ij, axis1 = 0, axis2=1)

        # Flatten to 1D Spectral Representation and remove 0s
        omega_n = omega_ij[triangle_indices_i, triangle_indices_j]
        omega_n = jnp.power(omega_n, beta / 2) / beta
        ee_n = 2 * ee_ij[triangle_indices_i, triangle_indices_j]

        s = jnp.stack((omega_n, ee_n), axis = 1)
        s = jnp.transpose(s, (0,1))
        
        # Sort and append 0
        indices = s[:,0].argsort()
        s = s[indices]
        s0 = jnp.zeros((1, 1))
        s1 = jnp.concatenate((s0, ee2 * jnp.ones((1,1))), axis = 1)
        s = jnp.concatenate((s1, s), axis = 0)

        

        return s.astype(dtype)

def weighted_sum(s, p = 2, max_index = None, inclusive = True):
    
    if max_index is None:
        return jnp.sum(s[:,1] * jnp.power(s[:,0], p), axis = -1)
    else:
        max_index = max_index + 1 if inclusive else max_index
        return jnp.sum(s[:max_index,1] * jnp.power(s[:max_index,0], p),axis = -1)
    
def find_indices_jax(X, Y):

    pairs = jnp.zeros((X.shape[0] + Y.shape[0], 2), dtype=int)

    j_indices = jnp.searchsorted(Y, X, side = 'left')
    i_indices = jnp.searchsorted(X, Y, side = 'left')

    # Update the 'pairs' array using the new 'at' method
    pairs = pairs.at[0:X.shape[0], 0].set(jnp.arange(X.shape[0]))
    pairs = pairs.at[0:X.shape[0], 1].set(j_indices)
    pairs = pairs.at[X.shape[0]:X.shape[0] + Y.shape[0], 0].set(i_indices)
    pairs = pairs.at[X.shape[0]:X.shape[0] + Y.shape[0], 1].set(jnp.arange(Y.shape[0]))

    return pairs
    
# Function to find indices on X and Y, then Y and X, and combine:
def find_indices(X, Y):
        
        x_size = X.shape[0]
        y_size = Y.shape[0]
    
        # Find indices on X and Y
        indices = find_indices_jax(X, Y)

        # Remove duplicates
        indices = jnp.unique(indices, size = (x_size + y_size), fill_value = -1, axis = 0)
        mask = (indices[:,0] > 0) * (indices[:,1] > 0) * (indices[:,0] < X.shape[0]) * (indices[:,1] < Y.shape[0])

        return indices, mask
    
def cross_term_improved(s1, s2):

    # Cross term
    omega1s = s1[:,0]
    omega2s = s2[:,0]

    E1s = s1[:,1]
    E2s = s2[:,1]

    # Calculate cumulative spectral functions
    cumulative_inclusive_1 = jnp.cumsum(E1s, axis = -1)
    cumulative_inclusive_2 = jnp.cumsum(E2s, axis = -1)

    Etot1 = jnp.sum(E1s)
    Etot2 = jnp.sum(E2s)

    # Exclusive = inclusive - 2EE
    cumulative_exclusive_1 = cumulative_inclusive_1 - E1s
    cumulative_exclusive_2 = cumulative_inclusive_2 - E2s

    # Determine which indices survive the theta function O(n^2)
    indices, mask = find_indices(cumulative_inclusive_1, cumulative_inclusive_2)
    # indices = jax.lax.stop_gradient(indices)
    # mask = jax.lax.stop_gradient(mask)
    i_indices, j_indices = indices[:,0], indices[:,1]

    
    # O(n2) parts -- calculate the cross term using the nonzero indices
    omega_n_omega_l = omega1s[i_indices] * omega2s[j_indices]
    minE = -jnp.maximum(Etot1 - cumulative_inclusive_1[i_indices], Etot2 - cumulative_inclusive_2[j_indices]) + jnp.minimum(Etot1 - cumulative_exclusive_1[i_indices], Etot2 -  cumulative_exclusive_2[j_indices])
    cross = omega_n_omega_l * minE

    # Get a mask of the places where omega_n_omega_l is nonzero
    nonzero = omega_n_omega_l > 0
    nonzero_cross = jnp.where(nonzero, cross, 0)
    nonzero_mask = jnp.where(mask, nonzero, 0)



    cross_term = jnp.sum(nonzero_cross * nonzero_mask, axis = (-1))


    return cross_term

def ds2(s1, s2):

    term1 = weighted_sum(s1)
    term2 = weighted_sum(s2)

    return term1 + term2 - 2*cross_term_improved(s1, s2)

def ds2_events1_events2(events1, events2):
    """Function to compute the spectral Earth Mover's Distance between two sets of events

    Args:
        events1 (ndarray): Array of events with shape (batch_size, pad1, 3)
        events2 (ndarray): Array of events with shape (batch_size, pad2, 3)

    Returns:
        ndarray: Spectral EMD between events1 and events2 of size (batch_size,)
    """

    s1 = compute_spectral_representation(events1)
    s2 = compute_spectral_representation(events2)

    return ds2(s1, s2)