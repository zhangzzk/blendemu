"""
General utilities: KDTree neighbor finding and bright-neighbor removal.
"""

import numpy as np
from scipy.spatial import KDTree


def kdt_neighbor_finder(pos1, pos2, r_min=0, r_max=10, k=30):
    """
    Find up to k neighbors of pos1 in pos2 within [r_min, r_max].

    Parameters
    ----------
    pos1, pos2 : np.ndarray, shape (N, 2)
        Positions (e.g. RA, DEC).
    r_min, r_max : float
        Min/max search radius.
    k : int
        Number of neighbors to query.

    Returns
    -------
    idx1 : np.ndarray
        Indices into pos1.
    idx2 : np.ndarray
        Indices into pos2.
    distances : np.ndarray
        Separations.
    """
    kdt_in = KDTree(pos2)
    dst, ind = kdt_in.query(pos1, k=k, distance_upper_bound=r_max, workers=-1)

    found_idx = np.where(
        (ind.reshape(-1) != pos2.shape[0]) & (dst.reshape(-1) > r_min)
    )[0]

    idx1 = np.repeat(np.arange(0, pos1.shape[0]), k)[found_idx]
    idx2 = ind.reshape(-1)[found_idx]

    return idx1, idx2, dst.reshape(-1)[found_idx]


def remove_detection_w_bright_neighbour(x, y, flux, ratio_max=10, r_min=0, r_max=5/3600):
    """
    Return indices of detections that have a neighbor brighter by > ratio_max.
    """
    pos = np.vstack((x, y)).T
    idx1, idx2, _ = kdt_neighbor_finder(pos, pos, r_min=r_min, r_max=r_max, k=30)
    flux_ratio = flux[idx2] / flux[idx1]
    reject_idx = idx1[np.where(flux_ratio > ratio_max)[0]]
    return np.unique(reject_idx)
