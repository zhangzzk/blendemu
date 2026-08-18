"""Intrinsic full-scene neighbour summaries used by scene-aware emulators.

The features in this module use only input-catalogue positions and magnitudes.
They never depend on detection, measured quantities, or emulator predictions.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


SCENE_FLUX_EDGES_ARCSEC = np.asarray([0.0, 1.0, 3.0, 10.0])
SCENE_FLUX_INPUT_COLUMNS = (
    "scene_logflux_near_0_1",
    "scene_logflux_mid_1_3",
    "scene_logflux_far_3_10",
)
SCENE_FLUX_PAIR_COLUMNS = tuple(
    f"{name}_input_p" for name in SCENE_FLUX_INPUT_COLUMNS
)
PAIR_OTHER_FLUX_COLUMNS = (
    "scene_logflux_other_near_0_1_input_p",
    "scene_logflux_other_mid_1_3_input_p",
    "scene_logflux_other_far_3_10_input_p",
)
PAIR_OTHER_ABSOLUTE_FLUX_COLUMNS = (
    "scene_logfluxabs_other_near_0_1",
    "scene_logfluxabs_other_mid_1_3",
    "scene_logfluxabs_other_far_3_10",
)


def scene_flux_ratios(ra_deg, dec_deg, magnitude, zero_point=30.0):
    """Return log10(1 + neighbour flux / primary flux) in three annuli.

    Every other input source inside 10 arcsec contributes, irrespective of
    detection or which half of a half-shear simulation it belongs to.  The
    primary itself is excluded because ``query_pairs`` returns distinct rows.

    The Euclidean degree-coordinate metric deliberately matches the existing
    SBSI neighbour-flux lookup.  These simulations occupy a small field near
    dec=-0.5 deg, where the omitted cos(dec) correction is negligible.
    """
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)
    mag = np.asarray(magnitude, dtype=float)
    if not (ra.ndim == dec.ndim == mag.ndim == 1):
        raise ValueError("ra, dec, and magnitude must be one-dimensional")
    if not (len(ra) == len(dec) == len(mag)):
        raise ValueError("ra, dec, and magnitude must have equal length")
    if not (np.isfinite(ra).all() and np.isfinite(dec).all() and np.isfinite(mag).all()):
        raise ValueError("scene inputs must all be finite")

    flux = 10.0 ** (-0.4 * (mag - float(zero_point)))
    shell_flux = np.zeros((3, len(ra)), dtype=np.float64)
    if len(ra) < 2:
        return shell_flux

    position = np.column_stack([ra, dec])
    pairs = cKDTree(position).query_pairs(
        r=SCENE_FLUX_EDGES_ARCSEC[-1] / 3600.0,
        output_type="ndarray",
    )
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        distance = np.hypot(
            position[i, 0] - position[j, 0],
            position[i, 1] - position[j, 1],
        ) * 3600.0
        shell = np.searchsorted(
            SCENE_FLUX_EDGES_ARCSEC, distance, side="right",
        ) - 1
        if np.any((shell < 0) | (shell >= 3)):
            raise RuntimeError("KD-tree returned a pair outside the declared annuli")
        for shell_index in range(3):
            use = shell == shell_index
            np.add.at(shell_flux[shell_index], i[use], flux[j[use]])
            np.add.at(shell_flux[shell_index], j[use], flux[i[use]])

    return np.log10(1.0 + shell_flux / flux[None, :])


def pair_other_flux_ratios(
    total_logflux_ratio,
    primary_magnitude,
    secondary_magnitude,
    pair_separation_arcsec,
    allow_outside=False,
):
    """Remove the designated pair secondary from full-scene shell flux.

    A response-catalogue row names one primary and one designated secondary.
    The returned context contains every *other* rendered input galaxy, so it
    can describe how third-or-later light changes that secondary's marginal
    blending response. It excludes no galaxy by detection or rank.

    ``total_logflux_ratio`` has shape ``(n_pair, 3)`` and contains per-primary
    ``log10(1 + F_all_neighbours/F_primary)`` in the standard 0--1, 1--3 and
    3--10 arcsec intrinsic-position shells. The other arguments are length-n
    vectors of intrinsic pair magnitudes and intrinsic-position separation.
    """
    total_log = np.asarray(total_logflux_ratio, dtype=float)
    mag_p = np.asarray(primary_magnitude, dtype=float)
    mag_s = np.asarray(secondary_magnitude, dtype=float)
    distance = np.asarray(pair_separation_arcsec, dtype=float)
    if total_log.ndim != 2 or total_log.shape[1] != 3:
        raise ValueError("total_logflux_ratio must have shape (n_pair, 3)")
    n_pair = total_log.shape[0]
    if any(x.ndim != 1 or len(x) != n_pair for x in (mag_p, mag_s, distance)):
        raise ValueError("pair magnitudes and separation must be length-n_pair vectors")
    if not (np.isfinite(total_log).all() and np.isfinite(mag_p).all()
            and np.isfinite(mag_s).all() and np.isfinite(distance).all()):
        raise ValueError("pair scene inputs must all be finite")

    shell = np.searchsorted(SCENE_FLUX_EDGES_ARCSEC, distance, side="right") - 1
    outside = (shell < 0) | (shell >= 3)
    if np.any(outside) and not allow_outside:
        bad = distance[outside][:5]
        raise ValueError(f"pair separation lies outside [0,10) arcsec: {bad.tolist()}")

    total_ratio = np.expm1(total_log * np.log(10.0))
    secondary_ratio = np.exp(-0.4 * np.log(10.0) * (mag_s - mag_p))
    row = np.flatnonzero(~outside)
    active_shell = shell[row]
    remainder = total_ratio[row, active_shell] - secondary_ratio[row]

    # Full-scene summaries are commonly stored as float32 log ratios. Their
    # inverse transform can undershoot a single-secondary shell by a few ulp.
    # Permit only relative roundoff; a material deficit is a mismatched pair.
    tolerance = 5.0e-5 * np.maximum(secondary_ratio[row], 1.0)
    materially_negative = remainder < -tolerance
    if np.any(materially_negative):
        bad_local = np.flatnonzero(materially_negative)[:5]
        detail = [
            (int(row[i]), float(remainder[i]), float(tolerance[i])) for i in bad_local
        ]
        raise ValueError(
            "full-scene shell lacks the designated secondary beyond float32 "
            f"roundoff; samples (row,remainder,tolerance)={detail}"
        )
    total_ratio[row, active_shell] = np.maximum(remainder, 0.0)
    return np.log10(1.0 + total_ratio)


def pair_other_absolute_flux(
    total_logflux_ratio,
    primary_magnitude,
    secondary_magnitude,
    pair_separation_arcsec,
    zero_point=30.0,
    allow_outside=False,
):
    """Return absolute other-galaxy shell flux after excluding this pair.

    The output is ``log10(1 + F_other)`` in simulation-count units.  It uses
    :func:`pair_other_flux_ratios` for the exact secondary subtraction, then
    removes the primary-flux normalization.  Absolute flux is deliberate: the
    held-out diagnostic found that ``F_other/F_primary`` mostly selected faint
    primaries, while absolute other light followed the expected suppression
    direction in all three shells.
    """
    mag_p = np.asarray(primary_magnitude, dtype=float)
    relative_log = pair_other_flux_ratios(
        total_logflux_ratio,
        mag_p,
        secondary_magnitude,
        pair_separation_arcsec,
        allow_outside=allow_outside,
    )
    primary_flux = np.exp(
        -0.4 * np.log(10.0) * (mag_p - float(zero_point))
    )
    other_ratio = np.expm1(relative_log * np.log(10.0))
    absolute_log = np.log10(1.0 + other_ratio * primary_flux[:, None])
    if not np.isfinite(absolute_log).all():
        raise ValueError("pair-specific absolute other flux is non-finite")
    return absolute_log


def add_scene_flux_features(catalogue, zero_point=30.0, copy=True):
    """Attach the three scene features to a one-row-per-input catalogue."""
    required = {"RA", "DEC", "r"}
    missing = required - set(catalogue.columns)
    if missing:
        raise KeyError(f"catalogue lacks scene-feature inputs: {sorted(missing)}")
    out = catalogue.copy() if copy else catalogue
    values = scene_flux_ratios(
        out["RA"].to_numpy(), out["DEC"].to_numpy(), out["r"].to_numpy(),
        zero_point=zero_point,
    )
    for index, name in enumerate(SCENE_FLUX_INPUT_COLUMNS):
        out[name] = values[index].astype(np.float32)
    return out
