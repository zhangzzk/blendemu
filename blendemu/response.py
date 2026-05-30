"""
Response and detection catalogue generation.

Build catalogues that pair primary galaxies with their neighbors and compute
the blending response (delta_et) from shape measurements at two shear values.
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from . import utils


SHEAR_COMPONENT_CONVENTION = 'sky_cos_sin'
LEGACY_SHEAR_COMPONENT_CONVENTION = 'sky_sin_cos'

DEFAULT_DETECTION_MEASURED_COLUMNS = [
    'X_IMAGE', 'Y_IMAGE', 'XWIN_IMAGE', 'YWIN_IMAGE',
    'X_WORLD', 'Y_WORLD',
    'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO',
    'FLUX_APER', 'FLUXERR_APER', 'MAG_APER', 'MAGERR_APER',
    'FLUX_RADIUS', 'FWHM_IMAGE',
    'A_IMAGE', 'B_IMAGE', 'A_WORLD', 'B_WORLD',
    'THETA_IMAGE', 'THETA_WORLD',
    'FLAGS', 'CLASS_STAR', 'ISOAREA_IMAGE',
]


def _normalize_shear_cases(shear_cases=None):
    """Return directory labels for the unsheared and sheared cases."""
    if shear_cases is None:
        return ['0.0', '0.1']
    if len(shear_cases) != 2:
        raise ValueError("shear_cases must contain exactly two shear labels")
    return [str(s) for s in shear_cases]


# --- Geometric helpers ---

def angle_between(pos1, pos2):
    """Polarization angle (radians) between two sets of positions."""
    sep = pos2 - pos1
    an = np.arctan2(sep[1, :], sep[0, :])
    return np.pi - an


def _normalize_shear_component_convention(convention):
    if convention in {None, '', SHEAR_COMPONENT_CONVENTION, 'usual', 'cos_sin'}:
        return SHEAR_COMPONENT_CONVENTION
    if convention in {LEGACY_SHEAR_COMPONENT_CONVENTION, 'legacy', 'sin_cos'}:
        return LEGACY_SHEAR_COMPONENT_CONVENTION
    raise ValueError(f"Unsupported shear component convention: {convention!r}")


def _frame_shear_component_convention(*frames, default=LEGACY_SHEAR_COMPONENT_CONVENTION):
    for frame in frames:
        if frame is None or 'shear_component_convention' not in frame.columns:
            continue
        values = frame['shear_component_convention'].dropna().unique()
        if len(values):
            return _normalize_shear_component_convention(values[0])
    return _normalize_shear_component_convention(default)


def _read_generated_catalog_convention(path, case, shear):
    generated_path = os.path.join(path, f'gals{case}_{shear}.feather')
    if not os.path.exists(generated_path):
        return LEGACY_SHEAR_COMPONENT_CONVENTION
    try:
        meta = pd.read_feather(
            generated_path,
            columns=['shear_component_convention'],
        )
    except Exception:
        return LEGACY_SHEAR_COMPONENT_CONVENTION
    return _frame_shear_component_convention(meta)


def _generated_catalog_count(path, case, shear):
    """Number of rows in the pre-simulation generated catalogue."""
    try:
        return pd.read_feather(
            os.path.join(path, f'gals{case}_{shear}.feather'),
            columns=['index'],
        ).shape[0]
    except Exception:
        return None


def _input_ids(icat):
    """Return stable input IDs for rows in a MultiBand_ImSim input catalogue."""
    if 'index_input' in icat.columns:
        return icat['index_input'].to_numpy(dtype=int)
    return np.arange(icat.shape[0], dtype=int)


def _input_id_lookup(icat):
    """Map stable input IDs back to row positions in ``icat``."""
    ids = _input_ids(icat)
    if pd.Index(ids).has_duplicates:
        raise ValueError("gals_info index_input values are not unique")
    return pd.Series(np.arange(ids.shape[0], dtype=int), index=ids), ids


def _lookup_input_rows(input_id_to_row, input_ids, label='input ids'):
    """Convert stable input IDs to row positions, failing loudly if absent."""
    input_ids = np.asarray(input_ids, dtype=int)
    rows = input_id_to_row.reindex(input_ids)
    missing = rows.isna()
    if missing.any():
        sample = input_ids[missing.to_numpy()][:5].tolist()
        raise KeyError(f"{label} not present in gals_info: {sample}")
    return rows.to_numpy(dtype=int)


def _primary_secondary_rows(icat, total_input_count=None):
    """Return row positions for unsheared primaries and sheared secondaries."""
    ids = _input_ids(icat)
    if total_input_count is None:
        total_input_count = int(ids.max()) + 1 if ids.size else 0
    split = int(total_input_count / 2)
    return np.where(ids < split)[0], np.where(ids >= split)[0]


def e2ang(e1, e2, convention=SHEAR_COMPONENT_CONVENTION):
    """Convert spin-2 components to magnitude and physical half-angle."""
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    convention = _normalize_shear_component_convention(convention)

    e = np.hypot(e1, e2)
    angle = np.full(e.shape, np.nan, dtype=float)
    good = np.isfinite(e) & (e > 0)
    if convention == LEGACY_SHEAR_COMPONENT_CONVENTION:
        angle[good] = 0.5 * np.arctan2(e1[good], e2[good])
    else:
        angle[good] = 0.5 * np.arctan2(e2[good], e1[good])
    angle[good] = np.mod(angle[good], np.pi)
    e[~good] = np.nan
    return e, angle


def spin2rot(e1, e2, angle):
    """Project spin-2 components parallel/cross to ``angle``."""
    parallel = e1 * np.cos(2 * angle) + e2 * np.sin(2 * angle)
    cross = -e1 * np.sin(2 * angle) + e2 * np.cos(2 * angle)
    return parallel, cross


# --- Response catalogue ---

def retrieve_response(case, r_max, r_min, k, real='real0',
                      tile_name='tile180.0_-0.5', data_path=None,
                      shear_cases=None):
    """
    Build the response catalogue for one simulation case.

    Cross-matches detections at the two configured shear values, finds secondary
    neighbors via KDTree, and computes delta_et (tangential shear response).

    Parameters
    ----------
    case : int
        Simulation case index.
    r_max : float
        Maximum neighbor search radius in arcsec.
    r_min : float
        Minimum neighbor search radius in arcsec.
    k : int
        Number of neighbors to query.
    real : str
        Realization tag.
    tile_name : str
        Tile identifier for file paths.
    data_path : str, optional
        Base data directory.

    Returns
    -------
    pd.DataFrame
        Response catalogue with galaxy features, angles, and delta_et.
    """
    if data_path is None:
        raise ValueError("data_path is required")
    path = data_path
    shear_case = _normalize_shear_cases(shear_cases)

    # Load input catalogue
    input_cat_path = os.path.join(
        path, f'case{case}_{shear_case[1]}', real,
        'catalogues/input', f'gals_info_{tile_name}.feather',
    )
    icat = pd.read_feather(input_cat_path)
    shear_component_convention = _frame_shear_component_convention(
        icat,
        default=_read_generated_catalog_convention(path, case, shear_case[1]),
    )

    total_input_count = _generated_catalog_count(path, case, shear_case[1])
    input_id_to_row, input_ids = _input_id_lookup(icat)
    primary_rows, secondary_rows = _primary_secondary_rows(icat, total_input_count)
    primary_ids = input_ids[primary_rows]

    def get_match(case_path_segment):
        """Load and cross-match shape/detection catalogues."""
        base_cat_path = os.path.join(path, case_path_segment, real, 'catalogues')
        shape_cat_path = os.path.join(
            base_cat_path, 'Shapes',
            f'shape_catalogue_detect_position_{tile_name}.feather',
        )
        match_cat_path = os.path.join(
            base_cat_path, 'CrossMatch', f'{tile_name}_rot0_matched.feather',
        )

        scat = pd.read_feather(shape_cat_path)
        mcat = pd.read_feather(match_cat_path)

        # Remove detections with very bright neighbors
        reject_idx = utils.remove_detection_w_bright_neighbour(
            scat['X_WORLD'].array, scat['Y_WORLD'].array,
            scat['FLUX_AUTO'].array,
            ratio_max=5, r_min=0, r_max=3 / 3600,
        )
        scat = scat.drop(reject_idx)

        _, ind1_, _ = np.intersect1d(mcat['id_detec'] - 1, scat.index, return_indices=True)
        mcat = mcat.iloc[ind1_].reset_index(drop=True)

        comm, ind1, _ = np.intersect1d(
            mcat['id_input'].to_numpy(dtype=int), primary_ids, return_indices=True
        )
        return scat, mcat, comm, ind1

    scat1, mcat1, comm1, ind1 = get_match(f'case{case}_{shear_case[0]}')
    scat2, mcat2, comm2, ind2 = get_match(f'case{case}_{shear_case[1]}')

    # Common primaries detected at both shear values
    comm, ind1_, ind2_ = np.intersect1d(comm1, comm2, return_indices=True)
    idx_det1 = mcat1['id_detec'].iloc[ind1[ind1_]].to_numpy() - 1
    idx_det2 = mcat2['id_detec'].iloc[ind2[ind2_]].to_numpy() - 1

    # --- Neighbor finding ---
    sec_pos = icat[['RA_input', 'DEC_input']].iloc[secondary_rows].to_numpy()
    kdt_in = KDTree(sec_pos)
    pri_pos = scat1[['X_WORLD', 'Y_WORLD']].loc[idx_det1]
    dst, ind = kdt_in.query(pri_pos, k=k, distance_upper_bound=r_max / 3600, workers=-1)

    found_mask = (ind.reshape(-1) != sec_pos.shape[0]) & (dst.reshape(-1) >= r_min / 3600)
    found_idx = np.where(found_mask)[0]

    sec_idx_relative = ind.reshape(-1)[found_idx]
    sec_idx_input = secondary_rows[sec_idx_relative]
    pri_idx_detec1 = np.repeat(idx_det1, k)[found_idx]
    pri_idx_detec2 = np.repeat(idx_det2, k)[found_idx]
    pri_idx_input = np.repeat(comm, k)[found_idx]
    pri_idx_rows = _lookup_input_rows(input_id_to_row, pri_idx_input, label='primary ids')

    # --- Build output catalogue ---
    R_cat = {'input_index': pri_idx_input}

    gal_features = [
        'RA_input', 'DEC_input', 'redshift_input', 'Re_input',
        'axis_ratio_input', 'position_angle_input', 'sersic_n_input', 'r_input',
    ]
    for gf in gal_features:
        R_cat[f'{gf}_p'] = icat[gf].iloc[pri_idx_rows].to_numpy()
        R_cat[f'{gf}_s'] = icat[gf].iloc[sec_idx_input].to_numpy()

    # Geometry
    pri_pos_arr = scat1[['X_WORLD', 'Y_WORLD']].loc[pri_idx_detec1].to_numpy()
    sec_pos_arr = icat[['RA_input', 'DEC_input']].iloc[sec_idx_input].to_numpy()

    pola_ang = angle_between(pri_pos_arr.T, sec_pos_arr.T)
    pola_ang += np.pi
    pola_ang[pola_ang > np.pi * 2] -= np.pi * 2

    R_cat['polarization_angle'] = pola_ang / np.pi * 180
    R_cat['distance'] = dst.reshape(-1)[found_idx] * 3600

    _, R_cat['shear_angle'] = e2ang(
        icat['gamma1_input'].iloc[sec_idx_input].to_numpy(),
        icat['gamma2_input'].iloc[sec_idx_input].to_numpy(),
        convention=shear_component_convention,
    )

    # Shape measurements and response
    e_c1 = scat1[['NGMIX_G1', 'NGMIX_G2']].loc[pri_idx_detec1].to_numpy().T
    e_c2 = scat2[['NGMIX_G1', 'NGMIX_G2']].loc[pri_idx_detec2].to_numpy().T

    et_c1 = np.array(spin2rot(e_c1[0, :], e_c1[1, :], R_cat['shear_angle']))
    et_c2 = np.array(spin2rot(e_c2[0, :], e_c2[1, :], R_cat['shear_angle']))

    R_cat['measured_e1_m'] = e_c1[0, :]
    R_cat['measured_e2_m'] = e_c1[1, :]
    R_cat['measured_e1_p'] = e_c2[0, :]
    R_cat['measured_e2_p'] = e_c2[1, :]
    R_cat['delta_et1'] = et_c2[0, :] - et_c1[0, :]
    R_cat['delta_et2'] = et_c2[1, :] - et_c1[1, :]

    R_cat['S/N_m'] = (scat1['FLUX_AUTO'] / scat1['FLUXERR_AUTO']).loc[pri_idx_detec1].to_numpy().T
    R_cat['S/N_p'] = (scat2['FLUX_AUTO'] / scat2['FLUXERR_AUTO']).loc[pri_idx_detec2].to_numpy().T

    R_cat['shear_angle'] = R_cat['shear_angle'] / np.pi * 180
    R_cat['shear_component_convention'] = np.full(len(pri_idx_detec1), shear_component_convention)
    R_cat['case'] = np.full(len(pri_idx_detec1), case)

    R_df = pd.DataFrame(data=R_cat)

    # Flag failed measurements
    fail_mask = ((e_c1[0, :] == -1.) & (e_c2[0, :] == -1.)) | \
                ((e_c1[0, :] == 0.) & (e_c2[0, :] == 0.))
    if np.any(fail_mask):
        R_df.loc[fail_mask, ['delta_et1', 'delta_et2']] = None

    return R_df


# --- Self-response catalogue ---

def retrieve_self_response(case, r_max, r_min, k, real='real0',
                           tile_name='tile180.0_-0.5', data_path=None,
                           secondary_shape_suffix='_secondaries',
                           nearest_only=True, include_isolated=False,
                           shear_cases=None):
    """
    Build the self-response catalogue for one simulation case.

    The "self-response" is how each *secondary* galaxy's own measured shape
    changes between the unsheared and sheared image
    pair. Unlike the blending response (which uses primaries as targets),
    this uses the sheared secondaries as targets and measures their direct
    response to the shear applied to themselves.

    For each secondary detected at both shear values:
      - Rotate measured e1, e2 into the secondary's own shear frame
      - Compute delta_et = e_t(g=0.1) - e_t(g=0.0)
      - <delta_et>/|g| should approach ~1 for high-quality isolated galaxies,
        with lower values expected from noise, blending, and multiplicative bias

    Features include the target secondary's own input properties (suffix _p)
    and neighbour properties (suffix _s), so downstream training code
    (data_utils.source_select_reg / rescale) works unchanged.

    By default this returns at most one row per target, using only the closest
    non-self neighbour inside ``r_max``.  If ``include_isolated=True``, targets
    with no neighbour inside ``r_max`` are retained with NaN neighbour
    properties and ``neighbored=False``.  Set ``nearest_only=False`` to recover
    the historical pair-shaped catalogue where one target may appear once for
    each of the ``k`` neighbours inside ``r_max``.

    Parameters
    ----------
    case : int
        Simulation case index.
    r_max : float
        Maximum neighbor search radius in arcsec.
    r_min : float
        Minimum neighbor search radius in arcsec.
    k : int
        Number of neighbors to query.
    real : str
        Realization tag.
    tile_name : str
        Tile identifier for file paths.
    data_path : str, optional
        Base data directory.
    secondary_shape_suffix : str
        Suffix on the shape-catalogue filename that distinguishes
        secondary-target measurements (default: '_secondaries', matching
        the run_shape.py --targets=secondaries output).
    nearest_only : bool
        If True, return one row per target with only its closest non-self
        neighbour inside ``r_max``. Default is True.
    include_isolated : bool
        In nearest-only mode, retain targets with no neighbour inside ``r_max``.

    Returns
    -------
    pd.DataFrame
        Self-response catalogue with columns:
        input_index, {features}_p (target secondary), {features}_s (neighbour),
        polarization_angle, distance, shear_angle,
        measured_e1_m, measured_e2_m, measured_e1_p, measured_e2_p,
        delta_et1, delta_et2, S/N_m, S/N_p, case.
    """
    if data_path is None:
        raise ValueError("data_path is required")
    path = data_path
    shear_case = _normalize_shear_cases(shear_cases)

    # Load input catalogue for the sheared case, which has secondaries'
    # applied shear.
    input_cat_path = os.path.join(
        path, f'case{case}_{shear_case[1]}', real,
        'catalogues/input', f'gals_info_{tile_name}.feather',
    )
    icat = pd.read_feather(input_cat_path)
    shear_component_convention = _frame_shear_component_convention(
        icat,
        default=_read_generated_catalog_convention(path, case, shear_case[1]),
    )

    total_input_count = _generated_catalog_count(path, case, shear_case[1])
    input_id_to_row, input_ids = _input_id_lookup(icat)
    _, secondary_rows = _primary_secondary_rows(icat, total_input_count)
    secondary_ids = input_ids[secondary_rows]  # target galaxies

    def get_match(case_path_segment):
        """Load shape/match catalogues, filter to secondary-target detections."""
        base_cat_path = os.path.join(path, case_path_segment, real, 'catalogues')
        shape_cat_path = os.path.join(
            base_cat_path, 'Shapes',
            f'shape_catalogue_detect_position{secondary_shape_suffix}_{tile_name}.feather',
        )
        match_cat_path = os.path.join(
            base_cat_path, 'CrossMatch', f'{tile_name}_rot0_matched.feather',
        )

        scat = pd.read_feather(shape_cat_path)
        mcat = pd.read_feather(match_cat_path)

        # Remove detections with very bright neighbors (same logic as blending
        # response; prevents catastrophic shape failures near very bright stars/gals)
        reject_idx = utils.remove_detection_w_bright_neighbour(
            scat['X_WORLD'].array, scat['Y_WORLD'].array,
            scat['FLUX_AUTO'].array,
            ratio_max=5, r_min=0, r_max=3 / 3600,
        )
        scat = scat.drop(reject_idx)

        _, ind1_, _ = np.intersect1d(mcat['id_detec'] - 1, scat.index, return_indices=True)
        mcat = mcat.iloc[ind1_].reset_index(drop=True)

        # Keep only matches whose stable input id is in the sheared secondary half.
        comm, ind1, _ = np.intersect1d(
            mcat['id_input'].to_numpy(dtype=int), secondary_ids, return_indices=True
        )
        return scat, mcat, comm, ind1

    scat1, mcat1, comm1, ind1 = get_match(f'case{case}_{shear_case[0]}')
    scat2, mcat2, comm2, ind2 = get_match(f'case{case}_{shear_case[1]}')

    # Common secondaries detected at both shear values
    comm, ind1_, ind2_ = np.intersect1d(comm1, comm2, return_indices=True)
    idx_det1 = mcat1['id_detec'].iloc[ind1[ind1_]].to_numpy() - 1
    idx_det2 = mcat2['id_detec'].iloc[ind2[ind2_]].to_numpy() - 1
    comm_rows = _lookup_input_rows(input_id_to_row, comm, label='secondary target ids')

    # --- Neighbor finding ---
    # For self-response, the relevant "nearest neighbor" is any nearby galaxy
    # (primary or other secondary) that may blend with the target.
    # Using all galaxies as the search pool, excluding the target itself.
    all_pos = icat[['RA_input', 'DEC_input']].to_numpy()
    kdt_in = KDTree(all_pos)
    # Query at the target's truth position. SExtractor positions can be
    # attached to a blended neighbour, while the neighbour features here are
    # truth-catalogue context for the target itself.
    tgt_pos = icat[['RA_input', 'DEC_input']].iloc[comm_rows]

    gal_features = [
        'RA_input', 'DEC_input', 'redshift_input', 'Re_input',
        'axis_ratio_input', 'position_angle_input', 'sersic_n_input', 'r_input',
    ]

    if nearest_only:
        # Query several neighbours so the non-self nearest is robust to exact
        # duplicates or zero-radius exclusions around the target.
        query_k = max(k + 1, 8)
        dst, ind = kdt_in.query(
            tgt_pos, k=query_k, distance_upper_bound=r_max / 3600, workers=-1,
        )
        dst = np.atleast_2d(dst)
        ind = np.atleast_2d(ind)
        if dst.shape[0] != comm.shape[0]:
            dst = dst.T
            ind = ind.T

        valid = np.isfinite(dst)
        valid &= ind != all_pos.shape[0]
        valid &= ind != comm_rows[:, None]
        valid &= dst >= r_min / 3600
        has_neighbor = np.any(valid, axis=1)

        neigh_idx_all = np.full(comm.shape[0], -1, dtype=int)
        neigh_dst_all = np.full(comm.shape[0], np.nan)
        if np.any(has_neighbor):
            first_valid = np.argmax(valid[has_neighbor], axis=1)
            row_idx = np.where(has_neighbor)[0]
            neigh_idx_all[row_idx] = ind[row_idx, first_valid]
            neigh_dst_all[row_idx] = dst[row_idx, first_valid]

        keep = np.ones(comm.shape[0], dtype=bool) if include_isolated else has_neighbor
        tgt_idx_input = comm[keep]
        tgt_idx_rows = comm_rows[keep]
        tgt_idx_detec1 = idx_det1[keep]
        tgt_idx_detec2 = idx_det2[keep]
        neigh_idx_rows = neigh_idx_all[keep]
        neigh_dst = neigh_dst_all[keep]
        neighbored = has_neighbor[keep]

        neighbour_input_ids = np.full(tgt_idx_input.shape[0], np.nan)
        if np.any(neighbored):
            neighbour_input_ids[neighbored] = input_ids[neigh_idx_rows[neighbored]]

        R_cat = {
            'input_index': tgt_idx_input,
            'input_index_sec': neighbour_input_ids,
            'neighbored': neighbored,
        }

        for gf in gal_features:
            R_cat[f'{gf}_p'] = icat[gf].iloc[tgt_idx_rows].to_numpy()
            neighbour_values = np.full(tgt_idx_input.shape[0], np.nan)
            if np.any(neighbored):
                neighbour_values[neighbored] = icat[gf].iloc[neigh_idx_rows[neighbored]].to_numpy()
            R_cat[f'{gf}_s'] = neighbour_values

        tgt_pos_arr = icat[['RA_input', 'DEC_input']].iloc[tgt_idx_rows].to_numpy()
        R_cat['polarization_angle'] = np.full(tgt_idx_input.shape[0], np.nan)
        if np.any(neighbored):
            neigh_pos_arr = icat[['RA_input', 'DEC_input']].iloc[neigh_idx_rows[neighbored]].to_numpy()
            pola_ang = angle_between(tgt_pos_arr[neighbored].T, neigh_pos_arr.T)
            pola_ang += np.pi
            pola_ang[pola_ang > np.pi * 2] -= np.pi * 2
            R_cat['polarization_angle'][neighbored] = pola_ang / np.pi * 180
        R_cat['distance'] = neigh_dst * 3600  # arcsec
    else:
        # k+1 because nearest neighbor will usually be the target itself.
        query_k = k + 1
        dst, ind = kdt_in.query(tgt_pos, k=query_k, distance_upper_bound=r_max / 3600, workers=-1)
        found_mask = (
            (ind.reshape(-1) != all_pos.shape[0])
            & (ind.reshape(-1) != np.repeat(comm_rows, query_k))
            & (dst.reshape(-1) >= r_min / 3600)
        )
        found_idx = np.where(found_mask)[0]

        neigh_idx_rows = ind.reshape(-1)[found_idx]
        tgt_idx_detec1 = np.repeat(idx_det1, query_k)[found_idx]
        tgt_idx_detec2 = np.repeat(idx_det2, query_k)[found_idx]
        tgt_idx_input = np.repeat(comm, query_k)[found_idx]
        tgt_idx_rows = np.repeat(comm_rows, query_k)[found_idx]

        R_cat = {
            'input_index': tgt_idx_input,
            'input_index_sec': input_ids[neigh_idx_rows],
        }

        for gf in gal_features:
            R_cat[f'{gf}_p'] = icat[gf].iloc[tgt_idx_rows].to_numpy()
            R_cat[f'{gf}_s'] = icat[gf].iloc[neigh_idx_rows].to_numpy()

        tgt_pos_arr = icat[['RA_input', 'DEC_input']].iloc[tgt_idx_rows].to_numpy()
        neigh_pos_arr = icat[['RA_input', 'DEC_input']].iloc[neigh_idx_rows].to_numpy()

        pola_ang = angle_between(tgt_pos_arr.T, neigh_pos_arr.T)
        pola_ang += np.pi
        pola_ang[pola_ang > np.pi * 2] -= np.pi * 2

        R_cat['polarization_angle'] = pola_ang / np.pi * 180
        R_cat['distance'] = dst.reshape(-1)[found_idx] * 3600  # arcsec
        R_cat['neighbored'] = np.full(len(tgt_idx_input), True)

    # KEY DIFFERENCE vs blending response: shear_angle is the TARGET
    # secondary's own applied shear direction (since the target is the one
    # being sheared).
    _, R_cat['shear_angle'] = e2ang(
        icat['gamma1_input'].iloc[tgt_idx_rows].to_numpy(),
        icat['gamma2_input'].iloc[tgt_idx_rows].to_numpy(),
        convention=shear_component_convention,
    )

    # Shape measurements and response (for the TARGET secondary)
    e_c1 = scat1[['NGMIX_G1', 'NGMIX_G2']].loc[tgt_idx_detec1].to_numpy().T
    e_c2 = scat2[['NGMIX_G1', 'NGMIX_G2']].loc[tgt_idx_detec2].to_numpy().T

    # Rotate into the target's own shear frame
    et_c1 = np.array(spin2rot(e_c1[0, :], e_c1[1, :], R_cat['shear_angle']))
    et_c2 = np.array(spin2rot(e_c2[0, :], e_c2[1, :], R_cat['shear_angle']))

    R_cat['measured_e1_m'] = e_c1[0, :]
    R_cat['measured_e2_m'] = e_c1[1, :]
    R_cat['measured_e1_p'] = e_c2[0, :]
    R_cat['measured_e2_p'] = e_c2[1, :]
    # delta_et is (sheared - unsheared) in the target's own shear frame
    R_cat['delta_et1'] = et_c2[0, :] - et_c1[0, :]  # tangential response
    R_cat['delta_et2'] = et_c2[1, :] - et_c1[1, :]  # cross response

    R_cat['S/N_m'] = (scat1['FLUX_AUTO'] / scat1['FLUXERR_AUTO']).loc[tgt_idx_detec1].to_numpy().T
    R_cat['S/N_p'] = (scat2['FLUX_AUTO'] / scat2['FLUXERR_AUTO']).loc[tgt_idx_detec2].to_numpy().T

    R_cat['shear_angle'] = R_cat['shear_angle'] / np.pi * 180
    R_cat['shear_component_convention'] = np.full(len(tgt_idx_detec1), shear_component_convention)
    R_cat['case'] = np.full(len(tgt_idx_detec1), case)

    R_df = pd.DataFrame(data=R_cat)

    # Flag failed measurements
    fail_mask = ((e_c1[0, :] == -1.) & (e_c2[0, :] == -1.)) | \
                ((e_c1[0, :] == 0.) & (e_c2[0, :] == 0.))
    if np.any(fail_mask):
        R_df.loc[fail_mask, ['delta_et1', 'delta_et2']] = None

    return R_df


# --- Detection catalogue ---

def retrieve_detection(case, shear='0.0', real='real0', r_max=10, r_min=0, k=10,
                       data_path=None, tile_name='tile180.0_-0.5',
                       include_measured=False, measured_columns=None,
                       attach_nearest_neighbor=False):
    """
    Build the detection catalogue for one simulation case.

    For each galaxy, finds neighbors via KDTree and annotates whether
    the galaxy was detected by SExtractor.

    Parameters
    ----------
    case : int
        Simulation case index.
    shear : str
        Shear value string.
    real : str
        Realization tag.
    r_max : float
        Maximum search radius in arcsec.
    r_min : float
        Minimum search radius in arcsec.
    k : int
        Number of neighbors.
    data_path : str, optional
        Base data directory.
    include_measured : bool, optional
        If True, join SExtractor measured quantities for matched detections.
        Unmatched/non-detected rows receive NaN measured values.  The default
        is False to preserve the historical blendemu detection catalogue.
    measured_columns : sequence of str or "all", optional
        SExtractor columns to join when include_measured=True.  If omitted,
        a default compact set of flux, magnitude, size, position, flag, and
        morphology columns is used.
    attach_nearest_neighbor : bool, optional
        If True, rows with no neighbour inside ``r_max`` still receive the
        nearest non-self catalogue neighbour's properties, distance, and pair
        angle.  ``neighbored`` remains False for these rows, preserving the
        close-neighbour/rendered-neighbour flag.

    Returns
    -------
    pd.DataFrame
        Detection catalogue with galaxy features, neighbor info, and detected flag.
    """
    if data_path is None:
        raise ValueError("data_path is required")
    path = data_path
    # Load catalogues
    input_cat_path = os.path.join(
        path, f'case{case}_{shear}', real,
        'catalogues/input', f'gals_info_{tile_name}.feather',
    )
    icat = pd.read_feather(input_cat_path)
    input_ids = _input_ids(icat)

    full_cat_path = os.path.join(path, f'gals{case}_{shear}.feather')
    icat_with_z = pd.read_feather(full_cat_path)
    shear_component_convention = _frame_shear_component_convention(icat_with_z)
    redshifts = icat_with_z['redshift'].iloc[input_ids].values
    del icat_with_z

    gal_num = icat.shape[0]
    row_index = np.arange(0, gal_num)

    # --- Neighbor finding ---
    sec_pos = icat[['RA_input', 'DEC_input']].to_numpy()
    kdt_in = KDTree(sec_pos)
    pri_pos = icat[['RA_input', 'DEC_input']].to_numpy()
    dst, ind = kdt_in.query(pri_pos, k=k, distance_upper_bound=r_max / 3600, workers=-1)

    valid_entries = (dst > r_min / 3600) & np.isfinite(dst)
    valid_mask = valid_entries.reshape(-1)
    found_idx = np.where(valid_mask)[0]
    not_found_mask = ~np.any(valid_entries, axis=1)
    not_found_idx = np.where(not_found_mask)[0]

    sec_idx_rows = ind.reshape(-1)[found_idx]
    pri_idx_rows = np.repeat(row_index, k)[found_idx]

    # --- Build output ---
    R_cat = {
        'input_index': input_ids[pri_idx_rows],
        'input_index_sec': input_ids[sec_idx_rows],
    }

    gal_features = ['Re_input', 'axis_ratio_input', 'RA_input', 'DEC_input',
                    'position_angle_input', 'sersic_n_input', 'r_input']
    optional_features = [
        'gamma1_input', 'gamma2_input',
        'e1_input_rot0', 'e2_input_rot0',
    ]
    gal_features.extend([gf for gf in optional_features if gf in icat.columns])

    for gf in gal_features:
        R_cat[f'{gf}_p'] = icat[gf].iloc[pri_idx_rows].to_numpy()
        R_cat[f'{gf}_s'] = icat[gf].iloc[sec_idx_rows].to_numpy()

    R_cat['redshift_input_p'] = redshifts[pri_idx_rows]
    R_cat['redshift_input_s'] = redshifts[sec_idx_rows]

    pri_pos_arr = icat[['RA_input', 'DEC_input']].iloc[pri_idx_rows].to_numpy()
    sec_pos_arr = icat[['RA_input', 'DEC_input']].iloc[sec_idx_rows].to_numpy()
    R_cat['polarization_angle'] = angle_between(pri_pos_arr.T, sec_pos_arr.T) / np.pi * 180
    R_cat['distance'] = dst.reshape(-1)[found_idx] * 3600

    if shear != '0.0':
        _, R_cat['shear_angle'] = e2ang(
            icat['gamma1_input'].iloc[sec_idx_rows].to_numpy(),
            icat['gamma2_input'].iloc[sec_idx_rows].to_numpy(),
            convention=shear_component_convention,
        )
        R_cat['shear_angle'] = R_cat['shear_angle'] / np.pi * 180

    # --- Append "no neighbor" entries ---
    R_cat['input_index'] = np.append(R_cat['input_index'], input_ids[not_found_idx])
    nan_array = np.full(not_found_idx.shape[0], np.nan)

    nearest_idx = nan_array
    nearest_dst = nan_array
    nearest_pola_ang = nan_array
    nearest_shear_angle = nan_array
    nearest_features = {
        gf: nan_array
        for gf in gal_features
    }
    nearest_redshift = nan_array

    if attach_nearest_neighbor and not_found_idx.shape[0] > 0:
        nearest_k = max(2, k)
        nearest_dst_all, nearest_ind_all = kdt_in.query(
            pri_pos[not_found_idx],
            k=nearest_k,
            workers=-1,
        )
        nearest_dst_all = np.atleast_2d(nearest_dst_all)
        nearest_ind_all = np.atleast_2d(nearest_ind_all)
        if nearest_dst_all.shape[0] != not_found_idx.shape[0]:
            nearest_dst_all = nearest_dst_all.T
            nearest_ind_all = nearest_ind_all.T

        valid_nearest = np.isfinite(nearest_dst_all)
        valid_nearest &= nearest_ind_all != not_found_idx[:, None]
        valid_nearest &= nearest_dst_all > r_min / 3600
        has_nearest = np.any(valid_nearest, axis=1)

        nearest_idx = np.full(not_found_idx.shape[0], np.nan)
        nearest_dst = np.full(not_found_idx.shape[0], np.nan)
        if np.any(has_nearest):
            first_valid = np.argmax(valid_nearest[has_nearest], axis=1)
            row_idx = np.where(has_nearest)[0]
            nearest_idx[row_idx] = nearest_ind_all[row_idx, first_valid]
            nearest_dst[row_idx] = nearest_dst_all[row_idx, first_valid]

        good = np.isfinite(nearest_idx)
        nearest_idx_int = nearest_idx[good].astype(int)
        nearest_features = {}
        for gf in gal_features:
            values = np.full(not_found_idx.shape[0], np.nan)
            values[good] = icat[gf].iloc[nearest_idx_int].to_numpy()
            nearest_features[gf] = values

        nearest_redshift = np.full(not_found_idx.shape[0], np.nan)
        nearest_redshift[good] = redshifts[nearest_idx_int]

        nearest_pola_ang = np.full(not_found_idx.shape[0], np.nan)
        if np.any(good):
            pri_pos_nearest = icat[['RA_input', 'DEC_input']].iloc[not_found_idx[good]].to_numpy()
            sec_pos_nearest = icat[['RA_input', 'DEC_input']].iloc[nearest_idx_int].to_numpy()
            nearest_pola_ang[good] = angle_between(pri_pos_nearest.T, sec_pos_nearest.T) / np.pi * 180

        nearest_shear_angle = np.full(not_found_idx.shape[0], np.nan)
        if shear != '0.0' and np.any(good):
            _, nearest_shear_angle_good = e2ang(
                icat['gamma1_input'].iloc[nearest_idx_int].to_numpy(),
                icat['gamma2_input'].iloc[nearest_idx_int].to_numpy(),
                convention=shear_component_convention,
            )
            nearest_shear_angle[good] = nearest_shear_angle_good / np.pi * 180

    nearest_input_ids = np.full(not_found_idx.shape[0], np.nan)
    good_nearest_id = np.isfinite(nearest_idx)
    if np.any(good_nearest_id):
        nearest_input_ids[good_nearest_id] = input_ids[nearest_idx[good_nearest_id].astype(int)]
    R_cat['input_index_sec'] = np.append(R_cat['input_index_sec'], nearest_input_ids)

    for gf in gal_features:
        R_cat[f'{gf}_p'] = np.append(R_cat[f'{gf}_p'], icat[gf].iloc[not_found_idx].to_numpy())
        R_cat[f'{gf}_s'] = np.append(R_cat[f'{gf}_s'], nearest_features[gf])

    R_cat['redshift_input_p'] = np.append(R_cat['redshift_input_p'], redshifts[not_found_idx])
    R_cat['redshift_input_s'] = np.append(R_cat['redshift_input_s'], nearest_redshift)

    R_cat['polarization_angle'] = np.append(R_cat['polarization_angle'], nearest_pola_ang)
    R_cat['distance'] = np.append(R_cat['distance'], nearest_dst * 3600)
    if shear != '0.0':
        R_cat['shear_angle'] = np.append(R_cat.get('shear_angle', []), nearest_shear_angle)

    R_cat['neighbored'] = np.concatenate([
        np.full(len(found_idx), True),
        np.full(len(not_found_idx), False),
    ])

    leng = len(R_cat['neighbored'])
    R_cat['shear_component_convention'] = np.full(leng, shear_component_convention)

    # --- Detection status ---
    match_cat_path = os.path.join(
        path, f'case{case}_{shear}', real,
        'catalogues/CrossMatch', f'{tile_name}_rot0_matched.feather',
    )
    mcat = pd.read_feather(match_cat_path)
    comm, _, _ = np.intersect1d(
        mcat['id_input'].to_numpy(dtype=int), input_ids, return_indices=True
    )

    R_cat['detected'] = np.full(leng, False)
    R_cat['detected'][np.isin(R_cat['input_index'], comm)] = True

    if include_measured:
        sex_cat_path = os.path.join(
            path, f'case{case}_{shear}', real,
            'catalogues/SExtractor', f'{tile_name}_bandr_rot0.feather',
        )
        _append_detection_measurements(
            R_cat=R_cat,
            mcat=mcat,
            sex_cat_path=sex_cat_path,
            primary_index=input_ids,
            measured_columns=measured_columns,
        )

    R_cat['case'] = np.full(leng, case)
    R_cat['shear_case'] = np.full(leng, float(shear))
    return pd.DataFrame(data=R_cat)


def _append_detection_measurements(
    R_cat,
    mcat,
    sex_cat_path,
    primary_index,
    measured_columns=None,
):
    """Attach SExtractor measurements to detection-catalogue rows in-place."""
    if measured_columns is None:
        measured_columns = DEFAULT_DETECTION_MEASURED_COLUMNS

    sexcat = pd.read_feather(sex_cat_path)
    if measured_columns == 'all':
        measured_columns = list(sexcat.columns)
    measured_columns = list(measured_columns)
    missing = [column for column in measured_columns if column not in sexcat.columns]
    if missing:
        raise KeyError(f"Missing SExtractor measured columns: {missing}")

    matched = mcat[mcat['id_input'].isin(primary_index)].copy()
    if 'distance_pixel_CM' in matched.columns:
        matched = matched.sort_values('distance_pixel_CM')
    matched = matched.drop_duplicates('id_input', keep='first')

    input_index = np.asarray(R_cat['input_index'], dtype=int)
    row_count = input_index.shape[0]

    def append_match_column(source_column, output_column, offset=0.0):
        values = np.full(row_count, np.nan)
        if source_column not in matched.columns:
            R_cat[output_column] = values
            return None
        lookup = pd.Series(
            matched[source_column].to_numpy(dtype=float) + offset,
            index=matched['id_input'].to_numpy(dtype=int),
        )
        mapped = lookup.reindex(input_index).to_numpy()
        R_cat[output_column] = mapped
        return mapped

    det_idx = append_match_column('id_detec', 'match_id_detec', offset=-1.0)
    append_match_column('distance_pixel_CM', 'match_distance_pixel_cm')
    append_match_column('dmag_CM', 'match_dmag_cm')

    valid = np.isfinite(det_idx)
    valid &= det_idx >= 0
    valid &= det_idx < len(sexcat)
    det_idx_int = det_idx[valid].astype(int)

    for column in measured_columns:
        values = np.full(row_count, np.nan)
        if np.any(valid):
            values[valid] = sexcat[column].iloc[det_idx_int].to_numpy()
        R_cat[f'measured_{column.lower()}'] = values
