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


# --- Geometric helpers ---

def angle_between(pos1, pos2):
    """Polarization angle (radians) between two sets of positions."""
    sep = pos2 - pos1
    an = np.arctan2(sep[1, :], sep[0, :])
    return np.pi - an


def e2ang(e1, e2):
    """Convert shear components to magnitude and half-angle."""
    e1_copy = e1.copy()
    e2_copy = e2.copy()
    e1_copy[np.where(e1 == 0.)] = np.nan
    e2_copy[np.where(e2 == 0.)] = np.nan

    e = np.sqrt(e1_copy**2 + e2_copy**2)
    an = np.arctan(e1_copy / e2_copy)

    an[(e2_copy > 0) & (e1_copy < 0)] += np.pi * 2
    an[(e2_copy < 0)] += np.pi
    an = an / 2
    return e, an


def spin2rot(e1, e2, angle):
    """Rotate spin-2 ellipticity into tangential/cross components."""
    ex = -e1 * np.cos(2 * angle) + e2 * np.sin(2 * angle)
    et = -e1 * np.sin(2 * angle) - e2 * np.cos(2 * angle)
    return -et, -ex


# --- Response catalogue ---

def retrieve_response(case, r_max, r_min, k, real='real0',
                      tile_name='tile180.0_-0.5', data_path=None):
    """
    Build the response catalogue for one simulation case.

    Cross-matches detections at shear=0.0 and shear=0.1, finds secondary
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
    shear_case = ['0.0', '0.1']

    # Load input catalogue
    input_cat_path = os.path.join(
        path, f'case{case}_{shear_case[1]}', real,
        'catalogues/input', f'gals_info_{tile_name}.feather',
    )
    icat = pd.read_feather(input_cat_path)

    gal_num = icat.shape[0]
    pri_num = int(gal_num / 2)
    primary_index = np.arange(0, pri_num)
    secondary_index = np.arange(gal_num - pri_num, gal_num)

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

        comm, ind1, _ = np.intersect1d(mcat['id_input'], primary_index, return_indices=True)
        return scat, mcat, comm, ind1

    scat1, mcat1, comm1, ind1 = get_match(f'case{case}_{shear_case[0]}')
    scat2, mcat2, comm2, ind2 = get_match(f'case{case}_{shear_case[1]}')

    # Common primaries detected at both shear values
    comm, ind1_, ind2_ = np.intersect1d(comm1, comm2, return_indices=True)
    idx_det1 = mcat1['id_detec'].iloc[ind1[ind1_]].to_numpy() - 1
    idx_det2 = mcat2['id_detec'].iloc[ind2[ind2_]].to_numpy() - 1

    # --- Neighbor finding ---
    sec_pos = icat[['RA_input', 'DEC_input']].iloc[secondary_index].to_numpy()
    kdt_in = KDTree(sec_pos)
    pri_pos = scat1[['X_WORLD', 'Y_WORLD']].loc[idx_det1]
    dst, ind = kdt_in.query(pri_pos, k=k, distance_upper_bound=r_max / 3600, workers=-1)

    found_mask = (ind.reshape(-1) != sec_pos.shape[0]) & (dst.reshape(-1) >= r_min / 3600)
    found_idx = np.where(found_mask)[0]

    sec_idx_input = ind.reshape(-1)[found_idx]
    pri_idx_detec1 = np.repeat(idx_det1, k)[found_idx]
    pri_idx_detec2 = np.repeat(idx_det2, k)[found_idx]
    pri_idx_input = np.repeat(comm, k)[found_idx]

    # --- Build output catalogue ---
    R_cat = {'input_index': pri_idx_input}

    gal_features = [
        'RA_input', 'DEC_input', 'redshift_input', 'Re_input',
        'axis_ratio_input', 'position_angle_input', 'sersic_n_input', 'r_input',
    ]
    for gf in gal_features:
        R_cat[f'{gf}_p'] = icat[gf].iloc[pri_idx_input].to_numpy()
        R_cat[f'{gf}_s'] = icat[gf].iloc[secondary_index].iloc[sec_idx_input].to_numpy()

    # Geometry
    pri_pos_arr = scat1[['X_WORLD', 'Y_WORLD']].loc[pri_idx_detec1].to_numpy()
    sec_pos_arr = icat[['RA_input', 'DEC_input']].iloc[secondary_index].iloc[sec_idx_input].to_numpy()

    pola_ang = angle_between(pri_pos_arr.T, sec_pos_arr.T)
    pola_ang += np.pi
    pola_ang[pola_ang > np.pi * 2] -= np.pi * 2

    R_cat['polarization_angle'] = pola_ang / np.pi * 180
    R_cat['distance'] = dst.reshape(-1)[found_idx] * 3600

    _, R_cat['shear_angle'] = e2ang(
        icat['gamma1_input'].iloc[secondary_index].iloc[sec_idx_input].to_numpy(),
        icat['gamma2_input'].iloc[secondary_index].iloc[sec_idx_input].to_numpy(),
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
                           secondary_shape_suffix='_secondaries'):
    """
    Build the self-response catalogue for one simulation case.

    The "self-response" is how each *secondary* galaxy's own measured shape
    changes between the unsheared (shear=0.0) and sheared (shear=0.1) image
    pair. Unlike the blending response (which uses primaries as targets),
    this uses the sheared secondaries as targets and measures their direct
    response to the shear applied to themselves.

    For each secondary detected at both shear values:
      - Rotate measured e1, e2 into the secondary's own shear frame
      - Compute delta_et = e_t(g=0.1) - e_t(g=0.0)
      - <delta_et>/|g| should be ~1 (with corrections from noise, blending,
        and multiplicative bias)

    Features include the target secondary's own input properties (suffix _p)
    and its nearest neighbor's properties (suffix _s), so downstream training
    code (data_utils.source_select_reg / rescale) works unchanged.

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

    Returns
    -------
    pd.DataFrame
        Self-response catalogue with columns:
        input_index, {features}_p (target secondary), {features}_s (nearest
        neighbor), polarization_angle, distance, shear_angle,
        measured_e1_m, measured_e2_m, measured_e1_p, measured_e2_p,
        delta_et1, delta_et2, S/N_m, S/N_p, case.
    """
    if data_path is None:
        raise ValueError("data_path is required")
    path = data_path
    shear_case = ['0.0', '0.1']

    # Load input catalogue (uses g=0.1 case, which has secondaries' applied shear)
    input_cat_path = os.path.join(
        path, f'case{case}_{shear_case[1]}', real,
        'catalogues/input', f'gals_info_{tile_name}.feather',
    )
    icat = pd.read_feather(input_cat_path)

    gal_num = icat.shape[0]
    sec_num = int(gal_num / 2)
    primary_index = np.arange(0, gal_num - sec_num)
    secondary_index = np.arange(gal_num - sec_num, gal_num)  # target galaxies

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

        # Keep only matches whose input index is a secondary (id_input in [num/2, num))
        comm, ind1, _ = np.intersect1d(mcat['id_input'], secondary_index, return_indices=True)
        return scat, mcat, comm, ind1

    scat1, mcat1, comm1, ind1 = get_match(f'case{case}_{shear_case[0]}')
    scat2, mcat2, comm2, ind2 = get_match(f'case{case}_{shear_case[1]}')

    # Common secondaries detected at both shear values
    comm, ind1_, ind2_ = np.intersect1d(comm1, comm2, return_indices=True)
    idx_det1 = mcat1['id_detec'].iloc[ind1[ind1_]].to_numpy() - 1
    idx_det2 = mcat2['id_detec'].iloc[ind2[ind2_]].to_numpy() - 1

    # --- Neighbor finding ---
    # For self-response, the relevant "nearest neighbor" is any nearby galaxy
    # (primary or other secondary) that may blend with the target.
    # Using all galaxies as the search pool, excluding the target itself.
    all_pos = icat[['RA_input', 'DEC_input']].to_numpy()
    kdt_in = KDTree(all_pos)
    # Query at the secondary's detected position (g=0.0 case, arbitrary but consistent)
    tgt_pos = scat1[['X_WORLD', 'Y_WORLD']].loc[idx_det1]

    # k+1 because nearest neighbor will be the target itself
    dst, ind = kdt_in.query(tgt_pos, k=k + 1, distance_upper_bound=r_max / 3600, workers=-1)
    # Drop the self-match (first column)
    dst = dst[:, 1:]
    ind = ind[:, 1:]

    found_mask = (ind.reshape(-1) != all_pos.shape[0]) & (dst.reshape(-1) >= r_min / 3600)
    found_idx = np.where(found_mask)[0]

    neigh_idx_input = ind.reshape(-1)[found_idx]
    tgt_idx_detec1 = np.repeat(idx_det1, k)[found_idx]
    tgt_idx_detec2 = np.repeat(idx_det2, k)[found_idx]
    tgt_idx_input = np.repeat(comm, k)[found_idx]  # input-space index (in [num/2, num))

    # --- Build output catalogue ---
    R_cat = {'input_index': tgt_idx_input}

    gal_features = [
        'RA_input', 'DEC_input', 'redshift_input', 'Re_input',
        'axis_ratio_input', 'position_angle_input', 'sersic_n_input', 'r_input',
    ]
    for gf in gal_features:
        R_cat[f'{gf}_p'] = icat[gf].iloc[tgt_idx_input].to_numpy()  # target secondary
        R_cat[f'{gf}_s'] = icat[gf].iloc[neigh_idx_input].to_numpy()  # nearest neighbor

    # Geometry: vector from target secondary to its neighbor
    tgt_pos_arr = scat1[['X_WORLD', 'Y_WORLD']].loc[tgt_idx_detec1].to_numpy()
    neigh_pos_arr = icat[['RA_input', 'DEC_input']].iloc[neigh_idx_input].to_numpy()

    pola_ang = angle_between(tgt_pos_arr.T, neigh_pos_arr.T)
    pola_ang += np.pi
    pola_ang[pola_ang > np.pi * 2] -= np.pi * 2

    R_cat['polarization_angle'] = pola_ang / np.pi * 180
    R_cat['distance'] = dst.reshape(-1)[found_idx] * 3600  # arcsec

    # KEY DIFFERENCE vs blending response: shear_angle is the TARGET
    # secondary's own applied shear direction (since the target is the one
    # being sheared).
    _, R_cat['shear_angle'] = e2ang(
        icat['gamma1_input'].iloc[tgt_idx_input].to_numpy(),
        icat['gamma2_input'].iloc[tgt_idx_input].to_numpy(),
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
                       data_path=None):
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
        Maximum search radius (in the same units as the catalogue positions).
    r_min : float
        Minimum search radius.
    k : int
        Number of neighbors.
    data_path : str, optional
        Base data directory.

    Returns
    -------
    pd.DataFrame
        Detection catalogue with galaxy features, neighbor info, and detected flag.
    """
    if data_path is None:
        raise ValueError("data_path is required")
    path = data_path
    tile_name = 'tile180.0_-0.5'

    # Load catalogues
    input_cat_path = os.path.join(
        path, f'case{case}_{shear}', real,
        'catalogues/input', f'gals_info_{tile_name}.feather',
    )
    icat = pd.read_feather(input_cat_path)

    full_cat_path = os.path.join(path, f'gals{case}_{shear}.feather')
    icat_with_z = pd.read_feather(full_cat_path)
    redshifts = icat_with_z['redshift'].iloc[icat['index_input']].values
    del icat_with_z

    gal_num = icat.shape[0]
    primary_index = np.arange(0, gal_num)
    secondary_index = np.arange(0, gal_num)

    # --- Neighbor finding ---
    sec_pos = icat[['RA_input', 'DEC_input']].iloc[secondary_index].to_numpy()
    kdt_in = KDTree(sec_pos)
    pri_pos = icat[['RA_input', 'DEC_input']].iloc[primary_index].to_numpy()
    dst, ind = kdt_in.query(pri_pos, k=k, distance_upper_bound=r_max, workers=-1)

    valid_mask = (dst.reshape(-1) > r_min) & (~np.isinf(dst.reshape(-1)))
    found_idx = np.where(valid_mask)[0]
    not_found_mask = (dst[:, 0] == r_min) & (np.isinf(dst[:, 1]))
    not_found_idx = np.where(not_found_mask)[0]

    sec_idx_input = ind.reshape(-1)[found_idx]
    pri_idx_input = np.repeat(primary_index, k)[found_idx]

    # --- Build output ---
    R_cat = {
        'input_index': pri_idx_input,
        'input_index_sec': sec_idx_input,
    }

    gal_features = ['Re_input', 'axis_ratio_input', 'RA_input', 'DEC_input',
                    'position_angle_input', 'sersic_n_input', 'r_input']

    for gf in gal_features:
        R_cat[f'{gf}_p'] = icat[gf].iloc[pri_idx_input].to_numpy()
        R_cat[f'{gf}_s'] = icat[gf].iloc[secondary_index].iloc[sec_idx_input].to_numpy()

    R_cat['redshift_input_p'] = redshifts[pri_idx_input]
    R_cat['redshift_input_s'] = redshifts[secondary_index][sec_idx_input]

    pri_pos_arr = icat[['RA_input', 'DEC_input']].iloc[pri_idx_input].to_numpy()
    sec_pos_arr = icat[['RA_input', 'DEC_input']].iloc[secondary_index].iloc[sec_idx_input].to_numpy()
    R_cat['polarization_angle'] = angle_between(pri_pos_arr.T, sec_pos_arr.T) / np.pi * 180
    R_cat['distance'] = dst.reshape(-1)[found_idx] * 3600

    if shear != '0.0':
        _, R_cat['shear_angle'] = e2ang(
            icat['gamma1_input'].iloc[secondary_index].iloc[sec_idx_input].to_numpy(),
            icat['gamma2_input'].iloc[secondary_index].iloc[sec_idx_input].to_numpy(),
        )
        R_cat['shear_angle'] = R_cat['shear_angle'] / np.pi * 180

    # --- Append "no neighbor" entries ---
    R_cat['input_index'] = np.append(R_cat['input_index'], not_found_idx)
    nan_array = np.full(not_found_idx.shape[0], np.nan)
    R_cat['input_index_sec'] = np.append(R_cat['input_index_sec'], nan_array)

    for gf in gal_features:
        R_cat[f'{gf}_p'] = np.append(R_cat[f'{gf}_p'], icat[gf].iloc[not_found_idx].to_numpy())
        R_cat[f'{gf}_s'] = np.append(R_cat[f'{gf}_s'], nan_array)

    R_cat['redshift_input_p'] = np.append(R_cat['redshift_input_p'], redshifts[not_found_idx])
    R_cat['redshift_input_s'] = np.append(R_cat['redshift_input_s'], nan_array)

    R_cat['polarization_angle'] = np.append(R_cat['polarization_angle'], nan_array)
    R_cat['distance'] = np.append(R_cat['distance'], nan_array)
    if shear != '0.0':
        R_cat['shear_angle'] = np.append(R_cat.get('shear_angle', []), nan_array)

    R_cat['neighbored'] = np.concatenate([
        np.full(len(found_idx), True),
        np.full(len(not_found_idx), False),
    ])

    leng = len(R_cat['neighbored'])

    # --- Detection status ---
    match_cat_path = os.path.join(
        path, f'case{case}_{shear}', real,
        'catalogues/CrossMatch', f'{tile_name}_rot0_matched.feather',
    )
    mcat = pd.read_feather(match_cat_path)
    comm, _, _ = np.intersect1d(mcat['id_input'], primary_index, return_indices=True)

    R_cat['detected'] = np.full(leng, False)
    R_cat['detected'][np.isin(R_cat['input_index'], comm)] = True

    R_cat['case'] = np.full(leng, case)
    return pd.DataFrame(data=R_cat)
