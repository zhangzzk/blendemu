"""
n(z) correction utilities for applying blending response emulators.

Computes the delta-n(z) correction to source redshift distributions
from predicted blending responses and detection probabilities.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import utils, data_utils

# Configuration
R_MAX = 7 / 3600
R_MAX_CLA = 3 / 3600
DETECTION_THRESH = 0.
CLASSIFICATION_NAMES_SEC = [
    'Re_input_s', 'r_input_s', 'sersic_n_input_s',
    'axis_ratio_input_s', 'distance',
]


def make_reg_features(icat1, icat2, r_min=0, r_max=99, k=30):
    """
    Build paired feature catalogue for regression by matching neighbors.
    """
    pos = ['RA', 'DEC']
    pri_idx, sec_idx, separation = utils.kdt_neighbor_finder(
        icat1[pos].to_numpy(), icat2[pos].to_numpy(),
        r_min=r_min, r_max=r_max, k=k,
    )

    sec_cat = icat2.iloc[sec_idx].reset_index(drop=True)
    sec_cat = sec_cat.rename(columns={f: f + '_input_s' for f in icat2.columns})

    pri_cat = icat1.iloc[pri_idx].reset_index(drop=True)
    pri_cat = pri_cat.rename(columns={f: f + '_input_p' for f in icat1.columns})

    response_features = pd.concat((pri_cat, sec_cat), axis=1)
    response_features['distance'] = separation
    return response_features


def icat2reg(icat_i_pri, icat_i_sec, model, conditions, cuts=None, r_max=None, k=None):
    """Prepare regression features from input catalogues.

    cuts / r_max (arcsec) / k default to the module constants when None, but the
    inference layer passes the values the emulator was TRAINED with (persisted in
    the model metadata) so train and inference use identical selection + aperture.
    """
    r_max_deg = R_MAX if r_max is None else r_max / 3600.0
    k = 30 if k is None else k
    reg_features_i = make_reg_features(icat_i_pri, icat_i_sec, r_max=r_max_deg, k=k)
    if cuts is None:
        reg_features_i = data_utils.source_select_reg(reg_features_i)
    else:
        reg_features_i = data_utils.source_select_reg(reg_features_i, cuts=cuts)
    reg_features_i['distance'] *= 3600

    reg_features_i = data_utils.rescale(
        reg_features_i,
        zero_mag=conditions['zero_point'],
        pixel_rms=conditions['pixel_rms'],
        pixel_size=conditions['pixel_size'],
        psf_fwhm=conditions['psf_fwhm'],
        moffat_beta=conditions['moffat_beta'],
    )
    return reg_features_i


def icat2cla(icat_i_pri, icat_i_sec, model, conditions, cla_k=2, predict=True, r_max=None):
    """Prepare classification features and optionally predict detection probability.

    r_max (arcsec) is the neighbour-search radius defining `neighbored`; defaults
    to the module constant R_MAX_CLA (3") when None. No magnitude cut is applied
    here on purpose: at inference we want a detection probability for EVERY input
    galaxy, not just the ones passing the training selection.
    """
    r_max_cla = R_MAX_CLA if r_max is None else r_max / 3600.0
    reg_features_i_cla = make_reg_features(icat_i_pri, icat_i_sec, r_max=999, k=cla_k)

    idx_cla = np.where(reg_features_i_cla['distance'] > r_max_cla)[0]
    reg_features_i_cla.loc[idx_cla, CLASSIFICATION_NAMES_SEC] = np.nan
    reg_features_i_cla['neighbored'] = np.full(reg_features_i_cla.shape[0], True)
    reg_features_i_cla.loc[idx_cla, 'neighbored'] = False

    reg_features_i_cla['distance'] *= 3600

    reg_features_i_cla = data_utils.rescale(
        reg_features_i_cla,
        zero_mag=conditions['zero_point'],
        pixel_rms=conditions['pixel_rms'],
        pixel_size=conditions['pixel_size'],
        psf_fwhm=conditions['psf_fwhm'],
        moffat_beta=conditions['moffat_beta'],
    )

    if predict:
        detect_pred = data_utils.xgb_pred(model, reg_features_i_cla)
        reg_features_i_cla['detection'] = detect_pred

    return reg_features_i_cla


# --- n(z) correction ---

def first_term(x, y, dz):
    idx = np.where((x['redshift_input_s'] > dz[0]) & (x['redshift_input_s'] < dz[1]))[0]
    detection_weights = x['detection'].array.copy()
    detection_weights[detection_weights < DETECTION_THRESH] = 0.
    return y[idx] * detection_weights[idx] / (dz[1] - dz[0])


def second_term(x, y, dz):
    idx = np.where((x['redshift_input_p'] > dz[0]) & (x['redshift_input_p'] < dz[1]))[0]
    detection_weights = x['detection'].array.copy()
    detection_weights[detection_weights < DETECTION_THRESH] = 0.
    return y[idx] * detection_weights[idx] / (dz[1] - dz[0])


def n_correction(x, y, dz, norm):
    """Compute the delta-n(z) correction across redshift bins."""
    def f(idx):
        return (np.sum(first_term(x, y, dz[idx])) - np.sum(second_term(x, y, dz[idx]))) / norm
    delta = Parallel(n_jobs=-1, backend='threading')(delayed(f)(i) for i in range(len(dz)))
    return np.array(delta)
