"""
Data loading, preprocessing, feature rescaling, and binning utilities.
"""

import numpy as np
import pandas as pd
import xgboost as xgb


# --- Source selection ---

def source_select_reg(dataset, cuts=[[18, 26], [18, 26], [0.1, 1.5], [0.1, 1.5], [0, 10]]):
    """Select sources for regression training based on magnitude, size, and distance cuts."""
    idx_sel = np.where(
        (dataset['r_input_s'] > cuts[0][0]) & (dataset['r_input_s'] < cuts[0][1])
        & (dataset['r_input_p'] > cuts[1][0]) & (dataset['r_input_p'] < cuts[1][1])
        & (dataset['Re_input_s'] > cuts[2][0]) & (dataset['Re_input_s'] < cuts[2][1])
        & (dataset['Re_input_p'] > cuts[3][0]) & (dataset['Re_input_p'] < cuts[3][1])
        & (dataset['distance'] > cuts[4][0]) & (dataset['distance'] < cuts[4][1])
    )[0]
    dataset = dataset.iloc[idx_sel].reset_index()
    return dataset


def source_select_cla(dataset, cuts=[[18, 26], [18, 26], [0.1, 1.5], [0.1, 1.5], [0, 10]]):
    """Select sources for classification training."""
    idx_sel = np.where(
        (dataset['r_input_p'] > cuts[1][0]) & (dataset['r_input_p'] < cuts[1][1])
        & (dataset['Re_input_p'] > cuts[3][0]) & (dataset['Re_input_p'] < cuts[3][1])
        & ((dataset['distance'] > cuts[4][0]) & (dataset['distance'] < cuts[4][1]) | (~dataset['neighbored']))
    )[0]
    dataset = dataset.iloc[idx_sel].reset_index()
    return dataset


# --- Unit conversions ---

def mag2flux(mag, zero_mag):
    return 10**(-0.4 * (mag - zero_mag))


def flux2mag(flux, zero_mag):
    return -2.5 * np.log10(flux) + zero_mag


def moffat_fwhm2Re(fwhm, beta):
    """Convert Moffat FWHM to half-light radius."""
    factor = np.sqrt((2**(1 / (beta - 1)) - 1) / (2**(1 / beta) - 1)) / 2
    return fwhm * factor


def moffat_Re2fwhm(re, beta):
    """Convert Moffat half-light radius to FWHM."""
    factor = np.sqrt((2**(1 / beta) - 1) / (2**(1 / (beta - 1)) - 1)) * 2
    return re * factor


def _convolved_size(r, psf_size):
    return np.sqrt(np.power(r, 2) + np.power(psf_size, 2))


def _deconvolved_size(r, psf_size):
    return np.sqrt(np.power(r, 2) - np.power(psf_size, 2))


# --- Feature rescaling ---

def rescale(dataset, pixel_rms=6, pixel_size=0.2,
            zero_mag=30, psf_fwhm=0.6, moffat_beta=2.4):
    """
    Rescale galaxy features to observation-condition-independent units.

    Deconvolves PSF from sizes, scales distances by primary size,
    converts magnitudes to S/N-like quantities.
    """
    psf_size = moffat_fwhm2Re(psf_fwhm, moffat_beta)
    aperture_rms = pixel_rms * (psf_size / pixel_size)**2 * np.pi

    post_Re_p = _convolved_size(dataset['Re_input_p'].array, psf_size)
    post_Re_s = _convolved_size(dataset['Re_input_s'].array, psf_size)

    dataset['distance_scaled'] = dataset['distance'] / post_Re_p
    dataset['Re_input_p_scaled'] = dataset['Re_input_p'] / post_Re_p
    dataset['Re_input_s_scaled'] = dataset['Re_input_s'] / post_Re_s

    flux_input_p = mag2flux(dataset['r_input_p'], zero_mag)
    flux_input_s = mag2flux(dataset['r_input_s'], zero_mag)
    dataset['flux_ratio'] = np.log10(flux_input_p / flux_input_s)

    dataset['r_input_p_scaled'] = flux2mag(flux_input_p / aperture_rms, zero_mag)
    dataset['r_input_s_scaled'] = flux2mag(flux_input_s / aperture_rms, zero_mag)

    return dataset


# --- Data loading ---

def load_data(path, target_name, select_cuts, rescale_params=None,
              normalized=False, shear=0.1):
    """
    Load a feather catalogue, apply source selection, optional rescaling,
    and split into features (x) and targets (y).

    Parameters
    ----------
    path : str
        Path to feather file.
    target_name : str or list
        Target column name(s).
    select_cuts : list of [min, max] pairs
        Source selection cuts (required — no survey-appropriate default).
    rescale_params : dict, optional
        If given, applies feature rescaling. Must contain keys:
        pixel_rms, pixel_size, zero_mag, psf_fwhm, moffat_beta.
    shear : float or None
        If float, selects regression sources and divides y by shear.
        If None, selects classification sources.
    normalized : bool
        If True, standardize y by (y - mean) / std.

    Returns
    -------
    x : pd.DataFrame
    y : pd.Series or pd.DataFrame
    """
    dataset = pd.read_feather(path)
    if shear:
        dataset = source_select_reg(dataset, cuts=select_cuts)
    else:
        dataset = source_select_cla(dataset, cuts=select_cuts)

    if rescale_params is not None:
        dataset = rescale(dataset, **rescale_params)

    x = dataset.drop(target_name, axis=1)
    y = dataset[target_name]

    if shear:
        y = y / shear

    if normalized:
        y_mean = np.mean(y)
        y_std = np.std(y, ddof=1)
        y = standardize(y, y_mean, y_std)
        print(f'Labels standardized with: y = (y - {y_mean})/{y_std} .')

    return x, y


# --- Standardization ---

def standardize(y, y_mean, y_std):
    return (y - y_mean) / y_std


def reverse_standardize(y, y_mean, y_std):
    return y * y_std + y_mean


# --- Binning ---

def get_bins(x, y, bins=11):
    """Bin y by x and return bin centres and [mean, stderr] per bin."""
    upper, lower = np.max(x), np.min(x)
    bins = np.linspace(upper, lower, bins)
    digitized = np.digitize(x, bins)
    bin_means = np.array([std_of_mean(y[digitized == i]) for i in range(1, len(bins))])
    return (bins[1:] + bins[:-1]) / 2, bin_means


def get_bins_sum(x, y, bins=11):
    """Sum y in each bin; return bin centres, sums, and bootstrap errors."""
    upper, lower = np.max(x), np.min(x)
    bins = np.linspace(lower, upper, bins)
    digitized = np.digitize(x, bins)
    bin_sums = np.array([np.sum(y[digitized == i]) for i in range(1, len(bins))])
    bin_sums_err = np.array([
        np.std(y[digitized == i], ddof=1) * np.sqrt(y[digitized == i].shape[0])
        for i in range(1, len(bins))
    ])
    return (bins[1:] + bins[:-1]) / 2, bin_sums, bin_sums_err


def std_of_mean(data, axis=0):
    """Return (mean, standard error of the mean), ignoring NaNs."""
    mean = np.nanmean(data, axis=axis)
    std_mean = np.nanstd(data, axis=axis, ddof=1) / np.sqrt(
        np.count_nonzero(~np.isnan(data), axis=axis)
    )
    return mean, std_mean


def std_of_the_weighted_mean(values, weights, axis=0):
    """Weighted mean and its standard deviation (Bessel-corrected)."""
    average = np.average(values, weights=weights, axis=axis)
    sum_w = np.sum(weights, axis=axis)
    sum_w2 = np.sum(weights**2, axis=axis)
    sample_variance = np.sum(weights * (values - average)**2, axis=axis) / (sum_w - sum_w2 / sum_w)
    variance = sample_variance * sum_w2 / sum_w**2
    return (average, np.sqrt(variance))


def polyfit(x, y):
    """Linear fit returning [slope, slope_err, intercept, intercept_err]."""
    coeffs, cov = np.polyfit(x, y, 1, cov=True)
    k, b = coeffs
    return [k, np.sqrt(cov[0, 0]), b, np.sqrt(cov[1, 1])]


# --- XGBoost prediction ---

def xgb_pred(model, cat):
    """Run XGBoost prediction on catalogue features matching model feature names."""
    features = cat[model.feature_names]
    DM = xgb.DMatrix(features)
    pred = model.predict(DM, iteration_range=[0, model.best_iteration])
    return pred
