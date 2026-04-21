"""
Configuration management for blendemu.

Loads a YAML config file, merges with defaults, and provides
a flat namespace for the full pipeline. CLI arguments override
config file values.
"""

import os
import copy
import yaml


DEFAULTS = {
    # --- Input catalogue ---
    'catalog': {
        'type': 'fs2',          # 'fs2' or 'galsbi'
        'path': None,           # path to base catalogue (required)
        'mag_column': 'lsst_r_mag',
        'mag_cut': 27,
        're_convention': 'circularized',  # 'circularized' or 'semi_major'
    },

    # --- Simulation output ---
    'simulation': {
        'output_path': None,    # where to write everything (required)
        'base_config': None,    # path to MultiBand_ImSim base .ini (required)
        'n_cases': 200,
        'case_offset': 0,
        'shear_values': [0.0, 0.1],
        'pixel_scale': 0.2,     # arcsec/pixel
    },

    # --- Noise & PSF ---
    'noise': {
        'source': 'csv',        # 'csv' or 'manual'
        'csv_path': None,       # path to multi-survey noise CSV
        'band': 'LSST_r',
        # manual values (used when source='manual')
        'rms': 6.0,
        'seeing': 0.6,
        'moffat_beta': 2.4,
    },

    # --- Shape measurement ---
    'shape_measurement': {
        'stamp_size': 48,
        'pixel_scale': 0.2,
        'use_pos': 'detect',    # 'true', 'detect', 'real0_detect', 'noshear'
        'tile_name': 'tile180.0_-0.5',
    },

    # --- Catalogue generation (step 4) ---
    'catalogues': {
        'response': {
            'r_max': 10,        # arcsec
            'r_min': 0,
            'k': 20,
        },
        'self_response': {
            'r_max': 10,        # arcsec, nearest-neighbor search for target secondary
            'r_min': 0,
            'k': 5,             # fewer neighbors than blending (only nearest matter for self)
            'shape_suffix': '_secondaries',  # matches run_shape.py --targets=secondaries
        },
        'detection': {
            'r_max_deg': 0.000833,  # degrees (~3 arcsec)
            'r_min_deg': 0.0,
            'k': 2,
        },
        'batch_size': 100,
        'n_jobs': 16,
    },

    # --- Emulator training (step 5) ---
    'training': {
        'model_dir': None,      # defaults to <output_path>/models/
        'model_tag': None,      # e.g. 'lsst_r', 'des_i'; appended to model filenames
        'n_trials': 200,
        'features': [
            'Re_input_p_scaled', 'Re_input_s_scaled',
            'r_input_p_scaled', 'r_input_s_scaled',
            'sersic_n_input_p', 'sersic_n_input_s',
            'distance_scaled',
        ],
        # Source selection cuts: [mag_s, mag_p, Re_s, Re_p, distance]
        'regression_cuts': [[18, 26], [18, 26], [0.1, 1.5], [0.1, 1.5], [0, 7]],
        'self_response_cuts': [[18, 28], [18, 26], [0.05, 3.0], [0.1, 1.5], [0, 10]],
        'classification_cuts': [[18, 26], [18, 26], [0.1, 1.5], [0.1, 1.5], [0, 5]],
        # Rescaling observing conditions (used by data_utils.rescale)
        'rescale': {
            'pixel_rms': 6.0,
            'pixel_size': 0.2,
            'zero_mag': 30,
            'psf_fwhm': 0.6,
            'moffat_beta': 2.4,
        },
        'test_size': 0.2,
        'random_state': 321,
    },

    # --- Inference (step 6) ---
    'inference': {
        'catalogue_path': None,       # default input catalogue
        'nz_file': None,              # target n(z) FITS file (e.g. HSC Y3 cosmosis input)
        'nz_hdu': 7,                  # FITS HDU containing nz table
        'nz_zcol': 'Z_MID',           # z-grid column name
        'resample_frac': 0.25,
        'seed': 42,
    },

    # --- Cluster / MPI ---
    'cluster': {
        'n_mpi': 50,
        'realizations': 'd,0,1',
    },
}


def _deep_merge(base, override):
    """Recursively merge override dict into base dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path):
    """
    Load a YAML config file and merge with defaults.

    Parameters
    ----------
    path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        Complete configuration with defaults filled in.
    """
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = _deep_merge(DEFAULTS, user_cfg)

    # Derived defaults
    if cfg['training']['model_dir'] is None:
        cfg['training']['model_dir'] = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'
        )  # defaults to blendemu/models/

    _validate(cfg)
    return cfg


def _validate(cfg):
    """Check required fields are set."""
    if not cfg['catalog']['path']:
        raise ValueError("config: catalog.path is required")
    if not cfg['simulation']['output_path']:
        raise ValueError("config: simulation.output_path is required")
    if cfg['noise']['source'] == 'csv' and not cfg['noise']['csv_path']:
        raise ValueError("config: noise.csv_path is required when noise.source='csv'")


def config_summary(cfg):
    """Return a human-readable summary string."""
    lines = [
        "blendemu configuration",
        "=" * 50,
        f"  Catalogue:   {cfg['catalog']['type']} ({os.path.basename(cfg['catalog']['path'])})",
        f"  Output:      {cfg['simulation']['output_path']}",
        f"  Cases:       {cfg['simulation']['case_offset']}..{cfg['simulation']['case_offset'] + cfg['simulation']['n_cases'] - 1}",
        f"  Shear:       {cfg['simulation']['shear_values']}",
        f"  Pixel scale: {cfg['simulation']['pixel_scale']} arcsec/pix",
    ]
    if cfg['noise']['source'] == 'csv':
        lines.append(f"  Noise:       {cfg['noise']['band']} from {os.path.basename(cfg['noise']['csv_path'])}")
    else:
        lines.append(f"  Noise:       manual (rms={cfg['noise']['rms']}, seeing={cfg['noise']['seeing']}\")")
    lines += [
        f"  Shape:       stamp={cfg['shape_measurement']['stamp_size']}px, pos={cfg['shape_measurement']['use_pos']}",
        f"  MPI ranks:   {cfg['cluster']['n_mpi']}",
        f"  Realization: {cfg['cluster']['realizations']}",
    ]
    return "\n".join(lines)
