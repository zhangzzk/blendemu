"""
Configuration management for blendemu.

Loads a YAML config file, merges with defaults, and provides
a flat namespace for the full pipeline. CLI arguments override
config file values.
"""

import os
import copy
import warnings
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
        'area_deg2': None,      # optional generated-catalogue sky area for small smoke runs
        'survey': None,         # optional override for [ImSim] survey in base_config
        'max_galaxies_per_realization': None,  # smoke-test cap; None keeps catalogue density
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
            'mode': 'nearest',   # 'nearest' or 'pairs'
            'include_isolated': False,
            'output_prefix': 'self_response_catalogue',
            'shape_suffix': '_secondaries',  # matches run_shape.py --targets=secondaries
        },
        'detection': {
            'r_max': 3.0,           # arcsec
            'r_min': 0.0,           # arcsec
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
        # Detection/classification can use a task-specific feature set.  By
        # default it follows the neighbour-aware regression feature convention.
        'classification_features': [
            'Re_input_p_scaled', 'Re_input_s_scaled',
            'r_input_p_scaled', 'r_input_s_scaled',
            'sersic_n_input_p', 'sersic_n_input_s',
            'distance_scaled',
        ],
        # Source selection cuts: [mag_s, mag_p, Re_s, Re_p, distance]
        'regression_cuts': [[18, 26], [18, 26], [0.1, 1.5], [0.1, 1.5], [0, 7]],
        'self_response_cuts': [[18, 28], [18, 26], [0.05, 3.0], [0.1, 1.5], [0, 10]],
        'classification_cuts': [[18, 28], [18, 28], [0.1, 1.5], [0.1, 1.5], [0, 5]],
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
        # Early stopping counts only improvements larger than min_delta.
        # Classification often has tiny late logloss gains, so give it a
        # task-specific threshold to avoid training hundreds of cosmetic trees.
        'early_stopping_rounds': 30,
        'early_stopping_min_delta': 0.0,
        'classification_early_stopping_min_delta': 1e-4,
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


def _resolve_config_path(value, config_dir):
    """Resolve a path value relative to the YAML config file."""
    if value is None or os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(config_dir, value))


_SIMULATION_OVERRIDE_KEYS = ('case_offset', 'n_cases', 'shear_values', 'shear_cases', 'shear_scale')


def _shear_label(g):
    label = f"{float(g):.3f}".rstrip('0').rstrip('.')
    if label == "-0":
        label = "0"
    if "." not in label:
        label += ".0"
    return label


def resolve_simulation_set(sim_cfg, set_name, overrides=None):
    """Return resolved case/shear settings for a named simulation set.

    ``simulation.response``/``simulation.blending_response`` and
    ``simulation.self_response`` can override the top-level legacy
    ``simulation.n_cases`` and ``simulation.shear_values`` keys.  Optional
    ``overrides`` keeps old catalogue-level case/shear settings working.
    """
    resolved = {
        'case_offset': sim_cfg.get('case_offset', 0),
        'n_cases': sim_cfg['n_cases'],
        'shear_values': sim_cfg['shear_values'],
    }
    aliases = {
        'response': ('response', 'blending_response'),
        'self_response': ('self_response',),
    }.get(set_name, (set_name,))
    for alias in aliases:
        if isinstance(sim_cfg.get(alias), dict):
            resolved.update(sim_cfg[alias])
    if overrides:
        for key in _SIMULATION_OVERRIDE_KEYS:
            if key in overrides:
                resolved[key] = overrides[key]
    return resolved


def simulation_case_window(section_cfg, label='simulation'):
    """Return ``(case_offset, n_cases)`` for a resolved simulation section."""
    offset = int(section_cfg.get('case_offset', 0))
    n_cases = int(section_cfg['n_cases'])
    if n_cases < 0:
        raise ValueError(f"{label} n_cases must be non-negative")
    return offset, n_cases


def simulation_shear_values(section_cfg, label='simulation'):
    """Return shear values for a resolved simulation section."""
    if 'shear_values' in section_cfg:
        values = section_cfg['shear_values']
    elif 'shear_cases' in section_cfg:
        values = section_cfg['shear_cases']
    else:
        raise ValueError(f"{label} must define shear_values or shear_cases")
    return [float(value) for value in values]


def simulation_shear_labels(section_cfg, label='simulation'):
    """Return compact shear labels for a resolved simulation section."""
    if 'shear_cases' in section_cfg:
        labels = [str(value) for value in section_cfg['shear_cases']]
        if len(labels) == 0:
            raise ValueError(f"{label} shear_cases must not be empty")
        return labels
    return [_shear_label(value) for value in simulation_shear_values(section_cfg, label=label)]


def simulation_shear_scale(section_cfg, label='simulation'):
    """Return the finite-difference shear scale for a two-shear section."""
    if 'shear_scale' in section_cfg:
        return float(section_cfg['shear_scale'])
    values = simulation_shear_values(section_cfg, label=label)
    if len(values) != 2:
        raise ValueError(f"{label} shear_values must contain exactly two values")
    return values[1] - values[0]


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
    path = os.path.abspath(path)
    config_dir = os.path.dirname(path)
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}

    cfg = _deep_merge(DEFAULTS, user_cfg)

    cfg['simulation']['base_config'] = _resolve_config_path(
        cfg['simulation']['base_config'], config_dir
    )
    _normalize_detection_radius(cfg, user_cfg)

    # Derived defaults
    if cfg['training']['model_dir'] is None:
        cfg['training']['model_dir'] = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'
        )  # defaults to blendemu/models/

    _validate(cfg)
    _warn_classification_mag_coverage(cfg)
    return cfg


def _normalize_detection_radius(cfg, user_cfg=None):
    """Keep old r_max_deg/r_min_deg configs working while storing arcsec."""
    detection = cfg['catalogues']['detection']
    user_detection = ((user_cfg or {}).get('catalogues') or {}).get('detection') or {}

    if 'r_max_deg' in detection:
        r_max_deg = detection.pop('r_max_deg')
        if 'r_max' not in user_detection:
            detection['r_max'] = r_max_deg * 3600.0

    if 'r_min_deg' in detection:
        r_min_deg = detection.pop('r_min_deg')
        if 'r_min' not in user_detection:
            detection['r_min'] = r_min_deg * 3600.0


def _validate(cfg):
    """Check required fields are set."""
    if not cfg['catalog']['path']:
        raise ValueError("config: catalog.path is required")
    if not cfg['simulation']['output_path']:
        raise ValueError("config: simulation.output_path is required")
    if cfg['noise']['source'] == 'csv' and not cfg['noise']['csv_path']:
        raise ValueError("config: noise.csv_path is required when noise.source='csv'")


def _warn_classification_mag_coverage(cfg):
    """Warn if the detection classifier is trained shallower than the input catalogue."""
    mag_cut = cfg['catalog'].get('mag_cut')
    cla_cuts = cfg['training'].get('classification_cuts')
    if mag_cut is None or not cla_cuts or len(cla_cuts) < 2:
        return

    primary_mag_max = cla_cuts[1][1]
    if primary_mag_max < mag_cut:
        warnings.warn(
            "classification_cuts primary-magnitude upper limit "
            f"({primary_mag_max}) is brighter than catalog.mag_cut ({mag_cut}). "
            "The detection emulator will extrapolate for fainter galaxies and can "
            "produce a flat/high faint-end detection histogram. Increase "
            "training.classification_cuts[1][1] to cover the inference catalogue "
            "and retrain the classification model.",
            RuntimeWarning,
            stacklevel=2,
        )


def config_summary(cfg):
    """Return a human-readable summary string."""
    sim_cfg = cfg['simulation']

    def _sim_set_summary(name):
        section = resolve_simulation_set(sim_cfg, name)
        start, n_cases = simulation_case_window(section, label=f'simulation.{name}')
        end = start + n_cases - 1
        shear_values = section.get('shear_values', section.get('shear_cases'))
        return f"  {name}: cases {start}..{end}, shear {shear_values}"

    lines = [
        "blendemu configuration",
        "=" * 50,
        f"  Catalogue:   {cfg['catalog']['type']} ({os.path.basename(cfg['catalog']['path'])})",
        f"  Output:      {sim_cfg['output_path']}",
        _sim_set_summary('response'),
        _sim_set_summary('self_response'),
        f"  Pixel scale: {sim_cfg['pixel_scale']} arcsec/pix",
    ]
    if sim_cfg.get('max_galaxies_per_realization'):
        lines.append(
            f"  Galaxy cap:  {sim_cfg['max_galaxies_per_realization']:,} per realization"
        )
    if sim_cfg.get('area_deg2'):
        lines.append(f"  Area:        {sim_cfg['area_deg2']} deg^2")
    if sim_cfg.get('survey'):
        lines.append(f"  Survey:      {sim_cfg['survey']}")
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
