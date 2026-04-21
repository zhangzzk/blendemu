"""
Galaxy catalogue generation for image simulations.

Handles loading a base galaxy catalogue, generating random realizations
with applied shear, writing simulation config files, and noise files.
"""

import os
import re
import csv
import numpy as np
import pandas as pd

# All paths below are supplied by the config YAML (see configs/*.example.yaml).
# They default to None so that misconfiguration fails loudly instead of
# silently falling back to a developer-specific absolute path.
BASE_CAT_PATH = None
FS2_CAT_PATH = None
FS2_NOISE_CSV = None
OUTPUT_PATH = None
BASE_CONFIG_PATH = None


def _require_path(path, name):
    if path is None:
        raise ValueError(
            f"{name} is not set. Pass it explicitly or configure it via the "
            f"pipeline config YAML (see configs/*.example.yaml)."
        )
    return path


def e1e2_to_q_phi(e1, e2):
    """
    Convert distortion ellipticity (e1, e2) to axis ratio q and position angle phi.

    Uses: e = (1 - q^2)/(1 + q^2) * exp(2i*phi)

    Returns
    -------
    q : np.ndarray
        Axis ratio b/a.
    phi : np.ndarray
        Position angle in radians.
    """
    e = np.hypot(e1, e2)
    e = np.clip(e, 0, 0.999999999)
    q = np.sqrt((1 - e) / (1 + e))
    phi = 0.5 * np.arctan2(e2, e1)
    return q, phi


def load_and_process_base_catalog(path=BASE_CAT_PATH):
    """
    Load the base galaxy catalogue and perform initial processing.

    Converts ellipticities to axis ratio / position angle,
    applies quality cuts, and computes number density.

    Returns
    -------
    gal_cat : pd.DataFrame
        Processed catalogue.
    n_degree2 : float
        Galaxy number density per square degree.
    """
    from cosmic_toolbox import arraytools as at

    path = _require_path(path, "catalog path")
    print(f"Loading base catalog from {path}...")
    gal_cat = at.rec2pd(at.load_hdf(path))

    gal_cat = gal_cat[['sersic_n', 'int_mag', 'int_r50_arcsec', 'z', 'e1', 'e2']].rename(columns={
        'sersic_n': 'shape/sersic_n',
        'int_mag': 'sdss_r',
        'int_r50_arcsec': 'Re_arcsec',
        'z': 'redshift',
    })

    BA, angle = e1e2_to_q_phi(gal_cat['e1'], gal_cat['e2'])
    gal_cat['BA'] = BA
    gal_cat['angle'] = angle

    gal_cat['angle'] = gal_cat['angle'] / np.pi * 180 + 90
    gal_cat['Re_arcsec'] /= np.sqrt(gal_cat['BA'])

    gal_cat = gal_cat.loc[
        (gal_cat['BA'] > 0.05) & (gal_cat['BA'] < 1.0) &
        (gal_cat['sdss_r'] < 28) & (gal_cat['sdss_r'] > 0) &
        (gal_cat['Re_arcsec'] < 5.) & (gal_cat['Re_arcsec'] > 0.01) &
        (gal_cat['shape/sersic_n'] < 6.) & (gal_cat['shape/sersic_n'] > 0.5)
    ].reset_index(drop=True)

    cat_area = 5.968310365946076
    n_degree2 = gal_cat.shape[0] / cat_area
    print(f"Number density: {n_degree2 / 60**2:.2f} arcmin^-2")

    return gal_cat, n_degree2


def load_fs2_catalog(path=FS2_CAT_PATH, mag_col='lsst_r_mag', mag_cut=27):
    """
    Load the Flagship 2 galaxy catalogue and standardize for the pipeline.

    Converts FS2 columns to the internal convention expected by
    ``generate_catalog_realization``:

    - ``Re_arcsec``: circularized → semi-major (divided by sqrt(q))
    - ``pa_gal``: [-90, 90] → [0, 180] deg
    - Drops 7 rows with NaN morphology

    Parameters
    ----------
    path : str
        Path to the FS2 feather file.
    mag_col : str
        Which magnitude column to use as the detection band (default: 'lsst_r_mag').
    mag_cut : float
        Faint magnitude cut.

    Returns
    -------
    gal_cat : pd.DataFrame
        Processed catalogue with columns:
        sdss_r, shape/sersic_n, Re_arcsec, redshift, BA, angle.
    n_degree2 : float
        Galaxy number density per square degree.
    """
    path = _require_path(path, "catalog path")
    print(f"Loading FS2 catalog from {path}...")
    raw = pd.read_feather(path)
    print(f"  Raw: {raw.shape[0]:,} galaxies")

    # Drop NaN morphology rows
    raw = raw.dropna(subset=['q2d_gal', 'pa_gal']).reset_index(drop=True)

    # Rename to standardized columns
    gal_cat = raw[[mag_col, 'sersic_n', 'Re_arcsec', 'observed_redshift_gal',
                    'q2d_gal', 'pa_gal']].rename(columns={
        mag_col: 'sdss_r',
        'sersic_n': 'shape/sersic_n',
        'observed_redshift_gal': 'redshift',
        'q2d_gal': 'BA',
        'pa_gal': 'angle',
    })

    # PA convention: FS2 [-90, 90] -> MultiBand_ImSim [0, 180]
    gal_cat['angle'] = gal_cat['angle'] + 90.0

    # Re convention: circularized -> semi-major axis for MultiBand_ImSim
    gal_cat['Re_arcsec'] = gal_cat['Re_arcsec'] / np.sqrt(gal_cat['BA'])

    # Quality cuts
    gal_cat = gal_cat.loc[
        (gal_cat['BA'] > 0.05) & (gal_cat['BA'] < 1.0) &
        (gal_cat['sdss_r'] < mag_cut) & (gal_cat['sdss_r'] > 0) &
        (gal_cat['Re_arcsec'] < 10.) & (gal_cat['Re_arcsec'] > 0.01) &
        (gal_cat['shape/sersic_n'] < 6.) & (gal_cat['shape/sersic_n'] > 0.5)
    ].reset_index(drop=True)

    # Number density from the FS2 footprint area
    area_deg2 = (raw['ra_gal'].max() - raw['ra_gal'].min()) * \
                (raw['dec_gal'].max() - raw['dec_gal'].min())
    n_degree2 = gal_cat.shape[0] / area_deg2

    print(f"  After cuts: {gal_cat.shape[0]:,} galaxies over {area_deg2:.0f} sq deg")
    print(f"  Number density: {n_degree2 / 3600:.1f} gal/arcmin^2")
    print(f"  z range:  [{gal_cat['redshift'].min():.2f}, {gal_cat['redshift'].max():.2f}]")
    print(f"  mag range: [{gal_cat['sdss_r'].min():.1f}, {gal_cat['sdss_r'].max():.1f}]")
    print(f"  Re range:  [{gal_cat['Re_arcsec'].min():.3f}, {gal_cat['Re_arcsec'].max():.3f}] arcsec (semi-major)")

    del raw
    return gal_cat, n_degree2


def write_noise_file_from_csv(noise_csv_path=FS2_NOISE_CSV, band='LSST_r',
                              output_path=OUTPUT_PATH):
    """
    Write a MultiBand_ImSim noise CSV from the multi-survey parameter file.

    Reads the per-band RMS, seeing, and Moffat beta from the master CSV
    and writes it in the format MultiBand_ImSim expects (single r-band).

    Parameters
    ----------
    noise_csv_path : str
        Path to the multi-survey noise CSV (e.g., Euclid_Q1_median_LSST10yr.csv).
    band : str
        Band key in the CSV (e.g., 'LSST_r').
    output_path : str
        Directory to write the noise.csv file.
    """
    noise_csv_path = _require_path(noise_csv_path, "noise.csv_path")
    output_path = _require_path(output_path, "simulation.output_path")
    params = pd.read_csv(noise_csv_path).iloc[0]

    rms = params[f'rms_{band}']
    seeing = params[f'InputSeeing_{band}']
    beta = params[f'InputBeta_{band}']

    labels = ['label', 'rmsExpo_r', 'InputSeeing_r', 'InputBeta_r',
              'seeing_e1_r', 'seeing_e2_r', 'chip_id', 'expo_id']
    sky_names = ['180.0_-0.5', '181.0_-0.5', '180.0_0.5', '181.0_0.5']

    noise_file = os.path.join(output_path, 'noise.csv')
    with open(noise_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(labels)
        for name in sky_names:
            writer.writerow([str(v) for v in [name, rms, seeing, beta, 0., 0., 0, 0]])

    print(f"Noise file written to {noise_file}")
    print(f"  Band: {band}, RMS={rms:.3f}, seeing={seeing:.3f}\", beta={beta:.3f}")


def generate_catalog_realization(gal_cat, num_sim, seed, g, shear_type='constant'):
    """
    Generate one realization of a galaxy catalogue with random sampling and applied shear.

    Parameters
    ----------
    gal_cat : pd.DataFrame
        Base galaxy catalogue.
    num_sim : int
        Number of galaxies (should be even).
    seed : int
        Random seed.
    g : float
        Shear magnitude applied to secondary half.
    shear_type : str
        'constant' applies shear to the second half with random orientations.

    Returns
    -------
    pd.DataFrame
        Realization catalogue with columns: index, cata_idx, RA, DEC, g1, g2,
        position_angle, redshift, Re, axis_ratio, sersic_n, r.
    """
    rng = np.random.RandomState(seed)
    idx = rng.choice(gal_cat.index.values, size=num_sim, replace=True)

    mag_CAT = gal_cat['sdss_r'].iloc[idx].to_numpy()
    n_CAT = gal_cat['shape/sersic_n'].iloc[idx].to_numpy()
    Re_CAT = gal_cat['Re_arcsec'].iloc[idx].to_numpy()
    BA_CAT = gal_cat['BA'].iloc[idx].to_numpy()
    z_CAT = gal_cat['redshift'].iloc[idx].to_numpy()
    Ang_CAT = gal_cat['angle'].iloc[idx].to_numpy()

    ra_CAT = rng.uniform(low=180, high=181, size=num_sim)
    dec_CAT = rng.uniform(low=0, high=1, size=num_sim)

    g_CAT = np.zeros((num_sim, 2))
    if shear_type == 'constant':
        angles = rng.uniform(low=0, high=np.pi, size=int(num_sim / 2))
        g_CAT[int(num_sim / 2):, 0] = g * np.sin(2 * angles)
        g_CAT[int(num_sim / 2):, 1] = g * np.cos(2 * angles)

    cat = {
        'index': np.arange(0, num_sim),
        'cata_idx': idx,
        'RA': ra_CAT,
        'DEC': dec_CAT,
        'g1': g_CAT[:, 0],
        'g2': g_CAT[:, 1],
        'position_angle': Ang_CAT,
        'redshift': z_CAT,
        'Re': Re_CAT,
        'axis_ratio': BA_CAT,
        'sersic_n': n_CAT,
        'r': mag_CAT,
    }
    return pd.DataFrame(data=cat)


def write_config_file(suffix, g, path=OUTPUT_PATH,
                      base_config_path=BASE_CONFIG_PATH, file_name_cat=None):
    """
    Read base simulation config, substitute paths, and write a new config file.

    Returns
    -------
    str
        Path to the written config file.
    """
    path = _require_path(path, "simulation.output_path")
    base_config_path = _require_path(base_config_path, "simulation.base_config")
    with open(base_config_path, 'r') as f:
        content = f.read()

    content = re.sub(
        r'(\[Paths\]([^\[])*out_dir\s=\s+).*',
        r'\1%s' % (path + f'case{suffix}_{g:.1f}'),
        content,
    )
    content = re.sub(
        r'(\[Paths\]([^\[])*tmp_dir\s=\s+).*',
        r'\1%s' % (path + f'case{suffix}_{g:.1f}_tmp'),
        content,
    )
    content = re.sub(
        r'(\[GalInfo\]([^\[])*cata_file\s=\s+).*',
        r'\1%s' % file_name_cat,
        content,
    )
    content = re.sub(
        r'(\[NoiseInfo\]([^\[])*cata_file\s=\s+).*',
        r'\1%s' % (path + 'noise.csv'),
        content,
    )

    output_config = path + f'sim_config_case{suffix}_{g:.1f}.ini'
    with open(output_config, 'w') as f:
        f.write(content)

    return output_config


def write_noise_file(path=OUTPUT_PATH):
    """Write the noise configuration CSV file for MultiBand_ImSim."""
    path = _require_path(path, "simulation.output_path")
    labels = ['label', 'rmsExpo_r', 'InputSeeing_r', 'InputBeta_r',
              'seeing_e1_r', 'seeing_e2_r', 'chip_id', 'expo_id']
    sky_names = ['180.0_-0.5', '181.0_-0.5', '180.0_0.5', '181.0_0.5']

    noise_file = os.path.join(path, 'noise.csv')
    with open(noise_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(labels)
        for name in sky_names:
            writer.writerow([str(v) for v in [name, 6., 0.6, 2.4, 0., 0., 0, 0]])

    print(f"Noise file written to {noise_file}")
