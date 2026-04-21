"""
Inference CLI: apply trained emulators to an input galaxy catalogue and
produce a blending-corrected n(z).

Usage:
  python run_inference.py --config ../configs/fs2_lsst_r.yaml \\
         --catalogue ../data/example_catalog.feather \\
         --nz-file path/to/nz.fits \\
         --output corrected_nz.fits

  # Or with a synthetic Gaussian target n(z)
  python run_inference.py --config ../configs/fs2_lsst_r.yaml \\
         --catalogue ../data/example_catalog.feather \\
         --nz-gaussian 0.7,0.1 --output corrected_nz.npz
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from blendemu.config import load_config, config_summary
from blendemu.inference import BlendingPredictor, force_norm


def load_target_nz(args):
    """Load the target lensing n(z) from one of the supported sources."""
    if args.nz_gaussian:
        mu, sigma = [float(x) for x in args.nz_gaussian.split(',')]
        z = np.linspace(0, 2, 201)
        dndz = np.exp(-(z - mu)**2 / (2 * sigma**2))
        dndz /= dndz.sum() * (z[1] - z[0])
        print(f"Using Gaussian target: mu={mu}, sigma={sigma}")
        return {'BIN_GAUSS': (dndz, z)}

    if args.nz_file.endswith('.fits'):
        from astropy.io import fits
        with fits.open(args.nz_file) as f:
            nz_data = f[args.nz_hdu].data
        bins = [c for c in nz_data.columns.names if c.startswith('BIN')]
        z = nz_data[args.nz_zcol]
        return {b: (nz_data[b], z) for b in bins}

    raise ValueError("Need --nz-file or --nz-gaussian")


def main():
    parser = argparse.ArgumentParser(description='Apply blendemu emulators to correct n(z)')
    parser.add_argument('--config', type=str, required=True, help='YAML config file')
    parser.add_argument('--catalogue', type=str, required=True, help='Input galaxy catalogue (feather)')
    parser.add_argument('--nz-file', type=str, default=None, help='Target n(z) FITS file')
    parser.add_argument('--nz-hdu', type=int, default=7, help='FITS HDU for n(z) (default: 7)')
    parser.add_argument('--nz-zcol', type=str, default='Z_MID', help='Column for z grid (default: Z_MID)')
    parser.add_argument('--nz-gaussian', type=str, default=None,
                        help='Use a Gaussian target n(z) "mean,sigma" instead of a file')
    parser.add_argument('--output', type=str, required=True, help='Output file (.fits or .npz)')
    parser.add_argument('--resample-frac', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    cfg = load_config(args.config)
    print(config_summary(cfg))
    print()

    # Load trained emulators
    conditions = {
        'pixel_size': cfg['simulation']['pixel_scale'],
        'zero_point': cfg['training']['rescale']['zero_mag'],
        'psf_fwhm': cfg['training']['rescale']['psf_fwhm'],
        'moffat_beta': cfg['training']['rescale']['moffat_beta'],
        'pixel_rms': cfg['training']['rescale']['pixel_rms'],
    }
    predictor = BlendingPredictor.load(
        model_dir=cfg['training']['model_dir'],
        tag=cfg['training'].get('model_tag'),
        conditions=conditions,
    )
    print(f"Loaded emulators from {cfg['training']['model_dir']}")

    # Load input catalogue
    print(f"Loading catalogue: {args.catalogue}")
    icat = pd.read_feather(args.catalogue)
    print(f"  {icat.shape[0]:,} galaxies, z range [{icat['redshift'].min():.2f}, {icat['redshift'].max():.2f}]")

    # Load target n(z) (possibly multiple tomographic bins)
    tomo_bins = load_target_nz(args)
    print(f"Target n(z): {len(tomo_bins)} bin(s)")

    # Apply correction per bin
    corrected = {}
    for bin_name, (dndz, z) in tomo_bins.items():
        print(f"\n--- {bin_name} ---")
        _, z_out, delta_n = predictor.correct_nz(icat, (dndz, z),
                                                 resample_frac=args.resample_frac,
                                                 seed=args.seed)
        new_nz = force_norm(dndz + delta_n, np.diff(z_out))
        corrected[bin_name] = {'z': z_out, 'original': dndz, 'corrected': new_nz,
                               'delta_n': delta_n}
        old_mean = np.trapz(dndz * z_out, z_out)
        new_mean = np.trapz(new_nz * z_out, z_out)
        print(f"  <z> shift: {old_mean:.4f} -> {new_mean:.4f} (delta = {new_mean-old_mean:+.4f})")

    # Save
    if args.output.endswith('.fits'):
        _save_fits(corrected, args.output)
    elif args.output.endswith('.npz'):
        _save_npz(corrected, args.output)
    else:
        raise ValueError(f"Unknown output format: {args.output}")
    print(f"\nSaved to {args.output}")


def _save_fits(corrected, path):
    from astropy.io import fits
    cols = []
    for bin_name, d in corrected.items():
        cols.append(fits.Column(name=f'{bin_name}_orig', array=d['original'], format='D'))
        cols.append(fits.Column(name=f'{bin_name}_corr', array=d['corrected'], format='D'))
        cols.append(fits.Column(name=f'{bin_name}_delta', array=d['delta_n'], format='D'))
    # Use the first bin's z grid
    first = next(iter(corrected.values()))
    cols.insert(0, fits.Column(name='Z_MID', array=first['z'], format='D'))
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto(path, overwrite=True)


def _save_npz(corrected, path):
    data = {}
    for bin_name, d in corrected.items():
        data[f'{bin_name}_z'] = d['z']
        data[f'{bin_name}_orig'] = d['original']
        data[f'{bin_name}_corr'] = d['corrected']
        data[f'{bin_name}_delta'] = d['delta_n']
    np.savez(path, **data)


if __name__ == '__main__':
    main()
