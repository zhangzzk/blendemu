"""
Shape Measurement Script.

Measures galaxy shapes in simulated images using GalSim and/or ngmix,
with MPI parallelization across simulation cases.

Usage:
    mpiexec -n 50 python run_shape.py /path/to/data --case_start 0 --shear_case 0.0 --stamp_size 48
"""

import os
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import wcs
from mpi4py import MPI

from blendemu import shape


def shear_label(g):
    """Compact shear label matching simulation directory names."""
    label = f"{float(g):.3f}".rstrip('0').rstrip('.')
    if label == "-0":
        label = "0"
    if "." not in label:
        label += ".0"
    return label


def get_shape(stamp, psf_im, scale=0.2, methods=['ngmix']):
    """Measure shape on a single stamp using specified methods."""
    g_galsim = np.array([-1., -1.])
    g_ngmix = np.array([-1., -1.])

    if 'galsim' in methods:
        try:
            g_galsim = shape.galsim_EstimateShear(stamp, psf_im)
        except Exception:
            pass

    if 'ngmix' in methods:
        try:
            obs = shape.make_obs(stamp, psf_im, scale)
            g_ngmix = shape.ngmix_psf_correct(obs)['g']
        except Exception:
            pass

    return np.append(g_galsim, g_ngmix)


def main():
    parser = argparse.ArgumentParser(description='Shape Measurement Pipeline')
    parser.add_argument('path', type=str, help='Image path')
    parser.add_argument('--case_start', type=int, default=0, help='Starting case index')
    parser.add_argument('--shear_case', type=float, default=0.1, help='Shear case value')
    parser.add_argument('--realizations', type=str, default='0,1', help='Realization range')
    parser.add_argument('--stamp_size', type=int, default=96, help='Stamp size in pixels')
    parser.add_argument('--use_pos', type=str, default='true', help='Position source: true/detect/real0_detect/noshear')
    parser.add_argument('--targets', type=str, default='primaries',
                        choices=['primaries', 'secondaries', 'all'],
                        help='Which input galaxies to measure: primaries (first half), '
                             'secondaries (second half), or all.')
    parser.add_argument('--tile_name', type=str, default="tile180.0_-0.5", help='Tile name')
    parser.add_argument('--pixel_scale', type=float, default=0.2, help='Pixel scale in arcsec/pix')

    args = parser.parse_args()

    path_ = args.path
    case_start = args.case_start
    shear_case = args.shear_case
    realizations = args.realizations.split(',')
    stamp_size = args.stamp_size
    use_pos = args.use_pos
    tile_name = args.tile_name
    scale = args.pixel_scale
    targets = args.targets

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Adaptive realization logic
    if 'd' in realizations:
        nreal0 = int(realizations[1])
        nreal = int(realizations[2])
        case = f'case{rank + case_start}'
    elif realizations == ['0', '1']:
        nreal0 = 0
        nreal = 1
        case = f'case{rank + case_start}'
    else:
        snr_cat_path = os.path.join(path_, 'snr_catalogue.feather')
        nreals = pd.read_feather(snr_cat_path)['nreal'].astype(int)
        nreal0 = 1
        idx_lower = int(realizations[0])
        idx_upper = int(realizations[1])
        nreals_idx = np.where((nreals >= idx_lower) & (nreals < idx_upper))[0]
        if rank == 0:
            print(f'Number of cases within this realization range: {nreals_idx.shape}')
        if rank + case_start < len(nreals_idx):
            nreal = nreals[nreals_idx[rank + case_start]]
            case = f'case{nreals_idx[rank + case_start]}'
        else:
            print(f"Rank {rank} has no work.")
            return

    case = f'{case}_{shear_label(shear_case)}'

    for i in range(nreal0, nreal):
        real = f'real{i}'

        # Construct paths
        if use_pos == 'real0_detect':
            base_dir = f"{path_}/{case}/real0/"
            path = f"{path_}/{case}/{real}/"
        elif use_pos == 'noshear':
            base_dir = f"{path_}/{case}_2/{real}/"
            path = f"{path_}/{case}/{real}/"
        else:
            path = f"{path_}/{case}/{real}/"
            base_dir = path

        if use_pos == 'true':
            out_name = '_true_position'
            catalogue_name = f"{path}catalogues/input/gals_info_{tile_name}.feather"
        elif use_pos == 'detect':
            out_name = '_detect_position'
            catalogue_name = f"{path}catalogues/SExtractor/{tile_name}_bandr_rot0.feather"
            match_name = f"{path}catalogues/CrossMatch/{tile_name}_rot0_matched.feather"
        elif use_pos == 'real0_detect':
            out_name = '_detect_position'
            catalogue_name = f"{base_dir}catalogues/SExtractor/{tile_name}_bandr_rot0.feather"
            match_name = f"{base_dir}catalogues/CrossMatch/{tile_name}_rot0_matched.feather"
        elif use_pos == 'noshear':
            out_name = '_noshear_position'
            catalogue_name = f"{base_dir}catalogues/SExtractor/{tile_name}_bandr_rot0.feather"
            match_name = f"{base_dir}catalogues/CrossMatch/{tile_name}_rot0_matched.feather"

        # Append target suffix to output name (keeps 'primaries' filename backward-compatible)
        if targets == 'secondaries':
            out_name += '_secondaries'
        elif targets == 'all':
            out_name += '_all'
        out_name += f'_{tile_name}'

        # Try background-subtracted image first, fall back to raw
        science_image_name = f"{path}images/original/-BACKGROUND/{tile_name}_bandr_rot0.fits"
        if not os.path.isfile(science_image_name):
            science_image_name = f"{path}images/original/{tile_name}_bandr_rot0.fits"
        psf_image_name = f"{path}images/original/psf_{tile_name}_bandr/psf_ima.fits"
        shape_path = f"{path}catalogues/Shapes/"

        if os.path.isfile(f"{shape_path}shape_catalogue{out_name}.feather"):
            print(f'Shape catalogue exists for {case}/{real}. Skipping.')
            continue

        os.makedirs(shape_path, exist_ok=True)

        try:
            gal_im, gal_im_header = fits.getdata(science_image_name, header=True)
            psf_im = fits.getdata(psf_image_name)
        except FileNotFoundError:
            print(f"Image not found: {science_image_name}")
            continue

        # Determine indices to measure
        input_cat_path = f"{path}catalogues/input/gals_info_{tile_name}.feather"
        if os.path.exists(input_cat_path):
            input_cat = pd.read_feather(input_cat_path)
            total_num = input_cat.shape[0]
            del input_cat
        else:
            total_num = 0

        if targets == 'primaries':
            index = np.arange(0, total_num // 2).astype(int)
        elif targets == 'secondaries':
            index = np.arange(total_num // 2, total_num).astype(int)
        else:  # 'all'
            index = np.arange(0, total_num).astype(int)

        # Get positions
        if use_pos == 'true':
            gal_cat_det = pd.read_feather(catalogue_name)
            matched = index
            ra, dec = gal_cat_det['RA_input'][matched].array, gal_cat_det['DEC_input'][matched].array
            w = wcs.WCS(gal_im_header)
            x, y = w.wcs_world2pix(ra, dec, 1)
            num = x.shape[0]
        else:
            gal_cat_det = pd.read_feather(catalogue_name)
            gal_cat_mat = pd.read_feather(match_name)
            comm_ids, ind1, ind2 = np.intersect1d(gal_cat_mat['id_input'], index, return_indices=True)
            matched = gal_cat_mat['id_detec'][ind1] - 1
            x, y = gal_cat_det['X_IMAGE'][matched].array, gal_cat_det['Y_IMAGE'][matched].array
            num = x.shape[0]
            del gal_cat_mat

        # Measure shapes
        gg = np.full((num, 4), -1.)
        for kk in range(num):
            stamp = shape.cutout(gal_im, x[kk], y[kk], stamp_size=stamp_size)
            gg[kk, :] = get_shape(stamp, psf_im, scale=scale)

        # Save results
        shapes_cat = np.full((gal_cat_det.shape[0], 4), -1.)
        shapes_cat[matched, :] = gg

        gal_cat_det['GALSIM_G1'] = shapes_cat[:, 0]
        gal_cat_det['GALSIM_G2'] = shapes_cat[:, 1]
        gal_cat_det['NGMIX_G1'] = shapes_cat[:, 2]
        gal_cat_det['NGMIX_G2'] = shapes_cat[:, 3]

        gal_cat_det.to_feather(f"{shape_path}shape_catalogue{out_name}.feather")


if __name__ == '__main__':
    main()
