"""Attach physical response gains to the frozen response-training pairs.

For the disk automorphism, L(e)=(1+|e|²)I-2eeᵀ. A scalar transport coefficient
b predicts parallel response s*b, where s=uᵀL(e)u. The original physical
labels and nine features remain unchanged; no calibration data enter.
"""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np

from scripts.prepare_output_conditioned_response import baseline_rows, sha256, write_json


def response_gains(shape, angle_degrees):
    shape = np.asarray(shape, dtype=np.float64)
    angle = 2*np.deg2rad(np.asarray(angle_degrees, dtype=np.float64))
    if shape.shape != (len(angle), 2) or not np.isfinite(shape).all() or not np.isfinite(angle).all():
        raise ValueError('aligned finite baseline shapes and shear angles required')
    radius2 = np.square(shape).sum(1)
    if np.any(radius2 >= 1):
        raise ValueError('baseline shapes must be strictly inside the disk')
    direction = np.column_stack((np.cos(angle), np.sin(angle)))
    parallel = np.sum(shape*direction, axis=1)
    transverse = -shape[:, 0]*direction[:, 1]+shape[:, 1]*direction[:, 0]
    # Positive form avoids cancellation near the radial disk boundary.
    gain = (1-radius2)*np.square(direction).sum(1)+2*np.square(transverse)
    cross_gain = -2*parallel*transverse
    if not np.isfinite(gain).all() or np.any(gain <= 0):
        raise ValueError('nonpositive or nonfinite physical gain')
    return gain, cross_gain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('full response-pair preparation requires Slurm')
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root/'gains').mkdir()
    source_manifest = args.prepared/'manifest.json'
    original = json.loads(source_manifest.read_text())
    if original['shear'] != .2 or sorted(map(int, original['counts'])) != list(range(200)):
        raise ValueError('expected original 200-case h=0.2 prepared sample')
    reports, parts = {}, []
    totals = {'fit': {'rows': 0, 'gain_sum': 0., 'label_sum': 0.},
              'development': {'rows': 0, 'gain_sum': 0., 'label_sum': 0.}}

    @lru_cache(maxsize=4)
    def load(case):
        rows, receipt = baseline_rows(args.source_root, case)
        if receipt != original['baseline_sources'][str(case)]:
            raise ValueError('baseline source differs from frozen physical labels')
        reports[str(case)] = receipt
        return rows

    for index, part in enumerate(original['parts']):
        path = args.prepared/part['path']
        if sha256(path) != part['sha256']:
            raise ValueError('changed physical-label part')
        with np.load(path) as data:
            shapes = load(part['case']).reindex(data['primary'])[['NGMIX_G1', 'NGMIX_G2']].to_numpy(float)
            if len(shapes) != part['rows'] or not np.isfinite(shapes).all():
                raise ValueError('unmatched baseline response identities')
            gain, cross = response_gains(shapes, data['shear_angle'])
            name = f'gains/{path.name}'
            output = args.output_root/name
            np.savez(output, gain=gain.astype(np.float32), cross_gain=cross.astype(np.float32))
            split = 'fit' if part['case'] >= 40 else 'development'
            totals[split]['rows'] += len(gain)
            totals[split]['gain_sum'] += float(gain.sum())
            totals[split]['label_sum'] += float(data['y'].astype(float).sum())
        parts.append(part | {'source_path': str(path.resolve()), 'gain_path': name,
                             'gain_sha256': sha256(output), 'gain_min': float(gain.min()),
                             'gain_max': float(gain.max())})
        if index == 0 or (index+1) % 200 == 0:
            print(f'DISK_MOMENT_PREP parts={index+1}/{len(original["parts"])}', flush=True)
    write_json(args.output_root/'manifest.json', {
        'format': 'disk_response_moment_data_v1', 'parts': parts, 'totals': totals,
        'original_manifest': str(source_manifest.resolve()), 'original_manifest_sha256': sha256(source_manifest),
        'baseline_sources': reports, 'features': original['features'], 'population': original['population'],
        'rescale': original['rescale'], 'shear': .2, 'y_mean': original['y_mean'], 'y_std': original['y_std'],
        'script_sha256': sha256(__file__), 'gain_definition': 'u^T[(1+|e0|^2)I-2e0e0^T]u',
        'limitations': ['first-order response moment from finite h=0.2 labels',
                        'same previously usable primary population as original physical-label fit',
                        'no calibration data, empirical offsets or post-fit rescaling']})
    print(f'DISK_MOMENT_PREP_COMPLETE {totals}', flush=True)


if __name__ == '__main__':
    main()
