"""Fit transport coefficients by conditional physical-response moments.

For b=mean+std*f, minimize E[(s*b²/2-z*b)/std²]. Its gradient in f is
s*f-(z-s*mean)/std and its Hessian is s>0. The fit uses the frozen trial9
218-tree recipe; neither development nor ConstGold chooses a checkpoint.
"""
import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import xgboost as xgb

from scripts.prepare_output_conditioned_response import sha256, write_json

# Frozen trial9 recipe, as in scripts/train_output_conditioned_response.py.
PARAMS = {"subsample": .7634540769409374, "colsample_bytree": .9651944092476105,
          "learning_rate": .03007562460248723, "max_depth": 11, "min_child_weight": 126,
          "reg_lambda": 3.255115768393047, "reg_alpha": .2015381955077309,
          "gamma": 4.552487941512809, "objective": "reg:squarederror", "eval_metric": "rmse",
          "seed": 20260816, "tree_method": "hist", "max_bin": 256}


def moment_gradient(prediction, matrix):
    gain, label = matrix.get_weight(), matrix.get_label()
    return gain*prediction-label, gain


def moment_metric(prediction, matrix):
    residual = matrix.get_weight()*prediction-matrix.get_label()
    return 'physical_moment_rmse', float(np.sqrt(np.mean(np.square(residual.astype(float)))))


class MomentIterator(xgb.DataIter):
    def __init__(self, root, parts, mean, std, features):
        self.root, self.parts = root, parts
        self.mean, self.std, self.features, self.position = mean, std, features, 0
        super().__init__(release_data=True)

    def reset(self):
        self.position = 0

    def next(self, input_data):
        if self.position == len(self.parts):
            return False
        part = self.parts[self.position]
        with np.load(part['source_path']) as data, np.load(self.root/part['gain_path']) as extra:
            gain = extra['gain']
            label = ((data['y'].astype(float)-gain.astype(float)*self.mean)/self.std).astype(np.float32)
            x = np.ascontiguousarray(data['x'])
        if (len(label) != part['rows'] or not np.isfinite(label).all() or not np.isfinite(gain).all()
                or not np.isfinite(x).all() or np.any(gain <= 0)):
            raise ValueError('invalid moment training part')
        input_data(data=x, label=label, weight=gain, feature_names=self.features)
        self.position += 1
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('response fitting requires Slurm')
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = args.prepared/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['format'] != 'disk_response_moment_data_v1' or manifest['shear'] != .2:
        raise ValueError('incompatible physical-moment data')
    for part in manifest['parts']:
        if sha256(part['source_path']) != part['sha256'] or sha256(args.prepared/part['gain_path']) != part['gain_sha256']:
            raise ValueError('changed moment-training part')
    mean, std = manifest['y_mean'], manifest['y_std']
    fit = [p for p in manifest['parts'] if 40 <= p['case'] <= 199]
    dev = [p for p in manifest['parts'] if 0 <= p['case'] <= 39]
    if not fit or not dev or len(fit)+len(dev) != len(manifest['parts']):
        raise ValueError('incomplete/disjoint fit/development split')
    nthread = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
    matrix = xgb.QuantileDMatrix(MomentIterator(args.prepared, fit, mean, std, manifest['features']),
                                nthread=nthread, max_bin=256)
    intercept = manifest['totals']['fit']['label_sum']/manifest['totals']['fit']['gain_sum']
    params = PARAMS | {'device': 'cpu', 'nthread': nthread, 'disable_default_eval_metric': True,
                       'base_score': (intercept-mean)/std}
    params.pop('eval_metric', None)
    print(f'DISK_MOMENT_TRAIN rows={matrix.num_row()} base_parameter={intercept}', flush=True)
    history = {}
    model = xgb.train(params, matrix, num_boost_round=218, obj=moment_gradient, custom_metric=moment_metric,
                      evals=[(matrix, 'train')], evals_result=history, verbose_eval=20)
    path = args.output_root/'model.json'
    model.save_model(path)
    del matrix
    gc.collect()
    reports = {}
    for part in dev:
        with np.load(part['source_path']) as data, np.load(args.prepared/part['gain_path']) as extra:
            x, label, null = data['x'], data['y'].astype(float), data['null'].astype(float)
            parameter = model.inplace_predict(np.ascontiguousarray(x)).astype(float)*std+mean
            gain, cross = extra['gain'].astype(float), extra['cross_gain'].astype(float)
            predicted, predicted_null = parameter*gain, parameter*cross
        if not np.isfinite(parameter).all():
            raise ValueError('nonfinite development parameter')
        cuts = {'uncut': np.ones(len(label), dtype=bool),
                'mag25.8_radius_gt0.60': (x[:, 8] < 25.8) & (x[:, 7] > np.log(3.)),
                'mag25.8_radius_ge0.75': (x[:, 8] < 25.8) & (x[:, 7] >= np.log(3.75))}
        for cut, keep in cuts.items():
            report = reports.setdefault(f'{part["case"]}/{cut}', {'rows': 0, 'label_sum': 0., 'predicted_sum': 0.,
                        'null_sum': 0., 'predicted_null_sum': 0., 'residual_square_sum': 0., 'parameter_sum': 0.})
            report['rows'] += int(keep.sum())
            for name, values in (('label_sum', label), ('predicted_sum', predicted), ('null_sum', null),
                                 ('predicted_null_sum', predicted_null), ('parameter_sum', parameter),
                                 ('residual_square_sum', np.square(predicted-label))):
                report[name] += float(values[keep].sum())
    write_json(args.output_root/'metadata.json', {
        'format': 'experimental_disk_response_moment_v1', 'arm': 'output_conditioned',
        'features': manifest['features'], 'model_sha256': sha256(path), 'y_mean': mean, 'y_std': std,
        'response_shear': .2, 'velocity_convention': 'radial_tanh(pair_sum_b_times_external_shear)',
        'moment_objective': 'E[(s*b^2/2-z*b)/label_std^2]', 'gain_definition': manifest['gain_definition'],
        'population': manifest['population'], 'rescale': manifest['rescale'], 'params': params, 'trees': 218,
        'fit_cases': [40, 199], 'development_cases': [0, 39], 'development_pair_sums': reports,
        'train_history': history, 'training_intercept': intercept,
        'prepared_manifest': str(manifest_path), 'prepared_manifest_sha256': sha256(manifest_path),
        'script_sha256': sha256(__file__), 'xgboost_version': xgb.__version__,
        'deployment': 'experimental Mobius transport coefficient; never use as an additive physical R_blend',
        'limitations': manifest['limitations']})
    print(f'DISK_MOMENT_TRAIN_COMPLETE {args.output_root}', flush=True)


if __name__ == '__main__':
    main()
