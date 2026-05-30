"""
blendemu pipeline runner.

All parameters are read from a YAML config file. CLI flags override
config values when provided.

Steps:
  1   catalog       Generate galaxy catalogues and simulation configs
  2   simulate      Run image simulations (MPI)
  3   measure       Run primary shape measurements (MPI)
  3b  measure_sec   Run secondary shape measurements (MPI; for self-response)
  4   response      Build blending-response and detection catalogues
  4b  self_response Build self-response catalogue
  5   train         Tune + train XGBoost emulators (delegates to train_emulator.py)

Usage:
  python run_pipeline.py --config ../configs/fs2_lsst_r.yaml --steps all
  python run_pipeline.py --config ../configs/fs2_lsst_r.yaml --steps 3,4 --n-mpi 100
  python run_pipeline.py --config ../configs/fs2_lsst_r.yaml --steps self_response
"""

import argparse
import glob
import os
import sys
import subprocess
import time
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from blendemu import catalog, response, data_utils
from blendemu.config import load_config, config_summary


def _remove_outputs(paths):
    """Remove stale derived outputs before rebuilding a catalogue step."""
    for path in paths:
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed stale output: {path}")


def _remove_output_glob(pattern):
    _remove_outputs(sorted(glob.glob(pattern)))


_SIMULATION_OVERRIDE_KEYS = ('case_offset', 'n_cases', 'shear_values', 'shear_cases', 'shear_scale')


def _simulation_set_cfg(sim_cfg, set_name):
    """Return resolved simulation settings for a named response set."""
    cfg = {
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
            cfg.update(sim_cfg[alias])
    return cfg


def _catalogue_sim_cfg(sim_cfg, section_cfg, set_name):
    """Resolve a catalogue's simulation set, with legacy catalogue overrides."""
    resolved = _simulation_set_cfg(sim_cfg, set_name)
    for key in _SIMULATION_OVERRIDE_KEYS:
        if key in section_cfg:
            resolved[key] = section_cfg[key]
    return resolved


def _case_window(section_cfg, label='simulation'):
    """Return the case offset and count for a resolved simulation section."""
    offset = int(section_cfg.get('case_offset', 0))
    n_total = int(section_cfg['n_cases'])
    if n_total < 0:
        raise ValueError(f"{label} n_cases must be non-negative")
    return offset, n_total


def _detection_case_window(detection_cfg, self_sim_cfg):
    """Return detection cases, defaulting to the self-response simulation set."""
    resolved = dict(self_sim_cfg)
    for key in ('case_offset', 'n_cases'):
        if key in detection_cfg:
            resolved[key] = detection_cfg[key]
    return _case_window(resolved, label='detection')


def _n_batches(n_total, batch_size):
    return (n_total + batch_size - 1) // batch_size


def _section_shear_values(section_cfg, label='simulation'):
    if 'shear_values' in section_cfg:
        shear_values = section_cfg['shear_values']
    elif 'shear_cases' in section_cfg:
        shear_values = section_cfg['shear_cases']
    else:
        raise ValueError(f"{label} must define shear_values or shear_cases")
    return [float(g) for g in shear_values]


def _paired_shear_settings(section_cfg, label='catalogue'):
    """Return shear labels and finite-difference scale for response catalogues."""
    if 'shear_cases' in section_cfg:
        shear_cases = [str(s) for s in section_cfg['shear_cases']]
        if len(shear_cases) != 2:
            raise ValueError("catalogue shear_cases must contain exactly two labels")
        if 'shear_scale' in section_cfg:
            shear_scale = float(section_cfg['shear_scale'])
        else:
            shear_scale = float(shear_cases[1]) - float(shear_cases[0])
        return shear_cases, shear_scale

    shear_values = _section_shear_values(section_cfg, label=label)
    if len(shear_values) != 2:
        raise ValueError(f"{label} shear_values must contain exactly two values")
    shear_cases = [catalog.shear_label(g) for g in shear_values]
    return shear_cases, shear_values[1] - shear_values[0]


def _detection_shear_label(detection_cfg, self_sim_cfg):
    """Return the detection shear, defaulting to the sheared self-response case."""
    if 'shear' in detection_cfg:
        return catalog.shear_label(float(detection_cfg['shear']))
    if 'shear_cases' in detection_cfg:
        shear_cases = detection_cfg['shear_cases']
        if len(shear_cases) != 2:
            raise ValueError("detection shear_cases must contain exactly two labels")
        return str(shear_cases[1])
    if 'shear_values' in detection_cfg:
        shear_values = detection_cfg['shear_values']
        if len(shear_values) != 2:
            raise ValueError("detection shear_values must contain exactly two values")
        return catalog.shear_label(float(shear_values[1]))
    shear_cases, _ = _paired_shear_settings(self_sim_cfg, label='simulation.self_response')
    return shear_cases[1]


def _simulation_work_items(sim_cfg, set_names=('response', 'self_response')):
    """Return unique case/shear jobs needed by the named simulation sets."""
    items = {}
    for set_name in set_names:
        section = _simulation_set_cfg(sim_cfg, set_name)
        offset, n_total = _case_window(section, label=f'simulation.{set_name}')
        for shear in _section_shear_values(section, label=f'simulation.{set_name}'):
            shear_label = catalog.shear_label(shear)
            key = (offset, n_total, shear_label)
            if key not in items:
                items[key] = {
                    'case_offset': offset,
                    'n_cases': n_total,
                    'shear': shear,
                    'shear_label': shear_label,
                    'sets': [set_name],
                }
            else:
                items[key]['sets'].append(set_name)
    return sorted(items.values(), key=lambda item: (item['shear'], item['case_offset'], item['n_cases']))


def _set_simulation_override(sim_cfg, key, value):
    """Apply a CLI simulation override to the global and configured named sets."""
    sim_cfg[key] = value
    for set_name in ('response', 'blending_response', 'self_response'):
        if isinstance(sim_cfg.get(set_name), dict):
            sim_cfg[set_name][key] = value


# ──────────────────────────────────────────────
# Step 1: Catalogue generation
# ──────────────────────────────────────────────

def step_catalog(cfg):
    print("\n" + "=" * 60)
    print("STEP 1: Generating galaxy catalogues")
    print("=" * 60)

    cat_cfg = cfg['catalog']
    sim_cfg = cfg['simulation']
    noise_cfg = cfg['noise']

    t0 = time.time()

    # Load base catalogue
    if cat_cfg['type'] == 'fs2':
        gal_cat, n_degree2 = catalog.load_fs2_catalog(
            path=cat_cfg['path'],
            mag_col=cat_cfg.get('mag_column', 'lsst_r_mag'),
            mag_cut=cat_cfg['mag_cut'],
        )
    elif cat_cfg['type'] == 'galsbi':
        gal_cat, n_degree2 = catalog.load_and_process_base_catalog(path=cat_cfg['path'])
    else:
        raise ValueError(f"Unknown catalog type: {cat_cfg['type']}")

    area_deg2 = sim_cfg.get('area_deg2')
    ra_range = (180.0, 181.0)
    dec_range = (0.0, 1.0)
    if area_deg2 is not None:
        area_deg2 = float(area_deg2)
        if area_deg2 <= 0:
            raise ValueError("simulation.area_deg2 must be positive")
        side_deg = area_deg2 ** 0.5
        ra_range = (180.0, 180.0 + side_deg)
        dec_range = (0.0, side_deg)
        num_sim = int(n_degree2 * area_deg2 / 2) * 2
        print(
            f"Generated sky area: {area_deg2:g} deg^2 "
            f"(RA {ra_range[0]:.6f}..{ra_range[1]:.6f}, "
            f"Dec {dec_range[0]:.6f}..{dec_range[1]:.6f})"
        )
    else:
        num_sim = int(n_degree2 / 2) * 2
    max_galaxies = sim_cfg.get('max_galaxies_per_realization')
    if max_galaxies is not None:
        max_galaxies = int(max_galaxies)
        if max_galaxies < 2:
            raise ValueError("simulation.max_galaxies_per_realization must be >= 2")
        capped = min(num_sim, max_galaxies)
        num_sim = int(capped / 2) * 2
        print(f"Galaxies per realization: {num_sim:,} (capped for smoke/testing)")
    else:
        print(f"Galaxies per realization: {num_sim:,}")

    # Write noise file
    out_path = sim_cfg['output_path']
    os.makedirs(out_path, exist_ok=True)

    if noise_cfg['source'] == 'csv':
        catalog.write_noise_file_from_csv(
            noise_csv_path=noise_cfg['csv_path'],
            band=noise_cfg['band'],
            output_path=out_path,
        )
    else:
        catalog.write_noise_file(path=out_path)

    # Generate realizations for the union of configured simulation sets.
    from tqdm import trange
    for item in _simulation_work_items(sim_cfg):
        g = item['shear']
        start = item['case_offset']
        end = start + item['n_cases']
        print(f"\nShear g={item['shear_label']} ({'+'.join(item['sets'])}), cases {start}..{end - 1}:")
        for i in trange(start, end, desc=f"g={item['shear_label']}"):
            seed = i + 123
            df = catalog.generate_catalog_realization(
                gal_cat, num_sim, seed, g, ra_range=ra_range, dec_range=dec_range
            )

            fname = os.path.join(out_path, f'gals{i}_{catalog.shear_label(g)}.feather')
            df.to_feather(fname)

            catalog.write_config_file(
                i, g,
                path=out_path,
                base_config_path=sim_cfg['base_config'],
                file_name_cat=fname,
                survey=sim_cfg.get('survey'),
                mag_cut=cat_cfg.get('mag_cut'),
                crossmatch_mag_faint_cut=cat_cfg.get('mag_cut'),
                pixel_scale=sim_cfg.get('pixel_scale'),
            )

    print(f"\nStep 1 complete ({time.time() - t0:.0f}s)")


# ──────────────────────────────────────────────
# Step 2: Image simulation (MPI)
# ──────────────────────────────────────────────

def step_simulate(cfg):
    print("\n" + "=" * 60)
    print("STEP 2: Running image simulations")
    print("=" * 60)

    sim_cfg = cfg['simulation']
    cl_cfg = cfg['cluster']

    t0 = time.time()
    run_sim_path = os.path.join(SCRIPT_DIR, 'run_sim.py')
    n_mpi = cl_cfg['n_mpi']
    work_items = _simulation_work_items(sim_cfg)

    total_batches = sum(_n_batches(item['n_cases'], n_mpi) for item in work_items)
    batch_i = 0

    for item in work_items:
        g = item['shear_label']
        start = item['case_offset']
        end = start + item['n_cases']
        print(f"\n--- Shear g={g} ({'+'.join(item['sets'])}), cases {start}..{end - 1} ---")
        for case_start in range(start, end, n_mpi):
            batch_i += 1
            n_this = min(n_mpi, end - case_start)
            elapsed = time.time() - t0
            print(f"  [{batch_i}/{total_batches}] Cases {case_start}..{case_start + n_this - 1} "
                  f"(elapsed {elapsed:.0f}s)")
            cmd = [
                'srun', '-n', str(n_this), '--mpi=pmi2',
                'python', run_sim_path, sim_cfg['output_path'],
                '--case_start', str(case_start),
                '--realizations', cl_cfg['realizations'],
                '--shear_case', g,
                '--actions', '1,3',
                '--loglevel', 'ERROR',
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  WARNING: srun returned {result.returncode}")

    print(f"\nStep 2 complete ({time.time() - t0:.0f}s)")


# ──────────────────────────────────────────────
# Step 3: Shape measurement (MPI)
# ──────────────────────────────────────────────

def _run_shape_measurement(cfg, targets, step_label, sim_set_name):
    """Dispatch MPI shape-measurement batches for the given target group."""
    sim_cfg = cfg['simulation']
    sm_cfg = cfg['shape_measurement']
    cl_cfg = cfg['cluster']

    t0 = time.time()
    run_shape_path = os.path.join(SCRIPT_DIR, 'run_shape.py')
    n_mpi = cl_cfg['n_mpi']
    work_items = _simulation_work_items(sim_cfg, set_names=(sim_set_name,))

    total_batches = sum(_n_batches(item['n_cases'], n_mpi) for item in work_items)
    batch_i = 0

    for item in work_items:
        g = item['shear_label']
        start = item['case_offset']
        end = start + item['n_cases']
        print(f"\n--- Shear g={g} ({'+'.join(item['sets'])}), cases {start}..{end - 1} ---")
        for case_start in range(start, end, n_mpi):
            batch_i += 1
            n_this = min(n_mpi, end - case_start)
            elapsed = time.time() - t0
            print(f"  [{batch_i}/{total_batches}] Cases {case_start}..{case_start + n_this - 1} "
                  f"(elapsed {elapsed:.0f}s)")
            cmd = [
                'srun', '-n', str(n_this), '--mpi=pmi2',
                'python', run_shape_path, sim_cfg['output_path'],
                '--case_start', str(case_start),
                '--realizations', cl_cfg['realizations'],
                '--shear_case', g,
                '--stamp_size', str(sm_cfg['stamp_size']),
                '--use_pos', sm_cfg['use_pos'],
                '--pixel_scale', str(sm_cfg['pixel_scale']),
                '--tile_name', sm_cfg['tile_name'],
                '--targets', targets,
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  WARNING: srun returned {result.returncode}")

    print(f"\n{step_label} complete ({time.time() - t0:.0f}s)")


def step_measure(cfg):
    """Step 3: shape measurement of primaries (blending-response targets)."""
    print("\n" + "=" * 60)
    print("STEP 3: Running shape measurements (primaries)")
    print("=" * 60)
    _run_shape_measurement(cfg, targets='primaries', step_label='Step 3', sim_set_name='response')


def step_measure_secondaries(cfg):
    """Step 3b: shape measurement of secondaries (self-response targets)."""
    print("\n" + "=" * 60)
    print("STEP 3b: Running shape measurements (secondaries)")
    print("=" * 60)
    _run_shape_measurement(cfg, targets='secondaries', step_label='Step 3b', sim_set_name='self_response')


# ──────────────────────────────────────────────
# Step 4: Response & detection catalogues
# ──────────────────────────────────────────────

def step_response(cfg):
    print("\n" + "=" * 60)
    print("STEP 4: Building response and detection catalogues")
    print("=" * 60)

    from joblib import Parallel, delayed
    from tqdm import tqdm

    sim_cfg = cfg['simulation']
    cat_cfg = cfg['catalogues']
    sm_cfg = cfg['shape_measurement']
    out_path = sim_cfg['output_path']

    t0 = time.time()
    batch_size = cat_cfg['batch_size']
    n_jobs = cat_cfg['n_jobs']

    # --- Response catalogue (R) ---
    r_cfg = cat_cfg['response']
    r_sim_cfg = _catalogue_sim_cfg(sim_cfg, r_cfg, 'response')
    r_offset, r_total = _case_window(r_sim_cfg, label='catalogues.response')
    r_batches = _n_batches(r_total, batch_size)
    shear_cases, shear_applied = _paired_shear_settings(r_sim_cfg, label='catalogues.response')
    response_scale = shear_applied if shear_applied else 1.0
    print(f"\n--- Response catalogue (R): {r_batches} batches, cases={r_offset}..{r_offset + r_total - 1}, "
          f"shear={shear_cases[0]}->{shear_cases[1]}, r_max={r_cfg['r_max']}\", k={r_cfg['k']} ---")
    _remove_output_glob(os.path.join(out_path, 'response_catalogue_[0-9]*.feather'))
    _remove_outputs([os.path.join(out_path, 'response_catalogue_train.feather')])

    def _save_batch_R(j):
        frames = []
        start = r_offset + j * batch_size
        end = min(start + batch_size, r_offset + r_total)
        for case in range(start, end):
            try:
                df = response.retrieve_response(
                    case=case, r_max=r_cfg['r_max'], r_min=r_cfg.get('r_min', 0),
                    k=r_cfg['k'], real='real0',
                    tile_name=sm_cfg['tile_name'], data_path=out_path,
                    shear_cases=shear_cases,
                )
                frames.append(df)
            except FileNotFoundError as e:
                print(f"  case {case}: {e}")
        if frames:
            batch = pd.concat(frames, ignore_index=True)
            path = os.path.join(out_path, f'response_catalogue_{j}.feather')
            batch.to_feather(path)
            return path
        return None

    Parallel(n_jobs=n_jobs)(delayed(_save_batch_R)(j) for j in range(r_batches))

    # Merge
    R_parts = []
    for j in tqdm(range(r_batches), desc="Merging R"):
        fp = os.path.join(out_path, f'response_catalogue_{j}.feather')
        if os.path.exists(fp):
            R_parts.append(pd.read_feather(fp))
    if R_parts:
        R_whole = pd.concat(R_parts, ignore_index=True)
        R_whole = R_whole[~np.isnan(R_whole['delta_et1'])].reset_index(drop=True)
        R_out = os.path.join(out_path, 'response_catalogue_train.feather')
        R_whole.to_feather(R_out)
        print(f"  Response: {R_whole.shape[0]:,} rows -> {R_out}")
        print(f"  <delta_et1>/gamma = {R_whole['delta_et1'].mean() / response_scale:.4f}")
        print(f"  <delta_et2>/gamma = {R_whole['delta_et2'].mean() / response_scale:.4f}")
        print(f"  distance range = {R_whole['distance'].min():.4f}..{R_whole['distance'].max():.4f}\"")
        del R_whole
    else:
        print("  WARNING: no response data produced (shape catalogues missing?)")

    # --- Detection catalogue (P) ---
    d_cfg = cat_cfg['detection']
    sr_sim_cfg = _catalogue_sim_cfg(sim_cfg, cat_cfg['self_response'], 'self_response')
    d_offset, d_total = _detection_case_window(d_cfg, sr_sim_cfg)
    d_batches = _n_batches(d_total, batch_size)
    detection_shear = _detection_shear_label(d_cfg, sr_sim_cfg)
    print(f"\n--- Detection catalogue (P): {d_batches} batches, cases={d_offset}..{d_offset + d_total - 1}, "
          f"shear={detection_shear}, r_max={d_cfg['r_max']:.1f}\", k={d_cfg['k']} ---")
    _remove_output_glob(os.path.join(out_path, 'detection_catalogue_[0-9]*.feather'))
    _remove_outputs([os.path.join(out_path, 'detection_catalogue_train.feather')])

    def _save_batch_P(j):
        frames = []
        start = d_offset + j * batch_size
        end = min(start + batch_size, d_offset + d_total)
        for case in range(start, end):
            df = response.retrieve_detection(
                case=case, shear=detection_shear, real='real0',
                r_max=d_cfg['r_max'], r_min=d_cfg.get('r_min', 0),
                k=d_cfg['k'], data_path=out_path,
                tile_name=sm_cfg['tile_name'],
            )
            frames.append(df)
        batch = pd.concat(frames, ignore_index=True)
        path = os.path.join(out_path, f'detection_catalogue_{j}.feather')
        batch.to_feather(path)
        return path

    Parallel(n_jobs=n_jobs)(delayed(_save_batch_P)(j) for j in range(d_batches))

    P_parts = []
    for j in tqdm(range(d_batches), desc="Merging P"):
        fp = os.path.join(out_path, f'detection_catalogue_{j}.feather')
        if os.path.exists(fp):
            P_parts.append(pd.read_feather(fp))
    if P_parts:
        P_whole = pd.concat(P_parts, ignore_index=True)
        P_out = os.path.join(out_path, 'detection_catalogue_train.feather')
        P_whole.to_feather(P_out)
        print(f"  Detection: {P_whole.shape[0]:,} rows -> {P_out}")
        del P_whole

    print(f"\nStep 4 complete ({time.time() - t0:.0f}s)")


# ──────────────────────────────────────────────
# Step 4b: Self-response catalogue
# ──────────────────────────────────────────────

def step_self_response(cfg):
    """Step 4b: build the self-response catalogue (target = secondaries)."""
    print("\n" + "=" * 60)
    print("STEP 4b: Building self-response catalogue")
    print("=" * 60)

    from joblib import Parallel, delayed
    from tqdm import tqdm

    sim_cfg = cfg['simulation']
    cat_cfg = cfg['catalogues']
    sm_cfg = cfg['shape_measurement']
    out_path = sim_cfg['output_path']

    sr_cfg = cat_cfg['self_response']
    sr_mode = str(sr_cfg.get('mode', 'nearest')).lower()
    nearest_only = bool(sr_cfg.get('nearest_only', sr_mode == 'nearest'))
    include_isolated = bool(sr_cfg.get('include_isolated', False))
    output_prefix = sr_cfg.get('output_prefix', 'self_response_catalogue')
    batch_size = cat_cfg['batch_size']
    n_jobs = cat_cfg['n_jobs']
    sr_sim_cfg = _catalogue_sim_cfg(sim_cfg, sr_cfg, 'self_response')
    offset, n_total = _case_window(sr_sim_cfg, label='catalogues.self_response')
    n_batches = _n_batches(n_total, batch_size)
    shear_cases, shear_applied = _paired_shear_settings(sr_sim_cfg, label='catalogues.self_response')

    response_scale = shear_applied if shear_applied else 1.0

    t0 = time.time()
    print(f"\n--- Self-response catalogue (S): {n_batches} batches, "
          f"cases={offset}..{offset + n_total - 1}, "
          f"shear={shear_cases[0]}->{shear_cases[1]}, "
          f"r_max={sr_cfg['r_max']}\", k={sr_cfg['k']}, "
          f"mode={'nearest' if nearest_only else 'pairs'}, "
          f"include_isolated={include_isolated} ---")
    _remove_output_glob(os.path.join(out_path, f'{output_prefix}_[0-9]*.feather'))
    _remove_outputs([os.path.join(out_path, f'{output_prefix}_train.feather')])

    def _save_batch_S(j):
        frames = []
        start = offset + j * batch_size
        end = min(start + batch_size, offset + n_total)
        for case in range(start, end):
            try:
                df = response.retrieve_self_response(
                    case=case, r_max=sr_cfg['r_max'],
                    r_min=sr_cfg.get('r_min', 0), k=sr_cfg['k'],
                    real='real0', tile_name=sm_cfg['tile_name'],
                    data_path=out_path,
                    secondary_shape_suffix=sr_cfg.get('shape_suffix', '_secondaries'),
                    nearest_only=nearest_only,
                    include_isolated=include_isolated,
                    shear_cases=shear_cases,
                )
                frames.append(df)
            except FileNotFoundError as e:
                print(f"  case {case}: {e}")
        if frames:
            batch = pd.concat(frames, ignore_index=True)
            path = os.path.join(out_path, f'{output_prefix}_{j}.feather')
            batch.to_feather(path)
            return path
        return None

    Parallel(n_jobs=n_jobs)(delayed(_save_batch_S)(j) for j in range(n_batches))

    S_parts = []
    for j in tqdm(range(n_batches), desc="Merging S"):
        fp = os.path.join(out_path, f'{output_prefix}_{j}.feather')
        if os.path.exists(fp):
            S_parts.append(pd.read_feather(fp))
    if S_parts:
        S_whole = pd.concat(S_parts, ignore_index=True)
        S_whole = S_whole[~np.isnan(S_whole['delta_et1'])].reset_index(drop=True)
        S_out = os.path.join(out_path, f'{output_prefix}_train.feather')
        S_whole.to_feather(S_out)
        print(f"  Self-response: {S_whole.shape[0]:,} rows -> {S_out}")
        print(f"  <delta_et1>/gamma = {S_whole['delta_et1'].mean() / response_scale:.4f} "
              f"(high-quality isolated galaxies should approach ~1)")
        print(f"  <delta_et2>/gamma = {S_whole['delta_et2'].mean() / response_scale:.4f}")
        print(f"  distance range = {S_whole['distance'].min():.4f}..{S_whole['distance'].max():.4f}\"")
        del S_whole
    else:
        print("  WARNING: no self-response data produced (secondary shape catalogues missing?)")

    print(f"\nStep 4b complete ({time.time() - t0:.0f}s)")


# ──────────────────────────────────────────────
# Step 5: Train XGBoost emulators (delegates to train_emulator.py)
# ──────────────────────────────────────────────

def step_train(cfg, _config_path):
    """Delegate to train_emulator.py so training is defined in one place."""
    print("\n" + "=" * 60)
    print("STEP 5: Training XGBoost emulators (via train_emulator.py)")
    print("=" * 60)

    t0 = time.time()
    train_script = os.path.join(SCRIPT_DIR, 'train_emulator.py')
    cmd = [
        sys.executable, train_script,
        '--config', _config_path,
        '--mode', 'tune',
        '--task', 'all',  # regression + self_response + classification
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"train_emulator.py failed with exit code {result.returncode}")

    print(f"\nStep 5 complete ({time.time() - t0:.0f}s)")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

STEP_MAP = {
    '1':  ('catalog',              step_catalog),
    '2':  ('simulate',             step_simulate),
    '3':  ('measure',              step_measure),
    '3b': ('measure_secondaries',  step_measure_secondaries),
    '4':  ('response',             step_response),
    '4b': ('self_response',        step_self_response),
    '5':  ('train',                step_train),
}


def parse_steps(s):
    """Parse step spec: 'all', 'self_response', '1,3,5', '2-4', '3b,4b'."""
    if s.lower() == 'all':
        return ['1', '2', '3', '3b', '4', '4b', '5']
    if s.lower() == 'self_response':
        # convenience: measure secondaries + build self-response catalogue
        return ['3b', '4b']
    steps = []
    for part in s.split(','):
        part = part.strip()
        # range like 2-4 (only numeric endpoints; letter-suffixed steps must be listed)
        if '-' in part and part.replace('-', '').isdigit():
            lo, hi = part.split('-')
            steps.extend(str(i) for i in range(int(lo), int(hi) + 1))
        else:
            steps.append(part)
    return steps


def main():
    parser = argparse.ArgumentParser(
        description='blendemu pipeline (config-driven)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  catalog    Generate galaxy catalogues and simulation configs
  2  simulate   Run image simulations (MPI)
  3  measure    Run shape measurements (MPI)
  3b measure2   Run secondary shape measurements (MPI)
  4  response   Build response and detection catalogues
  4b self_resp  Build self-response catalogue
  5  train      Train XGBoost emulators

Example:
  python run_pipeline.py --config ../configs/fs2_lsst_r.yaml --steps all
  python run_pipeline.py --config ../configs/fs2_lsst_r.yaml --steps 3,4 --n-mpi 100
        """,
    )
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--steps', type=str, default='all', help='Steps: "all", "1,3,5", "2-4"')

    # CLI overrides (all optional — config provides defaults)
    parser.add_argument('--n-cases', type=int, default=None)
    parser.add_argument('--case-offset', type=int, default=None)
    parser.add_argument('--n-mpi', type=int, default=None)
    parser.add_argument('--n-jobs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--n-trials', type=int, default=None)
    parser.add_argument('--output-path', type=str, default=None)
    parser.add_argument('--max-galaxies-per-realization', type=int, default=None,
                        help='Optional smoke-test cap for generated galaxies per realization.')
    parser.add_argument('--area-deg2', type=float, default=None,
                        help='Optional generated-catalogue sky area for small smoke runs.')
    parser.add_argument('--survey', type=str, default=None,
                        help='Optional [ImSim] survey override, e.g. simple_0.01sqdeg for smoke tests.')

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.n_cases is not None:
        _set_simulation_override(cfg['simulation'], 'n_cases', args.n_cases)
    if args.case_offset is not None:
        _set_simulation_override(cfg['simulation'], 'case_offset', args.case_offset)
    if args.n_mpi is not None:
        cfg['cluster']['n_mpi'] = args.n_mpi
    if args.n_jobs is not None:
        cfg['catalogues']['n_jobs'] = args.n_jobs
    if args.batch_size is not None:
        cfg['catalogues']['batch_size'] = args.batch_size
    if args.n_trials is not None:
        cfg['training']['n_trials'] = args.n_trials
    if args.output_path is not None:
        cfg['simulation']['output_path'] = args.output_path
    if args.max_galaxies_per_realization is not None:
        cfg['simulation']['max_galaxies_per_realization'] = args.max_galaxies_per_realization
    if args.area_deg2 is not None:
        cfg['simulation']['area_deg2'] = args.area_deg2
    if args.survey is not None:
        cfg['simulation']['survey'] = args.survey

    steps = parse_steps(args.steps)

    print(config_summary(cfg))
    print(f"\n  Steps: {', '.join(f'{s} ({STEP_MAP[s][0]})' for s in steps)}")

    t_total = time.time()
    for step_key in steps:
        if step_key not in STEP_MAP:
            print(f"Unknown step: {step_key}")
            sys.exit(1)
        _, func = STEP_MAP[step_key]
        # step_train needs the original config path to pass to train_emulator.py
        if step_key == '5':
            func(cfg, args.config)
        else:
            func(cfg)

    print("\n" + "=" * 60)
    print(f"Pipeline complete ({time.time() - t_total:.0f}s)")
    print("=" * 60)


if __name__ == '__main__':
    main()
