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

    num_sim = int(n_degree2 / 2) * 2
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

    # Generate realizations
    from tqdm import trange
    for g in sim_cfg['shear_values']:
        print(f"\nShear g={g}:")
        for i in trange(sim_cfg['case_offset'],
                        sim_cfg['case_offset'] + sim_cfg['n_cases'],
                        desc=f"g={g}"):
            seed = i + 123
            df = catalog.generate_catalog_realization(gal_cat, num_sim, seed, g)

            fname = os.path.join(out_path, f'gals{i}_{g:.1f}.feather')
            df.to_feather(fname)

            catalog.write_config_file(
                i, g,
                path=out_path,
                base_config_path=sim_cfg['base_config'],
                file_name_cat=fname,
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
    start = sim_cfg['case_offset']
    end = start + sim_cfg['n_cases']
    n_mpi = cl_cfg['n_mpi']

    n_batches = (end - start + n_mpi - 1) // n_mpi
    total_batches = n_batches * len(sim_cfg['shear_values'])
    batch_i = 0

    for g in sim_cfg['shear_values']:
        print(f"\n--- Shear g={g} ---")
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
                '--shear_case', f'{g}',
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

def _run_shape_measurement(cfg, targets, step_label):
    """Dispatch MPI shape-measurement batches for the given target group."""
    sim_cfg = cfg['simulation']
    sm_cfg = cfg['shape_measurement']
    cl_cfg = cfg['cluster']

    t0 = time.time()
    run_shape_path = os.path.join(SCRIPT_DIR, 'run_shape.py')
    start = sim_cfg['case_offset']
    end = start + sim_cfg['n_cases']
    n_mpi = cl_cfg['n_mpi']

    n_batches = (end - start + n_mpi - 1) // n_mpi
    total_batches = n_batches * len(sim_cfg['shear_values'])
    batch_i = 0

    for g in sim_cfg['shear_values']:
        print(f"\n--- Shear g={g} ---")
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
                '--shear_case', f'{g}',
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
    _run_shape_measurement(cfg, targets='primaries', step_label='Step 3')


def step_measure_secondaries(cfg):
    """Step 3b: shape measurement of secondaries (self-response targets)."""
    print("\n" + "=" * 60)
    print("STEP 3b: Running shape measurements (secondaries)")
    print("=" * 60)
    _run_shape_measurement(cfg, targets='secondaries', step_label='Step 3b')


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
    n_total = sim_cfg['n_cases']
    offset = sim_cfg['case_offset']
    batch_size = cat_cfg['batch_size']
    n_jobs = cat_cfg['n_jobs']
    n_batches = (n_total + batch_size - 1) // batch_size

    # --- Response catalogue (R) ---
    r_cfg = cat_cfg['response']
    print(f"\n--- Response catalogue (R): {n_batches} batches, r_max={r_cfg['r_max']}\", k={r_cfg['k']} ---")

    def _save_batch_R(j):
        frames = []
        start = offset + j * batch_size
        end = min(start + batch_size, offset + n_total)
        for case in range(start, end):
            try:
                df = response.retrieve_response(
                    case=case, r_max=r_cfg['r_max'], r_min=r_cfg.get('r_min', 0),
                    k=r_cfg['k'], real='real0',
                    tile_name=sm_cfg['tile_name'], data_path=out_path,
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

    Parallel(n_jobs=n_jobs)(delayed(_save_batch_R)(j) for j in range(n_batches))

    # Merge
    R_parts = []
    for j in tqdm(range(n_batches), desc="Merging R"):
        fp = os.path.join(out_path, f'response_catalogue_{j}.feather')
        if os.path.exists(fp):
            R_parts.append(pd.read_feather(fp))
    if R_parts:
        R_whole = pd.concat(R_parts, ignore_index=True)
        R_whole = R_whole[~np.isnan(R_whole['delta_et1'])].reset_index(drop=True)
        R_out = os.path.join(out_path, 'response_catalogue_train.feather')
        R_whole.to_feather(R_out)
        print(f"  Response: {R_whole.shape[0]:,} rows -> {R_out}")
        del R_whole
    else:
        print("  WARNING: no response data produced (shape catalogues missing?)")

    # --- Detection catalogue (P) ---
    d_cfg = cat_cfg['detection']
    print(f"\n--- Detection catalogue (P): {n_batches} batches, r_max={d_cfg['r_max_deg']*3600:.1f}\", k={d_cfg['k']} ---")

    def _save_batch_P(j):
        frames = []
        start = offset + j * batch_size
        end = min(start + batch_size, offset + n_total)
        for case in range(start, end):
            df = response.retrieve_detection(
                case=case, shear='0.0', real='real0',
                r_max=d_cfg['r_max_deg'], r_min=d_cfg.get('r_min_deg', 0),
                k=d_cfg['k'], data_path=out_path,
            )
            frames.append(df)
        batch = pd.concat(frames, ignore_index=True)
        path = os.path.join(out_path, f'detection_catalogue_{j}.feather')
        batch.to_feather(path)
        return path

    Parallel(n_jobs=n_jobs)(delayed(_save_batch_P)(j) for j in range(n_batches))

    P_parts = []
    for j in tqdm(range(n_batches), desc="Merging P"):
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
    batch_size = cat_cfg['batch_size']
    n_jobs = cat_cfg['n_jobs']
    n_total = sim_cfg['n_cases']
    offset = sim_cfg['case_offset']
    n_batches = (n_total + batch_size - 1) // batch_size

    shear_applied = sim_cfg['shear_values'][1]

    t0 = time.time()
    print(f"\n--- Self-response catalogue (S): {n_batches} batches, "
          f"r_max={sr_cfg['r_max']}\", k={sr_cfg['k']} ---")

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
                )
                frames.append(df)
            except FileNotFoundError as e:
                print(f"  case {case}: {e}")
        if frames:
            batch = pd.concat(frames, ignore_index=True)
            path = os.path.join(out_path, f'self_response_catalogue_{j}.feather')
            batch.to_feather(path)
            return path
        return None

    Parallel(n_jobs=n_jobs)(delayed(_save_batch_S)(j) for j in range(n_batches))

    S_parts = []
    for j in tqdm(range(n_batches), desc="Merging S"):
        fp = os.path.join(out_path, f'self_response_catalogue_{j}.feather')
        if os.path.exists(fp):
            S_parts.append(pd.read_feather(fp))
    if S_parts:
        S_whole = pd.concat(S_parts, ignore_index=True)
        S_whole = S_whole[~np.isnan(S_whole['delta_et1'])].reset_index(drop=True)
        S_out = os.path.join(out_path, 'self_response_catalogue_train.feather')
        S_whole.to_feather(S_out)
        print(f"  Self-response: {S_whole.shape[0]:,} rows -> {S_out}")
        print(f"  <delta_et1>/gamma = {S_whole['delta_et1'].mean() / shear_applied:.4f} "
              f"(expect ~1)")
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
        return ['1', '2', '3', '4', '5']
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
  4  response   Build response and detection catalogues
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
    parser.add_argument('--n-trials', type=int, default=None)
    parser.add_argument('--output-path', type=str, default=None)

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.n_cases is not None:
        cfg['simulation']['n_cases'] = args.n_cases
    if args.case_offset is not None:
        cfg['simulation']['case_offset'] = args.case_offset
    if args.n_mpi is not None:
        cfg['cluster']['n_mpi'] = args.n_mpi
    if args.n_jobs is not None:
        cfg['catalogues']['n_jobs'] = args.n_jobs
    if args.n_trials is not None:
        cfg['training']['n_trials'] = args.n_trials
    if args.output_path is not None:
        cfg['simulation']['output_path'] = args.output_path

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
