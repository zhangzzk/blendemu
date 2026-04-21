"""
Simulation Execution Script.

Runs the MultiBand_ImSim simulation pipeline using MPI parallelization.
Orchestrates image generation and source extraction based on config files.

Usage:
    mpiexec -n 50 python run_sim.py /path/to/data --case_start 0 --shear_case 0.0 --actions 1,3
"""

import argparse
import os
import subprocess
import numpy as np
import pandas as pd
from mpi4py import MPI

# Path to MultiBand_ImSim's Run.py. Set via the BLENDEMU_SIM_RUN env var
# (e.g., export BLENDEMU_SIM_RUN=/path/to/MultiBand_ImSim/modules/Run.py).
SIM_RUN_SCRIPT = os.environ.get("BLENDEMU_SIM_RUN")
if SIM_RUN_SCRIPT is None:
    raise EnvironmentError(
        "BLENDEMU_SIM_RUN is not set. Export it to the path of "
        "MultiBand_ImSim/modules/Run.py before running this script."
    )


def main():
    parser = argparse.ArgumentParser(description='Simulation Execution Wrapper')
    parser.add_argument('path', type=str, help='Work directory containing configs and outputs')
    parser.add_argument('--case_start', type=int, default=0, help='Starting case index')
    parser.add_argument('--shear_case', type=float, default=0.1, help='Shear case value')
    parser.add_argument('--realizations', type=str, default='0,1', help='Range/list of realizations')
    parser.add_argument('--actions', type=str, default='1', help="Actions: '1' (Sim), '3' (SExtractor)")
    parser.add_argument('--loglevel', type=str, default='ERROR', help='Log level')

    args = parser.parse_args()

    path = args.path
    case_start = args.case_start
    shear_case = args.shear_case
    realizations = args.realizations.split(',')
    actions = args.actions.split(',')
    loglevel = args.loglevel

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Determine realization and case indices
    if 'd' in realizations:
        nreal0 = int(realizations[1])
        nreal = int(realizations[2])
        case = rank + case_start
    elif realizations == ['0', '1']:
        nreal0 = 0
        nreal = 1
        case = rank + case_start
    else:
        snr_cat_path = os.path.join(path, 'snr_catalogue.feather')
        nreals = pd.read_feather(snr_cat_path)['nreal'].astype(int)
        nreal0 = 1

        idx_lower = int(realizations[0])
        idx_upper = int(realizations[1])
        nreals_idx = np.where((nreals >= idx_lower) & (nreals < idx_upper))[0]

        if rank == 0:
            print(f'Number of cases within this realization range: {nreals_idx.shape}')

        if rank + case_start < len(nreals_idx):
            nreal = nreals[nreals_idx[rank + case_start]]
            case = nreals_idx[rank + case_start]
        else:
            return

    config_name = f'sim_config_case{case}_{shear_case:.1f}.ini'

    for i in range(nreal0, nreal):
        real = f'real{i}'

        if '1' in actions:
            cmd = [
                "python", SIM_RUN_SCRIPT, "1",
                "--runTag", real,
                "--shear_columns", "g1", "g2",
                "--config", os.path.join(path, config_name),
                "--threads", "1",
                "--rng_seed", str(i),
                "--loglevel", loglevel,
                "--sep_running_log",
            ]
            result1 = subprocess.run(cmd, capture_output=True, text=True)
            if result1.stderr:
                print(f"[Rank {rank} | Case {case} | Sim]: {result1.stderr}")

        if '3' in actions:
            cmd = [
                "python", SIM_RUN_SCRIPT, "3",
                "--runTag", real,
                "--shear_columns", "g1", "g2",
                "--config", os.path.join(path, config_name),
                "--threads", "1",
                "--rng_seed", str(i),
                "--loglevel", loglevel,
                "--sep_running_log",
            ]
            result2 = subprocess.run(cmd, capture_output=True, text=True)
            if result2.stderr:
                print(f"[Rank {rank} | Case {case} | SExtractor]: {result2.stderr}")


if __name__ == "__main__":
    main()
