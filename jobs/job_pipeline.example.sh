#!/bin/bash
#SBATCH --job-name PIPELINE
#SBATCH --time=72:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=250G
#SBATCH --ntasks-per-node 50
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=your.email@example.org
#SBATCH --chdir=/path/to/blendemu/scripts
#SBATCH --output=/path/to/logs/pipeline.%j.%N.out
#SBATCH --partition=inter
#SBATCH -e /path/to/logs/pipeline.%j.%N.err

echo "START"
date

# Activate your Python environment (edit to match your setup).
source activate <your-conda-env>
module load sextractor

# Make the blendemu package importable and point to MultiBand_ImSim's Run.py.
export PYTHONPATH="/path/to/blendemu:$PYTHONPATH"
export BLENDEMU_SIM_RUN="/path/to/MultiBand_ImSim/modules/Run.py"

# ─── Full pipeline ───
# python run_pipeline.py --steps all --n-cases 200 --n-mpi 50

# ─── Or run individual steps ───
# Step 1: Generate catalogues (no MPI needed)
# python run_pipeline.py --steps 1

# Steps 2+3: Simulate + measure (needs MPI)
# python run_pipeline.py --steps 2,3 --n-mpi 50

# Step 4: Build response/detection catalogues (no MPI, uses joblib)
# python run_pipeline.py --steps 4 --n-jobs 16

# Step 5: Train emulators (needs GPU)
# python run_pipeline.py --steps 5 --n-trials 200

# ─── Default: run everything ───
python run_pipeline.py --config ../configs/fs2_lsst_r.yaml --steps all --n-cases 200 --n-mpi 50 --n-trials 200

echo "FINISH"
date
