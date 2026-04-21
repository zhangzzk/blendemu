#!/bin/bash
#SBATCH --job-name TRAIN
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=your.email@example.org
#SBATCH --chdir=/path/to/blendemu/scripts
#SBATCH --output=/path/to/logs/train.%j.out
#SBATCH --partition=inter
#SBATCH -e /path/to/logs/train.%j.err

echo "START - Train emulators"
date

eval "$(conda shell.bash hook)"
conda activate <your-conda-env>
export PYTHONPATH="/path/to/blendemu:$PYTHONPATH"

# Tune + train all three emulators (regression, self-response, classification)
python train_emulator.py \
    --config ../configs/fs2_lsst_r.yaml \
    --mode tune \
    --task all

echo "FINISH"
date
