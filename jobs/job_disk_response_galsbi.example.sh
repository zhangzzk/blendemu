#!/usr/bin/env bash
# Disk-response emulator on the GalSBI re-simulation training family (cases 0-199).
# Same three stages and frozen recipe as disk_response_moment_20260918_v1 (FS2);
# only the feature-scaling conditions change to the GalSBI LSST-i values
# (configs/galsbi_resim_training.yaml in SBSI). Submit one stage at a time:
#   sbatch --export=ALL,STAGE=pairs jobs/job_disk_response_galsbi.example.sh
#   sbatch --dependency=afterok:<pairs> --export=ALL,STAGE=gains jobs/job_disk_response_galsbi.example.sh
#   sbatch --dependency=afterok:<gains> --export=ALL,STAGE=train --cpus-per-task=32 --mem=48G --time=02:00:00 jobs/job_disk_response_galsbi.example.sh
#SBATCH --job-name=galsbi_disk_resp
#SBATCH --partition=cluster
#SBATCH --constraint=x86-64-v3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/project/ls-gruen/users/zekang.zhang/blendemu_runs/disk_response_moment_galsbi_20261103/logs/%x_%j.out
set -euo pipefail
repo=/home/z/Zekang.Zhang/blendemu_galsbi_resim
sims=/project/ls-gruen/users/zekang.zhang/sbsi/galsbi_resim/training/sims
run=/project/ls-gruen/users/zekang.zhang/blendemu_runs/disk_response_moment_galsbi_20261103
py=/project/ls-gruen/users/zekang.zhang/envs/py31/bin/python
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$repo"
echo "STAGE=$STAGE commit=$(git rev-parse HEAD) host=$(hostname)"
case "$STAGE" in
  pairs) $py -u -m scripts.prepare_output_conditioned_response --source-root "$sims" \
           --output-root "$run/pairs" --pixel-rms 0.6159 --psf-fwhm 0.71 --moffat-beta 2.458 ;;
  gains) $py -u -m scripts.prepare_disk_response_moment --prepared "$run/pairs" \
           --source-root "$sims" --output-root "$run/prepared" ;;
  train) $py -u -m scripts.train_disk_response_moment --prepared "$run/prepared" \
           --output-root "$run/model" ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
