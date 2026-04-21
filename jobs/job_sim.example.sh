#!/bin/bash
#SBATCH --job-name SIM
#SBATCH --time=36:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mem=156G
#SBATCH --ntasks-per-node 50
#SBATCH --nodes 1
#SBATCH --mail-user=your.email@example.org
#SBATCH --chdir=/path/to/blendemu/scripts
#SBATCH --output=/path/to/logs/sim.%j.%N.out
#SBATCH --partition=cluster
#SBATCH -e /path/to/logs/sim.%j.%N.err

echo "START"
date

source activate <your-conda-env>
module load sextractor

export PYTHONPATH="/path/to/blendemu:$PYTHONPATH"
export BLENDEMU_SIM_RUN="/path/to/MultiBand_ImSim/modules/Run.py"

CATA_FOLDER='/path/to/sim_outputs/galsbi_f24/'
echo "Work directory: $CATA_FOLDER"

REAL="d,0,1"

SIZE=50
START_CASE=0
END_CASE=199

for i in `seq $START_CASE $SIZE $END_CASE`
do
    echo $i
    echo "Running image simulation"
    srun -n $SIZE --mpi=pmi2 python run_sim.py $CATA_FOLDER --case_start $i --realizations $REAL --shear_case "0.0" --actions "1,3" --loglevel 'ERROR'
    echo "Running shape measurement"
    srun -n $SIZE --mpi=pmi2 python run_shape.py $CATA_FOLDER --case_start $i --realizations $REAL --shear_case "0.0" --stamp_size 48 --use_pos "detect" --pixel_scale "0.2"
done

for i in `seq $START_CASE $SIZE $END_CASE`
do
    echo $i
    echo "Running image simulation"
    srun -n $SIZE --mpi=pmi2 python run_sim.py $CATA_FOLDER --case_start $i --realizations $REAL --shear_case "0.1" --actions "1,3" --loglevel 'ERROR'
    echo "Running shape measurement"
    srun -n $SIZE --mpi=pmi2 python run_shape.py $CATA_FOLDER --case_start $i --realizations $REAL --shear_case "0.1" --stamp_size 48 --use_pos "detect" --pixel_scale "0.2"
done

echo "FINISH"
date
