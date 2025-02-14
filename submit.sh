#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on our clusters.
# ---------------------------------------------------------------------
#SBATCH --mail-user=kazemipour.alireza@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --signal=B:SIGTERM@180
#SBATCH --account=def-mtaylor3
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem=6000M
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --job-name="level_unknown_beluga"
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
# Run your simulation step here..

job="level_unknown_beluga"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cp -r ofu $SLURM_TMPDIR/
tar -xf venv.tar.gz -C $SLURM_TMPDIR/
cd $SLURM_TMPDIR/ || exit
venv/bin/python ofu/main.py -m hydra/launcher=joblib hydra/sweeper=manual_sweeper experiment.rng_seed="range(0, 30)" >/dev/null
tar -cavf data_$job.tar.xz data
cp data_$job.tar.xz ~/projects/def-mtaylor3/alirezak




