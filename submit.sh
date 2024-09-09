#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on our clusters.
# ---------------------------------------------------------------------
#SBATCH --mail-user=kazemipour.alireza@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-mtaylor3
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem=6000M
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --job-name="running_everything"
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
# Run your simulation step here..

job="running_everything"
module load python
cp -r ofu $SLURM_TMPDIR/
cp venv.tar.gz $SLURM_TMPDIR/
cd $SLURM_TMPDIR/ || exit
tar -xf venv.tar.gz
source venv/bin/activate
cd ofu/ || exit
python main.py -m hydra/launcher=joblib hydra/sweeper=manual_sweeper >/dev/null
tar -cavf data_$job.tar.xz data
cp data_$job.tar.xz ~/projects/def-mtaylor3/alirezak




