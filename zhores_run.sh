#!/bin/bash -l


# ------------------------------------
# --- Install manually before the run:

# module load python/anaconda3
# conda activate && conda remove --name superrelora --all -y
# conda create --name superrelora python=3.9 -y
# conda activate superrelora

# ---------------------------------
# --- How to use this shell script:
# Run this script as "sbatch zhores_run.sh"
# Check status as: "squeue"
# See results in "zhores_out.txt"
# Delete the task as "scancel NUMBER"


# ------------
# --- Options:

#SBATCH --job-name=t.glukhikh_superrelora
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --partition mem
##SBATCH --mem-per-cpu=6000MB
#SBATCH --mem=6GB
#SBATCH --mail-type=ALL
#SBATCH --output=zhores_out.txt


# ----------------
# --- Main script:
module rm *
module load python/anaconda3
conda activate superrelora


srun python3 run_func.py


exit 0

