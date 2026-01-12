#!/bin/bash
#SBATCH --job-name=energy
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=1-12
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

YEAR="2024"
MONTH=$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")

mkdir -p slurm/logs

# ---- Activate your environment (EDIT THIS PART) ----
# module load EESSI/2023.06
# source ~/.bashrc
# conda activate energy_mod
# ---------------------------------------------------

python -u scripts/energy_monthly.py --year "${YEAR}" --month "${MONTH}" --n-jobs-pv "${SLURM_CPUS_PER_TASK}"