#!/bin/bash
#SBATCH --array=0-1
#SBATCH --time=72:00:00
#SBATCH --job-name=gcast
#SBATCH --gpus=1
#SBATCH -C "fat"
#SBATCH --mem=200000
#SBATCH -o /proj/cvl/users/x_juska/slurm_logs/gcast_%a.out
#SBATCH -e /proj/cvl/users/x_juska/slurm_logs/gcast_%a.err


# List of Python commands
# options=(
#     "10 1 10"
#     "10 11 10"
#     "10 21 11"
#     "11 1 10"
#     "11 11 10"
#     "11 21 10"
#     "12 1 10"
#     "12 11 10"
#     "12 21 11"
# )

options=(
    "10 21 5"
    "10 26 6"
)

# Execute the command corresponding to the SLURM_ARRAY_TASK_ID
python -u inference/inference.py ${options[$SLURM_ARRAY_TASK_ID]}
