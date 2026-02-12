#!/bin/bash
#SBATCH --account=TODO
#SBATCH --partition=TODO
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1         # 1 processes per node
#SBATCH --time=2:00:00
#SBATCH --exclusive
#SBATCH --job-name=llama2_70B-MPMD-1F1B

# --- Configuration --- #

CONTAINER_IMAGE="TODO"
LOG_DIR="TODO"
mkdir -p "$LOG_DIR"

srun \
    --ntasks-per-node=1 \
    --output="${LOG_DIR}/node-%N-task-%t.log" \
    --container-image="$CONTAINER_IMAGE" \
    PATH_TO_MAXTEXT/src/scripts/run-no_dp-llama2_70b-1F1B.sh