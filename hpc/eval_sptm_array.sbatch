#!/bin/bash
# Local-maze evaluation of the SPTM agent (sptm_agent.py = --agent 1, the best scored version so far:
# 5/10 on maze 7 in logs/m3.txt), one maze per array task. Uses the training-free scorer, so no torch needed.
#   sbatch hpc/eval_sptm_array.sbatch                 # agent 1, seeds 7-11, budget 10000
#   AGENT=4 sbatch hpc/eval_sptm_array.sbatch         # another version, same mazes
#SBATCH --job-name=visnav-sptm
#SBATCH --account=rob_gy6203-a-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --array=7-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sptm_%A_%a.out
set -euo pipefail
REPO_DIR="${REPO_DIR:-$SCRATCH/vis_nav_player_work}"
cd "$REPO_DIR"
mkdir -p logs results
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SCRATCH/.uv-cache"
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=4
AGENT="${AGENT:-1}"
BUDGET="${BUDGET:-10000}"
uv run --no-sync source/eval_sptm.py --seeds "$SLURM_ARRAY_TASK_ID" --agent "$AGENT" \
    --budget "$BUDGET" --out "results/sptm${AGENT}_seed${SLURM_ARRAY_TASK_ID}.csv" \
    --log-dir "logs/sptm${AGENT}_seed${SLURM_ARRAY_TASK_ID}"
