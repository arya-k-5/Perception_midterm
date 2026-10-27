#!/bin/bash
# Local-maze evaluation of the current agent (path_agent, --agent 5), one maze per array task.
#   sbatch hpc/eval_path_array.sbatch                      # seeds 7-11 (the mazes with data in the repo)
#   AGENT=4 BUDGET=10000 sbatch hpc/eval_path_array.sbatch # other agent / budget
#SBATCH --job-name=visnav-path
#SBATCH --account=rob_gy6203-a-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --array=7-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/path_%A_%a.out
set -euo pipefail
REPO_DIR="${REPO_DIR:-$SCRATCH/vis_nav_player_work}"
cd "$REPO_DIR"
mkdir -p logs results
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$SCRATCH/.uv-cache"
export SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=4
AGENT="${AGENT:-5}"
BUDGET="${BUDGET:-5000}"
uv run --no-sync source/eval_sptm.py --seeds "$SLURM_ARRAY_TASK_ID" --agent "$AGENT" \
    --budget "$BUDGET" --out "results/agent${AGENT}_seed${SLURM_ARRAY_TASK_ID}.csv" \
    --log-dir "logs/agent${AGENT}_seed${SLURM_ARRAY_TASK_ID}"
