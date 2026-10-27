#!/bin/bash
# Run ONCE on an HPC LOGIN node (has internet). Compute nodes usually do not.
# Installs the project's Python environment with uv into $SCRATCH.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$SCRATCH/vis_nav_player_work}"
cd "$REPO_DIR"
export UV_CACHE_DIR="$SCRATCH/.uv-cache"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync                      # SDK + local simulator from the course package index
# Make sure the local simulator's assets (texture pack) are downloaded now, on the login node.
uv run --no-sync python -c "import sys; sys.path.insert(0, 'source'); import cli; print('cli ok')"
echo "setup done. Next: sbatch hpc/eval_path_array.sbatch"
