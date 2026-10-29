#!/usr/bin/env bash
# Runs every variant on every maze, one after the other, logs in logs/matrix/<variant>__<maze>.txt.
# From the repo root:   bash tools/run_matrix.sh            (defaults below)
#                       MAZES="8 52" BUDGET=3000 bash tools/run_matrix.sh
# Local runs only (free, no attempt spent). Afterwards:  python tools/summarize_runs.py
PY=${PY:-/Users/arya/vis_nav_player_work/.venv/bin/python}
MAZES=${MAZES:-"8 52 7"}
BUDGET=${BUDGET:-3000}
mkdir -p logs/matrix
export PYTHONIOENCODING=utf-8
for m in $MAZES; do
  while IFS='|' read -r name cmd; do
    out="logs/matrix/${name}__${m}.txt"
    [ -s "$out" ] && grep -q "result:" "$out" && { echo "skip $name maze $m (done)"; continue; }
    echo "== $name  maze $m  $(date +%H:%M:%S)"
    $PY $cmd --local "$m" --budget "$BUDGET" > "$out" 2>&1
    grep -E "^result:" "$out" | cut -c1-140
  done <<'V'
orig_2767|source/my_agent_2767.py
merge|source/nav_merge.py
think|source/nav_think.py
think_sw2|source/nav_think.py --switch-cost 2.0
think_nocommit|source/nav_think.py --no-commit
V
done
$PY tools/summarize_runs.py logs/matrix
