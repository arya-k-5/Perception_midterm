"""Runs variants x mazes one after another, each with a tick budget AND a wall-clock limit, logs in
logs/matrix/<variant>__<maze>.txt, then prints the table. From the repo root:

    /Users/arya/vis_nav_player_work/.venv/bin/python tools/run_matrix.py
    ... tools/run_matrix.py --mazes 8 52 --budget 3000 --minutes 25 --variants route think
Finished runs are skipped when you start it again."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

VARIANTS = {
    "orig": ["source/my_agent_2767.py"],
    "merge": ["source/nav_merge.py"],
    "route": ["source/nav_route.py"],
    "think": ["source/nav_think.py"],
    "replay": ["source/nav_replay.py"],
    "route2": ["source/nav_route2.py"],
    "wall": ["source/nav_wall.py"],
}
ap = argparse.ArgumentParser()
ap.add_argument("--mazes", nargs="+", default=["8", "10", "11"])
ap.add_argument("--variants", nargs="+", default=["wall"])
ap.add_argument("--budget", type=int, default=4000)
ap.add_argument("--minutes", type=float, default=25.0, help="wall-clock limit per run")
a = ap.parse_args()
Path("logs/matrix").mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONIOENCODING="utf-8")
for m in a.mazes:
    for v in a.variants:
        out = Path(f"logs/matrix/{v}__{m}.txt")
        if out.exists() and "result:" in out.read_text(errors="ignore"):
            print(f"skip {v} maze {m} (done)")
            continue
        cmd = [sys.executable, *VARIANTS[v], "--local", m, "--budget", str(a.budget)]
        print(f"== {v} maze {m} {time.strftime('%H:%M:%S')}  {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        with out.open("w") as fh:
            try:
                subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                               env=env, timeout=a.minutes * 60)
            except subprocess.TimeoutExpired:
                fh.write(f"\nTIMEOUT after {a.minutes} min\n")
                print("   TIMEOUT", flush=True)
        res = [l for l in out.read_text(errors="ignore").splitlines() if l.startswith("result:")]
        print(f"   {time.time() - t0:.0f} s  {res[0][:120] if res else 'no result'}", flush=True)
subprocess.run([sys.executable, "tools/summarize_runs.py", "logs/matrix"])
