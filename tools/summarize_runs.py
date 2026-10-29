"""Table of runs from logs/matrix/*.txt (file name: <agent>__<maze>.txt). Run from the repo root:
    python tools/summarize_runs.py [logs/matrix]
Columns: points, tier, ticks (nav_steps), SPL, jitter reversals, lost relocalisations, top tick consumers."""
import re
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/matrix")
rows = []
for f in sorted(d.glob("*.txt")):
    t = f.read_text(errors="ignore")
    r = re.search(r"result: Result\(goal_tier='(\w+)', nav_steps=(\d+), spl=([\d.]+), points=([\d.]+)", t)
    j = re.search(r"jitter: (\d+) turn reversals", t)
    lo = re.search(r"lost detector: (\d+) relocalisations", t)
    tb = re.search(r"ticks by reason: (.*)", t)
    agent, _, maze = f.stem.partition("__")
    top = ", ".join(tb.group(1).split(", ")[:4]) if tb else ""
    rows.append((maze, agent, r.group(4) if r else "?", r.group(1) if r else "no result",
                 r.group(2) if r else "?", r.group(3) if r else "?", j.group(1) if j else "-",
                 lo.group(1) if lo else "-", top))
print(f"{'maze':>5} {'agent':<22}{'pts':>5} {'tier':<9}{'ticks':>6} {'SPL':>5} {'jit':>4} {'lost':>4}  top tick consumers")
for m, a, p, tier, n, spl, jit, lost, top in sorted(rows, key=lambda x: (x[0], x[1])):
    print(f"{m:>5} {a:<22}{p:>5} {tier:<9}{n:>6} {spl:>5} {jit:>4} {lost:>4}  {top}")
