# kartik_best_run: snapshot of the working folder, 2026-10-29 ~01:50

This is the full working copy (`viz_nav_player_kartik_best/vis_nav_player_work-main`):
- Kartik's code at commit `20e6de9` of [kartikparameshwaran/vis_nav_player_work](https://github.com/kartikparameshwaran/vis_nav_player_work), plus everything we added;
- the trained models (`models/`) and SuperGlue (`third_party/SuperGlue`, without its .git);
- the run logs (`logs/`).

Left out:
- `data/` and `cache/` (about 750 MB; both are recreated automatically on the first run of each maze);
- `.env` (holds the API key);
- broken clone folders.

**The agent to use:** `frozen_1820lg_maze17_6of10/my_agent_1820lg.py`.
- **What it is:** Kartik's 1820, plus the 4-photo goal check, plus LightGlue.
- **Results:** sample challenge 7/10 (1,202 steps, SPL 0.63); local maze 17 6/10.
- **How to run it:** see `frozen_1820lg_maze17_6of10/README.txt`.

## Newer agents (added 2026-10-29 ~03:00), all in `source/`

| Agent | What it adds | Local results so far |
|---|---|---|
| `my_agent_1820lg.py` (= frozen) | Kartik 1820 + 4-photo goal check + LightGlue | **fastest so far:** sample challenge 7/10 in 1,202 steps (SPL 0.63); maze 17 6/10; failed 10, 67, 8; circled on 55 |
| `my_agent_1820lg2.py` | lg + 8 fixes: one-go undo of turns, look toward the route, near-goal search with scale check and time limit, hold far jumps, stick to the drive, LightGlue-scaled thresholds, last-metre damper, no check-in when far | maze 55 6/10 (3,757 steps, SPL 0.22), where the frozen agent failed |
| `my_agent_1820lg3.py` | lg2 + direction-aware shortest path (backing up costs x4) | experimental: did NOT reach the goal (maze 8 stuck at 17 hops, maze 55 circling) - do not use |
| `my_agent_1820lg4.py` | lg2 on Kartik's newest `sptm_agent.py` (commit 78c8d00) with his `--goal-place` and `--confirm-checkin` (strong check-in = 80 LightGlue inliers) | untested; the candidate for reaching the goal more reliably |

`source/sptm_agent.py` is now Kartik's commit 78c8d00 (needed by lg4 and `my_agent_1348.py`); the previous one is `source/sptm_agent_1820.py`.
The frozen folder keeps its own copy, so the frozen agent is unchanged.

Run (from the repo folder):
```
.venv/bin/python source/my_agent_1820lg4.py --local 8 --budget 10000 2>&1 | tee logs/lg4_local8.txt
.venv/bin/python source/my_agent_1820lg2.py --local 8 --budget 10000 2>&1 | tee logs/lg2_local8.txt
```

LightGlue install (once): `uv pip install git+https://github.com/cvg/LightGlue.git`
