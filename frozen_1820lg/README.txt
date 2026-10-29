FROZEN 2026-10-28 23:59 -- the agent that scored 6/10 on local maze 17 (0.02 m, 0.2 deg, 2,525 steps, SPL 0.43)
and 7/10 on the sample challenge (2026-10-29 01:24: 0.05 m, 1.4 deg, 1,202 steps, SPL 0.63; RUN_sample_challenge_7of10.json).

What it is: Kartik's my_agent_1820 (sptm_agent.py of his repo, commit 20e6de9, flags --smooth-steer --verbose
--steer-hold --detour-once --map-ticks --think-confused) + Claude's two additions:
  * 4-photo goal check (my_agent_2767g.py: GoalAgent) -- on maze 17 it moved the goal from photo 2106 to 1177
  * SuperPoint + LightGlue instead of SuperGlue for every match (lg_matcher.py)
Entry point: my_agent_1820lg.py

Run it from the repo root (it needs ../models, ../third_party/SuperGlue, ../.env, LightGlue installed):
  cd /Users/arya/viz_nav_player_kartik_best/vis_nav_player_work-main
  /Users/arya/vis_nav_player_work/.venv/bin/python frozen_1820lg_maze17_6of10/my_agent_1820lg.py --local 17 --budget 4000

LightGlue install (once): uv pip install --python /Users/arya/vis_nav_player_work/.venv/bin/python git+https://github.com/cvg/LightGlue.git
MD5SUMS.txt lists every file as frozen. Do not edit files in this folder; copy them elsewhere to change them.
