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

**Work in progress:** `source/my_agent_1820lg2.py`. It adds 7 fixes for the failures on mazes 10 and 67, and is not yet tested.

LightGlue install (once): `uv pip install git+https://github.com/cvg/LightGlue.git`
