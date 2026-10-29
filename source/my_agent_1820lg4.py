"""
my_agent_1820lg4.py -- lg2 (Kartik's 1820 + 4-photo goal check + LightGlue + 8 fixes, 6/10 on maze 55) running on
Kartik's NEWEST sptm_agent.py (commit 78c8d00, his 7/10 in 1,348 steps) with his two new switches ON:
  --goal-place       choose the goal photo among the top 64 candidates per goal photo (not 6)
  --confirm-checkin  check in only on a strong match confirmed on fresh readings; a weak "arrival" hands back
                     to the route. Kartik's threshold is 35 SuperGlue inliers (his correct check-ins: 43-122,
                     wrong: 21-25). With LightGlue counts are ~2.8x higher (our correct check-ins: 186-276,
                     wrong: 17-22), so CHECKIN_STRONG = 80 here.
NEW FILE. Direction-aware planning (lg3) is NOT included: it hurt on maze 55.

    uv run source/my_agent_1820lg4.py --local 8 --budget 10000
"""

from __future__ import annotations

import sys

import sg_matcher
import sptm_agent as base
from lg_matcher import LGMatcher
from my_agent import with_defaults
from my_agent_1820g import FLAGS
from my_agent_1820lg2 import LG_CHECKIN_MIN, LG_CHECKIN_STAGES, LG_GOTO_MIN, LG2Agent
from my_agent_2767g import GoalAgent
from sptm_agent import main

FLAGS4 = FLAGS + ("--goal-place", "--confirm-checkin")
LG_CHECKIN_STRONG = 80

if __name__ == "__main__":
    if not hasattr(base, "CHECKIN_STRONG"):
        raise SystemExit("source/sptm_agent.py is older than Kartik's commit 78c8d00 (no --confirm-checkin)")
    sg_matcher.SGMatcher = LGMatcher
    argv = sys.argv[1:]
    GoalAgent.goal_check = "--no-goal-check" not in argv
    LG2Agent.undo_on = "--no-undo1" not in argv
    LG2Agent.look_on = "--no-lookroute" not in argv
    LG2Agent.goal_on = "--no-goalnear" not in argv
    LG2Agent.lgth_on = "--no-lgthresh" not in argv
    if LG2Agent.lgth_on:
        base.CHECKIN_MIN, base.GOTO_MIN, base.CHECKIN_STAGES = LG_CHECKIN_MIN, LG_GOTO_MIN, LG_CHECKIN_STAGES
    base.CHECKIN_STRONG = LG_CHECKIN_STRONG
    argv = [x for x in argv if x not in ("--no-goal-check", "--no-undo1", "--no-lookroute", "--no-goalnear",
                                         "--no-lgthresh")]
    print(f"lg4: Kartik's --goal-place --confirm-checkin on, check-in needs {base.CHECKIN_STRONG} LightGlue inliers")
    sys.argv[1:] = with_defaults(argv, flags=FLAGS4, log_dir="logs/my_agent_1820lg4")
    main(LG2Agent)
