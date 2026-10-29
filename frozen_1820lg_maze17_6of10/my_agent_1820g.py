"""
my_agent_1820g.py -- EXACTLY Kartik's my_agent_1820 (same flags), plus the 4-photo goal check of my_agent_2767g.
NEW FILE.

Why: on local maze 17, 1820 drove through photos 1065-1150 around step 1300-1600 -- next to the place that matches
all four goal photos (photo 1169: 55/70/43/34 inliers) -- and kept going, because its goal came from one goal
photo's 28-inlier match (photo 2106). It ran out of budget 94 hops from that wrong goal.

    uv run source/my_agent_1820g.py --local 17 --budget 4000
"""

from __future__ import annotations

import sys

from my_agent import with_defaults
from my_agent_2767g import GoalAgent
from sptm_agent import main

FLAGS = ("--smooth-steer", "--verbose", "--steer-hold", "--detour-once", "--map-ticks", "--think-confused")

if __name__ == "__main__":
    argv = sys.argv[1:]
    GoalAgent.goal_check = "--no-goal-check" not in argv
    argv = [a for a in argv if a != "--no-goal-check"]
    sys.argv[1:] = with_defaults(argv, flags=FLAGS, log_dir="logs/my_agent_1820g")
    main(GoalAgent)
