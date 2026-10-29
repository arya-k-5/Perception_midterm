"""
my_agent_2767g.py -- EXACTLY my_agent_2767 (same class, same flags), plus ONE change: the goal is found with all
four goal photos.  NEW FILE; nothing existing is edited.

Why: on local maze 17 the base took its goal from one goal photo's 28-inlier match (photo 2106; all four goal
photos together scored 81), while another place matched all four goal photos (photo 1169: 55/70/43/34 = 202).
With the wrong goal every route leads somewhere else.

How: the goal spot was photographed in four directions, so at the real goal place several of the four photos
match exploration frames close to each other. For the best place-recognition candidates of each goal photo,
score the place by the best inliers of each of the four photos within +-GOAL_NBR frames on the same drive, and
add them up. The goal moves only if that place beats the base's goal by GOAL_BETTER inliers.

    uv run source/my_agent_2767g.py --local 17 --budget 4000
    uv run source/my_agent_2767g.py --no-goal-check ...     (= my_agent_2767)
"""

from __future__ import annotations

import sys

import networkx as nx
import numpy as np
from my_agent import MyAgent, with_defaults
from sptm_agent import main

FLAGS = ("--smooth-steer", "--verbose", "--steer-hold", "--detour-once")  # my_agent_2767's flags
GOAL_CANDS = 15
GOAL_NBR = 20
GOAL_STEP = 4
GOAL_BETTER = 20


class GoalAgent(MyAgent):
    goal_check = True

    def setup(self, info) -> None:
        super().setup(info)
        if self.goal_check and getattr(self, "sg_on", False):
            try:
                self._find_goal()
            except Exception as e:  # noqa: BLE001
                print(f"goal check failed ({e!r}); keeping the base's goal")

    def _find_goal(self) -> None:
        tembs = np.asarray(self.scorer.embed(list(self.targets)), np.float32)
        cache: dict[tuple[int, int], int] = {}

        def nin(i: int, k: int) -> int:
            if (i, k) not in cache:
                cache[(i, k)] = self.gm.match(self.tfeat[k], self._node_feat(i), self.focal).n
            return cache[(i, k)]

        def nbr(c: int) -> list[int]:
            s0, e0 = self.bounds[self.traj[c]]
            return list(range(max(s0, c - GOAL_NBR), min(e0, c + GOAL_NBR + 1), GOAL_STEP)) + [c]

        def score(c: int):
            per = [max(nin(j, k) for j in nbr(c)) for k in range(4)]
            front = max(nbr(c), key=lambda j: nin(j, 0))
            if nin(front, 0) < 15:
                front = max(nbr(c), key=lambda j: max(nin(j, k) for k in range(4)))
            return sum(per), per, front

        cands = set()
        for k in range(4):
            pk = self._prob_to_all(tembs[k])
            cands.update(int(i) for i in np.argsort(-pk)[:GOAL_CANDS])
        best = max(cands, key=lambda c: score(c)[0])
        s_new, per_new, front = score(best)
        s_old, per_old, _ = score(self.goal)
        print(f"GOAL check (all 4 goal photos, F/L/B/R): base goal {self.goal}: {s_old} = {per_old}; "
              f"best place {best}: {s_new} = {per_new}")
        if s_new >= s_old + GOAL_BETTER and front in self.paths and front != self.goal:
            self.goal = front
            self.paths = nx.single_source_dijkstra_path(self.G, self.goal, weight="weight")
            self.reach = np.zeros(self.n, bool)
            self.reach[list(self.paths)] = True
            self._local = {}
            print(f"GOAL moved to photo {front}")
        else:
            print("GOAL kept")


if __name__ == "__main__":
    argv = sys.argv[1:]
    GoalAgent.goal_check = "--no-goal-check" not in argv
    argv = [a for a in argv if a != "--no-goal-check"]
    sys.argv[1:] = with_defaults(argv, flags=FLAGS, log_dir="logs/my_agent_2767g")
    main(GoalAgent)
