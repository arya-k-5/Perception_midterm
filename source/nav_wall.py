"""
nav_wall.py -- nav_route2.py (in its --no-route form by default) plus what a person does in a maze.
NEW FILE; nothing existing is edited.

  1. LOOK BEFORE YOU STEP. Before any forward step chosen by the map follower (L network, measured steering,
     damper, spin-break), the floor ahead is measured in the camera image (the base's own _free_space: free floor
     below the horizon, 0 = wall at the feet, 1 = floor to the horizon). If the wall is close, the robot does not
     step: it turns, in one committed direction, until the way ahead is open again. The side: the one the map
     follower leans to if it is open, otherwise the more open side; if both are closed, it keeps turning
     (a dead end -> it comes out turned round). Maze 10, steps 194-616: 400 decisions of "FORWARD 1" into a wall.
  2. DON'T GO BACK WHERE YOU WENT. Photos the robot has stood at for a long time (>= STAY_LIMIT decisions) make
     the route through them costlier, so the planner sends it another way (the base's detour, used early:
     after 60 steps without progress instead of 150).
  3. MATCH WHERE YOU WANT TO GO: unchanged -- the base steers to the next route photo by the measured angle.

    uv run source/nav_wall.py --local 10 --budget 4000          (no map-first route; --route to plan it first)
    options: --no-wall --no-settle --no-damp --no-lost --no-spin --route
"""

from __future__ import annotations

import math
import sys

import networkx as nx
import numpy as np
import sptm_agent as base
from geometry import yaw_from_e
from my_agent import with_defaults
from nav_merge import FLAGS, with_budget
from nav_route2 import Route2Agent
from sptm_agent import main
from vis_nav_sdk import Action

WALL_STOP = 0.22  # floor depth ahead below which a forward step is not taken
WALL_CLEAR = 0.40  # ... and the committed turn goes on until the depth ahead is at least this
WALL_SIDE = 0.30  # a side counts as open from this depth
WALL_STEP = 6  # ticks per turn step (~21 deg)
WALL_MAX = 18  # turn steps at most (~370 deg), then the map follower gets the robot back
WALL_WHY = ("follow", "steer-fwd", "damp-fwd", "spin-break", "directed", "recover")
STALL_EARLY = 60  # base default 150

# FIND THE GOAL with all four goal photos (maze 17: the base took one photo's 28-inlier match). The goal spot
# was photographed in 4 directions; a real goal place shows several of them within a few frames of each other.
GOAL_CANDS = 15  # best place-recognition candidates per goal photo
GOAL_NBR = 20  # frames either side on the same drive that count as "the same place"
GOAL_STEP = 4  # check every 4th of those frames
GOAL_BETTER = 20  # combined inliers the new goal must beat the base's goal by

# UNDO IN ONE GO (maze 17: "undo LEFT" = RIGHT 2, repeated 670 times without reaching the previous photo).
# Going back over photos where the drive turned on the spot: measure where the camera points now against the
# photo we are at, and turn by the whole net amount at once; then we are at the photo before those turns.
UNDO_MIN = 25  # inliers with the current photo needed to measure the heading


class WallAgent(Route2Agent):
    wall_on = True

    def setup(self, info) -> None:
        super().setup(info)
        self.stall_steps = STALL_EARLY
        try:
            self._find_goal()
        except Exception as e:  # noqa: BLE001
            print(f"multi-view goal search failed ({e!r}); keeping the base's goal")
        if self.wall_on:
            self._check_floor()

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

        def score(c: int) -> tuple[int, list[int], int]:
            per = [max(nin(j, k) for j in nbr(c)) for k in range(4)]
            front = max(nbr(c), key=lambda j: nin(j, 0))
            if nin(front, 0) < 15:  # front photo not seen there: the frame matching any goal photo best
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
            print(f"GOAL moved to photo {front} (front-photo frame of that place)")
            if self.route_on:
                self._plan_map()

    def _undo_in_one_go(self):
        node = self.node
        if node < 0 or node not in self.paths or self._obs is None:
            return None
        path = self.paths[node][::-1]
        i, undo = 0, 0.0
        while (i + 1 < len(path) and path[i + 1] == path[i] - 1 and self.traj[path[i + 1]] == self.traj[node]
               and self.frames[path[i + 1]].action in ("LEFT", "RIGHT")):
            a = self.frames[path[i + 1]].action  # the drive went path[i+1] -> path[i] with this turn
            undo -= 10.3 if a == "LEFT" else -10.3
            i += 1
        if i == 0:
            return None  # not undoing a turn (e.g. undoing a FORWARD = backing up): the base does it
        e = path[i]
        m = self.gm.match(self._node_feat(node), self.gm.features(self._obs.image), self.focal)
        if m.n < UNDO_MIN:
            return None
        y = yaw_from_e(m, self.focal)
        if y is None:
            return None
        here = float(base.SIGN * y)  # + = camera turned left of the photo we are at
        need = (undo - here + 180.0) % 360.0 - 180.0
        self.node = e  # after this turn we face the way photo e was taken
        self.m_undo += 1
        self.state = f"undo {node}->{e} in one go: turn {need:+.0f} deg (drive turned {-undo:+.0f} there)"
        ticks = int(round(abs(need) / 3.44))
        if ticks == 0:
            return (Action.IDLE, 1)
        return (Action.LEFT if need > 0 else Action.RIGHT, min(ticks, 52))

    def _check_floor(self) -> None:
        """Does the floor detector work in this maze? Where the exploration drive went FORWARD there was floor
        ahead, so the detector must say "open" there; where it turned on the spot it often faced a wall."""
        import numpy as np
        fwd = [i for i in range(0, self.n - 1, 7) if self.frames[i].action == "FORWARD"
               and self.frames[i + 1].action == "FORWARD"][:60]
        if len(fwd) < 10:
            return
        depth = [self._free_space(self._frame_img(i))[0] for i in fwd]
        blocked = float(np.mean(np.asarray(depth) < WALL_STOP))
        print(f"wall guard check: floor ahead on {len(fwd)} forward-driving photos: median {np.median(depth):.2f}, "
              f"{blocked:.0%} would count as a wall")
        if blocked > 0.15:
            self.wall_on = False
            print("wall guard OFF for this maze: the floor detector does not match it")

    def _reset(self) -> None:
        super()._reset()
        self.w_dir, self.w_left, self.m_wall, self.m_wall_ticks = None, 0, 0, 0
        self.m_undo = 0

    def act(self, obs):
        self._obs = obs
        if self.wall_on and self.w_left > 0:
            left = obs.steps_left if self.budget is None else min(obs.steps_left, self.budget - obs.step)
            ahead, _, _ = self._free_space(obs.image)
            if left > 12 and ahead < WALL_CLEAR:
                self.t += 1
                self.w_left -= 1
                self.m_wall_ticks += WALL_STEP
                self.state = f"wall: turning {self.w_dir.name} until open (ahead={ahead:.2f})"
                return self._out((self.w_dir, WALL_STEP), "wall-turn")
            self.w_left, self.w_dir = 0, None  # open again (or out of turns): back to the map follower
        return super().act(obs)

    def _out(self, a, why: str):
        if why == "reverse":
            try:
                u = self._undo_in_one_go()
            except Exception as e:  # noqa: BLE001
                print(f"[{self.t}] undo error: {e!r}")
                u = None
            if u is not None:
                return super()._out(u, "undo-turn")
        act = a[0] if isinstance(a, tuple) else a
        if self.wall_on and why in WALL_WHY and act == Action.FORWARD and self._obs is not None:
            ahead, lft, rgt = self._free_space(self._obs.image)
            if ahead < WALL_STOP:
                lean = Action[self.l_side] if getattr(self, "l_side", None) in ("LEFT", "RIGHT") else None
                side_depth = {Action.LEFT: lft, Action.RIGHT: rgt}
                if lean is not None and side_depth[lean] >= WALL_SIDE:
                    turn = lean
                elif max(lft, rgt) >= WALL_SIDE:
                    turn = Action.LEFT if lft >= rgt else Action.RIGHT
                else:
                    turn = lean or Action.LEFT  # dead end: keep turning one way until it opens
                self.w_dir, self.w_left = turn, WALL_MAX - 1
                self.m_wall += 1
                self.m_wall_ticks += WALL_STEP
                self.state = f"wall ahead ({ahead:.2f}, L={lft:.2f} R={rgt:.2f}): turning {turn.name}"
                return super()._out((turn, WALL_STEP), "wall-turn")
        return super()._out(a, why)

    def finish(self, result) -> None:
        print(f"undo in one go: {self.m_undo} times")
        print(f"wall guard: {self.m_wall} forward steps into a wall avoided, {self.m_wall_ticks} ticks of wall turns")
        super().finish(result)


if __name__ == "__main__":
    argv = sys.argv[1:]
    flags = ("--no-wall", "--no-settle", "--no-damp", "--no-lost", "--no-spin", "--route")
    WallAgent.wall_on = "--no-wall" not in argv
    WallAgent.settle_on = "--no-settle" not in argv
    WallAgent.damp_on = "--no-damp" not in argv
    WallAgent.lost_on = "--no-lost" not in argv
    WallAgent.spin_on = "--no-spin" not in argv
    WallAgent.route_on = "--route" in argv
    WallAgent.lock_on = WallAgent.route_on
    argv = [x for x in argv if x not in flags]
    base.SEARCH_TICKS, base.SEARCH_AFTER = 45, 5
    sys.argv[1:] = with_defaults(with_budget(argv), flags=FLAGS, log_dir="logs/nav_wall")
    main(WallAgent)
