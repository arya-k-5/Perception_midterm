"""
nav_human.py -- my_agent_2767g (Kartik's my_agent_2767 + the 4-photo goal check) with the steering replaced by
"look, decide, then move", the way a person walks a maze with photos of the route.  NEW FILE; nothing edited.

Kept from the base: the map (photo graph + verified shortcuts), the shortest path (Dijkstra) to the goal,
localisation, the goal approach and check-in, the stuck detector.

Replaced -- every non-approach step is decided here, in this order, and the log says why:
  1. SEE   Match the camera with the next route photos (1..LOOK_AHEAD hops ahead) with SuperGlue; take the one
           that matches best. Its measured angle says where the route goes from here.
  2. FLOOR Measure the free floor ahead / left / right in the image (0 = wall at the feet).
  3. DECIDE
       a. route photo seen and more than TURN_DEG off   -> turn by the measured angle, in one action
       b. route photo seen and roughly ahead, floor open -> drive forward (longer when the floor is long)
       c. wall ahead                                     -> turn ~90 deg to the side that is open (the side the
                                                            route turns to if both are); both closed -> 180 deg
       d. route photo not seen                           -> turn the way the recorded drive turns next (from the
                                                            map), else to the more open side, ~60 deg, and look again
  4. NO BLIND UNDO The base's "undo the recorded move" (drive backwards along a drive) is only allowed while it
           makes progress; after UNDO_LIMIT undo decisions without reaching an earlier photo, the robot turns
           ~180 deg and steps 3-4 above take over (maze 17: 1,534 undo decisions, stuck at one photo).

    uv run source/nav_human.py --local 17 --budget 4000
    --no-human = my_agent_2767g
"""

from __future__ import annotations

import sys

import numpy as np
from geometry import yaw_from_e
from my_agent import with_defaults
from my_agent_2767g import FLAGS, GoalAgent
from sptm_agent import SIGN, main
from vis_nav_sdk import Action

LOOK_AHEAD = 6  # route photos ahead compared with the camera
SEE_MIN = 30  # inliers for "I see that route photo"
TURN_DEG = 12.0  # off by more than this -> turn first
WALL_STOP = 0.22  # free floor ahead below this = wall
SIDE_OPEN = 0.30  # a side with at least this much floor counts as open
FWD_SHORT, FWD_LONG = 3, 6  # forward ticks; long when the floor ahead is long (>= 0.6)
SEARCH_TURN = 18  # ticks (~60 deg) when the route photo is not in view
UNDO_LIMIT = 8  # undo decisions without reaching an earlier photo before giving up on undoing
REPLACE = ("follow", "steer-turn", "steer-fwd", "wp-search")
TICK_DEG = 3.44


class HumanAgent(GoalAgent):
    human_on = True

    def _reset(self) -> None:
        super()._reset()
        self._obs = None
        self.h_undo_node, self.h_undo_n = -1, 0
        self.h_counts: dict[str, int] = {}
        self.h_floor_ok = True

    def setup(self, info) -> None:
        super().setup(info)
        if self.human_on:
            self._check_floor()

    def _check_floor(self) -> None:
        fwd = [i for i in range(0, self.n - 1, 7)
               if self.frames[i].action == "FORWARD" and self.frames[i + 1].action == "FORWARD"][:60]
        if len(fwd) < 10:
            return
        depth = np.array([self._free_space(self._frame_img(i))[0] for i in fwd])
        blocked = float(np.mean(depth < WALL_STOP))
        self.h_floor_ok = blocked <= 0.15
        print(f"floor check: {len(fwd)} forward-driving photos, median floor ahead {np.median(depth):.2f}, "
              f"{blocked:.0%} look like a wall -> wall detection {'ON' if self.h_floor_ok else 'OFF'}")

    def act(self, obs):
        self._obs = obs
        return super().act(obs)

    # ---- the decision

    def _planned_turn(self, path: list[int]) -> float:
        """Net turn (deg, + left) the recorded drives make over the next few hops of the route."""
        total = 0.0
        for a, b in list(zip(path, path[1:]))[:LOOK_AHEAD]:
            if self.traj[a] == self.traj[b] and abs(a - b) == 1:
                act = self.frames[min(a, b)].action
                d = 10.3 if act == "LEFT" else -10.3 if act == "RIGHT" else 0.0
                total += d if b > a else -d
        return total

    def _decide(self):
        node = self.node
        if node < 0 or node not in self.paths or self._obs is None:
            return None
        path = self.paths[node][::-1]
        if len(path) < 2:
            return None
        live = self.gm.features(self._obs.image)
        best_n, best_k, best_m = -1, -1, None
        for k in path[1:LOOK_AHEAD + 1]:
            m = self.gm.match(self._node_feat(k), live, self.focal)
            if m.n > best_n:
                best_n, best_k, best_m = m.n, k, m
        ahead, lft, rgt = self._free_space(self._obs.image) if self.h_floor_ok else (1.0, 1.0, 1.0)
        seen = best_n >= SEE_MIN
        yaw = None
        if seen:
            y = yaw_from_e(best_m, self.focal)
            yaw = None if y is None else float(SIGN * y)  # + = route photo is to the right: turn right
        if seen and yaw is not None and abs(yaw) > TURN_DEG:
            ticks = int(np.clip(round(abs(yaw) / TICK_DEG), 1, 26))
            return (Action.RIGHT if yaw > 0 else Action.LEFT, ticks), \
                f"see {best_k} ({best_n}) {yaw:+.0f} deg off -> turn"
        if ahead < WALL_STOP:
            plan = self._planned_turn(path)
            pref = Action.LEFT if plan > 0 else Action.RIGHT
            if lft >= SIDE_OPEN and rgt >= SIDE_OPEN:
                turn = pref
            elif lft >= SIDE_OPEN or rgt >= SIDE_OPEN:
                turn = Action.LEFT if lft >= rgt else Action.RIGHT
            else:
                return (pref, 52), f"wall ahead, both sides closed (L={lft:.2f} R={rgt:.2f}) -> turn 180"
            return (turn, 26), f"wall ahead ({ahead:.2f}), L={lft:.2f} R={rgt:.2f} -> turn 90 {turn.name}"
        if seen:
            n = FWD_LONG if ahead >= 0.6 else FWD_SHORT
            return (Action.FORWARD, n), f"see {best_k} ({best_n}) ahead, floor {ahead:.2f} -> forward {n}"
        plan = self._planned_turn(path)
        if abs(plan) >= 20:
            turn = Action.LEFT if plan > 0 else Action.RIGHT
            why = f"route photo not in view; the route turns {plan:+.0f} deg -> look {turn.name}"
        else:
            turn = Action.LEFT if lft >= rgt else Action.RIGHT
            why = f"route photo not in view (best {best_n}) -> look {turn.name} (more open)"
        return (turn, SEARCH_TURN), why

    def _out(self, a, why: str):
        if not self.human_on:
            return super()._out(a, why)
        if why == "reverse":
            if self.node == self.h_undo_node:
                self.h_undo_n += 1
            else:
                self.h_undo_node, self.h_undo_n = self.node, 0
            if self.h_undo_n >= UNDO_LIMIT:
                self.h_undo_n = 0
                self.state = f"undo made no progress at {self.node} -> turn round and look"
                self.h_counts["undo-giveup"] = self.h_counts.get("undo-giveup", 0) + 1
                return super()._out((Action.LEFT, 52), "undo-giveup")
            return super()._out(a, why)
        if why in REPLACE:
            try:
                d = self._decide()
            except Exception as e:  # noqa: BLE001
                print(f"[{self.t}] decide error: {e!r}")
                d = None
            if d is not None:
                (act, n), reason = d
                tag = "h-" + reason.split(" -> ")[-1].split(" ")[0]
                self.h_counts[tag] = self.h_counts.get(tag, 0) + 1
                self.state = reason
                return super()._out((act, n), tag)
        return super()._out(a, why)

    def finish(self, result) -> None:
        print("decisions: " + ", ".join(f"{k} {v}" for k, v in sorted(self.h_counts.items(), key=lambda kv: -kv[1])))
        super().finish(result)


if __name__ == "__main__":
    argv = sys.argv[1:]
    HumanAgent.human_on = "--no-human" not in argv
    HumanAgent.goal_check = "--no-goal-check" not in argv
    argv = [a for a in argv if a not in ("--no-human", "--no-goal-check")]
    if not any(a == "--budget" or a.startswith("--budget=") for a in argv):
        argv += ["--budget", "4000"]
    sys.argv[1:] = with_defaults(argv, flags=FLAGS, log_dir="logs/nav_human")
    main(HumanAgent)
