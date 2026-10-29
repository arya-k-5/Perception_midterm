"""
nav_route.py -- MAP FIRST, THEN THE SHORTEST PATH, THEN FOLLOW IT.  NEW FILE; nothing existing is edited.

Base: Kartik's my_agent_2767 (reaches the goal) through nav_merge.py (lost detector, jitter counter,
reversal damper). What is new here:

  1. MAP FIRST. Before the first move, setup() builds the whole map: the topological graph of all
     exploration frames, the verified shortcuts (SuperGlue + sequence check), the spawn joins. Then
  2. PRICE THE MAP IN TICKS. The base charges every shortcut a flat cost, whatever it asks of the robot.
     A shortcut between two drives that face opposite ways needs a ~180 deg turn (52 ticks); one between
     two drives facing the same way needs none. For every shortcut on a candidate route this file measures
     the relative yaw of its two photos (essential matrix, lecture 4/5) and sets
         cost = 1 + |yaw| / 10.3 + SHORTCUT_RISK        (one hop = one recorded frame = 3 ticks = 10.3 deg)
     then re-plans, and repeats until the route stops changing (at most 4 rounds).
  3. DECIDE THE PATH ONCE, from the spawn (every run starts there), print it, and save it to
     logs/nav_route/route_*.json: number of hops, shortcuts used, estimated ticks, turn ticks.
  4. FOLLOW THE ROUTE. The robot's position estimate is kept on that route: while the camera still agrees
     with a photo of the route within [ri-2, ri+ROUTE_WINDOW], it is localised there (no more flipping
     between neighbouring photos / other drives, which is what made the base re-plan every step). If the
     camera clearly disagrees (lost detector) or the robot is forced off the route, the base's own planner
     takes over from wherever it is, and the robot rejoins the route when it comes back to it.

    uv run source/nav_route.py --local 8             # budget 4000 ticks unless --budget is given
    options: --no-route (price/plan as the base does), --no-damp, --no-lost
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
from geometry import yaw_from_e
from my_agent import with_defaults
from nav_merge import FLAGS, MergeAgent, with_budget
from sptm_agent import SIGN, main
from vis_nav_sdk import Action

TICKS_PER_HOP = 3.0  # one recorded frame
DEG_PER_HOP = 10.3  # 3 ticks x 3.44 deg
SHORTCUT_RISK = 1.5  # hops: price of the risk of switching drives (base: flat 4.0 + inlier term)
MIN_YAW_INLIERS = 20  # a yaw from fewer inliers is not trusted: keep the base's price
PLAN_ROUNDS = 4
ROUTE_WINDOW = 14  # photos ahead on the route in which the robot may be localised
ROUTE_P = 0.8  # place-recognition score a route photo needs to pull the estimate back onto the route


DIR_AHEAD = 3  # photos ahead whose planned turning the directed steering adds up
DIR_MIN = 20  # inliers with the photo we are at, needed to read our heading from it
DIR_DEAD = 10.0  # degrees: closer than this to the planned heading = drive straight
DIR_FOLLOW_RUN = 12  # L's turn ticks in a row before directed steering may take over from it
DIR_MAX_RUN = 40  # turn ticks in a row by directed steering before the base is allowed to try


class RouteAgent(MergeAgent):
    route_on = True
    directed_on = True
    _obs = None

    def _reset(self) -> None:
        super()._reset()
        self.m_dir, self.m_dir_run, self.m_follow_run = 0, 0, 0
        self.m_edge: dict[tuple[int, int], float | None] = {}
        self.route: list[int] = []
        self.route_idx: dict[int, int] = {}
        self.ri = 0
        self.m_costed: dict[tuple[int, int], float | None] = {}
        self.m_onroute = 0
        self.m_off = 0

    # ---- 1-3: the map and the path, before the first move

    def setup(self, info) -> None:
        super().setup(info)  # base: graph is built in __init__, goal found here, weighted Dijkstra
        self.m_costed = {}
        t0 = time.time()
        try:
            self._plan_map()
        except Exception as e:  # never let planning crash the run: the base's plan still stands
            print(f"map planning failed ({e!r}); using the base's route")
            self.route, self.route_idx = [], {}
        print(f"map planning took {time.time() - t0:.1f} s")

    def _turn_hops(self, a: int, b: int) -> float | None:
        m = self.gm.match(self._node_feat(a), self._node_feat(b), self.focal)
        if m.n < MIN_YAW_INLIERS:
            return None
        y = yaw_from_e(m, self.focal)
        return None if y is None else abs(float(y)) / DEG_PER_HOP

    def _plan_map(self) -> None:
        starts = sorted({s for s, _ in self.bounds})
        route: list[int] = []
        for rnd in range(PLAN_ROUNDS if self.route_on else 0):
            dist, paths = nx.single_source_dijkstra(self.G, self.goal, weight="weight")
            start = min((s for s in starts if s in dist), key=dist.get)
            route = paths[start][::-1]
            changed = 0
            for a, b in zip(route, route[1:]):
                if abs(a - b) == 1 and self.traj[a] == self.traj[b]:
                    continue  # a driven edge: its price is 1 hop
                e = (min(a, b), max(a, b))
                if e in self.m_costed:
                    continue
                turn = self._turn_hops(a, b)
                self.m_costed[e] = turn
                if turn is not None:
                    self.G.edges[e]["weight"] = 1.0 + turn + SHORTCUT_RISK
                    changed += 1
            if not changed:
                break
        # final plan with the priced map
        self.paths = nx.single_source_dijkstra_path(self.G, self.goal, weight="weight")
        self.reach = np.zeros(self.n, bool)
        self.reach[list(self.paths)] = True
        self._local = {}
        dist, paths = nx.single_source_dijkstra(self.G, self.goal, weight="weight")
        start = min((s for s in starts if s in dist), key=dist.get)
        route = paths[start][::-1]
        self.route, self.route_idx, self.ri = route, {n: i for i, n in enumerate(route)}, 0
        hops = len(route) - 1
        jumps = [(a, b) for a, b in zip(route, route[1:]) if not (abs(a - b) == 1 and self.traj[a] == self.traj[b])]
        turn_hops = sum(self.m_costed.get((min(a, b), max(a, b))) or 0.0 for a, b in jumps)
        rev = sum(1 for a, b in zip(route, route[1:]) if b == a - 1 and self.traj[a] == self.traj[b])
        print(f"MAP: {self.n} photos, {self.G.number_of_edges()} edges. ROUTE from spawn photo {route[0]}: "
              f"{hops} hops (~{hops * TICKS_PER_HOP:.0f} ticks), {len(jumps)} drive switches "
              f"(~{turn_hops * TICKS_PER_HOP:.0f} ticks of turning at them), {rev} hops driven in reverse")
        out = Path(self.log_dir or "logs/nav_route")
        out.mkdir(parents=True, exist_ok=True)
        (out / f"route_{int(time.time())}.json").write_text(json.dumps(dict(
            goal=int(self.goal), route=[int(x) for x in route], jumps=[[int(a), int(b)] for a, b in jumps],
            priced={f"{e[0]}-{e[1]}": c for e, c in self.m_costed.items()})))

    # ---- directed steering: read the planned turn from the map instead of spinning blind

    def act(self, obs):
        self._obs = obs
        return super().act(obs)

    def _edge_turn(self, a: int, b: int) -> float:
        """Heading change (degrees, + = left) when the route goes from photo a to photo b."""
        if self.traj[a] == self.traj[b] and abs(a - b) == 1:
            lo = min(a, b)
            act = self.frames[lo].action  # the action that took the drive from photo lo to lo+1
            d = DEG_PER_HOP if act == "LEFT" else -DEG_PER_HOP if act == "RIGHT" else 0.0
            return d if b > a else -d
        key = (a, b)
        if key not in self.m_edge:
            m = self.gm.match(self._node_feat(a), self._node_feat(b), self.focal)
            y = yaw_from_e(m, self.focal) if m.n >= MIN_YAW_INLIERS else None
            self.m_edge[key] = None if y is None else float(SIGN * y)
        return self.m_edge[key] or 0.0

    def _directed(self):
        node = self.node
        if node < 0 or node not in self.paths or self._obs is None:
            return None
        path = self.paths[node][::-1]
        if len(path) < 2:
            return None
        live = self.gm.features(self._obs.image)
        m = self.gm.match(self._node_feat(node), live, self.focal)
        if m.n < DIR_MIN:
            return None
        y = yaw_from_e(m, self.focal)
        if y is None:
            return None
        here = float(SIGN * y)  # + = the camera is turned left of the photo we are at
        steps = list(zip(path[:DIR_AHEAD + 1], path[1:DIR_AHEAD + 1]))
        if any(not (abs(a - b) == 1 and self.traj[a] == self.traj[b]) for a, b in steps):
            return None  # a drive switch just ahead: the recorded turns do not say where to face
        planned = sum(self._edge_turn(a, b) for a, b in steps)
        need = planned - here  # degrees to turn (+ = left) to face where the route goes
        self.state = f"directed here={here:+.0f} planned={planned:+.0f} need={need:+.0f} n={m.n}"
        if abs(need) <= DIR_DEAD:
            self.m_dir_run = 0
            return (Action.FORWARD, 2)
        ticks = int(np.clip(round(abs(need) / 3.44), 1, 6))
        if self.m_dir_run + ticks > DIR_MAX_RUN:
            self.m_dir_run = 0
            return None  # too long a turn in a row: let the base steer
        self.m_dir_run += ticks
        return (Action.LEFT if need > 0 else Action.RIGHT, ticks)

    def _out(self, a, why: str):
        act = a[0] if isinstance(a, tuple) else a
        n = a[1] if isinstance(a, tuple) else 1
        follow_turn = why == "follow" and act in (Action.LEFT, Action.RIGHT)
        self.m_follow_run = self.m_follow_run + n if follow_turn else 0
        recent = [x["why"] for x in self.log[-2:]]
        # only where the base would spin: the look-around, or a long run of L's turns. Never right after a
        # reversing step (maze 52: directed LEFT 6 and reverse RIGHT 2 x3 undid each other for 150 steps)
        if (self.directed_on and (why == "wp-search" or (follow_turn and self.m_follow_run >= DIR_FOLLOW_RUN))
                and "reverse" not in recent):
            if True:
                try:
                    d = self._directed()
                except Exception as e:  # noqa: BLE001
                    print(f"[{self.t}] directed steering error: {e!r}")
                    d = None
                if d is not None:
                    a, why = d, "directed"
                    self.m_dir += 1
        return super()._out(a, why)

    # ---- 4: follow it

    def _best_match_node(self, obs, q, p):
        before = self.m_reloc
        node = super()._best_match_node(obs, q, p)  # base localisation + the lost detector
        if not (self.route_on and self.route):
            return node
        if self.m_reloc != before:  # the lost detector moved us: re-anchor on the route if we are on it
            if node in self.route_idx:
                self.ri = self.route_idx[node]
            return node
        lo, hi = max(0, self.ri - 2), min(len(self.route), self.ri + ROUTE_WINDOW)
        i = self.route_idx.get(node, -1)
        if lo <= i < hi:
            self.ri, self.m_onroute = i, self.m_onroute + 1
            return node
        win = np.asarray(self.route[lo:hi], dtype=np.int64)
        if len(win):
            best = int(win[np.argmax(p[win])])
            if p[best] >= ROUTE_P:  # the camera still matches the route: stay on it
                self.ri, self.m_onroute = self.route_idx[best], self.m_onroute + 1
                return best
        self.m_off += 1
        return node

    def finish(self, result) -> None:
        print(f"directed steering: {self.m_dir} decisions replaced a blind turn")
        print(f"route: localised on the route in {self.m_onroute} decisions, off it in {self.m_off}")
        super().finish(result)


if __name__ == "__main__":
    argv = sys.argv[1:]
    RouteAgent.route_on = "--no-route" not in argv
    RouteAgent.damp_on = "--no-damp" not in argv
    RouteAgent.lost_on = "--no-lost" not in argv
    RouteAgent.spin_on = "--no-spin" not in argv
    RouteAgent.directed_on = "--no-directed" not in argv
    argv = [a for a in argv if a not in ("--no-route", "--no-damp", "--no-lost", "--no-spin", "--no-directed")]
    import sptm_agent as base
    base.SEARCH_TICKS, base.SEARCH_AFTER = 45, 5  # base 105 / 4: look-around spins held 681 of 2,988 ticks
    sys.argv[1:] = with_defaults(with_budget(argv), flags=FLAGS, log_dir="logs/nav_route")
    main(RouteAgent)
