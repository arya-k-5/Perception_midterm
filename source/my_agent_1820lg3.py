"""
my_agent_1820lg3.py -- my_agent_1820lg2 (8 fixes; 6/10 on maze 55) + DIRECTION-AWARE SHORTEST PATH. NEW FILE.

Why (ticks by reason): the sample-challenge run that scored 7/10 with SPL 0.63 spent 0% of its ticks driving a
drive backwards; lg2 on maze 55 (6/10, SPL 0.22) spent 34% (1,281 of 3,758 ticks) on "reverse" + "undo-turns".
Kartik's planner (Dijkstra on an undirected graph) charges going back along a drive the same as going forward,
but backing up is slow, blind (the camera faces away from where it goes) and gets stuck.

Change: every shortest-path computation of the agent (start-up, detours, dropped shortcuts, new links, goal check)
runs on a directed copy of the map in which going AGAINST a drive costs more:
    backing up over a recorded FORWARD step   x REV_BACK
    undoing a recorded turn on the spot       x REV_TURN   (lg2 does these in one measured turn)
    forward along a drive, shortcuts          unchanged
so routes prefer drives that go the right way. `hops` (and everything that uses it) stays the number of hops.

    uv run source/my_agent_1820lg3.py --local 55 --budget 10000
    --rev-back X --rev-turn Y to change the costs; --rev-back 1 --rev-turn 1 = lg2
"""

from __future__ import annotations

import sys

import networkx as nx
import sg_matcher
from lg_matcher import LGMatcher
from my_agent import with_defaults
from my_agent_1820g import FLAGS
from my_agent_1820lg2 import (LG_CHECKIN_MIN, LG_CHECKIN_STAGES, LG_GOTO_MIN, LG2Agent)
from my_agent_2767g import GoalAgent
import sptm_agent as base
from sptm_agent import main

REV_BACK = 4.0
REV_TURN = 1.5

_ORIG = nx.single_source_dijkstra_path


def _directed_paths(G, source, *args, weight="weight", **kw):
    """Same result format as nx.single_source_dijkstra_path(G, goal): {node: [goal, ..., node]}, but on a directed
    copy of G when G.graph carries the drive layout; otherwise exactly the original."""
    info = G.graph.get("lg3") if hasattr(G, "graph") else None
    if not info:
        return _ORIG(G, source, *args, weight=weight, **kw)
    traj, acts, rb, rt = info["traj"], info["acts"], info["rev_back"], info["rev_turn"]
    R = nx.DiGraph()  # edge x -> y in R  ==  the robot moves y -> x
    R.add_nodes_from(G.nodes)
    for u, v, d in G.edges(data=True):
        w = d.get(weight, 1.0)
        if traj[u] == traj[v] and abs(u - v) == 1:
            lo, hi = min(u, v), max(u, v)
            back = w * (rb if acts[lo] in ("FORWARD", "BACKWARD") else rt)
            R.add_edge(hi, lo, weight=w)  # robot lo -> hi: the way the drive went
            R.add_edge(lo, hi, weight=back)  # robot hi -> lo: against the drive
        else:
            R.add_edge(u, v, weight=w)
            R.add_edge(v, u, weight=w)
    return _ORIG(R, source, *args, weight="weight", **kw)


class LG3Agent(LG2Agent):
    rev_back = REV_BACK
    rev_turn = REV_TURN

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)  # the base builds the map here
        self.G.graph["lg3"] = dict(traj=self.traj, acts=[f.action for f in self.frames],
                                   rev_back=self.rev_back, rev_turn=self.rev_turn)
        print(f"direction-aware planning: backing up x{self.rev_back}, undoing a turn x{self.rev_turn}")

    def finish(self, result) -> None:
        super().finish(result)


def _pop_val(argv, flag, default):
    if flag in argv:
        i = argv.index(flag)
        v = float(argv[i + 1])
        del argv[i:i + 2]
        return v
    return default


if __name__ == "__main__":
    nx.single_source_dijkstra_path = _directed_paths  # every caller looks it up on the module at call time
    sg_matcher.SGMatcher = LGMatcher
    argv = sys.argv[1:]
    LG3Agent.rev_back = _pop_val(argv, "--rev-back", REV_BACK)
    LG3Agent.rev_turn = _pop_val(argv, "--rev-turn", REV_TURN)
    GoalAgent.goal_check = "--no-goal-check" not in argv
    LG2Agent.undo_on = "--no-undo1" not in argv
    LG2Agent.look_on = "--no-lookroute" not in argv
    LG2Agent.goal_on = "--no-goalnear" not in argv
    LG2Agent.lgth_on = "--no-lgthresh" not in argv
    if LG2Agent.lgth_on:
        base.CHECKIN_MIN, base.GOTO_MIN, base.CHECKIN_STAGES = LG_CHECKIN_MIN, LG_GOTO_MIN, LG_CHECKIN_STAGES
    argv = [x for x in argv if x not in ("--no-goal-check", "--no-undo1", "--no-lookroute", "--no-goalnear",
                                         "--no-lgthresh")]
    sys.argv[1:] = with_defaults(argv, flags=FLAGS, log_dir="logs/my_agent_1820lg3")
    main(LG3Agent)
