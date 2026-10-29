"""
nav_merge.py -- Kartik's navigation (source/my_agent_2767.py) + Arya's anti-jitter rules.

NEW FILE. Nothing that already exists is edited: this module imports MyAgent / main() from
my_agent.py and sptm_agent.py and only wraps the one place every decision leaves the agent (_out).

What stays exactly as in my_agent_2767 (this is what reaches the goal):
    map + verified shortcuts, SuperGlue localisation with far-jump checks, measured-angle steering
    (--steer-hold), detours (--detour-once), the four-photo goal approach and the check-in stages.

What is added (Arya's nav_v6 / v7 / v8 ideas):
    1. Turn-reversal damper (nav_v6 STEER_CONFIRM): a turn that would undo a turn made a moment ago
       (within REVERSE_WINDOW forward ticks) is not executed the first time it is asked for; the
       robot drives FORWARD 1 instead, and the reversal happens only if the NEXT decision asks for
       it again. One-off flips of L / the measured angle (noise) vanish, real corrections cost one
       forward tick. Deliberate manoeuvres (approach, goal scan, search, reversing, recover,
       stuck, check-in) are never touched.
    2. Jitter counter (nav_v7): printed at the end, "jitter: N turn reversals", plus how many
       were suppressed and the ticks spent turning / driving forward / backward.
    3. Photo gate (nav_v8): --goal-watch is on by default, so a clear view of a goal photo anywhere
       on the map starts the approach instead of finishing the planned route first.

    uv run source/nav_merge.py --local 8            # free, local maze
    uv run source/nav_merge.py                      # one real attempt (VIS_NAV_CHALLENGE in .env)
    uv run source/nav_merge.py --local 8 --no-damp  # same agent without rule 1 (for comparison)
"""

from __future__ import annotations

import sys

import numpy as np
from my_agent import MyAgent, with_defaults
from sptm_agent import main
from vis_nav_sdk import Action

# decisions the damper may soften: the routine "follow the map" turns only
DAMP_WHY = ("follow", "steer-turn")
REVERSE_WINDOW = 6  # a turn counts as a reversal if less than this many forward ticks lie between
MAX_DAMP_TICKS = 3  # only small turns (<= ~10 deg) are damped; a big turn is a decision, not noise
CONFIRM = 2  # a reversing turn must be asked for this many decisions in a row
JITTER_TICKS = 3  # the user's metric: a turn that undoes the previous turn within this many ticks

# lost detector (added after the first maze-8 run: the node estimate froze at 720 while the camera saw
# something else -- 0 SuperGlue inliers with that photo -- and the robot reversed for 2,700 ticks)
LOST_OK = 12  # live view matching the believed photo with at least this many inliers = not lost
LOST_AFTER = 8  # decisions in a row below LOST_OK before we call it lost
LOST_K = 40  # photos tried when relocalising (best by place-recognition score)
LOST_MIN = 35  # inliers a candidate needs to be believed (22 gave false relocalisations at 23-34)
LOST_LOOK = False  # look-around turns when lost: off -- 711 / 397 / 396 ticks on mazes 11 / 52 / 8 for 1-3 finds
LOST_NEAR = 12  # hops from the goal below which the lost detector is off
LOST_FAR_MIN = 60  # a relocalisation that puts us > LOST_FAR_HOPS further from the goal needs this many inliers
LOST_FAR_HOPS = 30
LOST_CONFIRM = 2  # ... on this many consecutive checks, same place (+-3 photos)
LOST_SPIN = (2, 6)  # not found: look around, 3 x LEFT 6 (~60 deg) per attempt
LOST_SPIN_MAX = 6  # attempts (~360 deg) before driving a little to a new viewpoint

def with_budget(argv: list[str], default: int = 4000) -> list[str]:
    """No --budget given: stop at ``default`` ticks (the challenge itself would allow 60,000)."""
    if any(a == "--budget" or a.startswith("--budget=") for a in argv):
        return argv
    return argv + ["--budget", str(default)]


# spin guard (maze-8 run: 31 turn runs of >= 20 ticks held 1,239 of 2,324 turning ticks, one of them
# 184 x "follow RIGHT 1" at node 374 while the robot did not move)
SPIN_MAX = 30  # turning ticks in a row (~100 deg) in follow/steer mode before a forward step is forced
SPIN_WHY = ("follow", "steer-turn")
SPIN_ESCAPE_AFTER = 3  # forced breaks at the same photo, then a bigger escape manoeuvre

# --goal-watch removed: it starts the goal approach wherever a goal photo matches >= 40 inliers, and on maze 10
# a look-alike place 93-97 hops from the goal matched with 66-88 inliers -> checked in there twice, 0 points.
# my_agent_2767 runs without it.
FLAGS = ("--smooth-steer", "--verbose", "--steer-hold", "--detour-once")


class MergeAgent(MyAgent):
    """my_agent_2767's agent with a turn-reversal damper and a jitter counter."""

    damp_on = True
    lost_on = True
    spin_on = True
    damp_why = DAMP_WHY
    max_damp_ticks = MAX_DAMP_TICKS

    def _reset(self) -> None:
        super()._reset()
        self.m_last_turn = 0  # +1 LEFT, -1 RIGHT, 0 none
        self.m_since = 10**6  # forward/backward ticks since that turn
        self.m_wait = 0  # consecutive requests for a reversing turn
        self.m_supp = 0  # reversals suppressed
        self.m_rev = 0  # executed turn reversals within JITTER_TICKS (the metric)
        self.m_lost, self.m_spins, self.m_reloc = 0, 0, 0
        self.m_spin, self.m_breaks, self.m_break_node = 0, 0, -1
        self.m_cand, self.m_cand_n = -1, 0
        self.m_why = {}  # ticks asked per reason: where the budget goes
        self.m_ticks = {"turn": 0, "fwd": 0, "back": 0}

    # ---- the one hook

    def _out(self, a, why: str):
        act, n = a if isinstance(a, tuple) else (a, 1)
        if act in (Action.LEFT, Action.RIGHT) and why in SPIN_WHY and self.spin_on:
            if self.m_spin + n > SPIN_MAX:
                self.m_breaks = self.m_breaks + 1 if self.node == self.m_break_node else 1
                self.m_break_node, self.m_spin = self.node, 0
                if self.m_breaks >= SPIN_ESCAPE_AFTER:  # still here after 3 breaks: turn away and drive
                    self.m_breaks = 0
                    other = Action.LEFT if act == Action.RIGHT else Action.RIGHT
                    self.queue = [(other, 10), (Action.FORWARD, 6)]
                    why = "spin-escape"
                else:
                    why = "spin-break"
                a, act, n = (Action.FORWARD, 3), Action.FORWARD, 3
            else:
                self.m_spin += n
        elif act in (Action.FORWARD, Action.BACKWARD):
            self.m_spin = 0
        if act in (Action.LEFT, Action.RIGHT):
            sign = 1 if act == Action.LEFT else -1
            reverses = self.m_last_turn == -sign and self.m_since < REVERSE_WINDOW
            if (self.damp_on and reverses and why in self.damp_why and n <= self.max_damp_ticks
                    and self.m_wait + 1 < CONFIRM):
                self.m_wait += 1
                self.m_supp += 1
                a, act, n, why = (Action.FORWARD, 1), Action.FORWARD, 1, "damp-fwd"
            else:
                self.m_wait = 0
                if self.m_last_turn == -sign and self.m_since <= JITTER_TICKS:
                    self.m_rev += 1
                self.m_last_turn, self.m_since = sign, 0
                self.m_ticks["turn"] += n
        if act in (Action.FORWARD, Action.BACKWARD):
            self.m_since += n
            self.m_ticks["fwd" if act == Action.FORWARD else "back"] += n
        self.m_why[why] = self.m_why.get(why, 0) + n
        return super()._out(a, why)

    # ---- lost detector

    def _best_match_node(self, obs, q, p):
        node = super()._best_match_node(obs, q, p)
        if not self.lost_on or not getattr(self, "sg_on", False) or node < 0 or self.t < 2:
            return node
        if self.log and self.log[-1]["why"] == "replay":  # replay itself just verified the view (>= 30 inliers)
            self.m_lost = 0
            return node
        # never during the last metre: in the approach the camera is turned away from the map photos on
        # purpose (maze 11: "lost" at 3 hops from the goal, relocalised 50 hops away, never came back)
        if (getattr(self, "lock", 0) > 0 or 0 < getattr(self, "scan", 0) < 105
                or getattr(self, "side_target", None) is not None or 0 <= getattr(self, "hops", -1) <= LOST_NEAR):
            self.m_lost = 0
            return node
        try:
            live = self.gm.features(obs.image)
            n_here = self.gm.match(self._node_feat(node), live, self.focal).n
            if n_here >= LOST_OK:
                self.m_lost = self.m_spins = 0
                return node
            self.m_lost += 1
            if self.m_lost < LOST_AFTER:
                return node
            cands = np.argsort(-np.where(self.reach, p, -1.0))[:LOST_K]
            best, best_n = node, n_here
            for c in cands:
                n = self.gm.match(self._node_feat(int(c)), live, self.focal).n
                if n > best_n:
                    best, best_n = int(c), n
            if best != node and best_n >= LOST_MIN and best_n < LOST_FAR_MIN and best in self.paths \
                    and node in self.paths and len(self.paths[best]) - len(self.paths[node]) > LOST_FAR_HOPS:
                best_n = 0  # a big step backwards on a modest match: do not believe it
            if best != node and best_n >= LOST_MIN:
                self.m_cand_n = self.m_cand_n + 1 if abs(best - self.m_cand) <= 3 else 1
                self.m_cand = best
            else:
                self.m_cand, self.m_cand_n = -1, 0
            if best != node and best_n >= LOST_MIN and self.m_cand_n >= LOST_CONFIRM:
                print(f"[{self.t}] lost at node {node} ({n_here} inliers): relocalised to {best} ({best_n} inliers)")
                self.m_lost = self.m_spins = 0
                self.m_reloc += 1
                self.jump_cand, self.jump_n = -1, 0
                self.rev_node, self.rev_n = -1, 0
                self.best_t = self.t  # the stall watchdog restarts from here
                return best
            if not LOST_LOOK:  # check + relocalise only; the base keeps driving
                return node
            if self.m_spins < LOST_SPIN_MAX:
                self.m_spins += 1
                self.queue = [(Action.LEFT, LOST_SPIN[1])] * LOST_SPIN[0]
            else:
                self.m_spins = 0
                self.queue = [(Action.FORWARD, 4)]
            self.state = f"lost at {node} ({n_here} inliers), best other {best_n}: looking around"
        except Exception as e:  # never let the safety net crash a run
            print(f"[{self.t}] lost detector error: {e!r}")
        return node

    def finish(self, result) -> None:
        print(f"jitter: {self.m_rev} turn reversals; {self.m_supp} reversals suppressed by the damper")
        top = sorted(self.m_why.items(), key=lambda kv: -kv[1])
        print("ticks by reason: " + ", ".join(f"{k} {v}" for k, v in top))
        print(f"lost detector: {self.m_reloc} relocalisations")
        print(f"ticks asked: turning {self.m_ticks['turn']}, forward {self.m_ticks['fwd']}, "
              f"backward {self.m_ticks['back']}")
        super().finish(result)


class UndampedAgent(MergeAgent):
    damp_on = False
    lost_on = False
    spin_on = False


if __name__ == "__main__":
    argv = sys.argv[1:]
    damp = "--no-damp" not in argv
    argv = [x for x in argv if x != "--no-damp"]
    sys.argv[1:] = with_defaults(with_budget(argv), flags=FLAGS, log_dir="logs/nav_merge")
    main(MergeAgent if damp else UndampedAgent)
