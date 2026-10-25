"""
Local-only ground truth: how far from the goal is the robot REALLY, tick by tick?

    import groundtruth; groundtruth.install()          # before run(...)
    ... run the agent on --local ...
    groundtruth.summary()                              # {'min_m':..., 'final_m':..., 'oracle_ticks':...}

Wraps ``LocalSession.step_all`` to read the simulator's true ``distance_heading()`` after every
step. The agent never sees this; it is for us to measure. Does nothing on a server session.
"""

from __future__ import annotations

TRACE: list[tuple[int, float, float]] = []
POSES: list[tuple[float, float, float]] = []
ORACLE: list[int] = []
MAZE: dict = {}


def install() -> None:
    from vis_nav_sdk import local

    if getattr(local.LocalSession, "_gt_installed", False):
        return
    original = local.LocalSession.step_all

    def step_all(self, action, repeat=1, *, frames="last"):
        out = original(self, action, repeat, frames=frames)
        d, h = self.simulator.distance_heading()
        TRACE.append((self.simulator.steps, float(d), float(h)))
        POSES.append(tuple(float(v) for v in self.simulator.pose()))
        if not ORACLE:
            ORACLE.append(int(self.simulator.oracle_steps))
            sim = self.simulator
            for key in ("grid", "spawn", "target", "cell_size", "spawn_cell", "target_cell"):
                try:
                    v = getattr(sim, key)
                    v = v() if callable(v) else v
                    MAZE[key] = list(v) if isinstance(v, tuple) else v
                except Exception:  # an optional extra; never break a run for a plot
                    pass
        return out

    local.LocalSession.step_all = step_all
    local.LocalSession._gt_installed = True


def reset() -> None:
    TRACE.clear()
    POSES.clear()
    ORACLE.clear()
    MAZE.clear()


def summary() -> dict:
    if not TRACE:
        return {}
    ds = [d for _, d, _ in TRACE]
    best = min(range(len(ds)), key=ds.__getitem__)
    return {
        "start_m": round(ds[0], 2),
        "min_m": round(ds[best], 2),
        "min_at_tick": TRACE[best][0],
        "final_m": round(ds[-1], 2),
        "final_heading_deg": round(TRACE[-1][2], 1),
        "oracle_ticks": ORACLE[0] if ORACLE else None,
        "ticks_within_1m": sum(1 for d in ds if d <= 1.0),
    }


def dump(path) -> None:
    import json
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"summary": summary(), "trace": TRACE, "poses": POSES, "maze": MAZE}, default=str))
