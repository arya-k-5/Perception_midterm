import sys
import types

import groundtruth


def test_wrapper_records_true_distance(monkeypatch):
    class Sim:
        steps = 0
        oracle_steps = 100

        def distance_heading(self):
            return (3.0 - 0.1 * self.steps, 12.0)

    class LocalSession:
        def __init__(self):
            self.simulator = Sim()

        def step_all(self, action, repeat=1, *, frames="last"):
            self.simulator.steps += repeat
            return ["obs"]

    mod = types.ModuleType("vis_nav_sdk")
    local = types.ModuleType("vis_nav_sdk.local")
    local.LocalSession = LocalSession
    mod.local = local
    monkeypatch.setitem(sys.modules, "vis_nav_sdk", mod)
    monkeypatch.setitem(sys.modules, "vis_nav_sdk.local", local)
    groundtruth.reset()
    groundtruth.install()
    s = LocalSession()
    for _ in range(5):
        s.step_all(1, 2)
    out = groundtruth.summary()
    assert out["start_m"] == 2.8 and out["final_m"] == 2.0 and out["oracle_ticks"] == 100
    assert out["min_at_tick"] == 10
