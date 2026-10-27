import numpy as np
from hybrid_scorer import HybridScorer


class FakeC:
    class ex:
        dim = 4

    ticks, dbg = 1, ""

    def prob(self, a, b):
        return (a * b).sum(1)  # in [0, 1] for the unit-ish vectors below

    def calibrate(self, emb, bounds):
        self.cal = emb.shape

    def embed(self, images):
        return np.ones((len(images), 4), np.float32) / 2


class FakeT:
    def prob(self, a, b):
        return np.full(len(a), 0.25, np.float32)

    def embed(self, images):
        return np.zeros((len(images), 3), np.float32)

    def locomote(self, cur, wp, node=-1):
        return "LEFT"


def test_probability_needs_both_and_l_steers():
    h = HybridScorer(FakeC(), FakeT(), use_l=True)
    emb = h.embed([0, 1])
    assert emb.shape == (2, 7)
    p = h.prob(emb[:1], emb[1:])  # p_vlad = 4 * 0.25 = 1.0, p_R = 0.25 -> sqrt = 0.5
    assert abs(float(p[0]) - 0.5) < 1e-6
    assert h.locomote(None, None) == "LEFT" and h.ticks == 2
    h.calibrate(emb, [(0, 2)])
    assert h.c.cal == (2, 4)
