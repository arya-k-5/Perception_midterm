"""ClassicalScorer: calibration and heading logic on synthetic images (no simulator, no torch)."""

import cv2
import numpy as np
from classical_scorer import ClassicalScorer


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    img = (rng.random((240, 320, 3)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    for _ in range(60):  # blurred noise has no corners; rectangles give SIFT something to hold
        x, y = int(rng.integers(0, 300)), int(rng.integers(0, 220))
        w, h = int(rng.integers(5, 30)), int(rng.integers(5, 30))
        cv2.rectangle(img, (x, y), (x + w, y + h), [int(c) for c in rng.integers(0, 255, 3)], -1)
    return img


def _shift(img, px):
    out = np.roll(img, px, axis=1)
    return out


def test_locomote_turns_toward_waypoint(tmp_path):
    sc = ClassicalScorer(tmp_path)
    sc.set_focal(160.0)
    wp = _scene()
    same = sc.locomote(_shift(wp, 3), wp, 1)  # identical images are degenerate for RANSAC
    assert same == "FORWARD"
    left_of = sc.locomote(_shift(wp, 100), wp, 1)  # picture shifted: camera is rotated
    right_of = sc.locomote(_shift(wp, -100), wp, 1)
    assert {left_of, right_of} == {"LEFT", "RIGHT"}


def test_no_match_keeps_turning(tmp_path):
    sc = ClassicalScorer(tmp_path)
    a = sc.locomote(_scene(1), _scene(2), 2)
    assert a in ("LEFT", "RIGHT") and sc.ticks >= 1


def test_prob_is_calibrated(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.normal(size=(300, 64)).astype(np.float32)
    emb = base.copy()
    for i in range(1, 300):  # a smooth walk: adjacent frames similar
        emb[i] = 0.97 * emb[i - 1] + 0.03 * base[i]
    sc = ClassicalScorer(tmp_path)
    sc.calibrate(emb, [(0, 150), (150, 300)])
    p_adj = sc.prob(emb[:-3], emb[3:]).mean()
    p_far = sc.prob(emb[:50], emb[200:250]).mean()
    assert p_adj > 0.8 and p_far < 0.2
