import numpy as np
from place_memory import PlaceMemory


def ring_with_branch():
    # 0-1-2-3-4-5 is the short route 0 -> 5; 0-6-7-8-9-10-5 is a longer detour
    e = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 5)]
    return np.array(e)


def test_penalty_reroutes():
    m = PlaceMemory(11, ring_with_branch(), goal=5)
    m.set_goal(5)
    assert m.path(0)[:3] == [0, 1, 2]
    m.punish([2], 8.0, 0)  # we got stuck at 2
    assert m.path(0)[:2] == [0, 6]  # now the detour is cheaper
    assert m.hops[0] == 5  # unweighted distance is unchanged


def test_stagnation_detected_only_when_circling():
    m = PlaceMemory(200, np.array([(i, i + 1) for i in range(199)]), goal=199, window=50, max_spots=4)
    for i in range(50):
        m.arrive(10 + (i % 6))  # circling in a small area
    assert m.stagnating()
    zone = m.escape_zone()
    assert zone and not m.stagnating()
    for i in range(50):
        m.arrive(i * 3)  # genuinely travelling
    assert not m.stagnating()


def test_turnaround_prior():
    acts = ["FORWARD"] * 5 + ["LEFT"] * 40 + ["FORWARD"] * 5
    m = PlaceMemory(50, np.array([(i, i + 1) for i in range(49)]), goal=49)
    assert m.mark_turnarounds(acts, [(0, 50)]) == 1
    assert m.pen[20] > 0 and m.pen[1] == 0


def test_false_shortcut_is_removed_and_route_goes_the_long_way():
    # a chain 0..19 with a look-alike jump 4 -> 15; the goal is node 19
    n = 20
    edges = [(i, i + 1) for i in range(n - 1)] + [(4, 15)]
    m = PlaceMemory(n, np.array(edges), goal=19)
    m.set_goal(19)
    assert m.hops[4] == 5
    assert m.ban_shortcuts([4]) == 1
    assert m.hops[4] == 15  # no more jump: walk the drive
    assert m.path(4, limit=30)[1] == 5


def test_reverse_driving_costs_more():
    # drive 0-1-2-3-4-5-6-7 (frame i+1 is after frame i) plus a look-alike jump 7 -> 0; goal 1, start 3
    edges = np.array([(i, i + 1) for i in range(7)] + [(0, 7)])
    cheap = PlaceMemory(8, edges, goal=1, rev_cost=1.0)
    assert cheap.path(3, limit=10)[1] == 2  # straight back along the drive
    dear = PlaceMemory(8, edges, goal=1, rev_cost=4.0)
    assert dear.path(3, limit=10)[1] == 4  # keep driving forward, around the loop
