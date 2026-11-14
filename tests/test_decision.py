import math
from smooth_graph.regression import WindowFit, decide_target

THR, AGE = 8.0, 7


def fit(a=0.01, t=12.0, vertex=-3.0):
    return WindowFit(a=a, t_stat=t, vertex=vertex)


def test_strong_recent_cup_means_buy_spy():
    assert decide_target(fit(a=0.01, t=12.0, vertex=-3.0), THR, AGE) == "SPY"


def test_strong_recent_cap_means_buy_gld():
    assert decide_target(fit(a=-0.01, t=-12.0, vertex=-3.0), THR, AGE) == "GLD"


def test_weak_bend_means_do_nothing():
    assert decide_target(fit(t=3.0), THR, AGE) is None
    assert decide_target(fit(a=-0.01, t=-3.0), THR, AGE) is None


def test_turn_too_long_ago_means_do_nothing():
    assert decide_target(fit(vertex=-20.0), THR, AGE) is None


def test_turn_in_the_future_means_do_nothing():
    assert decide_target(fit(vertex=3.0), THR, AGE) is None


def test_boundaries_are_inclusive():
    assert decide_target(fit(vertex=0.0), THR, AGE) == "SPY"          # turned today
    assert decide_target(fit(vertex=-7.0), THR, AGE) == "SPY"         # exactly max_age ago
    assert decide_target(fit(vertex=-7.01), THR, AGE) is None         # just too old
    assert decide_target(fit(t=8.0), THR, AGE) == "SPY"               # t exactly at threshold
    assert decide_target(fit(t=7.99), THR, AGE) is None


def test_nan_vertex_means_do_nothing():
    assert decide_target(fit(vertex=math.nan), THR, AGE) is None


def test_perfect_fit_with_infinite_t_still_works():
    assert decide_target(fit(t=math.inf), THR, AGE) == "SPY"
    assert decide_target(fit(a=-0.01, t=-math.inf), THR, AGE) == "GLD"


def test_returns_none_not_empty_string():
    assert decide_target(fit(t=0.0), THR, AGE) is None