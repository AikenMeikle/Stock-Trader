import math
import numpy as np
import pytest
from smooth_graph.regression import fit_window, MIN_WINDOW


def parabola(n, a, b, c, noise=0.0, seed=0):
    u = np.arange(-(n - 1), 1)
    rng = np.random.default_rng(seed)
    return a * u**2 + b * u + c + rng.normal(0, noise, n)


def test_recovers_known_vertex():
    # a=0.002, b=-0.03  ->  vertex at -b/(2a) = +7.5 (in the future)
    fit = fit_window(parabola(40, 0.002, -0.03, 1.5, noise=0.01))
    assert fit.a == pytest.approx(0.002, abs=3e-4)
    assert fit.vertex == pytest.approx(7.5, abs=2.0)


def test_minimum_8_bars_ago_has_negative_vertex_and_positive_a():
    # y = 0.01 * (u + 8)**2 has its minimum at u = -8
    u = np.arange(-39, 1)
    fit = fit_window(0.01 * (u + 8) ** 2 + np.random.default_rng(1).normal(0, 0.005, 40))
    assert fit.a > 0
    assert fit.vertex == pytest.approx(-8, abs=1.5)
    assert fit.t_stat > 0


def test_cap_shape_has_negative_a_and_negative_t():
    u = np.arange(-39, 1)
    fit = fit_window(-0.01 * (u + 8) ** 2 + np.random.default_rng(2).normal(0, 0.005, 40))
    assert fit.a < 0
    assert fit.t_stat < 0


def test_straight_line_is_not_significant():
    u = np.arange(-39, 1)
    fit = fit_window(0.02 * u + np.random.default_rng(3).normal(0, 0.01, 40))
    assert abs(fit.t_stat) < 4


def test_t_stat_and_vertex_ignore_the_scale_of_the_data():
    base = parabola(40, 0.002, -0.03, 1.5, noise=0.01)
    f1, f2 = fit_window(base), fit_window(base * 10)
    assert f2.t_stat == pytest.approx(f1.t_stat, rel=1e-6)
    assert f2.vertex == pytest.approx(f1.vertex, rel=1e-6)


def test_perfect_parabola_does_not_divide_by_zero():
    fit = fit_window(parabola(40, 0.002, -0.03, 1.5, noise=0.0))
    assert abs(fit.t_stat) > 1e6 or math.isinf(fit.t_stat)


def test_flat_window_gives_nan_vertex_not_a_crash():
    fit = fit_window(np.ones(20))
    assert math.isnan(fit.vertex) or abs(fit.a) < 1e-12


def test_too_short_window_raises():
    with pytest.raises(ValueError):
        fit_window(np.arange(MIN_WINDOW - 1))