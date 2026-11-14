import numpy as np
import pytest
from smooth_graph.regression import (
    calibrate_joint_threshold,
    calibrate_threshold,
    fit_window,
    noise_joint_rate,
)

STEPS = np.random.default_rng(42).normal(0, 0.01, 5000)   # stand-in for real daily log changes


def test_returns_a_positive_float():
    thr = calibrate_threshold(STEPS, window=30)
    assert isinstance(thr, float)
    assert thr > 0


def test_same_seed_same_answer():
    assert calibrate_threshold(STEPS, 30, seed=5) == calibrate_threshold(STEPS, 30, seed=5)


def test_higher_quantile_gives_higher_threshold():
    t90 = calibrate_threshold(STEPS, 30, quantile=90)
    t95 = calibrate_threshold(STEPS, 30, quantile=95)
    t99 = calibrate_threshold(STEPS, 30, quantile=99)
    assert t90 < t95 < t99


def test_longer_windows_need_bigger_thresholds():
    assert calibrate_threshold(STEPS, 60) > calibrate_threshold(STEPS, 20)


def test_ignores_the_volatility_of_the_data():
    # t-statistics are scale-free, so 10x bigger steps must give the same threshold
    a = calibrate_threshold(STEPS, 30, seed=7)
    b = calibrate_threshold(STEPS * 10, 30, seed=7)
    assert b == pytest.approx(a, rel=1e-6)


def test_ballpark_for_window_30():
    # I measured ~8.8 for W=30, q=95 on random-walk data. Anything near 2 means a bug.
    assert 6 < calibrate_threshold(STEPS, 30, quantile=95) < 12


def test_definition_noise_exceeds_threshold_about_5_percent_of_the_time():
    thr = calibrate_threshold(STEPS, 30, quantile=95, n_sims=3000, seed=1)
    rng = np.random.default_rng(999)                     # DIFFERENT seed: fresh noise
    exceed = [abs(fit_window(np.cumsum(rng.choice(STEPS, 30))).t_stat) > thr for _ in range(2000)]
    assert 0.03 < np.mean(exceed) < 0.07


# ---- joint-gate calibration ---------------------------------------------------------
def test_joint_threshold_achieves_the_requested_joint_rate():
    target = 0.02
    thr = calibrate_joint_threshold(STEPS, 30, max_age=7, target_rate=target,
                                    n_sims=8000, seed=3)
    achieved = noise_joint_rate(STEPS, 30, max_age=7, threshold=thr,
                                n_sims=8000, seed=99)     # fresh noise
    assert achieved == pytest.approx(target, abs=0.006)


def test_joint_threshold_is_higher_for_a_rarer_target():
    common = calibrate_joint_threshold(STEPS, 30, 7, target_rate=0.02, seed=0)
    rare = calibrate_joint_threshold(STEPS, 30, 7, target_rate=0.002, seed=0)
    assert rare > common


def test_joint_threshold_is_reproducible():
    a = calibrate_joint_threshold(STEPS, 30, 7, target_rate=0.01, seed=5)
    b = calibrate_joint_threshold(STEPS, 30, 7, target_rate=0.01, seed=5)
    assert a == b


def test_joint_threshold_rejects_an_unattainable_target():
    # With a one-bar recency gate, very few null vertices qualify; asking the
    # full rule to fire on half of all bars is impossible.
    with pytest.raises(ValueError, match="unattainable"):
        calibrate_joint_threshold(STEPS, 30, max_age=1, target_rate=0.5,
                                  n_sims=2000, seed=0)


def test_joint_threshold_rejects_a_bad_rate():
    with pytest.raises(ValueError):
        calibrate_joint_threshold(STEPS, 30, 7, target_rate=0.0)


def test_noise_joint_rate_is_not_the_marginal_t_rate():
    # The whole point: the joint rule fires far less often than "5% of |t|" suggests.
    marginal = calibrate_threshold(STEPS, 30, quantile=95)
    joint = noise_joint_rate(STEPS, 30, max_age=7, threshold=marginal, n_sims=8000, seed=0)
    assert joint < 0.005