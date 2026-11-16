"""Runs run_backtest() on SYNTHETIC data and spies on the calls it makes."""
import numpy as np
import pandas as pd
import pytest

import smooth_graph.backtest as backtest
from tests.helpers import N, curvy_data, random_walk_data


def run_with_spies(monkeypatch, data):
    ratio, history = data
    ratio_before = ratio.copy()
    monkeypatch.setattr(backtest, "load_price_data", lambda: (ratio, history))

    windows, calibrations, strategies, portfolios = [], [], [], []
    real_fit, real_cal = backtest.fit_window, backtest.calibrate_threshold
    RealStrategy, RealPortfolio = backtest.QuadraticStrategy, backtest.Portfolio

    def fit_spy(values):
        windows.append(np.array(values, dtype=float, copy=True))
        return real_fit(values)

    def cal_spy(log_steps, window, quantile=95.0, **kw):
        result = real_cal(log_steps, window, quantile, **kw)
        calibrations.append(dict(log_steps=np.array(log_steps, dtype=float, copy=True),
                                 window=window, quantile=quantile, result=result))
        return result

    rate_at_update = []     # the FX rate the portfolio held WHEN the strategy was about to trade

    class StrategySpy(RealStrategy):
        def __init__(self, threshold, max_age, fallback=None):
            strategies.append((threshold, max_age, fallback))
            super().__init__(threshold=threshold, max_age=max_age, fallback=fallback)

        def update(self, portfolio, fit, prices):
            rate_at_update.append(portfolio.rates[-1] if portfolio.rates else None)
            super().update(portfolio=portfolio, fit=fit, prices=prices)

    class PortfolioSpy(RealPortfolio):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.rates = []
            portfolios.append(self)

        def update_conversion(self, rate):
            self.rates.append(rate)
            super().update_conversion(rate)

    monkeypatch.setattr(backtest, "fit_window", fit_spy)
    monkeypatch.setattr(backtest, "calibrate_threshold", cal_spy)
    monkeypatch.setattr(backtest, "QuadraticStrategy", StrategySpy)
    monkeypatch.setattr(backtest, "Portfolio", PortfolioSpy)

    result = backtest.run_backtest()
    return dict(result=result, windows=windows, calibrations=calibrations, strategies=strategies,
                portfolios=portfolios, rate_at_update=rate_at_update, ratio=ratio, history=history, ratio_before=ratio_before)


@pytest.fixture
def quiet(monkeypatch):
    return run_with_spies(monkeypatch, random_walk_data())


@pytest.fixture
def curvy(monkeypatch):
    return run_with_spies(monkeypatch, curvy_data())


# ---- structure of the loop (random-walk data) -------------------------------
def test_every_bar_including_the_last_is_traded(quiet):
    r, ratio = quiet["result"], quiet["ratio"]
    assert len(r.dates) == N - backtest.WARMUP_BARS + 1
    assert r.dates[0] == ratio.index[backtest.WARMUP_BARS - 1]
    assert r.dates[-1] == ratio.index[-1]
    assert len(r.strategy_values) == len(r.baseline_values) == len(r.t_stats) == len(r.dates)


def test_windows_are_rolling_not_expanding(quiet):
    assert len(quiet["windows"]) == len(quiet["result"].dates)
    assert all(len(w) == backtest.WINDOW for w in quiet["windows"])


def test_each_window_is_the_log_ratio_ending_on_the_current_bar(quiet):
    W, S = backtest.WINDOW, backtest.WARMUP_BARS
    for k, w in enumerate(quiet["windows"]):
        cutoff = S + k
        np.testing.assert_allclose(w, np.log(quiet["ratio"].iloc[cutoff - W:cutoff].values))


def test_calibration_runs_once_on_warmup_bars_only(quiet):
    assert len(quiet["calibrations"]) == 1
    call = quiet["calibrations"][0]
    expected = np.diff(np.log(quiet["ratio"].iloc[:backtest.WARMUP_BARS].values))
    np.testing.assert_allclose(call["log_steps"], expected)
    assert call["window"] == backtest.WINDOW
    assert call["quantile"] == backtest.T_QUANTILE


def test_strategy_gets_the_calibrated_threshold_and_max_age(quiet):
    assert quiet["strategies"] == [
        (quiet["calibrations"][0]["result"], backtest.MAX_AGE, backtest.FALLBACK)
    ]


def test_ratio_series_is_left_untouched(quiet):
    pd.testing.assert_series_equal(quiet["ratio"], quiet["ratio_before"])


def test_result_records_vertices_and_the_luck_only_fire_rate(quiet):
    r = quiet["result"]
    assert len(r.vertices) == len(r.dates)
    assert 0.0 <= r.noise_joint_rate <= 1.0
    assert r.target_signal_rate is None
    assert r.fallback == backtest.FALLBACK


def test_fallback_can_be_disabled_explicitly():
    res = backtest.run_backtest(fallback=None, data=curvy_data())
    assert res.fallback is None
    assert len(res.trades) == res.trade_count


def test_target_signal_rate_calibrates_the_full_rule():
    res = backtest.run_backtest(target_signal_rate=0.02, data=curvy_data())
    assert res.target_signal_rate == 0.02
    assert 0.0 < res.noise_joint_rate < 0.10
    assert len(res.trades) == res.trade_count


def test_baseline_starts_on_the_first_traded_bar(quiet):
    first_baseline = quiet["result"].baseline_values[0]
    assert abs(first_baseline - backtest.INITIAL_BALANCE) / backtest.INITIAL_BALANCE < 0.005


# ---- behaviour when trades actually happen (sine-wave data) ------------------
def test_strategy_trades_on_clear_turning_points(curvy):
    assert curvy["result"].trade_count >= 2


def test_trade_log_matches_trade_count(curvy):
    r = curvy["result"]
    assert len(r.trades) == r.trade_count
    assert set(r.trades["bought"]) <= {"SPY", "GLD"}


def test_trades_alternate_between_the_two_tickers(curvy):
    bought = curvy["result"].trades["bought"].tolist()
    assert all(a != b for a, b in zip(bought, bought[1:]))


def test_fx_rate_is_current_at_the_moment_the_strategy_trades(curvy):
    # It does not matter how many times other code (e.g. valuation.py) sets the rate.
    # What matters: when strategy.update() runs, the portfolio already holds TODAY's rate.
    expected = curvy["history"]["GBPUSD"].iloc[backtest.WARMUP_BARS - 1:].tolist()
    assert curvy["rate_at_update"] == pytest.approx(expected)