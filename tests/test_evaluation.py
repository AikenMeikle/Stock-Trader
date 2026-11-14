import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import smooth_graph.backtest as backtest
import smooth_graph.evaluation as ev
from smooth_graph.backtest import BacktestResult, INITIAL_BALANCE
from smooth_graph.regression import fit_window
from tests.helpers import curvy_data, random_walk_data


@pytest.fixture(scope="module")
def curvy_result():
    return backtest.run_backtest(data=curvy_data())


def tiny_result(strategy, baseline, trades=(), log_ratio=None):
    n = len(strategy)
    dates = pd.bdate_range("2020-01-01", periods=n)
    if log_ratio is None:
        log_ratio = pd.Series(np.zeros(n), index=dates)
    trades = pd.DataFrame(list(trades), columns=["date", "bought"]).set_index("date")
    return BacktestResult(
        dates=dates, strategy_values=np.array(strategy, float), baseline_values=np.array(baseline, float),
        t_stats=np.zeros(n), prices=pd.DataFrame({"SPY": 1.0, "GLD": 1.0, "GBPUSD": 1.0}, index=dates),
        log_ratio=log_ratio, trades=trades, t_threshold=8.0, trade_count=len(trades),
        window_bars=30, t_quantile=95.0, max_age=7, warmup_bars=250)


# ---- statistics -----------------------------------------------------------------
def test_max_drawdown_known_values():
    assert ev.max_drawdown([100, 120, 90, 110]) == pytest.approx(-0.25)
    assert ev.max_drawdown([1, 2, 3]) == 0.0


def test_summarise_known_values():
    r = tiny_result([1000, 1100, 1210, 1331, 1464.1], [1000, 1000, 1000, 1000, 1000])
    table = ev.summarise(r)
    assert table.loc["Final value (GBP)", "Strategy"] == pytest.approx(1464.1)
    assert table.loc["Total return (%)", "Strategy"] == pytest.approx(46.41)
    assert table.loc["Total return (%)", "Baseline"] == pytest.approx(0.0)
    assert table.loc["Max drawdown (%)", "Strategy"] == 0.0
    assert table.loc["Time invested (%)", "Baseline"] == 100.0


def test_hit_rate_counts_trades_followed_by_the_expected_move():
    dates = pd.bdate_range("2020-01-01", periods=40)
    log_ratio = pd.Series(np.concatenate([np.linspace(0, 1, 20), np.linspace(1, 0, 20)]), index=dates)
    # SPY at bar 2 (ratio then rises: hit); SPY at bar 22 (ratio then falls: miss)
    trades = [(dates[2], "SPY"), (dates[22], "SPY")]
    r = tiny_result(np.ones(40) * 1000, np.ones(40) * 1000, trades, log_ratio)
    rate, n = ev.hit_rate(r, horizon=10)
    assert n == 2
    assert rate == pytest.approx(0.5)


# ---- the shifted-timeline control ---------------------------------------------------
def test_replaying_the_strategys_own_holdings_reproduces_its_final_value(curvy_result):
    replay = ev._replay_holdings(curvy_result.prices, ev.holding_array(curvy_result))
    assert replay == pytest.approx(curvy_result.strategy_values[-1], rel=1e-9)


def test_holding_array_is_cash_until_the_first_trade_then_follows_the_trades(curvy_result):
    held = ev.holding_array(curvy_result)
    first = curvy_result.dates.get_loc(curvy_result.trades.index[0])
    assert all(h is None for h in held[:first])
    assert held[first] == curvy_result.trades["bought"].iloc[0]
    assert held[-1] == curvy_result.trades["bought"].iloc[-1]


def test_shifted_control_has_one_value_per_run_and_is_reproducible(curvy_result):
    a = ev.shifted_timeline_control(curvy_result, n_runs=15, seed=3)
    b = ev.shifted_timeline_control(curvy_result, n_runs=15, seed=3)
    assert len(a) == 15
    np.testing.assert_allclose(a, b)


def test_shifted_control_never_contains_the_strategy_itself(curvy_result):
    # shift 0 would just replay the real timeline and inflate the control group
    finals = ev.shifted_timeline_control(curvy_result, n_runs=10_000, seed=0)   # uses every shift
    assert not np.any(np.isclose(finals, curvy_result.strategy_values[-1], rtol=0, atol=1e-9))


def test_shifted_control_with_no_trades_returns_empty():
    r = tiny_result([1000, 1000, 1000], [1000, 1000, 1000])
    assert len(ev.shifted_timeline_control(r)) == 0


# ---- the noise check -----------------------------------------------------------------------
def test_vectorised_window_stats_match_fit_window():
    path = np.cumsum(np.random.default_rng(5).normal(0, 0.01, 120))
    t, vertex, a = ev.window_stats_all(path, 30)
    for k in (0, 17, 60, 90):
        fit = fit_window(path[k:k + 30])
        assert t[k] == pytest.approx(fit.t_stat)
        assert vertex[k] == pytest.approx(fit.vertex)
        assert a[k] == pytest.approx(fit.a)


def test_trade_count_from_the_vectorised_rules_matches_the_real_strategy(curvy_result):
    r = curvy_result
    t, vertex, a = ev.window_stats_all(r.log_ratio.to_numpy(), r.window_bars)
    first = r.warmup_bars - r.window_bars          # index of the first traded window
    assert len(t[first:]) == len(r.dates)
    assert ev._count_trades(t[first:], vertex[first:], a[first:], r.t_threshold, r.max_age) == r.trade_count


def test_noise_trade_counts_one_per_sim_and_reproducible(curvy_result):
    a = ev.noise_trade_counts(curvy_result, n_sims=12, seed=1)
    b = ev.noise_trade_counts(curvy_result, n_sims=12, seed=1)
    assert len(a) == 12
    np.testing.assert_array_equal(a, b)


# ---- sensitivity ----------------------------------------------------------------------
def test_sensitivity_table_has_one_row_per_setting():
    table = ev.sensitivity_table(curvy_data(), windows=(20, 30), quantiles=(95.0,), age_fractions=(0.25,))
    assert len(table) == 2
    assert {"window", "quantile", "max_age", "trades", "strategy_GBP", "baseline_GBP"} <= set(table.columns)


# ---- charts ---------------------------------------------------------------------------
def test_flip_chart_marks_exactly_one_point_per_trade(curvy_result):
    fig = ev.plot_log_ratio_with_flips(curvy_result)
    markers = sum(len(c.get_offsets()) for c in fig.axes[0].collections)
    assert markers == len(curvy_result.trades) > 0


def test_all_charts_build_without_error(curvy_result):
    finals = ev.shifted_timeline_control(curvy_result, n_runs=10)
    assert ev.plot_values(curvy_result) is not None
    assert ev.plot_control(curvy_result, finals) is not None


def test_charts_survive_a_run_with_no_trades():
    r = tiny_result([1000, 1000, 1000, 1000], [1000, 1010, 1020, 1030])
    assert ev.plot_log_ratio_with_flips(r) is not None
    assert ev.plot_control(r, np.array([])) is not None