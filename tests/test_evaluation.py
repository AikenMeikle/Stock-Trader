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


def tiny_result(strategy, baseline, trades=(), log_ratio=None, fallback=None):
    n = len(strategy)
    dates = pd.bdate_range("2020-01-01", periods=n)
    if log_ratio is None:
        log_ratio = pd.Series(np.zeros(n), index=dates)
    trades = pd.DataFrame(list(trades), columns=["date", "bought"]).set_index("date")
    return BacktestResult(
        dates=dates, strategy_values=np.array(strategy, float), baseline_values=np.array(baseline, float),
        t_stats=np.zeros(n), vertices=np.full(n, -1.0),
        prices=pd.DataFrame({"SPY": 1.0, "GLD": 1.0, "GBPUSD": 1.0}, index=dates),
        log_ratio=log_ratio, trades=trades, t_threshold=8.0, noise_joint_rate=0.0,
        trade_count=len(trades), window_bars=30, t_quantile=95.0, max_age=7, warmup_bars=250,
        fallback=fallback)


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


def test_holding_periods_lists_each_contiguous_spell():
    dates = pd.bdate_range("2020-01-01", periods=6)
    r = tiny_result([1000] * 6, [1000] * 6,
                    trades=[(dates[1], "SPY"), (dates[4], "GLD")])
    periods = ev.holding_periods(r)
    assert list(periods["holding"]) == ["CASH", "SPY", "GLD"]
    assert list(periods["bars"]) == [1, 3, 2]
    assert periods["start"].iloc[0] == dates[0]
    assert periods["end"].iloc[0] == dates[0]
    assert periods["start"].iloc[2] == dates[4]
    assert periods["end"].iloc[2] == dates[5]
    assert periods["bars"].sum() == len(dates)


def test_holding_periods_of_an_empty_run_is_an_empty_table():
    periods = ev.holding_periods(tiny_result([], []))
    assert list(periods.columns) == ["start", "end", "holding", "bars"]
    assert len(periods) == 0


# ---- range metrics / IS-OOS -----------------------------------------------------------
def test_range_metrics_measure_from_the_bar_before_the_window():
    dates = pd.bdate_range("2020-01-01", periods=4)
    r = tiny_result([1000, 1100, 1210, 1331], [1000, 990, 980, 970],
                    trades=[(dates[2], "SPY")])
    m = ev.range_metrics(r, r.dates[1], r.dates[2])
    assert (m["start"], m["end"]) == (r.dates[1], r.dates[2])
    assert m["bars"] == 2                                   # dates[0] -> dates[2]
    assert m["strategy_return_pct"] == pytest.approx(21.0)  # 1210 / 1000 - 1
    assert m["baseline_return_pct"] == pytest.approx(-2.0)  # 980 / 1000 - 1
    assert m["strategy_max_drawdown_pct"] == 0.0
    assert m["trades"] == 1                                 # executed inside the window
    assert m["time_invested_pct"] == pytest.approx(50.0)    # bar 1 cash, bar 2 held
    assert m["beats_baseline"] is True


def test_range_metrics_without_trades_reports_no_exposure():
    r = tiny_result([1000, 1000, 1000], [1000, 1000, 1000])
    m = ev.range_metrics(r)
    assert m["trades"] == 0
    assert m["time_invested_pct"] == 0.0
    assert m["strategy_return_pct"] == 0.0
    assert m["beats_baseline"] is False                     # equal is not a win


def test_range_metrics_compares_growth_not_the_inherited_level():
    # The strategy enters the window far ahead, then grows more slowly than the
    # baseline. Final levels still favour the strategy; window growth does not.
    r = tiny_result([1000, 3000, 1200], [100, 100, 200])
    m = ev.range_metrics(r, r.dates[1], r.dates[2])
    assert m["strategy_final"] > m["baseline_final"]
    assert m["strategy_return_pct"] == pytest.approx(20.0)      # 1200 / 1000 - 1
    assert m["baseline_return_pct"] == pytest.approx(100.0)     # 200 / 100 - 1
    assert m["beats_baseline"] is False


def test_range_metrics_rejects_an_empty_window():
    r = tiny_result([1000, 1000], [1000, 1000])
    with pytest.raises(ValueError, match="no traded bars"):
        ev.range_metrics(r, r.dates[0] - pd.Timedelta(days=10), r.dates[0] - pd.Timedelta(days=1))


def test_in_and_out_of_sample_split_at_the_given_date():
    r = tiny_result([1000, 1100, 1210, 1331, 1464], [1000, 1000, 1000, 1000, 1000])
    split = r.dates[2]
    ins = ev.in_sample_metrics(r, split_date=split)
    oos = ev.out_of_sample_metrics(r, split_date=split)
    assert ins["end"] == split
    assert oos["start"] == r.dates[3]
    assert ins["strategy_return_pct"] == pytest.approx(21.0)          # 1210 / 1000 - 1
    assert oos["strategy_return_pct"] == pytest.approx((1464 / 1210 - 1) * 100)


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
    assert ev._count_trades(t[first:], vertex[first:], a[first:], r.t_threshold, r.max_age,
                            fallback=r.fallback) == r.trade_count


def test_vectorised_trade_count_models_the_fallback_exit():
    # One significant SPY entry, then no more signals. With no fallback the
    # position is held forever (one trade); with a fallback the stale vertex
    # rotates it out (two trades).
    t = np.array([10.0, 0.0, 0.0])
    vertex = np.array([-1.0, -20.0, -20.0])
    a = np.array([-0.01, -0.01, -0.01])   # significant cap -> buy GLD
    assert ev._count_trades(t, vertex, a, 8.0, 7, fallback=None) == 1
    assert ev._count_trades(t, vertex, a, 8.0, 7, fallback="SPY") == 2


def test_vectorised_trade_count_rejects_an_unknown_fallback():
    with pytest.raises(ValueError, match="fallback"):
        ev._count_trades(np.zeros(1), np.zeros(1), np.zeros(1), 8.0, 7, fallback="CASH")


def test_noise_trade_counts_one_per_sim_and_reproducible(curvy_result):
    a = ev.noise_trade_counts(curvy_result, n_sims=12, seed=1)
    b = ev.noise_trade_counts(curvy_result, n_sims=12, seed=1)
    assert len(a) == 12
    np.testing.assert_array_equal(a, b)


# ---- signal diagnostics ---------------------------------------------------------------
def test_signal_diagnostics_count_each_gate_separately():
    r = tiny_result([1000] * 4, [1000] * 4)
    r.t_stats = np.array([10.0, 2.0, 10.0, 10.0])    # bars 0, 2, 3 are significant
    r.vertices = np.array([-3.0, -3.0, -20.0, 3.0])  # bar 0 is recent, bar 2 too old, bar 3 in future
    d = ev.signal_diagnostics(r)
    assert d["bars"] == 4
    assert d["significant_bars"] == 3
    assert d["recent_vertex_bars"] == 2              # bars 0 and 1
    assert d["joint_bars"] == 1                      # only bar 0 passes both
    assert d["share_significant_with_recent_vertex_pct"] == pytest.approx(100 / 3)


def test_signal_diagnostics_survives_no_bars():
    r = tiny_result([], [])
    d = ev.signal_diagnostics(r)
    assert np.isnan(d["joint_rate_pct"])


# ---- sensitivity ----------------------------------------------------------------------
def test_sensitivity_table_has_one_row_per_setting():
    table = ev.sensitivity_table(curvy_data(), windows=(20, 30), quantiles=(95.0,), age_fractions=(0.25,))
    assert len(table) == 2
    assert {"window", "quantile", "max_age", "trades", "strategy_GBP", "baseline_GBP"} <= set(table.columns)


def test_sensitivity_table_reports_noise_and_control_columns():
    table = ev.sensitivity_table(curvy_data(), windows=(30,), quantiles=(95.0,), age_fractions=(0.25,),
                                 noise_sims=3, control_runs=3)
    assert {"noise_trades_median", "beats_control_pct"} <= set(table.columns)
    assert 0 <= table.loc[0, "beats_control_pct"] <= 100


def test_default_age_fraction_reproduces_the_default_max_age():
    assert max(1, round(backtest.WINDOW * backtest.MAX_AGE_FRACTION)) == backtest.MAX_AGE


def test_default_age_fractions_put_the_default_first_and_do_not_repeat_it():
    fractions = ev.default_age_fractions()
    assert fractions[0] == backtest.MAX_AGE_FRACTION
    assert len(fractions) == len(set(fractions))
    assert 0.35 in fractions and 0.5 in fractions and 0.65 in fractions


def test_default_sensitivity_cell_is_in_the_grid():
    table = ev.sensitivity_table(curvy_data(), windows=(backtest.WINDOW,),
                                 quantiles=(backtest.T_QUANTILE,),
                                 noise_sims=3, control_runs=3)
    assert len(table) == len(ev.default_age_fractions())
    assert table["max_age"].iloc[0] == backtest.MAX_AGE
    assert table["age_fraction"].iloc[0] == backtest.MAX_AGE_FRACTION


def test_sensitivity_table_has_in_and_out_of_sample_columns():
    table = ev.sensitivity_table(curvy_data(), windows=(30,), quantiles=(95.0,), age_fractions=(0.5,),
                                 noise_sims=2, control_runs=2)
    expected = {"IS_return_pct", "IS_baseline_pct", "IS_trades", "IS_beats_baseline",
                "OOS_return_pct", "OOS_baseline_pct", "OOS_trades", "OOS_beats_baseline",
                "age_fraction"}
    assert expected <= set(table.columns)
    assert table.loc[0, "IS_return_pct"] == table.loc[0, "IS_return_pct"]   # not NaN
    assert np.isnan(table.loc[0, "OOS_return_pct"])    # synthetic data ends before the split


def _grid(values):
    """A 3x3 grid (quantiles x age fractions) for one window."""
    quantiles = np.repeat([90.0, 95.0, 99.0], 3)
    fractions = np.tile([0.35, 0.5, 0.65], 3)
    return pd.DataFrame({
        "window": [30] * 9, "quantile": quantiles, "age_fraction": fractions,
        "IS_return_pct": values, "OOS_return_pct": [0.0] * 9,
    })


def test_neighbour_median_uses_the_3x3_region_and_ignores_other_windows():
    table = _grid([50, 50, 10, 50, 50, 10, 10, 10, 100])
    median = ev.is_neighbour_median(table)
    assert median.iloc[0] == pytest.approx(50)     # corner sees two 50s and two 50s
    assert median.iloc[4] == pytest.approx(50)     # centre sees all nine; median 50
    assert median.iloc[8] == pytest.approx(30)     # corner: [50, 10, 10, 100]

    other = _grid([0, 0, 0, 0, 0, 0, 0, 0, 0])
    other["window"] = 45
    both = pd.concat([table, other], ignore_index=True)
    np.testing.assert_allclose(ev.is_neighbour_median(both).iloc[9:].to_numpy(), 0.0)


def test_best_in_sample_cell_prefers_a_robust_region_over_an_isolated_spike():
    # Raw IS maximum is the bottom-right 100, but its neighbourhood median is 30
    # and the top-left region is consistently 50. Selection must choose the
    # region, not the spike.
    table = _grid([50, 50, 10, 50, 50, 10, 10, 10, 100])
    best = ev.best_in_sample_cell(table)
    assert best["IS_return_pct"] != 100
    assert best["IS_neighbour_median"] == pytest.approx(50)
    assert best["OOS_return_pct"] == 0.0


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