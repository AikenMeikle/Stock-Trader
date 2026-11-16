"""Tests for the trailing one-year return distribution (synthetic, no network)."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import smooth_graph.evaluation as ev
from smooth_graph.backtest import BacktestResult


def make_result(strategy, baseline, dates=None):
    strategy = np.asarray(strategy, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    n = len(strategy)
    if dates is None:
        dates = pd.bdate_range("2010-01-01", periods=n)
    dates = pd.DatetimeIndex(dates)
    prices = pd.DataFrame({"SPY": 1.0, "GLD": 1.0, "GBPUSD": 1.0}, index=dates)
    trades = pd.DataFrame({"bought": []}, index=pd.DatetimeIndex([]))
    return BacktestResult(
        dates=dates, strategy_values=strategy, baseline_values=baseline,
        t_stats=np.zeros(n), vertices=np.full(n, -1.0), prices=prices,
        log_ratio=pd.Series(np.zeros(n), index=dates), trades=trades,
        t_threshold=8.0, noise_joint_rate=0.0, trade_count=0,
        window_bars=30, t_quantile=95.0, max_age=15, warmup_bars=250,
        fallback="SPY",
    )


def test_constant_growth_gives_the_same_trailing_year_every_time():
    n = 300
    g = 0.001
    strategy = 1000.0 * (1.0 + g) ** np.arange(n)
    baseline = 1000.0 * (1.0 + 0.0005) ** np.arange(n)
    result = make_result(strategy, baseline)

    returns = ev.trailing_annual_returns(strategy)
    assert len(returns) == n - ev.BARS_PER_YEAR
    np.testing.assert_allclose(returns, (1.0 + g) ** ev.BARS_PER_YEAR - 1.0)

    stats = ev.annual_return_statistics(result)
    expected = (1.0 + g) ** ev.BARS_PER_YEAR - 1.0
    assert stats.loc["Strategy", "mean_pct"] == pytest.approx(100.0 * expected)
    assert stats.loc["Strategy", "sd_pct"] == pytest.approx(0.0, abs=1e-12)
    assert stats.loc["Active", "mean_pct"] == pytest.approx(
        100.0 * (expected - ((1.0 + 0.0005) ** ev.BARS_PER_YEAR - 1.0))
    )


def test_histogram_counts_sum_to_the_number_of_valid_windows():
    n = 300
    strategy = 1000.0 * (1.0 + 0.001) ** np.arange(n)
    baseline = 1000.0 * (1.0 + 0.0005) ** np.arange(n)
    result = make_result(strategy, baseline)
    returns = ev.trailing_annual_returns(strategy)

    counts, _ = np.histogram(returns, bins=ev._safe_bin_edges(returns, bins=40))
    assert counts.sum() == len(returns) == n - ev.BARS_PER_YEAR

    fig = ev.plot_annual_returns(result)
    panel1_height = sum(patch.get_height() for patch in fig.axes[0].patches)
    panel2_height = sum(patch.get_height() for patch in fig.axes[1].patches)
    assert panel1_height == pytest.approx(2 * len(returns))
    assert panel2_height == pytest.approx(len(returns))


def test_normal_curve_integrates_to_n_and_skips_zero_variance():
    edges = np.linspace(-0.5, 0.5, 101)
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = edges[1] - edges[0]
    curve = ev._normal_curve(centers, width, mean=0.0, sd=0.1, n=500)
    assert curve is not None
    assert curve.sum() == pytest.approx(500.0, rel=1e-6)
    assert ev._normal_curve(centers, width, mean=0.0, sd=0.0, n=500) is None


def test_calendar_year_table_matches_a_hand_calculation_and_flags_partials():
    dates = pd.bdate_range("2020-12-30", "2022-01-05")
    strategy = 1000.0 + 10.0 * np.arange(len(dates))
    baseline = 2000.0 + 5.0 * np.arange(len(dates))
    result = make_result(strategy, baseline, dates=dates)

    table = ev.calendar_year_returns(result)
    years = np.asarray(dates.year)
    last_2020 = int(np.flatnonzero(years == 2020)[-1])
    last_2021 = int(np.flatnonzero(years == 2021)[-1])
    row_2021 = table[table["year"] == 2021].iloc[0]
    expected_strategy = 100.0 * (strategy[last_2021] / strategy[last_2020] - 1.0)
    expected_baseline = 100.0 * (baseline[last_2021] / baseline[last_2020] - 1.0)
    assert row_2021["strategy_pct"] == pytest.approx(expected_strategy)
    assert row_2021["baseline_pct"] == pytest.approx(expected_baseline)
    assert row_2021["active_pp"] == pytest.approx(expected_strategy - expected_baseline)
    assert bool(row_2021["partial"]) is False

    assert bool(table[table["year"] == 2020].iloc[0]["partial"]) is True
    assert bool(table[table["year"] == 2022].iloc[0]["partial"]) is True


def test_fewer_than_a_year_gives_a_clear_message():
    n = 100
    result = make_result(np.linspace(1000, 1100, n), np.linspace(1000, 1050, n))
    assert len(ev.trailing_annual_returns(result.strategy_values)) == 0
    stats = ev.annual_return_statistics(result)
    assert int(stats["n"].max()) == 0
    fig = ev.plot_annual_returns(result)
    text = " ".join(t.get_text() for t in fig.texts)
    text += " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "fewer" in text.lower() or "cannot compute" in text.lower()


def test_zero_variance_skips_the_normal_curve_with_a_message():
    n = 300
    result = make_result(np.full(n, 1000.0), np.full(n, 1000.0))
    stats = ev.annual_return_statistics(result)
    assert stats.loc["Strategy", "sd_pct"] == pytest.approx(0.0)
    fig = ev.plot_annual_returns(result)
    text = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "zero variance" in text.lower()
