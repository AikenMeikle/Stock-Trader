"""Tests for the Phase 5 validation machinery (synthetic data only)."""
import numpy as np
import pandas as pd
import pytest

import smooth_graph.backtest as backtest
import smooth_graph.evaluation as ev
from smooth_graph.backtest import BacktestResult, FX_FEE_RATE, INITIAL_BALANCE
from tests.helpers import curvy_data, random_walk_data


def tiny_result(strategy, baseline, trades=(), log_ratio=None, prices=None):
    n = len(strategy)
    dates = pd.bdate_range("2020-01-01", periods=n)
    if log_ratio is None:
        log_ratio = pd.Series(np.zeros(n), index=dates)
    if prices is None:
        prices = pd.DataFrame({"SPY": 100.0, "GLD": 100.0, "GBPUSD": 1.0}, index=dates)
    trades = pd.DataFrame(list(trades), columns=["date", "bought"]).set_index("date")
    return BacktestResult(
        dates=dates, strategy_values=np.array(strategy, float),
        baseline_values=np.array(baseline, float),
        t_stats=np.zeros(n), vertices=np.full(n, -1.0), prices=prices,
        log_ratio=log_ratio, trades=trades, t_threshold=8.0, noise_joint_rate=0.0,
        trade_count=len(trades), window_bars=30, t_quantile=95.0, max_age=7,
        warmup_bars=250, fallback=None)


@pytest.fixture(scope="module")
def curvy():
    data = curvy_data()
    return data, backtest.run_backtest(data=data)


# ---- joint block bootstrap -----------------------------------------------------------
def test_block_bootstrap_keeps_shape_index_and_positive_prices():
    _, history = random_walk_data(n=120, seed=4)
    synthetic = ev.block_bootstrap_joint_prices(history, block_length=20, seed=7)
    assert synthetic.index.equals(history.index)
    assert list(synthetic.columns) == ["SPY", "GLD", "GBPUSD"]
    assert (synthetic > 0).all().all()
    # The series start from the real first levels, so the index/value alignment is sane.
    np.testing.assert_allclose(synthetic.iloc[0], history.iloc[0])


def test_block_bootstrap_is_reproducible_and_seed_dependent():
    _, history = random_walk_data(n=120, seed=5)
    a = ev.block_bootstrap_joint_prices(history, block_length=10, seed=11)
    b = ev.block_bootstrap_joint_prices(history, block_length=10, seed=11)
    c = ev.block_bootstrap_joint_prices(history, block_length=10, seed=12)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_block_bootstrap_rejects_bad_input():
    _, history = random_walk_data(n=5, seed=0)
    with pytest.raises(ValueError, match="at least two"):
        ev.block_bootstrap_joint_prices(history.iloc[:1])
    bad = history.copy()
    bad.iloc[0, 0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        ev.block_bootstrap_joint_prices(bad)


# ---- detour attribution --------------------------------------------------------------
def test_detour_rows_cover_every_non_spy_spell_and_reconcile(curvy):
    _, result = curvy
    detours = ev.detour_attribution(result, n_bootstrap=200, seed=1)
    rows = detours["rows"]
    assert len(rows) == detours["count"] > 0
    assert detours["gld_count"] > 0
    assert set(rows["holding"]) <= {"GLD", "CASH"}
    assert rows["bars"].sum() <= len(result.dates)
    assert detours["gld_count"] + detours["cash_count"] == detours["count"]
    assert 0 <= detours["share_time_in_gld_pct"] <= 100
    assert {"strategy_minus_baseline_gbp", "sum_active_log_return",
            "first_bar_log_mismatch", "log_final_ratio", "max_abs_log_error",
            "reconciled", "gbp_effect_by_holding"} <= set(detours["reconciliation"])
    assert detours["reconciliation"]["reconciled"] is True
    assert detours["reconciliation"]["max_abs_log_error"] <= 1e-9
    lo, hi = detours["bootstrap_ci"]
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi
    assert detours["in_sample"]["count"] + detours["out_of_sample"]["count"] == detours["gld_count"]


def test_detour_handles_an_empty_result():
    detours = ev.detour_attribution(tiny_result([], []))
    assert len(detours["rows"]) == 0
    assert detours["count"] == 0
    assert np.isnan(detours["mean_net_excess"])
    assert detours["exact"]["reconciled"] is True


def test_exact_attribution_reconciles_bar_by_bar(curvy):
    _, result = curvy
    exact = ev.exact_attribution(result)
    assert exact["reconciled"] is True
    assert exact["max_abs_log_error"] <= 1e-9
    assert exact["sum_active_log_return"] + exact["first_bar_log_mismatch"] == pytest.approx(
        exact["log_final_ratio"], abs=1e-9
    )
    assert exact["rows"]["active_log_return"].sum() == pytest.approx(
        exact["sum_active_log_return"], abs=1e-12
    )
    assert set(exact["by_holding"]["holding"]) <= {"CASH", "GLD", "SPY"}
    assert exact["by_holding"]["spells"].sum() == len(exact["rows"])


def test_detour_gross_includes_the_exit_swap_bar():
    # GLD is held on dates 1-3 and sold at the close of the swap bar (date 4).
    # Its 10% jump happens ON the swap bar, so a last-held-bar attribution would
    # report zero gross excess. The corrected window must capture the 10%.
    dates = pd.bdate_range("2020-01-01", periods=6)
    prices = pd.DataFrame({
        "SPY": [100.0] * 6,
        "GLD": [100.0, 100.0, 100.0, 100.0, 110.0, 110.0],
        "GBPUSD": [1.0] * 6,
    }, index=dates)
    f = FX_FEE_RATE
    gld_entry_value = 1000.0 / (1.0 + f)
    gld_exit_value = gld_entry_value * 1.10 * (1.0 - f) / (1.0 + f)
    baseline_value = 1000.0 / (1.0 + f)
    r = tiny_result(
        [1000.0, gld_entry_value, gld_entry_value, gld_entry_value,
         gld_exit_value, gld_exit_value],
        [baseline_value] * 6,
        trades=[(dates[1], "GLD"), (dates[4], "SPY")], prices=prices,
    )
    d = ev.detour_attribution(r, n_bootstrap=20, seed=0)
    gld = d["rows"][d["rows"]["holding"] == "GLD"].iloc[0]
    assert gld["exit_date"] == dates[3]                 # last held bar
    assert gld["exit_execution_date"] == dates[4]       # swap / sale bar
    assert gld["bars"] == 3
    assert gld["gross_excess_return"] == pytest.approx(0.10)
    exact_gld = d["exact"]["rows"][d["exact"]["rows"]["holding"] == "GLD"].iloc[0]
    assert exact_gld["return_bars"] == exact_gld["bars"] == 3
    assert exact_gld["active_log_return"] == pytest.approx(
        np.log(1.10) + np.log(1.0 - ev._switch_cost_fraction(FX_FEE_RATE)), abs=1e-12
    )


def test_detour_hand_built_cash_and_gld_spells():
    dates = pd.bdate_range("2020-01-01", periods=6)
    prices = pd.DataFrame({"SPY": 100.0, "GLD": 100.0, "GBPUSD": 1.0}, index=dates)
    r = tiny_result([1000] * 6, [1000] * 6,
                    trades=[(dates[1], "GLD"), (dates[4], "SPY")], prices=prices)
    d = ev.detour_attribution(r, n_bootstrap=50, seed=0)
    assert list(d["rows"]["holding"]) == ["CASH", "GLD"]
    assert d["rows"].loc[0, "bars"] == 1
    assert d["rows"].loc[1, "bars"] == 3
    assert d["share_time_in_gld_pct"] == pytest.approx(50.0)
    # Flat prices mean the only non-zero effect is the modelled switching cost.
    assert d["rows"].loc[1, "gross_excess_return"] == pytest.approx(0.0)
    assert d["rows"].loc[1, "net_excess_return"] < 0


def test_bootstrap_ci_brackets_the_mean():
    values = np.array([0.01, 0.02, 0.03, 0.04])
    lo, hi = ev._bootstrap_mean_ci(values, n_sims=500, seed=3)
    assert lo < values.mean() < hi


# ---- full-pipeline Monte Carlo -------------------------------------------------------
def test_full_pipeline_monte_carlo_reports_null_and_is_reproducible(curvy):
    data, result = curvy
    a = ev.full_pipeline_monte_carlo(result, data, n_sims=3, block_length=20,
                                     seed=2, n_jobs=1)
    b = ev.full_pipeline_monte_carlo(result, data, n_sims=3, block_length=20,
                                     seed=2, n_jobs=1)
    assert len(a["table"]) == 3
    assert {"annualised_excess_return", "trades", "strategy_max_drawdown"} <= set(a["table"].columns)
    assert 0.0 <= a["percentile"] <= 100.0
    assert 0.0 < a["p_value"] <= 1.0
    assert a["null_trades_p5"] <= a["null_trades_median"] <= a["null_trades_p95"]
    assert isinstance(a["passed"], bool)
    np.testing.assert_allclose(a["table"].to_numpy(), b["table"].to_numpy())


# ---- tracking error ------------------------------------------------------------------
def test_tracking_error_known_values():
    # Active daily returns are 1%, 2%, 1%; the baseline is flat.
    r = tiny_result([1000, 1010, 1030.2, 1040.502], [1000, 1000, 1000, 1000])
    t = ev.tracking_error_stats(r)
    mean_daily = np.mean([0.01, 0.02, 0.01])
    std_daily = np.std([0.01, 0.02, 0.01], ddof=1)
    ir = (mean_daily * ev.BARS_PER_YEAR) / (std_daily * np.sqrt(ev.BARS_PER_YEAR))
    assert t["bars"] == 3
    assert t["annualised_active_return_pct"] == pytest.approx(100 * mean_daily * ev.BARS_PER_YEAR)
    assert t["annualised_tracking_error_pct"] == pytest.approx(100 * std_daily * np.sqrt(ev.BARS_PER_YEAR))
    assert t["information_ratio"] == pytest.approx(ir)
    assert t["years_to_detect_2se"] == pytest.approx((2 / ir) ** 2)
    assert t["detectable_within_history"] is True


def test_tracking_error_without_a_positive_edge_is_infinite():
    r = tiny_result([1000, 990, 980], [1000, 1000, 1000])
    t = ev.tracking_error_stats(r)
    assert t["information_ratio"] < 0
    assert np.isinf(t["years_to_detect_2se"])
    assert t["detectable_within_history"] is False


# ---- friction ------------------------------------------------------------------------
def test_friction_sensitivity_reports_lead_and_break_even():
    data = random_walk_data(n=500, seed=2)
    f = ev.friction_sensitivity(data, extras=(0.0, 0.001), result=backtest.run_backtest(data=data))
    assert len(f["table"]) == 2
    assert {"extra_friction_pct_points", "lead_gbp", "beats_baseline"} <= set(f["table"].columns)
    assert f["table"].loc[0, "extra_friction_pct_points"] == 0.0
    assert f["table"].loc[1, "extra_friction_pct_points"] == pytest.approx(0.1)
    assert f["break_even_extra_friction"] is None or np.isfinite(f["break_even_extra_friction"])


def test_default_fee_is_unchanged():
    assert FX_FEE_RATE == pytest.approx(0.0015)


def test_fx_fee_argument_defaults_to_the_strategy_fee():
    data = curvy_data()
    default = backtest.run_backtest(data=data)
    explicit = backtest.run_backtest(data=data, fx_fee_rate=FX_FEE_RATE)
    np.testing.assert_allclose(default.strategy_values, explicit.strategy_values)
    np.testing.assert_allclose(default.baseline_values, explicit.baseline_values)


# ---- clean holdout -------------------------------------------------------------------
def test_forward_metrics_says_no_clean_data_before_the_holdout():
    r = tiny_result([1000, 1010, 1020], [1000, 1005, 1010])
    fwd = ev.forward_metrics(r)
    assert fwd["status"] == "no clean data yet"
    assert fwd["available"] is False
    assert fwd["bars"] == 0


def test_forward_metrics_uses_range_metrics_once_bars_exist():
    r = tiny_result([1000, 1100, 1210, 1331], [1000, 1000, 1000, 1000])
    fwd = ev.forward_metrics(r, start=r.dates[2])
    assert fwd["status"] == "ok" and fwd["available"] is True
    assert fwd["bars"] > 0
    assert fwd["strategy_return_pct"] == pytest.approx(21.0)


# ---- neighbourhood -------------------------------------------------------------------
def _neighbour_table(full_true=6, oos_true=9):
    rows = []
    quantiles = [90.0, 95.0, 99.0]
    # Deliberately put the default fraction first, as sensitivity_table does.
    fractions = [0.5, 0.35, 0.65]
    n = 0
    for q in quantiles:
        for frac in fractions:
            rows.append({
                "window": 30, "quantile": q, "age_fraction": frac,
                "beats_baseline": n < full_true,
                "OOS_beats_baseline": n < oos_true,
            })
            n += 1
    return pd.DataFrame(rows)


def test_neighbourhood_check_requires_six_wins_in_both_samples():
    passed = ev.neighbourhood_check(_neighbour_table(6, 6))
    assert passed["found"] is True
    assert passed["cells"] == 9
    assert passed["passed"] is True

    failed_full = ev.neighbourhood_check(_neighbour_table(5, 9))
    assert failed_full["passed"] is False
    failed_oos = ev.neighbourhood_check(_neighbour_table(9, 5))
    assert failed_oos["passed"] is False


def test_neighbourhood_check_fails_when_the_default_cell_is_missing():
    table = _neighbour_table()
    table = table[~np.isclose(table["age_fraction"], 0.5)]
    result = ev.neighbourhood_check(table)
    assert result["found"] is False and result["passed"] is False


# ---- overall verdict -----------------------------------------------------------------
def test_validation_verdict_requires_all_checks():
    table = _neighbour_table(9, 9)
    doubled = tiny_result([1200, 1300], [1000, 1000])
    detours = {"bootstrap_ci": (0.01, 0.02), "mean_net_excess": 0.015}
    mc_pass = {"passed": True, "real_annualised_excess_pct": 2.0,
               "null_percentile_pct": 1.0, "p_value": 0.01}
    mc_fail = dict(mc_pass, passed=False, p_value=0.5)

    good = ev.validation_verdict(tiny_result([1000, 1000], [1000, 1000]), None,
                                 table=table, mc=mc_pass, detours=detours,
                                 doubled_result=doubled)
    assert good["validated"] is True
    assert [c["number"] for c in good["checks"]] == [1, 2, 3, 4]

    bad = ev.validation_verdict(tiny_result([1000, 1000], [1000, 1000]), None,
                                table=table, mc=mc_fail, detours=detours,
                                doubled_result=doubled)
    assert bad["validated"] is False
    assert bad["checks"][0]["passed"] is False
