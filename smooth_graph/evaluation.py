"""Evaluation of a BacktestResult: statistics, controls and charts."""
import math
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from . import backtest
from .backtest import FX_FEE_RATE, INITIAL_BALANCE, BacktestResult, run_backtest
from .portfolio import Portfolio
from .regression import window_stats_all

BARS_PER_YEAR = 252
GREEN, GOLD = "tab:green", "goldenrod"
OVERLAP_WARNING = (
    "Windows overlap: about 5,000 daily observations are roughly 21 independent "
    "years. Do not read the smoothness as precision."
)
# Trade-level statistics are meaningless on a handful of trades; the report
# suppresses the hit rate below this many.
MIN_TRADES_FOR_STATS = 10
# Pre-registered in-sample / out-of-sample boundary. It is deliberately fixed
# before looking at any result: the first ~9 years are for choosing settings,
# the remaining ~12 are evaluated once. Do not move it after reading a grid.
SPLIT_DATE = pd.Timestamp("2014-12-31")

# ---------------------------------------------------------------------------
# Phase 5 validation gates. PRE-REGISTERED before any real-data validation run
# and recorded in DIAGNOSIS.md. Do not edit these values after seeing results.
# A signal is VALIDATED only when all four checks below pass.
# ---------------------------------------------------------------------------
# Data before this date informed Phases 1-3 (including the out-of-sample
# window), so only bars on/after it count as clean evidence.
CLEAN_HOLDOUT_START = pd.Timestamp("2026-11-14")

# Check 1: full-pipeline Monte Carlo null.
MC_SIMS = 300
MC_BLOCK_BARS = 20          # joint block bootstrap of daily log returns
MC_SEED = 0
MC_PERCENTILE = 95.0        # real annualised excess must meet/beat this percentile

# Check 2: detour attribution bootstrap.
DETOUR_BOOTSTRAP_SIMS = 10_000
DETOUR_BOOTSTRAP_SEED = 0
DETOUR_CI_PCT = 95.0

# Check 3: friction. The strategy must still beat the baseline when the FX fee
# is this multiple of the unchanged default.
FRICTION_FEE_MULTIPLE = 2.0
# Extra friction per swap leg, in decimal (0.0005 = 0.05 percentage points).
FRICTION_EXTRAS = (0.0, 0.0005, 0.0010, 0.0020, 0.0030)

# Check 4: neighbourhood. The default cell must beat the baseline in at least
# this many of its 3x3 neighbours, in BOTH the full sample and OOS.
NEIGHBOURHOOD_MIN_BEATS = 6
NEIGHBOURHOOD_TOTAL_CELLS = 9


# ---------------------------------------------------------------- statistics
def holding_timeline(result: BacktestResult) -> pd.Series:
    """What the strategy held on each traded bar: "SPY", "GLD" or NaN (cash)."""
    held = pd.Series(index=result.dates, dtype=object)
    for date, bought in result.trades["bought"].items():
        held.loc[date:] = bought
    return held


def max_drawdown(values) -> float:
    """Largest peak-to-trough fall, as a negative fraction (-0.25 = -25%)."""
    values = np.asarray(values, dtype=float)
    return float((values / np.maximum.accumulate(values) - 1).min())


def holding_periods(result: BacktestResult) -> pd.DataFrame:
    """One row per contiguous holding spell.

    Columns: `start`, `end` (inclusive dates), `holding` ("SPY", "GLD" or
    "CASH"), `bars`. The longest non-fallback spell is the staleness measure that
    motivated the Phase 3 exit rule.
    """
    held = holding_timeline(result)
    if len(held) == 0:
        return pd.DataFrame(columns=["start", "end", "holding", "bars"])

    def same(a, b) -> bool:
        return (pd.isna(a) and pd.isna(b)) or a == b

    values = held.to_numpy()
    rows = []
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or not same(values[i], values[start]):
            rows.append({
                "start": held.index[start], "end": held.index[i - 1],
                "holding": "CASH" if pd.isna(values[start]) else values[start],
                "bars": i - start,
            })
            start = i
    return pd.DataFrame(rows)


def _annualised(values) -> float:
    years = (len(values) - 1) / BARS_PER_YEAR
    return float((values[-1] / INITIAL_BALANCE) ** (1 / years) - 1) if years > 0 else float("nan")


def summarise(result: BacktestResult) -> pd.DataFrame:
    """Side-by-side statistics for the strategy and the baseline."""
    s, b = result.strategy_values, result.baseline_values
    mid = len(s) // 2
    invested = holding_timeline(result).notna().mean() * 100
    rows = {
        "Final value (GBP)": (s[-1], b[-1]),
        "Total return (%)": (100 * (s[-1] / INITIAL_BALANCE - 1), 100 * (b[-1] / INITIAL_BALANCE - 1)),
        "Annualised return (%)": (100 * _annualised(s), 100 * _annualised(b)),
        "Max drawdown (%)": (100 * max_drawdown(s), 100 * max_drawdown(b)),
        "Return, first half (%)": (100 * (s[mid] / s[0] - 1), 100 * (b[mid] / b[0] - 1)),
        "Return, second half (%)": (100 * (s[-1] / s[mid] - 1), 100 * (b[-1] / b[mid] - 1)),
        "Trades": (len(result.trades), 0),
        "Time invested (%)": (invested, 100.0),
    }
    return pd.DataFrame(rows, index=["Strategy", "Baseline"]).T


def trailing_annual_returns(values, bars_per_year: int = BARS_PER_YEAR) -> np.ndarray:
    """Trailing one-year simple return on every bar with a full year behind it.

    `values[t] / values[t - bars_per_year] - 1`, for t >= bars_per_year. Returns
    an empty array when the history is shorter than one year plus one bar.
    """
    if bars_per_year < 1:
        raise ValueError("bars_per_year must be positive")
    values = np.asarray(values, dtype=float)
    if len(values) <= bars_per_year:
        return np.array([], dtype=float)
    return values[bars_per_year:] / values[:-bars_per_year] - 1.0


def _distribution_stats(values) -> dict:
    """Mean/median/SD/skew/excess-kurtosis/percentiles for one return series."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "n": 0, "mean_pct": float("nan"), "median_pct": float("nan"),
            "sd_pct": float("nan"), "skewness": float("nan"),
            "excess_kurtosis": float("nan"), "p5_pct": float("nan"),
            "worst_pct": float("nan"), "best_pct": float("nan"),
            "share_negative_pct": float("nan"),
        }
    series = pd.Series(values)
    return {
        "n": int(len(values)),
        "mean_pct": 100.0 * float(np.mean(values)),
        "median_pct": 100.0 * float(np.median(values)),
        "sd_pct": 100.0 * float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        "skewness": float(series.skew()) if len(values) >= 3 else float("nan"),
        "excess_kurtosis": float(series.kurt()) if len(values) >= 4 else float("nan"),
        "p5_pct": 100.0 * float(np.percentile(values, 5)),
        "worst_pct": 100.0 * float(np.min(values)),
        "best_pct": 100.0 * float(np.max(values)),
        "share_negative_pct": 100.0 * float((values < 0).mean()),
    }


def annual_return_statistics(result: BacktestResult) -> pd.DataFrame:
    """Distribution table for trailing one-year strategy, baseline and active returns.

    Active return is `strategy_pct - baseline_pct` (percentage points). The
    windows overlap heavily, so these are not independent annual observations;
    the report prints that caveat next to the table and chart.
    """
    s = trailing_annual_returns(result.strategy_values)
    b = trailing_annual_returns(result.baseline_values)
    active = s - b if len(s) == len(b) else np.array([], dtype=float)
    table = pd.DataFrame({
        "Strategy": _distribution_stats(s),
        "Baseline": _distribution_stats(b),
        "Active": _distribution_stats(active),
    }).T
    return table


def calendar_year_returns(result: BacktestResult) -> pd.DataFrame:
    """Calendar-year strategy/baseline returns and their difference in pp.

    A full year uses the last bar of the previous year as its base. The first
    year is partial when trading starts after 1 January; the last year is partial
    when trading ends before 31 December. Partial years are flagged and must be
    excluded from summary statistics.
    """
    columns = ["year", "strategy_pct", "baseline_pct", "active_pp", "partial"]
    dates = result.dates
    if len(dates) == 0:
        return pd.DataFrame(columns=columns)

    s = np.asarray(result.strategy_values, dtype=float)
    b = np.asarray(result.baseline_values, dtype=float)
    years = np.asarray(dates.year)
    unique_years = sorted(set(int(y) for y in years))
    first_year, last_year = unique_years[0], unique_years[-1]
    rows = []
    for year in unique_years:
        positions = np.flatnonzero(years == year)
        first_pos = int(positions[0])
        last_pos = int(positions[-1])
        base_pos = first_pos - 1 if first_pos > 0 else first_pos
        if last_pos > base_pos:
            strategy_return = s[last_pos] / s[base_pos] - 1.0
            baseline_return = b[last_pos] / b[base_pos] - 1.0
        else:
            strategy_return = 0.0
            baseline_return = 0.0
        partial = (
            (year == first_year and pd.Timestamp(dates[first_pos]) > pd.Timestamp(year, 1, 1))
            or (year == last_year and pd.Timestamp(dates[last_pos]) < pd.Timestamp(year, 12, 31))
        )
        rows.append({
            "year": year,
            "strategy_pct": 100.0 * float(strategy_return),
            "baseline_pct": 100.0 * float(baseline_return),
            "active_pp": 100.0 * float(strategy_return - baseline_return),
            "partial": bool(partial),
        })
    return pd.DataFrame(rows, columns=columns)


def _safe_bin_edges(values, bins: int = 60) -> np.ndarray:
    """Histogram edges that stay finite when all values are identical."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.linspace(-1.0, 1.0, int(bins) + 1)
    low, high = float(np.min(values)), float(np.max(values))
    if low == high:
        pad = 0.5 if low == 0 else max(abs(low) * 0.05, 0.5)
        low, high = low - pad, high + pad
    return np.linspace(low, high, int(bins) + 1)


def _normal_curve(bin_centers, bin_width: float, mean: float, sd: float, n: int):
    """Normal density scaled to histogram counts: pdf(x) * n * bin_width.

    Returns None when the sample has zero (or non-finite) variance, so callers
    can skip the curve with an explicit message.
    """
    if not np.isfinite(sd) or sd <= 0 or n <= 0:
        return None
    centers = np.asarray(bin_centers, dtype=float)
    z = (centers - mean) / sd
    pdf = np.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))
    return pdf * float(n) * float(bin_width)


def range_metrics(result: BacktestResult, start=None, end=None) -> dict:
    """Strategy vs baseline statistics over a date window.

    Returns are measured from the last traded bar *before* `start` (or the first
    bar of the window when there is none), so the window's first bar already
    carries a day of return. Trades are those executed inside the window.

    This is the building block for the pre-registered in-sample / out-of-sample
    split: choose settings on `range_metrics(result, end=SPLIT_DATE)` and report
    `range_metrics(result, start=SPLIT_DATE + 1 day)` exactly once.
    """
    dates = result.dates
    if len(dates) == 0:
        raise ValueError("cannot measure an empty backtest")
    start = dates[0] if start is None else pd.Timestamp(start)
    end = dates[-1] if end is None else pd.Timestamp(end)

    first = int(dates.searchsorted(start, side="left"))
    last = int(dates.searchsorted(end, side="right")) - 1
    if first > last:
        raise ValueError(f"no traded bars between {start.date()} and {end.date()}")

    base = max(first - 1, 0)
    bars = last - base

    def growth(values):
        return float(values[last] / values[base] - 1)

    def annualised(values):
        return float((1 + growth(values)) ** (BARS_PER_YEAR / bars) - 1) if bars > 0 else float("nan")

    def drawdown(values):
        window = values[base:last + 1]
        return float((window / np.maximum.accumulate(window) - 1).min())

    s, b = result.strategy_values, result.baseline_values
    traded = result.trades.index
    n_trades = int(((traded > dates[base]) & (traded <= dates[last])).sum())
    invested = holding_timeline(result).iloc[first:last + 1].notna().mean()

    return {
        "start": dates[first],
        "end": dates[last],
        "bars": int(bars),
        "strategy_return_pct": 100 * growth(s),
        "baseline_return_pct": 100 * growth(b),
        "excess_return_pct": 100 * (growth(s) - growth(b)),
        "strategy_annualised_pct": 100 * annualised(s),
        "baseline_annualised_pct": 100 * annualised(b),
        "strategy_max_drawdown_pct": 100 * drawdown(s),
        "baseline_max_drawdown_pct": 100 * drawdown(b),
        "strategy_final": float(s[last]),
        "baseline_final": float(b[last]),
        "trades": n_trades,
        "time_invested_pct": 100 * float(invested),
        # Compare growth over THIS window, not final levels: the strategy may
        # enter the window with a large lead carried over from earlier.
        "beats_baseline": bool(growth(s) > growth(b)),
    }


def in_sample_metrics(result: BacktestResult, split_date=SPLIT_DATE) -> dict:
    """Statistics from the start of trading up to the pre-registered split."""
    return range_metrics(result, result.dates[0], split_date)


def out_of_sample_metrics(result: BacktestResult, split_date=SPLIT_DATE) -> dict:
    """Statistics from the first bar after the pre-registered split to the end."""
    start = max(pd.Timestamp(split_date) + pd.Timedelta(days=1), result.dates[0])
    return range_metrics(result, start, result.dates[-1])


def hit_rate(result: BacktestResult, horizon: int = 15):
    """After each trade, did the log ratio move the way the trade expected over the
    next `horizon` bars? (SPY expects it to rise, GLD expects it to fall.)
    Returns (fraction_correct, number_of_trades_measured)."""
    log_ratio = result.log_ratio
    pos = log_ratio.index.get_indexer(result.trades.index)
    bought = result.trades["bought"].to_numpy()
    ok = pos + horizon < len(log_ratio)
    pos, bought = pos[ok], bought[ok]
    if len(pos) == 0:
        return float("nan"), 0
    values = log_ratio.to_numpy()
    change = values[pos + horizon] - values[pos]
    return float(((change > 0) == (bought == "SPY")).mean()), int(len(pos))


# --------------------------------------------- shifted-timeline control
def holding_array(result: BacktestResult) -> list:
    """What the strategy held on each traded bar: "SPY", "GLD" or None (cash)."""
    return [None if pd.isna(x) else x for x in holding_timeline(result)]


def _replay_holdings(prices: pd.DataFrame, holdings: list) -> float:
    """Replay a per-bar holding list (None = cash) with the SAME costs as the strategy.
    Returns the final GBP value."""
    portfolio = Portfolio(fx_fee_rate=FX_FEE_RATE, initial_balance=INITIAL_BALANCE)
    spy = prices["SPY"].to_numpy()
    gld = prices["GLD"].to_numpy()
    fx = prices["GBPUSD"].to_numpy()
    for i, target in enumerate(holdings):
        portfolio.update_conversion(float(fx[i]))
        current = portfolio.get_current_stock()
        if target != current:
            px = {"SPY": float(spy[i]), "GLD": float(gld[i])}
            if current is not None:
                portfolio.sell(px[current], current)
            if target is not None:
                portfolio.buy(px[target], target)
    held = portfolio.get_current_stock()
    if held is None:
        return portfolio.get_balance()
    return portfolio.get_portfolio_value(float(prices[held].iloc[-1]), held)


def shifted_timeline_control(result: BacktestResult, n_runs: int = 300, seed: int = 0) -> np.ndarray:
    """Final values of the SAME holding timeline rotated in time by a random amount.

    Each run spends exactly as long in cash, SPY and GLD as the real strategy did, with
    the same switches, but at the wrong times. If the strategy cannot beat most of
    these, WHEN it held each asset added nothing. (Unlike a control with random switch
    dates, this one cannot be handicapped by sitting in cash for a different length of time.)
    """
    if len(result.trades) == 0:
        return np.array([])
    holdings = holding_array(result)
    n = len(holdings)
    rng = np.random.default_rng(seed)
    shifts = rng.choice(np.arange(1, n), size=min(n_runs, n - 1), replace=False)
    return np.array([
        _replay_holdings(result.prices, holdings[-int(s):] + holdings[:-int(s)]) for s in shifts
    ])


# ------------------------------------------------ noise check: signal frequency
def _count_trades(t, vertex, a, threshold: float, max_age: int,
                  fallback: str | None = None) -> int:
    """How many trades the strategy rules would make on these per-bar statistics
    (cash start, one-bar delay, never queue what is already held).

    `fallback` models the Phase 3 stale-signal exit: if a position is held and
    the fitted vertex is outside `[-max_age, 0]`, rotate into the fallback
    asset. `None` disables the exit rule.
    """
    fire = (np.abs(t) >= threshold) & (vertex >= -max_age) & (vertex <= 0)
    targets = np.where(fire, np.where(a > 0, 1, -1), 0)   # 1 = SPY, -1 = GLD
    if fallback == "SPY":
        fallback_target = 1
    elif fallback == "GLD":
        fallback_target = -1
    elif fallback is None:
        fallback_target = 0
    else:
        raise ValueError(f"fallback must be None, 'SPY' or 'GLD', got {fallback!r}")

    held, pending, trades = 0, 0, 0
    for target, v in zip(targets, vertex):
        if pending:
            held, pending = pending, 0
            trades += 1
        if target != 0:
            if target != held:
                pending = target
        elif held != 0 and fallback_target != 0 and held != fallback_target:
            if not (-max_age <= v <= 0):
                pending = fallback_target
    return trades


def noise_trade_counts(result: BacktestResult, n_sims: int = 200, seed: int = 0) -> np.ndarray:
    """Number of trades the SAME rules make on `n_sims` histories built by shuffling
    (resampling) the real daily changes. This is what pure luck produces."""
    steps = np.diff(result.log_ratio.to_numpy())
    n_points = len(result.dates) + result.window_bars - 1
    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(n_sims):
        path = np.cumsum(rng.choice(steps, size=n_points, replace=True))
        t, vertex, a = window_stats_all(path, result.window_bars)
        counts.append(_count_trades(t, vertex, a, result.t_threshold, result.max_age,
                                    fallback=result.fallback))
    return np.array(counts)


# ------------------------------------------------------- signal diagnostics
def signal_diagnostics(result: BacktestResult) -> dict[str, float]:
    """How often each gate fires, and how often luck alone would fire.

    These are the numbers that explain a low trade count. In particular:
    `share_significant_with_recent_vertex_pct` is usually tiny because a large
    curvature t-statistic selects vertices near the middle of the window, while
    `median_vertex_when_significant` shows where they actually sit.
    """
    t = np.asarray(result.t_stats, dtype=float)
    vertex = np.asarray(result.vertices, dtype=float)
    n = len(t)
    if n == 0:
        return {k: float("nan") for k in (
            "bars", "significant_bars", "significant_rate_pct",
            "recent_vertex_bars", "recent_vertex_rate_pct",
            "joint_bars", "joint_rate_pct", "noise_joint_rate_pct",
            "noise_joint_events", "median_vertex_when_significant",
            "share_significant_with_recent_vertex_pct",
        )}

    significant = np.abs(t) >= result.t_threshold
    recent = (vertex >= -result.max_age) & (vertex <= 0)
    joint = significant & recent
    significant_vertices = vertex[significant & np.isfinite(vertex)]

    return {
        "bars": float(n),
        "significant_bars": float(significant.sum()),
        "significant_rate_pct": float(100 * significant.mean()),
        "recent_vertex_bars": float(recent.sum()),
        "recent_vertex_rate_pct": float(100 * recent.mean()),
        "joint_bars": float(joint.sum()),
        "joint_rate_pct": float(100 * joint.mean()),
        "noise_joint_rate_pct": float(100 * result.noise_joint_rate),
        "noise_joint_events": float(result.noise_joint_rate * n),
        "median_vertex_when_significant": (
            float(np.median(significant_vertices)) if len(significant_vertices) else float("nan")
        ),
        "share_significant_with_recent_vertex_pct": (
            float(100 * joint.sum() / significant.sum()) if significant.sum() else float("nan")
        ),
    }


# ------------------------------------------------------ sensitivity grid
def default_age_fractions() -> tuple[float, ...]:
    """Age fractions the grid always includes, with the default first.

    Deduplicated so changing the default to a value already in the grid (e.g.
    0.5) does not produce a repeated row.
    """
    fractions = (backtest.MAX_AGE_FRACTION, 0.35, 0.5, 0.65)
    unique = []
    for frac in fractions:
        if not any(abs(frac - seen) < 1e-9 for seen in unique):
            unique.append(frac)
    return tuple(unique)


def is_neighbour_median(table: pd.DataFrame, column: str = "IS_return_pct") -> pd.Series:
    """Median `column` over each cell's 3x3 neighbourhood in the grid.

    The neighbourhood spans +/- one quantile and +/- one age fraction inside the
    same window. A cell with a good value but bad neighbours is a spike, not a
    finding; the median is the robust version of "is the region good?".

    Returns a Series aligned with `table.index` (NaN where a cell is missing).
    """
    out = pd.Series(np.nan, index=table.index, dtype=float)
    for _window, group in table.groupby("window", sort=False):
        pivot = group.pivot(index="quantile", columns="age_fraction", values=column)
        # Sort the axes by VALUE, not by row order. The grid puts the default
        # age fraction first, so without sorting the default cell would sit at
        # an edge and see only a 2x3 block instead of its full 3x3 neighbourhood.
        pivot = pivot.sort_index().sort_index(axis=1)
        quantiles = list(pivot.index)
        fractions = list(pivot.columns)
        for qi, q in enumerate(quantiles):
            for fi, frac in enumerate(fractions):
                block = pivot.iloc[max(0, qi - 1):qi + 2,
                                   max(0, fi - 1):fi + 2].to_numpy().ravel()
                value = float(np.nanmedian(block))
                rows = group.index[
                    (group["quantile"] == q) & np.isclose(group["age_fraction"], frac)
                ]
                out.loc[rows] = value
    return out


def best_in_sample_cell(table: pd.DataFrame, column: str = "IS_return_pct") -> pd.Series:
    """Pick a cell using in-sample data only, by neighbour-robust median.

    This is the honest selection rule: rank on the 3x3 neighbourhood median of
    the in-sample column, not the single best in-sample cell, then break ties by
    the cell's own in-sample value. The caller reports the chosen cell's
    out-of-sample column once; it must not feed back into the choice.
    """
    if table.empty:
        raise ValueError("cannot select from an empty table")
    if "IS_neighbour_median" not in table.columns:
        table = table.assign(IS_neighbour_median=is_neighbour_median(table, column))
    ranked = table.sort_values(["IS_neighbour_median", column], ascending=False, kind="stable")
    return ranked.iloc[0]


def sensitivity_table(data, windows=(20, 30, 45), quantiles=(90.0, 95.0, 99.0),
                      age_fractions=None, noise_sims: int = 100,
                      control_runs: int = 100) -> pd.DataFrame:
    """Re-run the backtest over a grid of settings. Judge the STABILITY of the
    pattern, not the single best cell (picking the best cell is just overfitting).

    The first age fraction defaults to `backtest.MAX_AGE_FRACTION`, so the
    default configuration is always one of the cells. `noise_trades_median` and
    `beats_control_pct` say whether a cell's action is distinguishable from luck;
    a high final value with a noise-like trade count is not evidence.

    `IS_*` and `OOS_*` columns split at the pre-registered `SPLIT_DATE`; they are
    for evaluation, not for choosing settings. Selection must use the in-sample
    columns only (`best_in_sample_cell`).
    """
    if age_fractions is None:
        age_fractions = default_age_fractions()

    rows = []
    for w in windows:
        for q in quantiles:
            for frac in age_fractions:
                res = run_backtest(window_bars=w, t_quantile=q, max_age=max(1, round(w * frac)),
                                   warmup_bars=backtest.WARMUP_BARS, data=data)
                noise = noise_trade_counts(res, n_sims=noise_sims)
                finals = shifted_timeline_control(res, n_runs=control_runs)
                try:
                    in_sample = in_sample_metrics(res)
                except ValueError:      # split outside this result's range
                    in_sample = {}
                try:
                    out_sample = out_of_sample_metrics(res)
                except ValueError:
                    out_sample = {}
                rows.append({
                    "window": w, "quantile": q, "age_fraction": frac, "max_age": res.max_age,
                    "trades": res.trade_count,
                    "strategy_GBP": round(float(res.strategy_values[-1]), 1),
                    "baseline_GBP": round(float(res.baseline_values[-1]), 1),
                    "beats_baseline": bool(res.strategy_values[-1] > res.baseline_values[-1]),
                    "IS_return_pct": round(in_sample.get("strategy_return_pct", np.nan), 1),
                    "IS_baseline_pct": round(in_sample.get("baseline_return_pct", np.nan), 1),
                    "IS_trades": in_sample.get("trades", np.nan),
                    "IS_beats_baseline": in_sample.get("beats_baseline", np.nan),
                    "OOS_return_pct": round(out_sample.get("strategy_return_pct", np.nan), 1),
                    "OOS_baseline_pct": round(out_sample.get("baseline_return_pct", np.nan), 1),
                    "OOS_trades": out_sample.get("trades", np.nan),
                    "OOS_beats_baseline": out_sample.get("beats_baseline", np.nan),
                    "noise_trades_median": float(np.median(noise)) if len(noise) else float("nan"),
                    "beats_control_pct": (
                        round(100 * float((res.strategy_values[-1] > finals).mean()), 1)
                        if len(finals) else float("nan")
                    ),
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------ Phase 5 validation
def _bootstrap_mean_ci(values, n_sims: int = DETOUR_BOOTSTRAP_SIMS,
                       seed: int = DETOUR_BOOTSTRAP_SEED,
                       level: float = DETOUR_CI_PCT) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of `values`.

    Pure and seeded. Returns `(nan, nan)` when there are fewer than two values,
    because a bootstrap interval from one observation carries no information.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(int(n_sims), len(values)), replace=True)
    means = draws.mean(axis=1)
    tail = (100.0 - float(level)) / 2.0
    return float(np.percentile(means, tail)), float(np.percentile(means, 100.0 - tail))


def block_bootstrap_joint_prices(prices: pd.DataFrame, block_length: int = MC_BLOCK_BARS,
                                 seed: int = MC_SEED) -> pd.DataFrame:
    """One synthetic price history from a JOINT block bootstrap of log returns.

    SPY, GLD and GBPUSD are resampled in the SAME blocks, so their
    cross-correlation and volatility clustering survive. The synthetic history
    has the same index and length as `prices`, starts from the real first price
    of each series, and is therefore a drop-in replacement for `run_backtest`.
    """
    columns = ["SPY", "GLD", "GBPUSD"]
    px = prices.loc[:, columns].astype(float)
    if len(px) < 2:
        raise ValueError("need at least two price rows to bootstrap returns")
    if (px <= 0).any().any():
        raise ValueError("prices must be strictly positive to bootstrap log returns")

    log_returns = np.log(px).diff().iloc[1:].to_numpy()
    n_returns = len(log_returns)
    block = max(1, min(int(block_length), n_returns))
    starts = np.arange(0, n_returns - block + 1)
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n_returns / block))
    chosen = rng.choice(starts, size=n_blocks, replace=True)
    sampled = np.concatenate(
        [log_returns[start:start + block] for start in chosen], axis=0
    )[:n_returns]

    first_log_levels = np.log(px.iloc[0].to_numpy(dtype=float))
    levels = np.exp(first_log_levels + np.cumsum(sampled, axis=0))
    synthetic = pd.DataFrame(levels, index=px.index[1:], columns=columns)
    synthetic = pd.concat([px.iloc[[0]], synthetic])
    return synthetic.loc[px.index, columns]


def _monte_carlo_one(args) -> dict:
    """Worker: bootstrap one synthetic history and re-run the FULL pipeline."""
    prices, block_length, sim_seed = args
    synthetic = block_bootstrap_joint_prices(prices, block_length=block_length, seed=sim_seed)
    ratio = synthetic["SPY"] / synthetic["GLD"]
    res = run_backtest(data=(ratio, synthetic))
    return {
        "annualised_excess_return": _annualised(res.strategy_values) - _annualised(res.baseline_values),
        "trades": int(res.trade_count),
        "strategy_max_drawdown": max_drawdown(res.strategy_values),
    }


def full_pipeline_monte_carlo(result: BacktestResult, data, n_sims: int = MC_SIMS,
                              block_length: int = MC_BLOCK_BARS, seed: int = MC_SEED,
                              n_jobs: int = 1) -> dict:
    """Null distribution from re-running the whole strategy on synthetic histories.

    Each synthetic history jointly block-bootstraps the daily log returns of
    SPY, GLD and GBPUSD (default block 20 bars), then runs `run_backtest` with
    the DEFAULT settings, including threshold calibration on that history's own
    warm-up. The comparison is therefore full-pipeline, not just the signal
    statistic.

    The pre-registered check is that the real strategy's annualised excess
    return over buy-and-hold SPY is at or above the `MC_PERCENTILE`-th
    percentile of this null. The one-sided p-value is
    `(1 + #{null >= real}) / (n_sims + 1)`.
    """
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    prices = data[1].loc[:, ["SPY", "GLD", "GBPUSD"]].astype(float)
    rng = np.random.default_rng(seed)
    sim_seeds = rng.integers(0, 2**32 - 1, size=int(n_sims), dtype=np.int64)
    jobs = [(prices, int(block_length), int(s)) for s in sim_seeds]

    if n_jobs and n_jobs > 1:
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as pool:
            records = list(pool.map(_monte_carlo_one, jobs))
    else:
        records = [_monte_carlo_one(job) for job in jobs]

    table = pd.DataFrame(records)
    null = table["annualised_excess_return"].to_numpy(dtype=float)
    real = _annualised(result.strategy_values) - _annualised(result.baseline_values)
    p95 = float(np.percentile(null, MC_PERCENTILE))
    # Percentile of the real result in the null distribution (higher is better).
    percentile = 100.0 * float((null < real).mean())
    p_value = (1.0 + float((null >= real).sum())) / (len(null) + 1.0)
    trade_low, trade_median, trade_high = (
        float(x) for x in np.percentile(table["trades"], [5, 50, 95])
    )

    return {
        "table": table,
        "n_sims": int(n_sims),
        "block_length": int(block_length),
        "seed": int(seed),
        "real_annualised_excess_pct": 100.0 * real,
        "null_median_pct": 100.0 * float(np.median(null)),
        "null_percentile_pct": 100.0 * p95,
        "percentile": percentile,
        "p_value": p_value,
        "passed": bool(real >= p95),
        "null_trades_p5": trade_low,
        "null_trades_median": trade_median,
        "null_trades_p95": trade_high,
        "null_max_drawdown_median_pct": 100.0 * float(
            np.median(table["strategy_max_drawdown"])
        ),
    }


def _gbp_growth(prices: pd.DataFrame, entry, exit, ticker: str) -> float:
    """Gross GBP growth of a USD asset from `entry` close to `exit` close."""
    p0, p1 = float(prices.loc[entry, ticker]), float(prices.loc[exit, ticker])
    fx0, fx1 = float(prices.loc[entry, "GBPUSD"]), float(prices.loc[exit, "GBPUSD"])
    return (p1 / p0) * (fx0 / fx1)


def _switch_cost_fraction(fx_fee_rate: float) -> float:
    """Value lost in one asset-to-asset swap, as a fraction of pre-swap value.

    Sell: pay `f` of proceeds. Buy: the remaining budget is divided by `(1+f)`.
    So the post-swap value is `(1-f)/(1+f)` of the pre-swap value.
    """
    return 2.0 * fx_fee_rate / (1.0 + fx_fee_rate)


def _buy_cost_fraction(fx_fee_rate: float) -> float:
    """Value lost buying from cash, as a fraction of the pre-buy balance."""
    return fx_fee_rate / (1.0 + fx_fee_rate)


def _log_gbp_growth(prices: pd.DataFrame, entry, exit, ticker: str) -> float:
    """log of the GBP growth of a USD asset from `entry` close to `exit` close."""
    return math.log(_gbp_growth(prices, entry, exit, ticker))


def exact_attribution(result: BacktestResult) -> dict:
    """Exact bar-by-bar decomposition of the strategy-minus-baseline log gap.

    For every traded bar ``t > 0`` define

        active[t] = log(strategy[t] / strategy[t-1])
                    - log(baseline[t] / baseline[t-1]).

    The interval ending on bar ``t`` is attributed to the asset held on bar
    ``t-1``, i.e. the asset that earned the return over ``(t-1, t]``. A switch
    executed at the close of bar ``t`` is therefore charged to the spell being
    *left*. This is deliberate: a GLD position is sold at the close of the swap
    bar, so the swap bar's GLD-vs-SPY return belongs to the GLD spell. The old
    last-held-bar attribution stopped one bar early and missed it.

    The only term not attached to a spell is the first-bar entry mismatch
    ``log(strategy[0] / baseline[0])``: the baseline has already paid its entry
    fee on the first traded bar while the strategy is still in cash. Adding it
    makes the reconciliation telescoping and exact:

        sum(active over all intervals) + log(s[0]/b[0])
            == log(s[-1]/b[-1]).

    The returned ``gbp_effect_to_end`` for each spell is the standalone effect
    of neutralising that spell's active log return, grown to the final strategy
    value: ``s_final * (1 - exp(-active_log))``. These per-spell GBP effects are
    not additive; use the combined effect from ``by_holding`` (or the log sums)
    for additive comparisons.
    """
    columns = [
        "holding", "entry_date", "last_held_date", "exit_execution_date",
        "bars", "return_bars", "active_log_return", "gross_log_return",
        "cost_log_return", "gbp_effect_to_end", "in_sample",
    ]
    dates = result.dates
    n = len(dates)
    if n == 0:
        return {
            "rows": pd.DataFrame(columns=columns),
            "by_holding": pd.DataFrame(
                columns=["holding", "spells", "bars", "active_log_return",
                         "gross_log_return", "cost_log_return", "gbp_effect_to_end"]
            ),
            "first_bar_log_mismatch": float("nan"),
            "first_bar_gbp_effect": float("nan"),
            "sum_active_log_return": 0.0,
            "log_final_ratio": float("nan"),
            "max_abs_log_error": 0.0,
            "reconciled": True,
            "strategy_final": float("nan"),
            "baseline_final": float("nan"),
        }

    s = np.asarray(result.strategy_values, dtype=float)
    b = np.asarray(result.baseline_values, dtype=float)
    held = holding_timeline(result).to_numpy()
    periods = holding_periods(result)

    active = np.zeros(n, dtype=float)
    for t in range(1, n):
        active[t] = math.log(s[t] / s[t - 1]) - math.log(b[t] / b[t - 1])

    first_mismatch = math.log(s[0] / b[0])
    s_final, b_final = float(s[-1]), float(b[-1])
    rows = []
    for _, spell in periods.iterrows():
        holding = spell["holding"]
        entry = pd.Timestamp(spell["start"])
        last_held = pd.Timestamp(spell["end"])
        start_pos = int(dates.get_loc(entry))
        end_pos = int(dates.get_loc(last_held))
        # The exit swap (if any) executes at the close of the next bar, so its
        # return is earned while still holding this spell's asset.
        exit_pos = min(end_pos + 1, n - 1)
        exit_execution = pd.Timestamp(dates[exit_pos])

        first_t = max(start_pos + 1, 1)
        last_t = min(end_pos + 1, n - 1)
        if first_t <= last_t:
            spell_active = float(active[first_t:last_t + 1].sum())
            return_bars = int(last_t - first_t + 1)
        else:
            spell_active = 0.0
            return_bars = 0

        if holding == "GLD":
            gross = (_log_gbp_growth(result.prices, entry, exit_execution, "GLD")
                     - _log_gbp_growth(result.prices, entry, exit_execution, "SPY"))
        elif holding == "SPY":
            # Strategy and baseline both hold SPY, so the asset return cancels
            # exactly; only switch costs can remain.
            gross = 0.0
        else:  # CASH earns no return; baseline is in SPY over the same interval.
            gross = -_log_gbp_growth(result.prices, entry, exit_execution, "SPY")

        rows.append({
            "holding": holding,
            "entry_date": entry,
            "last_held_date": last_held,
            "exit_execution_date": exit_execution,
            "bars": int(spell["bars"]),
            "return_bars": return_bars,
            "active_log_return": spell_active,
            "gross_log_return": float(gross),
            "cost_log_return": float(spell_active - gross),
            "gbp_effect_to_end": float(s_final * (-math.expm1(-spell_active))),
            "in_sample": bool(entry <= SPLIT_DATE),
        })

    rows_df = pd.DataFrame(rows, columns=columns)
    grouped = []
    for holding, frame in rows_df.groupby("holding", sort=False):
        total_active = float(frame["active_log_return"].sum())
        grouped.append({
            "holding": holding,
            "spells": int(len(frame)),
            "bars": int(frame["bars"].sum()),
            "active_log_return": total_active,
            "gross_log_return": float(frame["gross_log_return"].sum()),
            "cost_log_return": float(frame["cost_log_return"].sum()),
            "gbp_effect_to_end": float(s_final * (-math.expm1(-total_active))),
        })
    by_holding = pd.DataFrame(grouped)
    if not by_holding.empty:
        order = {"CASH": 0, "GLD": 1, "SPY": 2}
        by_holding["_order"] = by_holding["holding"].map(order).fillna(99)
        by_holding = (by_holding.sort_values("_order")
                      .drop(columns="_order").reset_index(drop=True))

    sum_active = float(rows_df["active_log_return"].sum()) if len(rows_df) else 0.0
    log_final_ratio = math.log(s_final / b_final)
    max_error = abs(sum_active + first_mismatch - log_final_ratio)
    return {
        "rows": rows_df,
        "by_holding": by_holding,
        "first_bar_log_mismatch": float(first_mismatch),
        "first_bar_gbp_effect": float(s_final * (-math.expm1(-first_mismatch))),
        "sum_active_log_return": sum_active,
        "log_final_ratio": float(log_final_ratio),
        "max_abs_log_error": float(max_error),
        "reconciled": bool(max_error <= 1e-9),
        "strategy_final": s_final,
        "baseline_final": b_final,
    }


def detour_attribution(result: BacktestResult, fx_fee_rate: float = FX_FEE_RATE,
                       n_bootstrap: int = DETOUR_BOOTSTRAP_SIMS,
                       seed: int = DETOUR_BOOTSTRAP_SEED) -> dict:
    """Attribute the strategy's deviations from buy-and-hold SPY, spell by spell.

    One row per non-SPY holding spell (`holding` is "GLD" or "CASH"). A CASH
    spell can only occur before the first trade; it is included so the exact
    decomposition covers the whole gap, but the headline "detour" statistics use
    GLD spells only. For each row:
      - `entry_date` / `exit_date`: first and last bar on which the spell's asset
        was held.
      - `exit_execution_date`: the bar on which the next asset is bought (the
        swap bar). A GLD position is sold at this bar's close, so its return
        belongs to the spell. `gross_excess_return` and `ratio_log_change` cover
        entry close to this exit-execution close.
      - `gross_excess_return`: holding return in GBP minus SPY's GBP return over
        the same dates, before switching costs.
      - `switch_cost`: modelled cost of the swaps at the spell boundaries.
      - `net_excess_return`: gross minus switch cost. This is the metric used by
        the pre-registered detour check.
      - `notional_gbp` / `net_excess_gbp`: the local GBP effect at the spell
        (not grown to the end, and not used for reconciliation).

    The summary bootstraps the mean net excess of the GLD detours. Reconciliation
    is now exact: `exact_attribution` sums log active returns bar by bar, adds the
    first-bar entry mismatch, and asserts the result equals
    `log(final strategy / final baseline)` to within 1e-9. The old additive
    notional sum is retained only as `legacy_*` for comparison.
    """
    columns = [
        "holding", "entry_date", "exit_date", "exit_execution_date", "bars",
        "ratio_log_change", "gross_excess_return", "switch_cost",
        "net_excess_return", "notional_gbp", "net_excess_gbp", "in_sample",
    ]
    exact = exact_attribution(result)
    actual_gap = (float(result.strategy_values[-1] - result.baseline_values[-1])
                  if len(result.dates) else 0.0)
    attributed = 0.0
    reconciliation = {
        "strategy_minus_baseline_gbp": actual_gap,
        "sum_active_log_return": exact["sum_active_log_return"],
        "first_bar_log_mismatch": exact["first_bar_log_mismatch"],
        "log_final_ratio": exact["log_final_ratio"],
        "max_abs_log_error": exact["max_abs_log_error"],
        "reconciled": exact["reconciled"],
        "gbp_effect_by_holding": (
            exact["by_holding"].set_index("holding")["gbp_effect_to_end"].to_dict()
            if len(exact["by_holding"]) else {}
        ),
        # Kept only to show how far the old additive-notional method was off.
        "legacy_sum_net_excess_gbp": attributed,
        "legacy_gap_gbp": actual_gap - attributed,
    }
    empty_summary = {
        "count": 0, "gld_count": 0, "cash_count": 0,
        "share_time_in_gld_pct": float("nan"),
        "mean_net_excess": float("nan"), "median_net_excess": float("nan"),
        "share_positive": float("nan"), "bootstrap_ci": (float("nan"), float("nan")),
        "in_sample": {"count": 0, "mean_net_excess": float("nan"),
                      "median_net_excess": float("nan"), "share_positive": float("nan"),
                      "bootstrap_ci": (float("nan"), float("nan"))},
        "out_of_sample": {"count": 0, "mean_net_excess": float("nan"),
                          "median_net_excess": float("nan"), "share_positive": float("nan"),
                          "bootstrap_ci": (float("nan"), float("nan"))},
        "reconciliation": reconciliation,
        "exact": exact,
    }
    if len(result.dates) == 0:
        return {"rows": pd.DataFrame(columns=columns), **empty_summary}

    dates = result.dates
    periods = holding_periods(result)
    holdings = holding_timeline(result).to_numpy()
    rows = []
    for _, spell in periods.iterrows():
        if spell["holding"] == "SPY":
            continue
        entry, last_held = spell["start"], spell["end"]
        start_pos = int(dates.get_loc(entry))
        end_pos = int(dates.get_loc(last_held))
        exit_pos = min(end_pos + 1, len(dates) - 1)
        exit_execution = pd.Timestamp(dates[exit_pos])
        bars = int(spell["bars"])
        holding = spell["holding"]

        # Include the exit swap bar: the position is still held during that
        # bar's return and is only sold at its close.
        spy_growth = _gbp_growth(result.prices, entry, exit_execution, "SPY")
        if holding == "GLD":
            gross_excess = (_gbp_growth(result.prices, entry, exit_execution, "GLD")
                            - spy_growth)
        else:  # CASH earns no GBP return
            gross_excess = 1.0 - spy_growth

        # Entry swap: a CASH spell starts in cash and has no entry trade. A GLD
        # spell normally follows SPY; a buy from cash is charged the buy leg only.
        entry_cost = 0.0
        source = holdings[start_pos - 1] if start_pos > 0 else None
        if holding != "CASH":
            entry_cost = (_buy_cost_fraction(fx_fee_rate) if source is None
                          else _switch_cost_fraction(fx_fee_rate))
        # Exit swap: only charge it when there is a later bar to trade on.
        exit_cost = 0.0
        if end_pos + 1 < len(dates):
            target = holdings[end_pos + 1]
            if target is not None and target != holding:
                exit_cost = (_buy_cost_fraction(fx_fee_rate) if holding == "CASH"
                             else _switch_cost_fraction(fx_fee_rate))

        switch_cost = entry_cost + exit_cost
        net_excess = gross_excess - switch_cost
        notional = (float(result.strategy_values[start_pos - 1])
                    if start_pos > 0 else float(INITIAL_BALANCE))
        rows.append({
            "holding": holding,
            "entry_date": pd.Timestamp(entry),
            "exit_date": pd.Timestamp(last_held),
            "exit_execution_date": exit_execution,
            "bars": bars,
            "ratio_log_change": float(
                result.log_ratio.loc[exit_execution] - result.log_ratio.loc[entry]
            ),
            "gross_excess_return": float(gross_excess),
            "switch_cost": float(switch_cost),
            "net_excess_return": float(net_excess),
            "notional_gbp": notional,
            "net_excess_gbp": float(notional * net_excess),
            "in_sample": bool(pd.Timestamp(entry) <= SPLIT_DATE),
        })

    rows_df = pd.DataFrame(rows, columns=columns)
    gld = rows_df[rows_df["holding"] == "GLD"]
    gld_net = gld["net_excess_return"].to_numpy(dtype=float)

    def subset(frame: pd.DataFrame) -> dict:
        values = frame["net_excess_return"].to_numpy(dtype=float)
        return {
            "count": int(len(values)),
            "mean_net_excess": float(np.mean(values)) if len(values) else float("nan"),
            "median_net_excess": float(np.median(values)) if len(values) else float("nan"),
            "share_positive": float((values > 0).mean()) if len(values) else float("nan"),
            "bootstrap_ci": _bootstrap_mean_ci(values, n_sims=n_bootstrap, seed=seed),
        }

    total_bars = len(dates)
    gld_bars = int(periods.loc[periods["holding"] == "GLD", "bars"].sum())
    attributed = float(rows_df["net_excess_gbp"].sum())
    reconciliation["legacy_sum_net_excess_gbp"] = attributed
    reconciliation["legacy_gap_gbp"] = actual_gap - attributed
    summary = {
        "count": int(len(rows_df)),
        "gld_count": int(len(gld)),
        "cash_count": int((rows_df["holding"] == "CASH").sum()),
        "share_time_in_gld_pct": 100.0 * gld_bars / total_bars if total_bars else float("nan"),
        "mean_net_excess": float(np.mean(gld_net)) if len(gld_net) else float("nan"),
        "median_net_excess": float(np.median(gld_net)) if len(gld_net) else float("nan"),
        "share_positive": float((gld_net > 0).mean()) if len(gld_net) else float("nan"),
        "bootstrap_ci": _bootstrap_mean_ci(gld_net, n_sims=n_bootstrap, seed=seed),
        "in_sample": subset(gld[gld["in_sample"]]),
        "out_of_sample": subset(gld[~gld["in_sample"]]),
        "reconciliation": reconciliation,
        "exact": exact,
    }
    return {"rows": rows_df, **summary}


def tracking_error_stats(result: BacktestResult) -> dict:
    """Annualised tracking error, information ratio and time-to-detect.

    Active return is the daily strategy return minus the daily baseline return.
    The information ratio is the annualised mean active return divided by the
    annualised standard deviation of active returns. At an observed IR, a
    two-standard-error detection needs `(2 / IR) ** 2` years. If the mean active
    return is not positive, `years_to_detect_2se` is infinite.
    """
    s = np.asarray(result.strategy_values, dtype=float)
    b = np.asarray(result.baseline_values, dtype=float)
    if len(s) < 2:
        return {
            "bars": 0, "years_available": 0.0,
            "annualised_active_return_pct": float("nan"),
            "annualised_tracking_error_pct": float("nan"),
            "information_ratio": float("nan"),
            "years_to_detect_2se": float("inf"),
            "detectable_within_history": False,
        }
    active = (s[1:] / s[:-1] - 1.0) - (b[1:] / b[:-1] - 1.0)
    n = len(active)
    mean_daily = float(np.mean(active))
    std_daily = float(np.std(active, ddof=1)) if n > 1 else float("nan")
    te_ann = std_daily * math.sqrt(BARS_PER_YEAR)
    active_ann = mean_daily * BARS_PER_YEAR
    ir = active_ann / te_ann if te_ann > 0 and np.isfinite(te_ann) else float("nan")
    years_available = n / BARS_PER_YEAR
    years_to_detect = (2.0 / ir) ** 2 if np.isfinite(ir) and ir > 0 else float("inf")
    return {
        "bars": int(n),
        "years_available": float(years_available),
        "annualised_active_return_pct": 100.0 * active_ann,
        "annualised_tracking_error_pct": 100.0 * te_ann,
        "information_ratio": float(ir),
        "years_to_detect_2se": float(years_to_detect),
        "detectable_within_history": bool(np.isfinite(years_to_detect)
                                          and years_to_detect <= years_available),
    }


def friction_sensitivity(data, extras: tuple[float, ...] = FRICTION_EXTRAS,
                         base_fee: float = FX_FEE_RATE,
                         result: BacktestResult | None = None) -> dict:
    """Final-value lead over the baseline as extra per-leg friction is added.

    `extras` are decimal additions to the FX fee on EACH leg of a swap; the
    default fee itself is never changed. The break-even extra friction is the
    linear interpolation where the lead crosses zero, or None when it does not
    cross inside the supplied grid. `result`, when supplied, is reused for the
    zero-extra run.
    """
    extras = tuple(sorted(float(x) for x in extras))
    if not extras:
        raise ValueError("extras must contain at least one value")
    rows = []
    for extra in extras:
        if result is not None and abs(extra) < 1e-15:
            res = result
        else:
            res = run_backtest(data=data, fx_fee_rate=base_fee + extra)
        lead = float(res.strategy_values[-1] - res.baseline_values[-1])
        rows.append({
            "extra_friction_pct_points": 100.0 * extra,
            "total_fx_fee_pct": 100.0 * (base_fee + extra),
            "strategy_final": float(res.strategy_values[-1]),
            "baseline_final": float(res.baseline_values[-1]),
            "lead_gbp": lead,
            "beats_baseline": bool(lead > 0),
        })
    table = pd.DataFrame(rows)
    leads = table["lead_gbp"].to_numpy(dtype=float)
    break_even = None
    if leads[0] <= 0:
        note = "does not beat the baseline even at zero extra friction"
    elif leads[-1] > 0:
        note = "still ahead at the highest grid point; break-even is beyond the grid"
    else:
        note = "lead crosses zero inside the grid"
        for i in range(1, len(leads)):
            if leads[i] <= 0 < leads[i - 1]:
                x0, x1 = extras[i - 1], extras[i]
                y0, y1 = leads[i - 1], leads[i]
                break_even = x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)
                break
    return {
        "table": table,
        "break_even_extra_friction": break_even,
        "break_even_total_fee": (base_fee + break_even) if break_even is not None else None,
        "note": note,
    }


def forward_metrics(result: BacktestResult, start=CLEAN_HOLDOUT_START) -> dict:
    """`range_metrics` on clean post-design data, or an explicit empty status.

    Data before `CLEAN_HOLDOUT_START` informed the design (including the
    out-of-sample window), so only later bars are clean evidence. Until at
    least one traded bar exists on/after that date this returns
    `{"status": "no clean data yet", "bars": 0, ...}` instead of raising.
    """
    start = pd.Timestamp(start)
    if len(result.dates) == 0 or start > result.dates[-1]:
        return {
            "status": "no clean data yet",
            "start": start,
            "end": result.dates[-1] if len(result.dates) else None,
            "bars": 0,
            "available": False,
        }
    metrics = range_metrics(result, start, result.dates[-1])
    return {"status": "ok", "available": True, **metrics}


def _axis_index(axis, value) -> int | None:
    arr = np.asarray(axis, dtype=float)
    hits = np.where(np.isclose(arr, float(value)))[0]
    return int(hits[0]) if len(hits) else None


def neighbourhood_check(table: pd.DataFrame, window: int = backtest.WINDOW,
                        quantile: float = backtest.T_QUANTILE,
                        age_fraction: float = backtest.MAX_AGE_FRACTION) -> dict:
    """Does the default cell's 3x3 neighbourhood beat the baseline?

    Full-sample and out-of-sample counts are checked separately, as
    pre-registered. A missing/partial neighbourhood cannot pass: the check
    needs all nine cells and at least `NEIGHBOURHOOD_MIN_BEATS` wins in each.
    """
    required = {"window", "quantile", "age_fraction", "beats_baseline", "OOS_beats_baseline"}
    missing = required - set(table.columns)
    if missing:
        return {"found": False, "cells": 0, "full_beats": 0, "oos_beats": 0,
                "passed": False, "reason": f"missing columns: {sorted(missing)}"}
    group = table[np.isclose(table["window"].astype(float), float(window))]
    if group.empty:
        return {"found": False, "cells": 0, "full_beats": 0, "oos_beats": 0,
                "passed": False, "reason": "default window not in table"}
    quantiles = sorted(pd.unique(group["quantile"].astype(float)))
    fractions = sorted(pd.unique(group["age_fraction"].astype(float)))
    qi = _axis_index(quantiles, quantile)
    fi = _axis_index(fractions, age_fraction)
    if qi is None or fi is None:
        return {"found": False, "cells": 0, "full_beats": 0, "oos_beats": 0,
                "passed": False, "reason": "default quantile/age fraction not in table"}
    q_block = quantiles[max(0, qi - 1):qi + 2]
    f_block = fractions[max(0, fi - 1):fi + 2]
    cells = group[
        group["quantile"].astype(float).isin(q_block)
        & group["age_fraction"].astype(float).isin(f_block)
    ]
    full_beats = int(cells["beats_baseline"].fillna(False).astype(bool).sum())
    oos_beats = int((cells["OOS_beats_baseline"] == True).fillna(False).sum())  # noqa: E712
    n_cells = int(len(cells))
    passed = (n_cells == NEIGHBOURHOOD_TOTAL_CELLS
              and full_beats >= NEIGHBOURHOOD_MIN_BEATS
              and oos_beats >= NEIGHBOURHOOD_MIN_BEATS)
    return {
        "found": True, "cells": n_cells,
        "full_beats": full_beats, "oos_beats": oos_beats,
        "required": NEIGHBOURHOOD_MIN_BEATS,
        "passed": bool(passed),
        "reason": "ok" if passed else "not enough neighbourhood wins",
    }


def validation_verdict(result: BacktestResult, data, table: pd.DataFrame | None = None,
                       mc: dict | None = None, detours: dict | None = None,
                       doubled_result: BacktestResult | None = None,
                       n_sims: int = MC_SIMS, n_jobs: int = 1) -> dict:
    """Evaluate the four pre-registered gates and return PASS/FAIL per check.

    Callers that already computed a piece (the sensitivity grid, the Monte
    Carlo run, the detour attribution) can pass it in to avoid re-running it.
    `VALIDATED` requires every check to pass; there is no partial credit.
    """
    if detours is None:
        detours = detour_attribution(result)
    if mc is None:
        mc = full_pipeline_monte_carlo(result, data, n_sims=n_sims, n_jobs=n_jobs)
    if table is None:
        table = sensitivity_table(data)
    if doubled_result is None:
        doubled_result = run_backtest(data=data, fx_fee_rate=FRICTION_FEE_MULTIPLE * FX_FEE_RATE)

    neighbourhood = neighbourhood_check(table)
    ci_low, ci_high = detours["bootstrap_ci"]
    doubled_lead = float(doubled_result.strategy_values[-1] - doubled_result.baseline_values[-1])
    checks = [
        {
            "number": 1,
            "name": "Full-pipeline Monte Carlo",
            "passed": bool(mc["passed"]),
            "detail": (f"real annualised excess {mc['real_annualised_excess_pct']:+.2f}% vs "
                       f"null {MC_PERCENTILE:g}th percentile "
                       f"{mc['null_percentile_pct']:+.2f}% (p={mc['p_value']:.4f})"),
        },
        {
            "number": 2,
            "name": "Detour attribution",
            "passed": bool(np.isfinite(ci_low) and ci_low > 0),
            "detail": (f"mean net detour excess {100 * detours['mean_net_excess']:+.2f}% "
                       f"with {DETOUR_CI_PCT:g}% bootstrap CI "
                       f"[{100 * ci_low:+.2f}%, {100 * ci_high:+.2f}%]"),
        },
        {
            "number": 3,
            "name": f"Friction at {FRICTION_FEE_MULTIPLE:g}x FX fee",
            "passed": bool(doubled_lead > 0),
            "detail": (f"final-value lead {doubled_lead:+,.0f} GBP at total FX fee "
                       f"{100 * FRICTION_FEE_MULTIPLE * FX_FEE_RATE:.2f}%"),
        },
        {
            "number": 4,
            "name": "Neighbourhood (3x3)",
            "passed": bool(neighbourhood["passed"]),
            "detail": (f"default cell beats baseline in {neighbourhood['full_beats']}/"
                       f"{neighbourhood['cells']} neighbours full-sample and "
                       f"{neighbourhood['oos_beats']}/{neighbourhood['cells']} OOS "
                       f"(need {NEIGHBOURHOOD_MIN_BEATS}/{NEIGHBOURHOOD_TOTAL_CELLS} in both)"),
        },
    ]
    return {
        "checks": checks,
        "validated": bool(all(check["passed"] for check in checks)),
        "monte_carlo": mc,
        "detours": detours,
        "neighbourhood": neighbourhood,
        "doubled_fee_result": doubled_result,
    }


# ------------------------------------------------------------------ charts
def plot_values(result: BacktestResult) -> Figure:
    """Strategy vs baseline portfolio value (GBP), with drawdowns underneath."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(result.dates, result.strategy_values, label="Strategy", lw=1.6)
    ax1.plot(result.dates, result.baseline_values, label="Baseline: buy & hold SPY", lw=1.6, alpha=0.85)
    ax1.set_ylabel("Portfolio value (GBP)")
    ax1.set_title("Strategy vs buy-and-hold baseline (both start on the same day with GBP 1,000)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.25)

    for values, label in ((result.strategy_values, "Strategy"), (result.baseline_values, "Baseline")):
        drawdown = 100 * (values / np.maximum.accumulate(values) - 1)
        ax2.plot(result.dates, drawdown, label=label, lw=1)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_log_ratio_with_flips(result: BacktestResult) -> Figure:
    """log(SPY/GLD) with every switch marked, holding periods shaded, and the
    t-statistic (with its calibrated threshold) underneath."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.3]})
    log_ratio = result.log_ratio
    ax1.plot(log_ratio.index, log_ratio.values, color="black", lw=0.8, label="log(SPY / GLD)")
    ax1.axvspan(log_ratio.index[0], result.dates[0], color="0.88", label="Warm-up (threshold calibration only)")

    # shade each holding period
    edges = [result.dates[0], *result.trades.index, result.dates[-1]]
    holdings = [None, *result.trades["bought"].tolist()]
    shade = {"SPY": (GREEN, "Holding SPY"), "GLD": (GOLD, "Holding GLD")}
    used = set()
    for start, end, held in zip(edges[:-1], edges[1:], holdings):
        if held is None:
            continue
        colour, label = shade[held]
        ax1.axvspan(start, end, color=colour, alpha=0.12, label=None if held in used else label)
        used.add(held)

    to_spy = result.trades[result.trades["bought"] == "SPY"]
    to_gld = result.trades[result.trades["bought"] == "GLD"]
    ax1.scatter(to_spy.index, log_ratio.loc[to_spy.index], marker="^", s=110, color=GREEN,
                edgecolor="black", zorder=3, label=f"Switched to SPY ({len(to_spy)})")
    ax1.scatter(to_gld.index, log_ratio.loc[to_gld.index], marker="v", s=110, color=GOLD,
                edgecolor="black", zorder=3, label=f"Switched to GLD ({len(to_gld)})")
    ax1.set_ylabel("log(SPY / GLD)")
    ax1.set_title("Every switch the strategy made")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.plot(result.dates, result.t_stats, color="0.3", lw=0.7, label="t-statistic of the curvature")
    ax2.axhline(result.t_threshold, color="red", ls="--", lw=0.9, label=f"+/- calibrated threshold ({result.t_threshold:.1f})")
    ax2.axhline(-result.t_threshold, color="red", ls="--", lw=0.9)
    ax2.set_ylabel("t-statistic")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_control(result: BacktestResult, finals: np.ndarray) -> Figure:
    """Where the real strategy's final value sits among shifted-timeline versions of itself."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(finals):
        ax.hist(finals, bins=30, color="0.7", edgecolor="white", label="Same time in cash / SPY / GLD, shifted in time")
    ax.axvline(result.strategy_values[-1], color="tab:blue", lw=2, label=f"Strategy ({result.strategy_values[-1]:,.0f})")
    ax.axvline(result.baseline_values[-1], color="tab:orange", lw=2, ls="--", label=f"Baseline ({result.baseline_values[-1]:,.0f})")
    beaten = f"{(result.strategy_values[-1] > finals).mean():.0%}" if len(finals) else "n/a"
    ax.set_title(f"Strategy beat {beaten} of shifted-timeline runs")
    ax.set_xlabel("Final value (GBP)")
    ax.set_ylabel("Number of runs")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_annual_returns(result: BacktestResult) -> Figure:
    """Trailing one-year return distributions for strategy, baseline and active.

    Panel 1 overlays strategy and baseline histograms on identical bins with a
    fitted normal curve for each. Panel 2 does the same for the active return
    (strategy minus baseline, percentage points). The y-axis is a count of
    trading days; the normal curve is scaled as pdf * N * bin width.
    """
    s = trailing_annual_returns(result.strategy_values)
    b = trailing_annual_returns(result.baseline_values)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Trailing one-year returns on every traded bar with a full year behind it")

    if len(s) == 0 or len(b) == 0:
        message = (
            f"Fewer than {BARS_PER_YEAR + 1} traded bars: cannot compute a trailing "
            f"one-year return. Need at least {BARS_PER_YEAR + 1} bars."
        )
        ax1.text(0.5, 0.5, message, ha="center", va="center", transform=ax1.transAxes,
                 wrap=True)
        ax2.axis("off")
        fig.text(0.5, 0.02, OVERLAP_WARNING, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        return fig

    s_pct, b_pct = 100.0 * s, 100.0 * b
    active_pct = s_pct - b_pct
    combined = np.concatenate([s_pct, b_pct])
    bins = _safe_bin_edges(combined, bins=60)
    width = float(bins[1] - bins[0]) if len(bins) > 1 else 1.0
    centers = (bins[:-1] + bins[1:]) / 2.0

    notes = []
    for values, colour, label in (
        (s_pct, "tab:blue", "Strategy"),
        (b_pct, "tab:orange", "Baseline"),
    ):
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        share_neg = 100.0 * float((values < 0).mean())
        ax1.hist(values, bins=bins, alpha=0.45, color=colour,
                 label=f"{label}: mean {mean:+.1f}%, {share_neg:.0f}% negative years")
        ax1.axvline(mean, color=colour, ls="--", lw=1.5)
        curve = _normal_curve(centers, width, mean, sd, len(values))
        if curve is None:
            notes.append(f"{label} normal curve skipped: zero variance")
        else:
            ax1.plot(centers, curve, color=colour, lw=1.5)
    ax1.axvline(0.0, color="black", lw=1.5, label="0%")
    ax1.set_xlabel("Trailing one-year return (%)")
    ax1.set_ylabel("Number of trading days")
    ax1.set_title("Strategy vs baseline (identical bins)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.25)
    if notes:
        ax1.text(0.99, 0.02, "\n".join(notes), ha="right", va="bottom",
                 transform=ax1.transAxes, fontsize=8)

    mean_active = float(np.mean(active_pct))
    sd_active = float(np.std(active_pct, ddof=1)) if len(active_pct) > 1 else float("nan")
    share_ahead = 100.0 * float((active_pct > 0).mean())
    active_bins = _safe_bin_edges(active_pct, bins=60)
    active_width = (float(active_bins[1] - active_bins[0])
                    if len(active_bins) > 1 else 1.0)
    active_centers = (active_bins[:-1] + active_bins[1:]) / 2.0
    ax2.hist(active_pct, bins=active_bins, alpha=0.55, color="tab:green",
             label=f"Active: mean {mean_active:+.1f} pp, {share_ahead:.0f}% of days ahead")
    ax2.axvline(mean_active, color="tab:green", ls="--", lw=1.5)
    ax2.axvline(0.0, color="black", lw=1.5, label="0 pp")
    curve_active = _normal_curve(active_centers, active_width, mean_active,
                                 sd_active, len(active_pct))
    if curve_active is None:
        ax2.text(0.99, 0.02, "Active normal curve skipped: zero variance",
                 ha="right", va="bottom", transform=ax2.transAxes, fontsize=8)
    else:
        ax2.plot(active_centers, curve_active, color="tab:green", lw=1.5)
    ax2.set_xlabel("Trailing one-year active return (percentage points)")
    ax2.set_ylabel("Number of trading days")
    ax2.set_title("Active return: strategy minus baseline")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.25)

    fig.text(0.5, 0.02, OVERLAP_WARNING, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


def plot_mc_null(mc: dict) -> Figure:
    """Monte Carlo null distribution of annualised excess returns."""
    fig, ax = plt.subplots(figsize=(10, 5))
    table = mc.get("table")
    if table is None or len(table) == 0:
        ax.text(0.5, 0.5, "No Monte Carlo null available", ha="center", va="center",
                transform=ax.transAxes)
        fig.tight_layout()
        return fig
    null = 100.0 * table["annualised_excess_return"].to_numpy(dtype=float)
    ax.hist(null, bins=30, color="0.7", edgecolor="white",
            label=f"Synthetic null ({len(null)} runs)")
    real = float(mc["real_annualised_excess_pct"])
    p95 = float(mc["null_percentile_pct"])
    ax.axvline(real, color="tab:blue", lw=2, label=f"Real ({real:+.2f}%)")
    ax.axvline(p95, color="tab:red", lw=2, ls="--",
               label=f"{MC_PERCENTILE:g}th percentile ({p95:+.2f}%)")
    ax.set_xlabel("Annualised excess return vs synthetic buy-and-hold SPY (%)")
    ax.set_ylabel("Number of synthetic histories")
    ax.set_title("Full-pipeline Monte Carlo null")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig