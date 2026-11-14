"""Evaluation of a BacktestResult: statistics, controls and charts."""
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
# Trade-level statistics are meaningless on a handful of trades; the report
# suppresses the hit rate below this many.
MIN_TRADES_FOR_STATS = 10
# Pre-registered in-sample / out-of-sample boundary. It is deliberately fixed
# before looking at any result: the first ~9 years are for choosing settings,
# the remaining ~12 are evaluated once. Do not move it after reading a grid.
SPLIT_DATE = pd.Timestamp("2014-12-31")


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
def _count_trades(t, vertex, a, threshold: float, max_age: int) -> int:
    """How many trades the strategy rules would make on these per-bar statistics
    (cash start, one-bar delay, never queue what is already held)."""
    fire = (np.abs(t) >= threshold) & (vertex >= -max_age) & (vertex <= 0)
    targets = np.where(fire, np.where(a > 0, 1, -1), 0)   # 1 = SPY, -1 = GLD
    held, pending, trades = 0, 0, 0
    for target in targets:
        if pending:
            held, pending = pending, 0
            trades += 1
        if target != 0 and target != held:
            pending = target
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
        counts.append(_count_trades(t, vertex, a, result.t_threshold, result.max_age))
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