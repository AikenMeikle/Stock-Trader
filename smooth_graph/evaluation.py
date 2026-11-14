"""Evaluation of a BacktestResult: statistics, controls and charts."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.lib.stride_tricks import sliding_window_view

from . import backtest
from .backtest import FX_FEE_RATE, INITIAL_BALANCE, BacktestResult, run_backtest
from .portfolio import Portfolio

BARS_PER_YEAR = 252
GREEN, GOLD = "tab:green", "goldenrod"


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
def window_stats_all(values, window: int):
    """t-statistic, vertex and curvature for EVERY rolling window at once.
    A vectorised version of regression.fit_window (a test checks they agree)."""
    values = np.asarray(values, dtype=float)
    u = np.arange(-(window - 1), 1).astype(float)
    X = np.vander(u, 3)                                  # columns: u**2, u, 1
    xtx_inv = np.linalg.inv(X.T @ X)
    windows = sliding_window_view(values, window)        # one row per window
    coef = windows @ (xtx_inv @ X.T).T
    residuals = windows - coef @ X.T
    s2 = (residuals ** 2).sum(axis=1) / (window - 3)
    a, b = coef[:, 0], coef[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = a / np.sqrt(s2 * xtx_inv[0, 0])
        vertex = -b / (2 * a)
    return t, vertex, a


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


# ------------------------------------------------------ sensitivity grid
def sensitivity_table(data, windows=(20, 30, 45), quantiles=(90.0, 95.0, 99.0),
                      age_fractions=(0.25, 0.5)) -> pd.DataFrame:
    """Re-run the backtest over a grid of settings. Judge the STABILITY of the
    pattern, not the single best cell (picking the best cell is just overfitting)."""
    rows = []
    for w in windows:
        for q in quantiles:
            for frac in age_fractions:
                res = run_backtest(window_bars=w, t_quantile=q, max_age=max(1, round(w * frac)),
                                   warmup_bars=backtest.WARMUP_BARS, data=data)
                rows.append({
                    "window": w, "quantile": q, "max_age": res.max_age,
                    "trades": res.trade_count,
                    "strategy_GBP": round(float(res.strategy_values[-1]), 1),
                    "baseline_GBP": round(float(res.baseline_values[-1]), 1),
                    "beats_baseline": bool(res.strategy_values[-1] > res.baseline_values[-1]),
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