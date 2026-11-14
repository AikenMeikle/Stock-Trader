# %%
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from smooth_graph.portfolio import Portfolio


# %% ---------------------------------------------------------------------
# 1. DATA
# ------------------------------------------------------------------------
def get_data(start="2005-01-01"):
    """Returns (log_ratio, data).
    log_ratio : Series, ln(SPY / GLD)
    data      : DataFrame with columns SPY, GLD, GBPUSD on the same dates
    """
    spy = yf.download("SPY", start=start)["Close"]["SPY"]
    gld = yf.download("GLD", start=start)["Close"]["GLD"]
    fx = yf.download("GBPUSD=X", start=start)["Close"]["GBPUSD=X"]

    data = pd.concat([spy.rename("SPY"), gld.rename("GLD")], axis=1).dropna()
    data["GBPUSD"] = fx.reindex(data.index, method="ffill")
    data = data.dropna()

    print("NaNs remaining:", int(data.isna().sum().sum()))
    log_ratio = np.log(data["SPY"] / data["GLD"])
    return log_ratio, data


# %% ---------------------------------------------------------------------
# 2. THE QUADRATIC FIT  (this replaces the smoothing spline)
# ------------------------------------------------------------------------
def window_stats(values):
    """Fit L = a*u^2 + b*u + c to the window, where u = -(W-1) ... -1, 0
    (u = 0 is today).  Returns (t_stat, vertex_day, a).

    a      : curvature (>0 cup shape, <0 cap shape)
    t_stat : a / standard_error(a)
    vertex : u* = -b / (2a), the turning point in days relative to today
             (negative = days ago, positive = in the future)
    """
    W = len(values)
    u = np.arange(-(W - 1), 1)
    (a, b, c), cov = np.polyfit(u, values, 2, cov=True)

    se_a = np.sqrt(cov[0, 0])
    t_stat = a / se_a if se_a > 0 else 0.0
    vertex = -b / (2 * a) if a != 0 else np.nan
    return t_stat, vertex, a


# %% ---------------------------------------------------------------------
# 3. CALIBRATING THE SIGNIFICANCE THRESHOLD
# ------------------------------------------------------------------------
def calibrate_threshold(log_steps, W, q=95, n_sims=3000, seed=0):
    """What does |t| look like on paths with NO real curvature?
    Build fake paths by resampling the real daily changes, fit the same
    quadratic, and return the q-th percentile of |t|.
    """
    rng = np.random.default_rng(seed)
    abs_t = np.empty(n_sims)
    for i in range(n_sims):
        fake_path = np.cumsum(rng.choice(log_steps, size=W, replace=True))
        t, _, _ = window_stats(fake_path)
        abs_t[i] = abs(t)
    return float(np.percentile(abs_t, q))


# %% ---------------------------------------------------------------------
# 4. SIGNALS  (one row per day)
# ------------------------------------------------------------------------
def compute_signals(log_ratio, W, thr, m, start, end=None):
    """For each day i in [start, end): fit the last W days (ending TODAY, i).
    The gate fires if  |t| >= thr  AND  -m <= vertex <= 0.
    Target is SPY if a > 0 (a minimum, ratio now rising), else GLD.
    """
    end = len(log_ratio) if end is None else end
    assert start >= W - 1, "start must be at least W-1 so the first window is full"

    vals = log_ratio.values
    rows = []
    for i in range(start, end):
        window = vals[i - W + 1: i + 1]
        t, vertex, a = window_stats(window)

        # comparisons with NaN are False, so a == 0 never fires
        fire = bool(abs(t) >= thr and (-m <= vertex <= 0))
        target = ("SPY" if a > 0 else "GLD") if fire else None
        rows.append((log_ratio.index[i], t, vertex, a, fire, target))

    return pd.DataFrame(rows, columns=["date", "t", "vertex", "a", "fire", "target"]).set_index("date")


# %% ---------------------------------------------------------------------
# 5. PORTFOLIO SIMULATION  (fresh Portfolio objects every call)
# ------------------------------------------------------------------------
def portfolio_value(p, px):
    held = p.get_current_stock()
    if held is None:
        return p.get_balance()
    return p.get_portfolio_value(held, px[held])


def simulate(signals, data, cooldown=10, fee=0.0015, initial=1000,
             start_in_spy=True, randomise_seed=None):
    """Walks day by day through `signals`.
    - Signal seen at today's close  ->  trade executes at TOMORROW's close.
    - Only trades if the target differs from what is held.
    - `cooldown` = minimum days between trades.
    - Baseline (buy & hold SPY) and profile BOTH start on the first date.
    - randomise_seed: if given, replaces every fired direction with a coin flip
      (used for the random-control test).
    """
    profile = Portfolio(fx_fee_rate=fee, initial_balance=initial)
    baseline = Portfolio(fx_fee_rate=fee, initial_balance=initial)

    dates = signals.index
    px_df = data.loc[dates]
    spy_arr, gld_arr, fx_arr = px_df["SPY"].values, px_df["GLD"].values, px_df["GBPUSD"].values

    targets = signals["target"].tolist()
    if randomise_seed is not None:
        rng = np.random.default_rng(randomise_seed)
        targets = [rng.choice(["SPY", "GLD"]) if t is not None else None for t in targets]

    pending = None
    last_trade_i = -10 ** 9
    prof_vals, base_vals, held_log, trades = [], [], [], []

    for i, date in enumerate(dates):
        px = {"SPY": spy_arr[i], "GLD": gld_arr[i]}
        profile.update_conversion(float(fx_arr[i]))
        baseline.update_conversion(float(fx_arr[i]))

        if i == 0:
            baseline.buy(px["SPY"], "SPY")
            if start_in_spy:
                profile.buy(px["SPY"], "SPY")

        # (a) execute yesterday's signal at today's close
        if pending is not None:
            if pending != profile.get_current_stock():
                other = "GLD" if pending == "SPY" else "SPY"
                profile.sell(px[other], other)
                profile.buy(px[pending], pending)
                trades.append((date, pending))
                last_trade_i = i
            pending = None

        # (b) queue a new signal from today's close
        target = targets[i]
        if (target is not None
                and target != profile.get_current_stock()
                and i - last_trade_i >= cooldown):
            pending = target

        prof_vals.append(portfolio_value(profile, px))
        base_vals.append(portfolio_value(baseline, px))
        held_log.append(profile.get_current_stock())

    result = pd.DataFrame({"profile": prof_vals, "baseline": base_vals, "held": held_log}, index=dates)
    trades = pd.DataFrame(trades, columns=["date", "bought"]).set_index("date")
    return result, trades


# %% ---------------------------------------------------------------------
# 6. EVALUATION HELPERS
# ------------------------------------------------------------------------
def summarise(result, trades):
    out = {}
    for name in ("profile", "baseline"):
        v = result[name]
        out[name] = {
            "final_GBP": round(float(v.iloc[-1]), 2),
            "return_%": round(100 * (v.iloc[-1] / v.iloc[0] - 1), 1),
            "max_drawdown_%": round(100 * float((v / v.cummax() - 1).min()), 1),
        }
    out["trades"] = len(trades)
    return out


def hit_rate(signals, log_ratio, horizon=15):
    """After each fired signal, did the ratio move the predicted way over the
    next `horizon` days?  (Measured from the execution day, i+1.)"""
    fired = signals[signals["fire"]]
    pos = log_ratio.index.get_indexer(fired.index)
    ok = pos + 1 + horizon < len(log_ratio)
    pos, fired = pos[ok], fired[ok]
    if len(pos) == 0:
        return np.nan, 0
    change = log_ratio.values[pos + 1 + horizon] - log_ratio.values[pos + 1]
    hits = (change > 0) == (fired["target"] == "SPY").values
    return float(np.mean(hits)), int(len(pos))


def random_control(signals, data, n=100, **sim_kwargs):
    """Same fire days, random directions. Returns the final profile values."""
    finals = []
    for seed in range(n):
        res, _ = simulate(signals, data, randomise_seed=seed, **sim_kwargs)
        finals.append(res["profile"].iloc[-1])
    return np.array(finals)


# %% ---------------------------------------------------------------------
# 7. TUNING ON THE TRAINING SLICE ONLY
# ------------------------------------------------------------------------
def tune(log_ratio, data, grid, train_start, train_end, cooldown=10):
    """grid = list of (W, frac, q). Uses ONLY rows < train_end."""
    train_lr = log_ratio.iloc[:train_end]
    steps = np.diff(train_lr.values)
    thr_cache, rows = {}, []

    for W, frac, q in grid:
        if (W, q) not in thr_cache:
            thr_cache[(W, q)] = calibrate_threshold(steps, W, q)
        thr = thr_cache[(W, q)]

        sig = compute_signals(train_lr, W, thr, m=round(W * frac), start=train_start, end=train_end)
        res, trades = simulate(sig, data, cooldown=cooldown)
        hr, n_hr = hit_rate(sig, train_lr)
        rows.append({
            "W": W, "frac": frac, "q": q, "thr": round(thr, 1),
            "signal_days": int(sig["fire"].sum()), "trades": len(trades),
            "hit_rate": round(hr, 2) if n_hr else np.nan,
            "strategy_GBP": round(res["profile"].iloc[-1], 1),
            "baseline_GBP": round(res["baseline"].iloc[-1], 1),
        })
    return pd.DataFrame(rows)


# %% ---------------------------------------------------------------------
# 8. PLOTS
# ------------------------------------------------------------------------
def plot_results(signals, result, trades, thr, title=""):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})

    ax = axes[0]
    ax.plot(result.index, result["profile"], label="Strategy (GBP)")
    ax.plot(result.index, result["baseline"], label="Buy & hold SPY (GBP)", alpha=0.8)
    if not trades.empty:
        tv = result.loc[trades.index, "profile"]
        buys_spy = trades["bought"] == "SPY"
        ax.scatter(tv.index[buys_spy], tv[buys_spy], marker="^", color="tab:green", s=50, zorder=3, label="Switch to SPY")
        ax.scatter(tv.index[~buys_spy], tv[~buys_spy], marker="v", color="goldenrod", s=50, zorder=3, label="Switch to GLD")
    ax.set_ylabel("Portfolio value (GBP)")
    ax.set_title(title or "Strategy vs buy & hold")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(signals.index, signals["t"], color="grey", lw=0.7, label="t-stat of curvature")
    ax.axhline(thr, color="red", ls="--", lw=0.8, label=f"+/- threshold ({thr:.1f})")
    ax.axhline(-thr, color="red", ls="--", lw=0.8)
    fired = signals[signals["fire"]]
    ax.scatter(fired.index, fired["t"], c=fired["target"].map({"SPY": "tab:green", "GLD": "goldenrod"}), s=25, zorder=3)
    ax.set_ylabel("t-statistic")
    ax.legend(loc="upper left")

    fig.tight_layout()
    plt.show()


# %% ---------------------------------------------------------------------
# 9. RUN IT
# ------------------------------------------------------------------------
log_ratio, data = get_data()
N = len(log_ratio)

TRAIN_START = 250                  # first day of the training backtest
TRAIN_END = int(N * 0.6)           # tune on [TRAIN_START, TRAIN_END), test on [TRAIN_END, N)

# %% 9a. TUNE (training slice only). Look at the table, then pick ONE row.
grid = [(W, frac, q) for W in (20, 30, 45, 60) for frac in (0.25, 0.5) for q in (90, 95)]
table = tune(log_ratio, data, grid, TRAIN_START, TRAIN_END)
print(table.sort_values("strategy_GBP", ascending=False).to_string(index=False))

# %% 9b. FINAL TEST: run ONCE on the untouched slice with the parameters you chose
W, FRAC, Q = 30, 0.25, 95          # <- replace with your chosen row from the table

steps = np.diff(log_ratio.values[:TRAIN_END])          # training data only
thr = calibrate_threshold(steps, W, Q)
print("Calibrated threshold:", round(thr, 2))

signals = compute_signals(log_ratio, W, thr, m=round(W * FRAC), start=TRAIN_END)
result, trades = simulate(signals, data)

print(summarise(result, trades))
hr, n_hr = hit_rate(signals, log_ratio)
print(f"Hit rate over next 15 days: {hr:.2f}  ({n_hr} signal days)")

plot_results(signals, result, trades, thr, title=f"Test slice  (W={W}, frac={FRAC}, q={Q})")

# %% 9c. RANDOM CONTROL: is the strategy better than coin-flip directions?
finals = random_control(signals, data, n=100)
mine = result["profile"].iloc[-1]
print(f"Strategy final: {mine:.1f} GBP")
print(f"Random-direction runs: median {np.median(finals):.1f}, 5th-95th pct "
      f"{np.percentile(finals, 5):.1f} to {np.percentile(finals, 95):.1f}")
print(f"Strategy beat {np.mean(mine > finals):.0%} of random runs")