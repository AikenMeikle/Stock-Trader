"""Run the strategy and produce the full report:   python report.py

Everything is printed AND saved to the report_output/ folder:
    summary.txt            all the printed numbers
    1_values.png           strategy vs baseline portfolio value
    2_flips.png            log(SPY/GLD) with every switch marked
    3_shifted_control.png  strategy vs shifted-timeline versions of itself
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import smooth_graph.evaluation as ev
from smooth_graph.backtest import run_backtest
from smooth_graph.data import load_price_data

RUN_SENSITIVITY = True     # re-runs the backtest ~18 times; takes about a minute
OUTPUT_DIR = Path("report_output")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    lines = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    data = load_price_data()
    result = run_backtest(data=data)

    out(f"Window {result.window_bars} bars | quantile {result.t_quantile} | "
        f"max age {result.max_age} | calibrated t-threshold {result.t_threshold:.2f}")
    out(f"Warm-up bars (calibration only): {result.warmup_bars}")
    out(f"Traded {result.dates[0].date()} -> {result.dates[-1].date()} ({len(result.dates)} bars)")
    out()
    out(ev.summarise(result).round(1).to_string())

    hit, n = ev.hit_rate(result)
    out()
    out(f"Hit rate: the ratio moved the expected way over the next 15 bars after {hit:.0%} of {n} trades")

    finals = ev.shifted_timeline_control(result)
    if len(finals):
        out(f"Shifted-timeline control: strategy beat {(result.strategy_values[-1] > finals).mean():.0%} "
            f"of {len(finals)} runs that spend the same time in cash/SPY/GLD at the wrong times "
            f"(median {sorted(finals)[len(finals) // 2]:,.0f} GBP)")

    noise = ev.noise_trade_counts(result)
    low, median, high = (int(x) for x in np.percentile(noise, [5, 50, 95]))
    out(f"Noise check: real data made {result.trade_count} trades; the same rules on shuffled "
        f"versions of your own daily changes make a median of {median} [{low}-{high}] (5th-95th percentile)")

    out()
    out("Trades (execution date, ticker bought):")
    out(result.trades.to_string() if len(result.trades) else "  none")

    if RUN_SENSITIVITY:
        out()
        out("Sensitivity (judge the pattern, not the best row):")
        out(ev.sensitivity_table(data).to_string(index=False))

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    figures = {
        "1_values.png": ev.plot_values(result),
        "2_flips.png": ev.plot_log_ratio_with_flips(result),
        "3_shifted_control.png": ev.plot_control(result, finals),
    }
    for name, fig in figures.items():
        fig.savefig(OUTPUT_DIR / name, dpi=120)
    print(f"\nSaved summary.txt and {len(figures)} charts to {OUTPUT_DIR.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()