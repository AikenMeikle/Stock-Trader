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

RUN_SENSITIVITY = True     # re-runs the backtest over a 3x3x3 grid plus noise/control checks (a few minutes)
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
    if result.target_signal_rate is not None:
        out(f"Joint calibration: target {100 * result.target_signal_rate:.2f}% of bars "
            f"(the FULL rule, not just |t|)")
    out(f"Warm-up bars (calibration only): {result.warmup_bars}")
    out(f"Traded {result.dates[0].date()} -> {result.dates[-1].date()} ({len(result.dates)} bars)")
    out()
    out(ev.summarise(result).round(1).to_string())

    diag = ev.signal_diagnostics(result)
    out()
    out("Signal diagnostics (why this many trades?):")
    out(f"  Bars with |t| >= {result.t_threshold:.2f}: {diag['significant_bars']:.0f} "
        f"({diag['significant_rate_pct']:.2f}% of {diag['bars']:.0f})")
    out(f"  Bars with a vertex within {result.max_age} bars: {diag['recent_vertex_bars']:.0f} "
        f"({diag['recent_vertex_rate_pct']:.2f}%)")
    out(f"  Bars passing BOTH gates: {diag['joint_bars']:.0f} ({diag['joint_rate_pct']:.3f}%)")
    out(f"  Luck alone passes both at {diag['noise_joint_rate_pct']:.3f}% "
        f"(~{diag['noise_joint_events']:.1f} events over this period)")
    out(f"  Median vertex when significant: {diag['median_vertex_when_significant']:.1f} bars "
        f"(the gate asks for <= {result.max_age})")
    out(f"  Share of significant bars that also pass the age gate: "
        f"{diag['share_significant_with_recent_vertex_pct']:.1f}%")

    out()
    out(f"In-sample / out-of-sample (pre-registered split {ev.SPLIT_DATE.date()}):")
    in_sample = ev.in_sample_metrics(result)
    out_sample = ev.out_of_sample_metrics(result)
    out(f"  {'':14s}{'Return':>9s}{'Baseline':>10s}{'Max DD':>9s}{'Trades':>8s}")
    for name, metrics in (("In-sample", in_sample), ("Out-of-sample", out_sample)):
        out(f"  {name:14s}{metrics['strategy_return_pct']:+8.1f}%"
            f"{metrics['baseline_return_pct']:+9.1f}%"
            f"{metrics['strategy_max_drawdown_pct']:8.1f}%{metrics['trades']:8d}")
    out("  The split is fixed before looking at results; the defaults were chosen "
        "structurally, not by IS return ranking.")

    out()
    if len(result.trades) >= ev.MIN_TRADES_FOR_STATS:
        hit, n = ev.hit_rate(result)
        out(f"Hit rate: the ratio moved the expected way over the next 15 bars "
            f"after {hit:.0%} of {n} trades")
    else:
        out(f"Hit rate: suppressed — {len(result.trades)} trades is below the "
            f"{ev.MIN_TRADES_FOR_STATS} needed for a meaningful read")

    finals = ev.shifted_timeline_control(result)
    if len(finals):
        weak = "" if len(result.trades) >= ev.MIN_TRADES_FOR_STATS else " (weak: very few switches)"
        out(f"Shifted-timeline control: strategy beat {(result.strategy_values[-1] > finals).mean():.0%} "
            f"of {len(finals)} runs that spend the same time in cash/SPY/GLD at the wrong times "
            f"(median {sorted(finals)[len(finals) // 2]:,.0f} GBP){weak}")

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
        table = ev.sensitivity_table(data)
        out(table.to_string(index=False))

        out()
        out("Pre-registered selection check (choose on IS only, then evaluate once on OOS):")
        best = ev.best_in_sample_cell(table)
        out(f"  Neighbour-robust IS pick: window {int(best['window'])}, "
            f"quantile {best['quantile']:g}, age fraction {best['age_fraction']:g} "
            f"(max age {int(best['max_age'])})")
        out(f"    IS {best['IS_return_pct']:+.1f}% vs baseline {best['IS_baseline_pct']:+.1f}% "
            f"({int(best['IS_trades'])} trades); 3x3 IS neighbourhood median "
            f"{best['IS_neighbour_median']:+.1f}%")
        verdict = "beat" if best["OOS_beats_baseline"] else "did NOT beat"
        out(f"    OOS {best['OOS_return_pct']:+.1f}% vs baseline {best['OOS_baseline_pct']:+.1f}% "
            f"({int(best['OOS_trades'])} trades) -> {verdict} the baseline")
        valid = table[table["OOS_beats_baseline"].notna()]
        n_beat = int((valid["OOS_beats_baseline"] == True).sum())     # noqa: E712
        out(f"  {n_beat} of {len(valid)} settings beat the baseline out of sample "
            f"(baseline OOS {out_sample['baseline_return_pct']:+.1f}%).")

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