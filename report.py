"""Run the strategy and produce the full report:   python report.py

Everything is printed AND saved to the report_output/ folder:
    summary.txt            all the printed numbers
    1_values.png           strategy vs baseline portfolio value
    2_flips.png            log(SPY/GLD) with every switch marked
    3_shifted_control.png  strategy vs shifted-timeline versions of itself
    4_annual_returns.png   trailing one-year return distributions (strategy/baseline/active)
    5_mc_null.png          full-pipeline Monte Carlo null vs the real excess return
"""
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np

import smooth_graph.evaluation as ev
from smooth_graph.backtest import run_backtest
from smooth_graph.data import load_price_data

RUN_SENSITIVITY = True     # prints the 3x3x3 grid and selection check; the grid is computed for the validation verdict either way
MC_JOBS = max(1, min(4, os.cpu_count() or 1))
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
    if result.fallback:
        out(f"Exit rule: rotate a stale position to {result.fallback} when the fitted "
            f"vertex leaves [-{result.max_age}, 0]")
    else:
        out("Exit rule: disabled (hold until the opposite signal)")
    if result.target_signal_rate is not None:
        out(f"Joint calibration: target {100 * result.target_signal_rate:.2f}% of bars "
            f"(the FULL rule, not just |t|)")
    out(f"Warm-up bars (calibration only): {result.warmup_bars}")
    out(f"Traded {result.dates[0].date()} -> {result.dates[-1].date()} ({len(result.dates)} bars)")
    out()
    out(ev.summarise(result).round(1).to_string())

    periods = ev.holding_periods(result)
    if len(periods):
        longest = periods.loc[periods["bars"].idxmax()]
        out()
        out(f"Longest holding period: {longest['holding']} for {int(longest['bars'])} bars "
            f"({longest['start'].date()} -> {longest['end'].date()})")
        if result.fallback:
            invested = periods[~periods["holding"].isin([result.fallback, "CASH"])]
            if len(invested):
                l = invested.loc[invested["bars"].idxmax()]
                out(f"Longest non-{result.fallback} invested holding: {l['holding']} for "
                    f"{int(l['bars'])} bars ({l['start'].date()} -> {l['end'].date()})")

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

    detours = ev.detour_attribution(result)
    out()
    out("Detour attribution (every non-SPY holding spell; GLD spells are the detours):")
    if len(detours["rows"]):
        view = detours["rows"].copy()
        view["entry_date"] = view["entry_date"].dt.date
        view["exit_date"] = view["exit_date"].dt.date
        view["exit_execution_date"] = view["exit_execution_date"].dt.date
        for column in ("ratio_log_change", "gross_excess_return", "switch_cost",
                       "net_excess_return"):
            view[column] = (100 * view[column]).round(2)
        view["notional_gbp"] = view["notional_gbp"].round(0)
        view["net_excess_gbp"] = view["net_excess_gbp"].round(0)
        out(view[["holding", "entry_date", "exit_date", "exit_execution_date", "bars",
                  "ratio_log_change", "gross_excess_return", "switch_cost",
                  "net_excess_return", "notional_gbp", "net_excess_gbp"]].to_string(index=False))
    else:
        out("  none")
    out(f"  Spells: {detours['count']} ({detours['gld_count']} GLD, "
        f"{detours['cash_count']} cash); time in GLD {detours['share_time_in_gld_pct']:.1f}%")
    out(f"  GLD detours: mean net excess {100 * detours['mean_net_excess']:+.2f}%, "
        f"median {100 * detours['median_net_excess']:+.2f}%, "
        f"share positive {100 * detours['share_positive']:.1f}%")
    ci_low, ci_high = detours["bootstrap_ci"]
    out(f"  {ev.DETOUR_CI_PCT:g}% bootstrap CI of mean net excess: "
        f"[{100 * ci_low:+.2f}%, {100 * ci_high:+.2f}%]")
    ins, oos = detours["in_sample"], detours["out_of_sample"]
    out(f"  In-sample: {ins['count']} detours, mean {100 * ins['mean_net_excess']:+.2f}%, "
        f"median {100 * ins['median_net_excess']:+.2f}%, share positive "
        f"{100 * ins['share_positive']:.1f}%")
    out(f"  Out-of-sample: {oos['count']} detours, mean {100 * oos['mean_net_excess']:+.2f}%, "
        f"median {100 * oos['median_net_excess']:+.2f}%, share positive "
        f"{100 * oos['share_positive']:.1f}%")

    exact = detours["exact"]
    out()
    out("Exact bar-by-bar attribution (active log return = log(strategy_t/strategy_{t-1}) -")
    out("log(baseline_t/baseline_{t-1}); grouped by the asset held on bar t-1, so a switch")
    out("at bar t is charged to the spell being left and the exit swap bar belongs to it):")
    agg = exact["by_holding"].copy()
    if len(agg):
        agg_view = agg.copy()
        for column in ("active_log_return", "gross_log_return", "cost_log_return"):
            agg_view[column] = (100 * agg_view[column]).round(3)
        agg_view["gbp_effect_to_end"] = agg_view["gbp_effect_to_end"].round(0)
        agg_view = agg_view.rename(columns={
            "holding": "holding", "spells": "spells", "bars": "bars",
            "active_log_return": "active_log_%", "gross_log_return": "gross_log_%",
            "cost_log_return": "cost_log_%", "gbp_effect_to_end": "GBP effect to end",
        })
        out(agg_view.to_string(index=False))
    out(f"  First-bar entry mismatch log(strategy0/baseline0): "
        f"{exact['first_bar_log_mismatch']:+.9f} "
        f"({exact['first_bar_gbp_effect']:+,.0f} GBP grown to end)")
    out(f"  Sum of spell active log returns: {exact['sum_active_log_return']:+.9f}")
    out(f"  Sum + first-bar mismatch: "
        f"{exact['sum_active_log_return'] + exact['first_bar_log_mismatch']:+.9f}")
    out(f"  log(final strategy / final baseline): {exact['log_final_ratio']:+.9f}")
    out(f"  Reconciled: {'yes' if exact['reconciled'] else 'no'} "
        f"(max abs error {exact['max_abs_log_error']:.1e})")

    gld_exact = exact["rows"][exact["rows"]["holding"] == "GLD"].copy()
    if len(gld_exact):
        gv = gld_exact.copy()
        gv["entry_date"] = gv["entry_date"].dt.date
        gv["last_held_date"] = gv["last_held_date"].dt.date
        gv["exit_execution_date"] = gv["exit_execution_date"].dt.date
        for column in ("active_log_return", "gross_log_return", "cost_log_return"):
            gv[column] = (100 * gv[column]).round(2)
        gv["gbp_effect_to_end"] = gv["gbp_effect_to_end"].round(0)
        gv = gv[["entry_date", "last_held_date", "exit_execution_date", "bars",
                 "active_log_return", "gross_log_return", "cost_log_return",
                 "gbp_effect_to_end"]]
        gv.columns = ["entry", "last_held", "exit_exec", "bars", "active_log_%",
                      "gross_log_%", "cost_log_%", "GBP effect to end"]
        out("  GLD spells (exact, including the exit swap bar):")
        out(gv.to_string(index=False))

    spy = agg[agg["holding"] == "SPY"] if len(agg) else agg
    if len(spy):
        row = spy.iloc[0]
        out(f"  SPY spells: {int(row['spells'])} spells, {int(row['bars'])} bars, "
            f"active log return {100 * row['active_log_return']:+.3f}% "
            f"(gross SPY-vs-SPY {100 * row['gross_log_return']:+.3f}%, switch costs "
            f"{100 * row['cost_log_return']:+.3f}%), GBP effect to end "
            f"{row['gbp_effect_to_end']:+,.0f}")
        if abs(row["gross_log_return"]) > 1e-9:
            out("  NOTE: SPY gross active return is not zero; something other than "
                "switch costs affected the SPY spells.")

    if len(gld_exact):
        top = gld_exact.reindex(
            gld_exact["active_log_return"].abs().sort_values(ascending=False).index
        ).head(3)
        out("  Leave-one-out (neutralise one GLD spell's exact active log return):")
        for _, row in top.iterrows():
            new_final = exact["strategy_final"] * np.exp(-row["active_log_return"])
            out(f"    {row['entry_date'].date()} -> {row['exit_execution_date'].date()}: "
                f"active {100 * row['active_log_return']:+.2f}%, final "
                f"{new_final:,.0f} GBP, lead {new_final - exact['baseline_final']:+,.0f} GBP")

    stats = ev.annual_return_statistics(result)
    out()
    out("Annual return distribution (trailing one-year return on every traded bar "
        "with a full year behind it):")
    if len(stats) == 0 or int(stats["n"].max()) == 0:
        out(f"  fewer than {ev.BARS_PER_YEAR + 1} traded bars: cannot compute a "
            f"trailing one-year return.")
    else:
        view = stats[["n", "mean_pct", "median_pct", "sd_pct", "skewness",
                      "excess_kurtosis", "p5_pct", "worst_pct", "best_pct",
                      "share_negative_pct"]]
        out(view.round(2).to_string())
        for name, row in stats.iterrows():
            if ((np.isfinite(row["skewness"]) and row["skewness"] < -0.5)
                    or (np.isfinite(row["excess_kurtosis"]) and row["excess_kurtosis"] > 1)):
                out(f"  NOTE: {name} skewness {row['skewness']:+.2f}, excess kurtosis "
                    f"{row['excess_kurtosis']:+.2f}; the normal curve understates the "
                    f"left tail.")
    out("  " + ev.OVERLAP_WARNING)

    cal = ev.calendar_year_returns(result)
    out()
    out("Calendar-year returns (partial first/last years marked and excluded from "
        "the statistics):")
    if len(cal):
        cal_view = cal.copy()
        for column in ("strategy_pct", "baseline_pct", "active_pp"):
            cal_view[column] = cal_view[column].round(1)
        out(cal_view.to_string(index=False))
        full_years = cal[~cal["partial"]]
        if len(full_years):
            out(f"  Full-year means (excluding partial years): strategy "
                f"{full_years['strategy_pct'].mean():+.1f}%, baseline "
                f"{full_years['baseline_pct'].mean():+.1f}%, active "
                f"{full_years['active_pp'].mean():+.1f} pp")
        else:
            out("  No full calendar years; statistics not computed.")
    else:
        out("  no traded bars")

    te = ev.tracking_error_stats(result)
    out()
    out("Tracking error (strategy minus baseline daily returns):")
    out(f"  Annualised active return {te['annualised_active_return_pct']:+.2f}%; "
        f"annualised tracking error {te['annualised_tracking_error_pct']:.2f}%; "
        f"information ratio {te['information_ratio']:.2f}")
    if np.isfinite(te["years_to_detect_2se"]):
        out(f"  Years needed to detect the observed edge at 2 standard errors: "
            f"{te['years_to_detect_2se']:.1f}; years of data available: "
            f"{te['years_available']:.1f} -> "
            f"{'detectable' if te['detectable_within_history'] else 'NOT detectable'}")
    else:
        out(f"  Years needed to detect the observed edge at 2 standard errors: n/a "
            f"(no positive active return); years of data available: {te['years_available']:.1f}")

    friction = ev.friction_sensitivity(data, result=result)
    out()
    out("Friction sensitivity (extra cost per swap leg; default FX fee unchanged):")
    ftable = friction["table"].copy()
    for column in ("strategy_final", "baseline_final", "lead_gbp"):
        ftable[column] = ftable[column].round(0)
    out(ftable.to_string(index=False))
    if friction["break_even_extra_friction"] is not None:
        out(f"  Break-even extra friction: "
            f"{100 * friction['break_even_extra_friction']:.2f} percentage points per leg "
            f"(total FX fee {100 * friction['break_even_total_fee']:.2f}%)")
    else:
        out(f"  Break-even extra friction: none in the grid — {friction['note']}")

    fwd = ev.forward_metrics(result)
    out()
    out(f"Clean holdout from {ev.CLEAN_HOLDOUT_START.date()} "
        f"(data before it informed the design):")
    if not fwd["available"]:
        out(f"  no clean data yet (last traded bar "
            f"{result.dates[-1].date() if len(result.dates) else 'n/a'})")
    else:
        out(f"  {fwd['bars']} bars {fwd['start'].date()} -> {fwd['end'].date()}: "
            f"strategy {fwd['strategy_return_pct']:+.1f}% vs baseline "
            f"{fwd['baseline_return_pct']:+.1f}% ({fwd['trades']} trades)")

    out()
    out("Trades (execution date, ticker bought):")
    out(result.trades.to_string() if len(result.trades) else "  none")

    table = ev.sensitivity_table(data)
    mc = ev.full_pipeline_monte_carlo(result, data, n_sims=ev.MC_SIMS,
                                      block_length=ev.MC_BLOCK_BARS, seed=ev.MC_SEED,
                                      n_jobs=MC_JOBS)
    out()
    out(f"Full-pipeline Monte Carlo null ({mc['n_sims']} synthetic histories, "
        f"joint block bootstrap of {mc['block_length']} bars):")
    out(f"  Real annualised excess {mc['real_annualised_excess_pct']:+.2f}%; "
        f"null median {mc['null_median_pct']:+.2f}%, "
        f"{ev.MC_PERCENTILE:g}th percentile {mc['null_percentile_pct']:+.2f}%")
    out(f"  Null trades median {mc['null_trades_median']:.0f} "
        f"[{mc['null_trades_p5']:.0f}-{mc['null_trades_p95']:.0f}] (5th-95th); "
        f"null max drawdown median {mc['null_max_drawdown_median_pct']:.1f}% "
        f"(real: {result.trade_count} trades, drawdown "
        f"{100 * ev.max_drawdown(result.strategy_values):.1f}%)")
    out(f"  Real result at percentile {mc['percentile']:.1f}; one-sided p-value "
        f"{mc['p_value']:.4f} -> {'PASS' if mc['passed'] else 'FAIL'}")

    if RUN_SENSITIVITY:
        out()
        out("Sensitivity (judge the pattern, not the best row):")
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

    verdict = ev.validation_verdict(result, data, table=table, mc=mc, detours=detours,
                                    n_jobs=MC_JOBS)
    out()
    out("Validation verdict (pre-registered; ALL four checks must pass):")
    for check in verdict["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        out(f"  [{check['number']}] {status}  {check['name']}: {check['detail']}")
    out(f"  OVERALL: {'VALIDATED' if verdict['validated'] else 'NOT VALIDATED'}")

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    figures = {
        "1_values.png": ev.plot_values(result),
        "2_flips.png": ev.plot_log_ratio_with_flips(result),
        "3_shifted_control.png": ev.plot_control(result, finals),
        "4_annual_returns.png": ev.plot_annual_returns(result),
        "5_mc_null.png": ev.plot_mc_null(mc),
    }
    for name, fig in figures.items():
        fig.savefig(OUTPUT_DIR / name, dpi=120)
    print(f"\nSaved summary.txt and {len(figures)} charts to {OUTPUT_DIR.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()