import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals

import ManualStrategy as ms
import StrategyLearner as sl

def author():
    return 'bflores9'

def plot(symbol="JPM"):
    # Variables
    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)
    sv_in = 100000

    sd_out = dt.datetime(2010, 1, 1)
    ed_out = dt.datetime(2011, 12, 31)
    sv_out = 100000

    commission = 9.95
    impact = 0.005

    # In sample
    manual = ms.ManualStrategy()
    strategy = sl.StrategyLearner(impact=impact, commission=commission)
    strategy.add_evidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv_in)

    manual_trades = manual.testPolicy(sd=sd_in, ed=ed_in, sv=sv_in).to_frame(name=symbol)
    strategy_trades = strategy.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv_in)
    benchmark_trades = manual_trades * 0.0
    benchmark_trades.iloc[0] = 1000.0

    manual_portvals = compute_portvals(manual_trades, start_val=sv_in, commission=commission, impact=impact)
    strategy_portvals = compute_portvals(strategy_trades, start_val=sv_in, commission=commission, impact=impact)
    benchmark_portvals = compute_portvals(benchmark_trades, start_val=sv_in, commission=commission, impact=impact)

    # Normalize
    manual_portvals = manual_portvals / manual_portvals.iloc[0]
    strategy_portvals = strategy_portvals / strategy_portvals.iloc[0]
    benchmark_portvals = benchmark_portvals / benchmark_portvals.iloc[0]

    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(manual_portvals, label="Manual Strategy")
    plt.plot(strategy_portvals, label="Strategy Learner")
    plt.plot(benchmark_portvals, label="Benchmark")
    plt.title(f"Manual Strategy vs. Strategy Learner vs. Benchmark Portfolio Value for ${symbol} (In-Sample)")
    plt.xlabel("Dates")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f"images/exp_1_in_sample.png")

    plt.clf()

    # Out of sample
    # manual = ms.ManualStrategy()
    # strategy = sl.StrategyLearner(impact=impact, commission=commission)
    # strategy.add_evidence(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv_out)

    manual_trades = manual.testPolicy(sd=sd_out, ed=ed_out, sv=sv_out).to_frame(name=symbol)
    strategy_trades = strategy.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv_out)
    benchmark_trades = manual_trades * 0.0
    benchmark_trades.iloc[0] = 1000.0

    manual_portvals = compute_portvals(manual_trades, start_val=sv_out, commission=commission, impact=impact)
    strategy_portvals = compute_portvals(strategy_trades, start_val=sv_out, commission=commission, impact=impact)
    benchmark_portvals = compute_portvals(benchmark_trades, start_val=sv_out, commission=commission, impact=impact)

    # Normalize
    manual_portvals = manual_portvals / manual_portvals.iloc[0]
    strategy_portvals = strategy_portvals / strategy_portvals.iloc[0]
    benchmark_portvals = benchmark_portvals / benchmark_portvals.iloc[0]

    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(manual_portvals, label="Manual Strategy")
    plt.plot(strategy_portvals, label="Strategy Learner")
    plt.plot(benchmark_portvals, label="Benchmark")
    plt.title(f"Manual Strategy vs. Strategy Learner vs. Benchmark Portfolio Value for ${symbol} Out-of-Sample")
    plt.xlabel("Dates")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f"images/exp_1_out_sample.png")
    plt.clf()

if __name__ == "__main__":
    plot()

