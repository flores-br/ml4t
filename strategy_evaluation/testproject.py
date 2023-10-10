import pandas as pd

import ManualStrategy as ms
import StrategyLearner as sl
from marketsimcode import compute_portvals
import experiment1 as exp1
import experiment2 as exp2
from matplotlib import pyplot as plt
import datetime as dt

def plot_manual_benchmark_stats(symbol="JPM"):
    # In sample
    strategy = ms.ManualStrategy()
    df_trades = strategy.testPolicy().to_frame(name=symbol)
    benchmark_trades = df_trades * 0.0
    benchmark_trades.iloc[0] = 1000.0
    portvals = compute_portvals(df_trades, start_val=100000, commission=9.95, impact=0.005)
    benchmark_portvals = compute_portvals(benchmark_trades, start_val=100000, commission=9.95, impact=0.005)

    # Normalize
    portvals = portvals / portvals.iloc[0]
    benchmark_portvals = benchmark_portvals / benchmark_portvals.iloc[0]

    # Print stats
    cr = portvals["value"].iloc[-1] / portvals["value"].iloc[0] - 1
    benchmark_cr = benchmark_portvals["value"].iloc[-1] / benchmark_portvals["value"].iloc[0] - 1
    dr = portvals["value"].pct_change()
    benchmark_dr = benchmark_portvals["value"].pct_change()
    std = dr.std()
    benchmark_std = benchmark_dr.std()
    avg = dr.mean()
    benchmark_avg = benchmark_dr.mean()
    d = {
        "cumulative_ret": [cr, benchmark_cr],
        "std": [std, benchmark_std],
        "avg": [avg, benchmark_avg]
    }

    df = pd.DataFrame(d, index=["manual", "benchmark"])
    df = df.round(6)
    print(f"\nManual Strategy In-Sample Stats for ${symbol}\n")
    print(df, "\n")

    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(portvals, label="Manual Strategy", color="red")
    plt.plot(benchmark_portvals, label="Benchmark", color="purple")
    plt.title(f"Manual Strategy vs. Benchmark Portfolio Value for ${symbol} (In-Sample)")
    plt.xlabel("Dates")
    plt.ylabel("Value")
    plt.legend()

    # Plot trade entries
    for index, row in df_trades.iterrows():
        if row[symbol] == 1000:
            plt.axvline(x=index, color="blue", linestyle="--")
        elif row[symbol] == -1000:
            plt.axvline(x=index, color="black", linestyle="--")

    plt.savefig(f"images/manual_strategy_{symbol}_in_sample.png")

    # Out of sample
    strategy = ms.ManualStrategy()
    df_trades = strategy.testPolicy(sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31)).to_frame(
        name=symbol)
    benchmark_trades = df_trades * 0.0
    benchmark_trades.iloc[0] = 1000.0
    portvals = compute_portvals(df_trades, start_val=100000, commission=9.95, impact=0.005)
    benchmark_portvals = compute_portvals(benchmark_trades, start_val=100000, commission=9.95, impact=0.005)

    # Normalize
    portvals = portvals / portvals.iloc[0]
    benchmark_portvals = benchmark_portvals / benchmark_portvals.iloc[0]

    # Print stats
    cr = portvals["value"].iloc[-1] / portvals["value"].iloc[0] - 1
    benchmark_cr = benchmark_portvals["value"].iloc[-1] / benchmark_portvals["value"].iloc[0] - 1
    dr = portvals["value"].pct_change()
    benchmark_dr = benchmark_portvals["value"].pct_change()
    std = dr.std()
    benchmark_std = benchmark_dr.std()
    avg = dr.mean()
    benchmark_avg = benchmark_dr.mean()
    d = {
        "cumulative_ret": [cr, benchmark_cr],
        "std": [std, benchmark_std],
        "avg": [avg, benchmark_avg]
    }

    df = pd.DataFrame(d, index=["manual", "benchmark"])
    df = df.round(6)
    print(f"\nManual Strategy Out-Sample Stats for ${symbol}\n")
    print(df, "\n")

    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(portvals, label="Manual Strategy", color="red")
    plt.plot(benchmark_portvals, label="Benchmark", color="purple")
    plt.title(f"Manual Strategy vs. Benchmark Portfolio Value for ${symbol} (Out-of-Sample)")
    plt.xlabel("Dates")
    plt.ylabel("Value")
    plt.legend()

    # Plot trade entries
    for index, row in df_trades.iterrows():
        if row[symbol] == 1000:
            plt.axvline(x=index, color="blue", linestyle="--")
        elif row[symbol] == -1000:
            plt.axvline(x=index, color="black", linestyle="--")

    plt.savefig(f"images/manual_strategy_{symbol}_out_of_sample.png")
    plt.clf()

def author():
    return 'bflores9'

if __name__ == "__main__":
    # plot_manual_benchmark_stats()
    exp1.plot()
    # exp2.plot()