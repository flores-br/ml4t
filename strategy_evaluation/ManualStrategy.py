import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters

import indicators as ind
from marketsimcode import compute_portvals
from util import get_data

register_matplotlib_converters()


class ManualStrategy(object):
    def __init__(self):
        pass

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        prices = get_data([symbol], pd.date_range(sd, ed))
        prices = prices[[symbol]]  # remove SPY
        df = pd.DataFrame(index=prices.index)

        # Indicators
        _, bb_upper, bb_lower = ind.bollinger_bands(prices)
        _, _, hist = ind.macd(prices)
        k, d = ind.stochastic(prices)

        df["close"] = prices[symbol]
        df["bb_upper"] = bb_upper
        df["bb_lower"] = bb_lower
        df["hist"] = hist
        df["k"] = k
        df["d"] = d
        df["trades"] = 0

        buy = ((df["hist"] > 0) & (df["hist"].shift() <= 0)) & (
              (df['k'] > df['d']) & (df['k'].shift() < df['d'].shift())) & (df['d'] < 20) | (
                    (df['close'] < df['bb_lower']) & (df['close'].shift() >= df['bb_lower'].shift()))
        sell = ((df["hist"] < 0) & (df["hist"].shift() >= 0)) & (
              (df['k'] < df['d']) & (df['k'].shift() > df['d'].shift())) & (df['d'] > 80) | (
                     (df['close'] > df['bb_upper']) & (df['close'].shift() <= df['bb_upper'].shift()))
        hold = ~(buy | sell)

        df.loc[buy, "trades"] = 1000
        df.loc[sell, "trades"] = -1000
        df.loc[hold, "trades"] = 0

        return df["trades"]

    def author(self):
        return "bflores9"


if __name__ == "__main__":
    symbol = "JPM"

    # In sample
    df_trades = ManualStrategy.testPolicy().to_frame(name=symbol)
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

    plt.savefig(f"manual_strategy_{symbol}_in_sample.png")

    # Out of sample
    df_trades = ManualStrategy.testPolicy(sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31)).to_frame(name=symbol)
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

    plt.savefig(f"manual_strategy_{symbol}_out_of_sample.png")



