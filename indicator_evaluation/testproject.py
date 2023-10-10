import TheoreticallyOptimalStrategy as tos
import marketsimcode as ms
import indicators as ind
import datetime as dt
from util import get_data

import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters

def optimal_vs_benchmark(start_date, end_date, start_val, commission, impact):
    df_trades = tos.testPolicy(symbol="JPM", sd=start_date, ed=end_date, sv=start_val)
    optimal = ms.compute_portvals(df_trades, start_val, commission, impact)

    benchmark_trades = df_trades * 0.0
    benchmark_trades.iloc[0] = 1000.0
    benchmark = ms.compute_portvals(benchmark_trades, start_val, commission, impact)

    # normalized
    optimal = optimal / optimal.iloc[0]
    benchmark = benchmark / benchmark.iloc[0]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(optimal, label="Optimal")
    ax.plot(benchmark, label="Benchmark")
    ax.set_title(f"Optimal vs. Benchmark Portfolio Value for $JPM")
    ax.set_xlabel("Dates")
    ax.set_ylabel("Value")
    ax.legend()
    plt.savefig("optimal_vs_benchmark.png")
    plt.clf()

def print_stats():
    df_trades = tos.testPolicy(symbol="JPM", sd=start_date, ed=end_date, sv=start_val)
    optimal = ms.compute_portvals(df_trades, start_val, commission, impact)

    benchmark_trades = df_trades * 0.0
    benchmark_trades.iloc[0] = 1000.0
    benchmark = ms.compute_portvals(benchmark_trades, start_val, commission, impact)

    benchmark = benchmark["value"]
    optimal = optimal["value"]

    # cumulative returns
    optimal_cr = optimal.iloc[-1] / optimal.iloc[0] - 1
    benchmark_cr = benchmark.iloc[-1] / benchmark.iloc[0] - 1

    # daily returns
    optimal_dr = optimal.pct_change()
    benchmark_dr = benchmark.pct_change()

    # std daily returns
    optimal_std = optimal_dr.std()
    benchmark_std = benchmark_dr.std()

    # mean daily returns
    optimal_avg = optimal_dr.mean()
    benchmark_avg = benchmark_dr.mean()

    d = {
        "cumulative_ret": [optimal_cr, benchmark_cr],
        "std": [optimal_std, benchmark_std],
        "avg": [optimal_avg, benchmark_avg]
    }

    df = pd.DataFrame(d, index=["optimal", "benchmark"])
    df = df.round(6)

def plot_bollinger(price_df):
    bb_mean, bb_upper, bb_lower = ind.bollinger_bands(price_df)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(price_df[symbol], label='Price')
    ax.plot(bb_mean, label='SMA')
    ax.plot(bb_upper, label='Upper Band')
    ax.plot(bb_lower, label='Lower Band')
    ax.set_title('Bollinger Bands')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    plt.savefig("bollinger.png")
    plt.clf()

def plot_rsi(price_df):
    symbol = price_df.columns.values[0]
    rsi = ind.rsi(price_df)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(price_df, label='Price')
    ax.plot(rsi, label='RSI')
    ax.axhline(70, color='r', linestyle='--', label='Overbought')
    ax.axhline(30, color='g', linestyle='--', label='Oversold')
    ax.set_title(f'Relative Strength Index (RSI) for ${symbol}')
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI')
    ax.legend()
    plt.savefig("rsi.png")

def plot_stochastic(stock_data, window=14):
    k, d = ind.stochastic(stock_data)
    symbol = stock_data.columns.values[0]
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 8), sharex=True)
    ax1.plot(stock_data, label=symbol)
    ax1.set_ylabel('Price')
    ax2.plot(k, label='%K')
    ax2.plot(d, label='%D')
    ax2.axhline(80, color='r', linestyle='--', label='Overbought')
    ax2.axhline(20, color='g', linestyle='--', label='Oversold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Stochastic Oscillator')
    ax1.set_title('Stochastic Oscillator')
    ax1.legend()
    ax2.legend()
    plt.savefig("stochastic.png")
    plt.clf()

def plot_ppi(stock_data):
    symbol = stock_data.columns.values[0]
    ppo, signal, hist = ind.ppi(stock_data)
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(12, 8))
    ax1.plot(stock_data, label=f"Price for ${symbol}")
    ax1.set_title(f"Percentage Price Indicator (PPI) for {symbol}")
    ax2.plot(ppo, label="PPO")
    ax2.plot(signal, label="Signal", color="r")
    ax2.set_title(f"PPO and Signal Line for {symbol}")
    ax2.bar(stock_data.index.to_pydatetime(), hist[symbol], label="Histogram", color='gray', alpha=0.5)
    ax2.set_xlabel('Date')
    ax2.set_title('Histogram: Distance between lines')
    ax1.legend()
    ax2.legend()
    plt.savefig("ppi.png")
    plt.clf()

def plot_macd(stock_data):
    symbol = stock_data.columns.values[0]
    macd, signal, hist = ind.macd(stock_data)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 8), sharex=True)
    ax1.plot(stock_data, label=f'Price for ${symbol}')
    ax2.plot(macd, label='MACD Line')
    ax2.plot(signal, label='Signal Line')
    ax2.bar(stock_data.index.to_pydatetime(), hist[symbol], label='Histogram', color='gray', alpha=0.5)
    ax2.set_xlabel('Date')
    ax1.set_title(f'Moving Average Convergence Divergence (MACD) for ${symbol}')
    ax1.legend()
    ax2.legend()
    plt.savefig("macd.png")


if __name__ == "__main__":
    register_matplotlib_converters()
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009,12,31)
    start_val = 100000
    commission = 0.0
    impact = 0.0
    symbol = "JPM"
    price_df = get_data([symbol], pd.date_range(start_date, end_date), addSPY=False).dropna()

    # optimal_vs_benchmark(start_date, end_date, start_val, commission, impact)

    # 1. Bollinger Bands
    plot_bollinger(price_df)

    # 2. RSI
    plot_rsi(price_df)

    # 3. Stochastic
    plot_stochastic(price_df)

    # 4. Percentage Price Indicator
    plot_ppi(price_df)

    # # 5. MACD
    plot_macd(price_df)

