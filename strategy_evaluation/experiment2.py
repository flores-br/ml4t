
# 3.3.4 Implement Experiment 2
#
#
# Conduct an experiment with your StrategyLearner that shows how changing the value of impact should affect in-sample trading behavior.
#
# Select two metrics, and generate tests that will provide you with at least 3 measurements when trading JPM on the in-sample period with a commission of $0.00. Generate charts that support your tests and show your results.
#
# The code that implements this experiment and generates the relevant charts and data should be submitted as experiment2.py.

import StrategyLearner as sl
import pandas as pd
from marketsimcode import compute_portvals
from matplotlib import pyplot as plt

def plot(symbol="JPM"):
    learner = sl.StrategyLearner(impact=0.005, commission=0)
    learner.add_evidence()
    df_trades = learner.testPolicy()

    learner1 = sl.StrategyLearner(impact = 0.0005, commission=0)
    learner1.add_evidence()
    df_trades1 = learner1.testPolicy()

    learner2 = sl.StrategyLearner(impact = 0.00025, commission=0)
    learner2.add_evidence()
    df_trades2 = learner2.testPolicy()

    portvals = compute_portvals(df_trades, start_val=100000, commission=0, impact=0.005)
    portvals1 = compute_portvals(df_trades1, start_val=100000, commission=0, impact=0.0005)
    portvals2 = compute_portvals(df_trades2, start_val=100000, commission=0, impact=0.00025)

    # Normalize
    portvals = portvals / portvals.iloc[0]
    portvals1 = portvals1 / portvals1.iloc[0]
    portvals2 = portvals2 / portvals2.iloc[0]

    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(portvals, label="Impact = 0.005")
    plt.plot(portvals1, label="Impact = 0.0005")
    plt.plot(portvals2, label="Impact = 0.00025")
    plt.title(f"Strategy Learner Portfolio Value for ${symbol} (In-Sample)")
    plt.xlabel("Dates")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(f"images/exp_2")
    plt.clf()

def author():
    return 'bflores9'

if __name__ == "__main__":
    plot()