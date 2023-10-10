""""""  		  	   		  		 			  		 			 	 	 		 		 	
"""  		  	   		  		 			  		 			 	 	 		 		 	
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		  		 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			 	 	 		 		 	
or edited.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Student Name: Bryan Flores  		  	   		  		 			  		 			 	 	 		 		 	
GT User ID: bflores9		  	   		  		 			  		 			 	 	 		 		 	
GT ID: 903848430		  	   		  		 			  		 			 	 	 		 		 	
"""  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
import datetime as dt  		  	   		  		 			  		 			 	 	 		 		 	
import random
  		  	   		  		 			  		 			 	 	 		 		 	
import pandas as pd
import numpy as np
import util as ut

import RTLearner as rt
import BagLearner as bl
import indicators as ind

from marketsimcode import compute_portvals
from matplotlib import pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
class StrategyLearner(object):  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  		 			  		 			 	 	 		 		 	
        If verbose = False your code should not generate ANY output.  		  	   		  		 			  		 			 	 	 		 		 	
    :type verbose: bool  		  	   		  		 			  		 			 	 	 		 		 	
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  		 			  		 			 	 	 		 		 	
    :type impact: float  		  	   		  		 			  		 			 	 	 		 		 	
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  		 			  		 			 	 	 		 		 	
    :type commission: float  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    # constructor  		  	   		  		 			  		 			 	 	 		 		 	
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		  		 			  		 			 	 	 		 		 	
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Constructor method  		  	   		  		 			  		 			 	 	 		 		 	
        """  		  	   		  		 			  		 			 	 	 		 		 	
        self.verbose = verbose  		  	   		  		 			  		 			 	 	 		 		 	
        self.impact = impact  		  	   		  		 			  		 			 	 	 		 		 	
        self.commission = commission
        self.learner = bl.BagLearner(learner=rt.RTLearner, bags=20, kwargs={"leaf_size": 5}, boost=False, verbose=verbose)
  		  	   		  		 			  		 			 	 	 		 		 	
    # this method should create a QLearner, and train it for trading  		  	   		  		 			  		 			 	 	 		 		 	
    def add_evidence(
        self,
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=100000,
    ):  		  	   		  		 			  		 			 	 	 		 		 	
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Trains your strategy learner over a given time frame.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
        :param symbol: The stock symbol to train on  		  	   		  		 			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		  		 			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		  		 			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		  		 			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		  		 			  		 			 	 	 		 		 	
        :type sv: int  		  	   		  		 			  		 			 	 	 		 		 	
        """  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
        # add your code to do learning here  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
        # example usage of the old backward compatible util function  		  	   		  		 			  		 			 	 	 		 		 	
        syms = [symbol]  		  	   		  		 			  		 			 	 	 		 		 	
        dates = pd.date_range(sd, ed)  		  	   		  		 			  		 			 	 	 		 		 	
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY  		  	   		  		 			  		 			 	 	 		 		 	
        prices = prices_all[syms]  # only portfolio symbols  		  	   		  		 			  		 			 	 	 		 		 	
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		  		 			  		 			 	 	 		 		 	
        # if self.verbose:
        #     print(prices)
        window = 12
  		  	   		  		 			  		 			 	 	 		 		 	
        # Indicators
        bb_mean, bb_upper, bb_lower = ind.bollinger_bands(prices, window=window)
        macd, macd_signal, _ = ind.macd(prices)
        k, d = ind.stochastic(prices, window=window)
        df = pd.DataFrame(index=prices.index)

        df["bb_mean"] = bb_mean
        df["bb_upper"] = bb_upper
        df["bb_lower"] = bb_lower
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["k"] = k
        df["d"] = d
        df.fillna(0, inplace=True)

        x_train = df.to_numpy()[window:]

        returns = prices[symbol].to_numpy()[window:] / prices[symbol].to_numpy()[:-window] - 1
        y_train = np.zeros(prices.shape[0])

        for i, ret in enumerate(returns):
            if ret > 0.002 + self.impact:
                y_train[i] = 1
            elif ret < -0.002 - self.impact:
                y_train[i] = -1

        self.learner.add_evidence(x_train, y_train)
  		  	   		  		 			  		 			 	 	 		 		 	
    # this method should use the existing policy and test it against new data  		  	   		  		 			  		 			 	 	 		 		 	
    def testPolicy(
        self,
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=10000,  		  	   		  		 			  		 			 	 	 		 		 	
    ):  		  	   		  		 			  		 			 	 	 		 		 	
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Tests your learner using data outside of the training data  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
        :param symbol: The stock symbol that you trained on on  		  	   		  		 			  		 			 	 	 		 		 	
        :type symbol: str  		  	   		  		 			  		 			 	 	 		 		 	
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			 	 	 		 		 	
        :type sd: datetime  		  	   		  		 			  		 			 	 	 		 		 	
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			 	 	 		 		 	
        :type ed: datetime  		  	   		  		 			  		 			 	 	 		 		 	
        :param sv: The starting value of the portfolio  		  	   		  		 			  		 			 	 	 		 		 	
        :type sv: int  		  	   		  		 			  		 			 	 	 		 		 	
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  		 			  		 			 	 	 		 		 	
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  		 			  		 			 	 	 		 		 	
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  		 			  		 			 	 	 		 		 	
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  		 			  		 			 	 	 		 		 	
        :rtype: pandas.DataFrame  		  	   		  		 			  		 			 	 	 		 		 	
        """  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
        # here we build a fake set of trades  		  	   		  		 			  		 			 	 	 		 		 	
        # your code should return the same sort of data  		  	   		  		 			  		 			 	 	 		 		 	
        dates = pd.date_range(sd, ed)  		  	   		  		 			  		 			 	 	 		 		 	
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY  		  	   		  		 			  		 			 	 	 		 		 	
        trades = prices_all[[symbol,]]  # only portfolio symbols  		  	   		  		 			  		 			 	 	 		 		 	
        trades_SPY = prices_all["SPY"]  # only SPY, for comparison later

        # Prepare test data
        window = 12
        bb_mean, bb_upper, bb_lower = ind.bollinger_bands(prices_all, window=window)
        macd, macd_signal, _ = ind.macd(prices_all)
        k, d = ind.stochastic(prices_all, window=window)
        df = pd.DataFrame(index=prices_all.index, columns=["bb_mean", "bb_upper", "bb_lower", "macd", "macd_signal", "k", "d"])

        df["bb_mean"] = bb_mean
        df["bb_upper"] = bb_upper
        df["bb_lower"] = bb_lower
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["k"] = k
        df["d"] = d
        df.fillna(0, inplace=True)

        x_test = df.to_numpy()[window:]
        y_test = self.learner.query(x_test)

        # Trades
        trades = pd.DataFrame(index=prices_all.index, columns=[symbol])
        trades[symbol] = 0.0
        position = 0

        for i, action in enumerate(y_test):
            if position == 0 and action == 1:  # Buy
                trades[symbol].iloc[i] = 1000
                position = 1
            elif position == 0 and action == -1:  # Sell
                trades[symbol].iloc[i] = -1000
                position = -1
            elif position == 1 and action == -1:  # Sell
                trades[symbol].iloc[i] = -1000
                position = -1
            elif position == -1 and action == 1:  # Buy
                trades[symbol].iloc[i] = 1000
                position = 1
            elif position == 1 and action == 0:  # Sell
                trades[symbol].iloc[i] = -1000
                position = 0
            elif position == -1 and action == 0:  # Buy
                trades[symbol].iloc[i] = 1000
                position = 0

        return trades

    def author(self):
        return "bflores9"
  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		  		 			  		 			 	 	 		 		 	
    print("One does not simply think up a strategy")
    sl = StrategyLearner()
    sl.add_evidence()
    df_trades = sl.testPolicy()
    benchmark_trades = df_trades * 0.0
    benchmark_trades.iloc[0] = 1000.0

    portvals = compute_portvals(df_trades, start_val=100000, commission=9.95, impact=0.005)
    benchmark_portvals = compute_portvals(benchmark_trades, start_val=100000, commission=9.95, impact=0.005)

    # Normalize
    portvals = portvals / portvals.iloc[0]
    benchmark_portvals = benchmark_portvals / benchmark_portvals.iloc[0]

    # Plot
    plt.plot(portvals, label="Strategy Learner")
    plt.plot(benchmark_portvals, label="Benchmark")
    plt.legend()
    # plt.show()

