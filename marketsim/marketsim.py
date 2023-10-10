""""""  		  	   		  		 			  		 			 	 	 		 		 	
"""MC2-P1: Market simulator.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
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
import os  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
import numpy as np  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
import pandas as pd  		  	   		  		 			  		 			 	 	 		 		 	
from util import get_data, plot_data  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    """
        Computes the portfolio values.

        :param orders_file: Path of the order file or the file object
        :type orders_file: str or file object
        :param start_val: The starting value of the portfolio
        :type start_val: int
        :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
        :type commission: float
        :param impact: The amount the price moves against the trader compared to the historical data at each transaction
        :type impact: float
        :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
        :rtype: pandas.DataFrame
    """

    # get order data
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    symbols = orders_df["Symbol"].unique()
    start_date = orders_df.index.array[0]
    end_date = orders_df.index.array[-1]

    prices = get_data(symbols, dates=pd.date_range(start_date, end_date))
    prices["Cash"] = pd.Series(data=np.ones(prices.shape[0]), index=prices.index.array)
    shares = prices * 0.0

    # initialize start value
    shares.iloc[0,-1] = start_val

    for index, row in orders_df.iterrows():
        stock_name, order_type, num_shares = row
        order_price = prices.loc[index, stock_name]

        sign = -1 if order_type == "BUY" else 1

        shares.loc[index, stock_name] += -1 * num_shares * sign
        # account for commission and market impact
        shares.loc[index, "Cash"] += num_shares * order_price * sign - commission - num_shares * order_price * impact

    # update shares
    for i in range(1, len(shares)):
        shares.iloc[i, :] += shares.iloc[i - 1, :]

    values = prices * shares
    port_vals = values.sum(axis=1)

    return port_vals

  		  	   		  		 			  		 			 	 	 		 		 	
def test_code():  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    Helper function to test code  		  	   		  		 			  		 			 	 	 		 		 	
    """  		  	   		  		 			  		 			 	 	 		 		 	
    # this is a helper function you can use to test your code  		  	   		  		 			  		 			 	 	 		 		 	
    # note that during autograding his function will not be called.  		  	   		  		 			  		 			 	 	 		 		 	
    # Define input parameters  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
    of = "./orders/orders2.csv"  		  	   		  		 			  		 			 	 	 		 		 	
    sv = 1000000  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
    # Process orders  		  	   		  		 			  		 			 	 	 		 		 	
    # portvals = compute_portvals(orders_file=of, start_val=sv)
    portvals = compute_portvals()
    # if isinstance(portvals, pd.DataFrame):
    #     portvals = portvals[portvals.columns[0]]  # just get the first column
    # else:
    #     "warning, code did not return a DataFrame"
  	#
    # # Get portfolio stats
    # # Here we just fake the data. you should use your code from previous assignments.
    # start_date = dt.datetime(2008, 1, 1)
    # end_date = dt.datetime(2008, 6, 1)
    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
    #     0.2,
    #     0.01,
    #     0.02,
    #     1.5,
    # ]
    # cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
    #     0.2,
    #     0.01,
    #     0.02,
    #     1.5,
    # ]
  	#
    # # Compare portfolio against $SPX
    # print(f"Date Range: {start_date} to {end_date}")
    # print()
    # print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    # print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    # print()
    # print(f"Cumulative Return of Fund: {cum_ret}")
    # print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    # print()
    # print(f"Standard Deviation of Fund: {std_daily_ret}")
    # print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    # print()
    # print(f"Average Daily Return of Fund: {avg_daily_ret}")
    # print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    # print()
    # print(f"Final Portfolio Value: {portvals[-1]}")

def author():
  return "bflores9"
  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		  		 			  		 			 	 	 		 		 	
    test_code()  		  	   		  		 			  		 			 	 	 		 		 	
