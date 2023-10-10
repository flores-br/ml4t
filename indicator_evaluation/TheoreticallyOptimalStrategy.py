"""
Student Name: Bryan Flores
GT User ID: bflores9
GT ID: 903848430
"""
import pandas as pd

from util import get_data

import datetime as dt

def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
    prices = get_data([symbol], pd.date_range(sd, ed)).ffill().bfill()
    prices = prices[[symbol]]  # remove SPY

    trades = prices * 0.0

    max_purchase = 1000.0
    holdings = 0

    for i in range(len(prices) - 1):
        cur_price = prices.iloc[i, 0]
        next_price = prices.iloc[i + 1, 0]

        if next_price > cur_price and holdings <= 0:
            trades.iloc[i, 0] = max_purchase
            holdings += max_purchase
        elif next_price < cur_price and holdings >= 0:
            trades.iloc[i, 0] = -max_purchase
            holdings -= max_purchase

        # since trades is filled with 0.0, no need to check for
        # next_price == cur_price

    return trades

def author():
    return "bflores9"

if __name__ == "__main__":
    testPolicy()