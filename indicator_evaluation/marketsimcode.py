import datetime as dt
import os

import numpy as np

import pandas as pd
from util import get_data, plot_data

def compute_portvals(trades, start_val=1000000, commission=0.0, impact=0.0):
    start_date, end_date = trades.index.array[0], trades.index.array[-1]
    symbol = trades.columns.values[0]

    prices = get_data([symbol], pd.date_range(start_date, end_date))
    prices = prices.ffill().bfill()

    cash = start_val
    shares_owned = 0

    portvals = pd.DataFrame(index=prices.index, columns=['value'])
    portvals['value'][0] = start_val

    for date, trade in trades.iterrows():
        shares = trade[symbol]
        price = prices.loc[date, symbol]
        transaction_cost = commission + (impact * price * abs(shares))

        if shares > 0:  # BUY
            cash -= (price * shares + transaction_cost)
            shares_owned += shares
        elif shares < 0:  # SELL
            cash += (price * abs(shares) - transaction_cost)
            shares_owned += shares

        portvals.loc[date, 'value'] = cash + (shares_owned * price)

    return portvals


def author():
    return "bflores9"