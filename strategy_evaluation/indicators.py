def bollinger_bands(stock_data, window=20, std=2):
    symbols = stock_data.columns.values
    rolling_mean = stock_data[symbols].rolling(window).mean()
    rolling_std = stock_data[symbols].rolling(window).std()
    upper = rolling_mean + rolling_std * std
    lower = rolling_mean - rolling_std * std
    return rolling_mean, upper, lower

def rsi(stock_data, window=14):
    symbols = stock_data.columns.values
    diff = stock_data[symbols].diff().dropna()

    up_change, down_change = diff.copy(), diff.copy()

    up_change[up_change < 0] = 0
    down_change[down_change > 0] = 0

    up_mean = up_change.rolling(window).mean()
    down_mean = down_change.rolling(window).mean().abs()

    return 100 * up_mean / (up_mean + down_mean)

def stochastic(stock_data, window=14):
    symbols = stock_data.columns.values
    highest = stock_data[symbols].rolling(window).max()
    lowest = stock_data[symbols].rolling(window).min()
    k = 100 * ((stock_data[symbols] - lowest) / (highest - lowest))
    d = k.rolling(window=3).mean()
    return k, d

def ppi(stock_data, fast_window=12, slow_window=26, signal_window=9):
    symbols = stock_data.columns.values

    fast_df = stock_data[symbols].ewm(span=fast_window).mean()
    slow_df = stock_data[symbols].ewm(span=slow_window).mean()

    ppo = 100 * (fast_df - slow_df) / slow_df
    signal = ppo.ewm(span=signal_window).mean()
    hist = ppo - signal

    return ppo, signal, hist

def macd(stock_data, fast_window=12, slow_window=26, signal_window=9):
    symbols = stock_data.columns.values

    fast_df = stock_data[symbols].ewm(span=fast_window).mean()
    slow_df = stock_data[symbols].ewm(span=slow_window).mean()

    macd = fast_df - slow_df
    signal = macd.ewm(span=signal_window).mean()
    hist = macd - signal

    return macd, signal, hist

def author():
    return 'bflores9'