import plotly.graph_objects as go
import random

def ma(n, df, fig=None):
    df['MA' + str(n)] = df['close'].rolling(window=n).mean()
    if fig:
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'yellow']
        color = random.choice(colors)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA' + str(n)], mode='lines', name='MA' + str(n), line=dict(color=color, width=2)))
    return df

def rolling_std(n, df):
    df['rolling_std' + str(n)] = df['close'].rolling(window=n).std()
    return df

def calculate_bollinger_bands(df, window=20, num_std_dev=2, fig=None):
    df = ma(window, df)
    df = rolling_std(window, df)
    df['upper_band'] = df['MA' + str(window)] + (df['rolling_std' + str(window)] * num_std_dev)
    df['lower_band'] = df['MA' + str(window)] - (df['rolling_std' + str(window)] * num_std_dev)
    df['band_width'] = df['upper_band'] - df['lower_band']
    if fig:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['upper_band'], mode='lines', name='Upper Bollinger Band', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['lower_band'], mode='lines', name='Lower Bollinger Band', line=dict(color='blue')))
    return df


# MACD STRATEGY

def ema_column(data,i):
    data[f'ema_{i}'] = data['close'].ewm(span=i, adjust=False).mean()

# Function to calculate MACD, Signal Line, and MACD Histogram
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    ema_column(data, short_window)
    ema_column(data, long_window)
    # data['ema_short'] = data['close'].ewm(span=short_window, adjust=False).mean()
    # data['ema_long'] = data['close'].ewm(span=long_window, adjust=False).mean()
    data['macd_12_26'] = data['ema_12'] - data['ema_26']
    data['signal_line_12_26'] = data['macd_12_26'].ewm(span=signal_window, adjust=False).mean()
    data['macd_histogram_12_26'] = data['macd_12_26'] - data['signal_line_12_26']
    return data
