import plotly.graph_objects as go
import random

def ma(n, df,fig=None):
    df['MA'+str(n)] = df['close'].rolling(window=n).mean()
    if not fig:
        pass
    else:
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'yellow']
        color = random.choice(colors)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA'+str(n)], mode='lines', name='MA'+str(n),line=dict(color=color, width=2)))
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
