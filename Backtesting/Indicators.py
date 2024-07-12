import plotly.graph_objects as go
import random

def ma(n, df, fig=None):
    df['MA' + str(n)] = df['close'].rolling(window=n).mean()
    if fig:
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'yellow']
        color = random.choice(colors)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA' + str(n)], mode='lines', name='MA' + str(n), line=dict(color=color, width=2)))
    return df

#BOLLINGER STRATEGY
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

def calculate_macd(data, short_window, long_window, signal_window):
    ema_column(data, short_window)
    ema_column(data, long_window)

    # Properly format the column names
    macd_col = f'macd_{short_window}_{long_window}'
    signal_col = f'signal_line_{short_window}_{long_window}'
    histogram_col = f'macd_histogram_{short_window}_{long_window}'

    data[macd_col] = data[f'ema_{short_window}'] - data[f'ema_{long_window}']
    data[signal_col] = data[macd_col].ewm(span=signal_window, adjust=False).mean()
    data[histogram_col] = data[macd_col] - data[signal_col]
    return data

def add_macd_trace(fig, df, short_window, long_window):
    # Add MACD line to the third subplot
    fig.add_trace(go.Scatter(x=df['Date'], y=df[f'macd_{short_window}_{long_window}'], mode='lines', name='MACD'), row=3, col=1)
    # Add signal line to the third subplot
    fig.add_trace(go.Scatter(x=df['Date'], y=df[f'signal_line_{short_window}_{long_window}'], mode='lines', name='Signal Line'), row=3, col=1)
    # Add MACD histogram to the third subplot
    fig.add_trace(go.Bar(x=df['Date'], y=df[f'macd_histogram_{short_window}_{long_window}'], name='MACD Histogram'), row=3, col=1)
    return fig