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
def rolling_std(df, window):
    df[f'rolling_std_{window}'] = df['close'].rolling(window=window).std()
    return df

def ma(window, df):
    df[f'MA_{window}'] = df['close'].rolling(window=window).mean()
    return df

def calculate_bollinger_bands(df, window, num_std_dev, fig=None):
    df = ma(window, df)
    df = rolling_std(df, window)
    
    upper_band = f'upper_band_{window}_{num_std_dev}'
    lower_band = f'lower_band_{window}_{num_std_dev}'
    band_width = f'band_width_{window}_{num_std_dev}'

    df[upper_band] = df[f'MA_{window}'] + (df[f'rolling_std_{window}'] * num_std_dev)
    df[lower_band] = df[f'MA_{window}'] - (df[f'rolling_std_{window}'] * num_std_dev)
    df[band_width] = df[upper_band] - df[lower_band]

    if fig:
        fig.add_trace(go.Scatter(x=df['Date'], y=df[upper_band], mode='lines', name=upper_band, line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df[lower_band], mode='lines', name=lower_band, line=dict(color='blue')))
    return df



# MACD STRATEGY

def ema_column(data,i):
    data[f'ema_{i}'] = data['close'].ewm(span=i, adjust=False).mean()

def calculate_macd_and_add_trace(data, short_window=12, long_window=26, signal_window=9, fig=None):
    # Calculate MACD
    ema_column(data, short_window)
    ema_column(data, long_window)

    macd_col = f'macd_{short_window}_{long_window}'
    signal_col = f'signal_line_{short_window}_{long_window}'
    histogram_col = f'macd_histogram_{short_window}_{long_window}'

    data[macd_col] = data[f'ema_{short_window}'] - data[f'ema_{long_window}']
    data[signal_col] = data[macd_col].ewm(span=signal_window, adjust=False).mean()
    data[histogram_col] = data[macd_col] - data[signal_col]

    if fig is None:
        return data
    else:
        # Add MACD line to the third subplot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[macd_col], mode='lines', name='MACD'), row=3, col=1)
        # Add signal line to the third subplot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[signal_col], mode='lines', name='Signal Line'), row=3, col=1)
        # Add MACD histogram to the third subplot
        fig.add_trace(go.Bar(x=data['Date'], y=data[histogram_col], name='MACD Histogram'), row=3, col=1)
        return fig
