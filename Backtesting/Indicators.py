import plotly.graph_objects as go
import random
import pandas as pd

# from ta.momentum import RSIIndicator
def ma(n, df, fig=None):
    df['MA_' + str(n)] = df['close'].rolling(window=n).mean()
    if fig:
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'yellow']
        color = random.choice(colors)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_' + str(n)], mode='lines', name='MA_' + str(n), line=dict(color=color, width=2)))
    return df

#BOLLINGER STRATEGY
def rolling_std(df, window):
    df[f'rolling_std_{window}'] = df['close'].rolling(window=window).std()

def ma(window, df):
    df[f'MA_{window}'] = df['close'].rolling(window=window).mean()

def calculate_bollinger_bands(df, window, num_std_dev, fig=None):
    ma(window, df)
    rolling_std(df, window)
    
    upper_band = f'upper_band_{window}_{num_std_dev}'
    lower_band = f'lower_band_{window}_{num_std_dev}'
    band_width = f'band_width_{window}_{num_std_dev}'

    df[upper_band] = df[f'MA_{window}'] + (df[f'rolling_std_{window}'] * num_std_dev)
    df[lower_band] = df[f'MA_{window}'] - df[f'rolling_std_{window}'] * num_std_dev
    df[band_width] = df[upper_band] - df[lower_band]

    if fig:
        fig.add_trace(go.Scatter(x=df['Date'], y=df[upper_band], mode='lines', name='Upper Band', line=dict(color='red')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df[lower_band], mode='lines', name='Lower Band', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df[f'MA_{window}'], mode='lines', name='Moving Average', line=dict(color='green')), row=3, col=1)
    
    return df

    



# MACD STRATEGY

def ema_column(data,i):
    data[f'ema_{i}'] = data['close'].ewm(span=i, adjust=False).mean()

def calculate_macd_and_add_trace(data, short_window=12, long_window=26, signal_window=9, fig=None):
    # Calculate MACD
    ema_column(data, short_window)
    ema_column(data, long_window)
    #print("Printing data124444:",data)
    macd_col = f'macd_{short_window}_{long_window}'
    signal_col = f'signal_line_{short_window}_{long_window}'
    histogram_col = f'macd_histogram_{short_window}_{long_window}'
    #print("Printing data1255555:",data)
    data[macd_col] = data[f'ema_{short_window}'] - data[f'ema_{long_window}']
    data[signal_col] = data[macd_col].ewm(span=signal_window, adjust=False).mean()
    data[histogram_col] = data[macd_col] - data[signal_col]
    #print("Printing data1212212:",data)
    if fig:
        # Add MACD line to the third subplot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[macd_col], mode='lines', name='MACD'), row=3, col=1)
        # Add signal line to the third subplot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[signal_col], mode='lines', name='Signal Line'), row=3, col=1)
        # Add MACD histogram to the third subplot
        fig.add_trace(go.Bar(x=data['Date'], y=data[histogram_col], name='MACD Histogram'), row=3, col=1)
        
    
#RSI Strategy

def calculate_RSI(data, window=14,fig=None):
    #rsi = RSIIndicator(close=data['close'], window=window)
    #data['RSI'] = rsi.rsi()
    if fig:
            # Add RSI indicator to the third subplot
        fig.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], mode='lines', name='RSI'), row=3, col=1)

        # Add overbought (upper bound) and oversold (lower bound) lines to the RSI subplot
        fig.add_shape(
            type="line", line=dict(color="red", width=1, dash="dash"),
            x0=data['Date'].iloc[0], y0=70, x1=data['Date'].iloc[-1], y1=70,  # Upper bound (overbought)
            row=3, col=1
        )
        fig.add_shape(
            type="line", line=dict(color="blue", width=1, dash="dash"),
            x0=data['Date'].iloc[0], y0=30, x1=data['Date'].iloc[-1], y1=30,  # Lower bound (oversold)
            row=3, col=1
        )
    #return data
        
# Stochastic Oscillator
def calculate_and_add_trace_stochastic_oscillator(data, k_window=14, d_window=3,fig=None):
    # Calculate %K
    data[f'lowest_flow_{k_window}_{d_window}'] = data['low'].rolling(window=k_window).min()
    data[f'highest_hfigh_{k_window}_{d_window}'] = data['high'].rolling(window=k_window).max()
    data[f'%K_{k_window}_{d_window}'] = 100 * ((data['close'] - data[f'lowest_flow_{k_window}_{d_window}']) / (data[f'highest_hfigh_{k_window}_{d_window}'] - data[f'lowest_flow_{k_window}_{d_window}']))
    
    # Calculate %D
    data[f'%D_{k_window}_{d_window}'] = data[f'%K_{k_window}_{d_window}'].rolling(window=d_window).mean()
    if fig:
        # Add %K line to the subplot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'%K_{k_window}_{d_window}'], mode='lines', name=f'%K_{k_window}_{d_window}'), row=3, col=1)
        # Add %D line to the subplot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'%D_{k_window}_{d_window}'], mode='lines', name=f'%D_{k_window}_{d_window}'), row=3, col=1)


#ICHIMOKU
def calculate_ichimoku(df, tenkan_sen_period, kijun_sen_period, senkou_span_b_period, senkou_shift, fig=None):
    # Tenkan-sen (Conversion Line)
    df['tenkan_sen'] = (df['high'].rolling(window=tenkan_sen_period).max() + df['low'].rolling(window=tenkan_sen_period).min()) / 2
    
    # Kijun-sen (Base Line)
    df['kijun_sen'] = (df['high'].rolling(window=kijun_sen_period).max() + df['low'].rolling(window=kijun_sen_period).min()) / 2
    
    # Senkou Span A (Leading Span A)
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(senkou_shift)
    
    # Senkou Span B (Leading Span B)
    df['senkou_span_b'] = (df['high'].rolling(window=senkou_span_b_period).max() + df['low'].rolling(window=senkou_span_b_period).min()) / 2
    df['senkou_span_b'] = df['senkou_span_b'].shift(senkou_shift)
    
    # Chikou Span (Lagging Span)
    df['chikou_span'] = df['close'].shift(-senkou_shift)
    
    # Plotting if fig is provided
    if fig:
        # Create a subplot for Ichimoku Cloud
        fig.add_trace(go.Scatter(x=df.index, y=df['tenkan_sen'], mode='lines', name='Tenkan-sen', line=dict(color='red')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['kijun_sen'], mode='lines', name='Kijun-sen', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['senkou_span_a'], mode='lines', name='Senkou Span A', line=dict(color='green')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['senkou_span_b'], mode='lines', name='Senkou Span B', line=dict(color='orange')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['chikou_span'], mode='lines', name='Chikou Span', line=dict(color='purple')), row=3, col=1)

    return df
