import plotly.graph_objects as go
import random
import pandas as pd
import numpy as np
# import ta

# import pandas as pd

# from ta.momentum import RSIIndicator
def ma(n, df, fig=None):
    df['MA_' + str(n)] = df['close'].rolling(window=n).mean()
    if fig:
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'yellow']
        color = random.choice(colors)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_' + str(n)], mode='lines', name='MA_' + str(n), line=dict(color=color, width=2)))
    return df

#BOLLINGER STRATEGY
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        fig.update_layout(height=900, width=900)  # Adjust the figure size as needed

# Example of creating a figure with subplots
def create_figure_with_subplots():
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("Price", "Bollinger Bands", "MACD"),
                        vertical_spacing=0.1)





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
        
# Define the ema_column function to calculate and append the EMA column to the data DataFrame.
def ema_column_incremental(data, i):
    ema_col = f'ema_{i}'
    if ema_col in data.columns:
        last_date = data.dropna(subset=[ema_col]).index[-1]
        data.loc[last_date:, ema_col] = data['close'].loc[last_date:].ewm(span=i, adjust=False).mean()
    else:
        data[ema_col] = data['close'].ewm(span=i, adjust=False).mean()

# Define the calculate_macd_and_add_trace function to calculate MACD, signal line, and histogram, and optionally add traces to the provided figure.
def calculate_macd_and_add_trace_incremental(data, short_window=12, long_window=26, signal_window=9, fig=None):
    # Calculate the EMAs
    ema_column(data, short_window)
    ema_column(data, long_window)

    # Define column names for MACD, signal line, and histogram
    macd_col = f'macd_{short_window}_{long_window}'
    signal_col = f'signal_line_{short_window}_{long_window}'
    histogram_col = f'macd_histogram_{short_window}_{long_window}'

    if macd_col in data.columns:
        last_date = data.dropna(subset=[macd_col]).index[-1]
        data.loc[last_date:, macd_col] = data[f'ema_{short_window}'].loc[last_date:] - data[f'ema_{long_window}'].loc[last_date:]
    else:
        data[macd_col] = data[f'ema_{short_window}'] - data[f'ema_{long_window}']

    if signal_col in data.columns:
        last_date = data.dropna(subset=[signal_col]).index[-1]
        data.loc[last_date:, signal_col] = data[macd_col].loc[last_date:].ewm(span=signal_window, adjust=False).mean()
    else:
        data[signal_col] = data[macd_col].ewm(span=signal_window, adjust=False).mean()

    data[histogram_col] = data[macd_col] - data[signal_col]

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data[macd_col], mode='lines', name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data[signal_col], mode='lines', name='Signal Line'), row=3, col=1)
        fig.add_trace(go.Bar(x=data['Date'], y=data[histogram_col], name='MACD Histogram'), row=3, col=1)


#RSI Strategy

def calculate_RSI(data, window=14,fig=None):
    rsi = ta.RSIIndicator(close=data['close'], window=window)
    data['RSI'] = rsi.rsi()
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

def calculate_and_add_trace_stochastic_oscillator_incremental(data, k_window=14, d_window=3, fig=None):
    low_col = f'lowest_flow_{k_window}_{d_window}'
    high_col = f'highest_hfigh_{k_window}_{d_window}'
    k_col = f'%K_{k_window}_{d_window}'
    d_col = f'%D_{k_window}_{d_window}'

    if low_col in data.columns:
        last_date = data.dropna(subset=[low_col]).index[-1]
        data.loc[last_date:, low_col] = data['low'].loc[last_date:].rolling(window=k_window).min()
    else:
        data[low_col] = data['low'].rolling(window=k_window).min()

    if high_col in data.columns:
        last_date = data.dropna(subset=[high_col]).index[-1]
        data.loc[last_date:, high_col] = data['high'].loc[last_date:].rolling(window=k_window).max()
    else:
        data[high_col] = data['high'].rolling(window=k_window).max()

    data[k_col] = 100 * ((data['close'] - data[low_col]) / (data[high_col] - data[low_col]))

    if d_col in data.columns:
        last_date = data.dropna(subset=[d_col]).index[-1]
        data.loc[last_date:, d_col] = data[k_col].loc[last_date:].rolling(window=d_window).mean()
    else:
        data[d_col] = data[k_col].rolling(window=d_window).mean()

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data[k_col], mode='lines', name=f'%K_{k_window}_{d_window}'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data[d_col], mode='lines', name=f'%D_{k_window}_{d_window}'), row=3, col=1)


#ICHIMOKU
def cal_ichimoku(df, tenkan_sen_period, kijun_sen_period, senkou_span_b_period, senkou_shift, fig=None):
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


#Fibonacci Retracement
def calculate_and_add_fibonacci_levels(data, fig=None):
    max_price = data['close'].max()
    min_price = data['close'].min()

    diff = max_price - min_price
    levels = {
        '0%': max_price,
        '23.6%': max_price - 0.236 * diff,
        '38.2%': max_price - 0.382 * diff,
        '50%': max_price - 0.5 * diff,
        '61.8%': max_price - 0.618 * diff,
        '100%': min_price
    }

    for level, value in levels.items():
        data[f'fibo_{level}'] = value

    if fig:
        for level, value in levels.items():
            fig.add_hline(y=value, line_dash="dash", line_color="blue", annotation_text=f"Fibo {level}", row=1, col=1)

def calculate_and_add_fibonacci_levels_incremental(data, fig=None):
    max_price = data['close'].max()
    min_price = data['close'].min()

    diff = max_price - min_price
    levels = {
        '0%': max_price,
        '23.6%': max_price - 0.236 * diff,
        '38.2%': max_price - 0.382 * diff,
        '50%': max_price - 0.5 * diff,
        '61.8%': max_price - 0.618 * diff,
        '100%': min_price
    }

    for level, value in levels.items():
        data[f'fibo_{level}'] = value

    if fig:
        for level, value in levels.items():
            fig.add_hline(y=value, line_dash="dash", line_color="blue", annotation_text=f"Fibo {level}", row=1, col=1)


# ADX

def calculate_adx_and_add_trace(data, period=14, fig=None):
    # Calculate True Range
    data['high_diff'] = data['high'].diff()
    data['low_diff'] = data['low'].diff()
    data['close_diff'] = data['close'].diff()
    
    data['tr'] = data[['high_diff', 'low_diff', 'close_diff']].max(axis=1).abs()
    
    # Calculate +DI and -DI
    data['+DI'] = 100 * (data['high'].diff(periods=1).where(data['high'].diff(periods=1) > data['low'].diff(periods=1), 0).rolling(window=period).mean() / data['tr'].rolling(window=period).mean())
    data['-DI'] = 100 * (data['low'].diff(periods=1).where(data['low'].diff(periods=1) > data['high'].diff(periods=1), 0).rolling(window=period).mean() / data['tr'].rolling(window=period).mean())
    
    # Calculate ADX
    data['adx'] = 100 * (data['+DI'] - data['-DI']).abs().rolling(window=period).mean()

    # Add traces to the figure if provided
    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['adx'], mode='lines', name='ADX'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['+DI'], mode='lines', name='+DI'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['-DI'], mode='lines', name='-DI'), row=3, col=1)


def calculate_adx_and_add_trace_incremental(data, period=14, fig=None):
    # Check if ADX values already exist in the data
    if 'adx' in data.columns:
        last_defined_index = data[data['adx'].notna()].index[-1]
        new_data = data.loc[last_defined_index + 1:]
    else:
        new_data = data

    if not new_data.empty:
        # Calculate True Range
        new_data['high_diff'] = new_data['high'].diff()
        new_data['low_diff'] = new_data['low'].diff()
        new_data['close_diff'] = new_data['close'].diff()

        new_data['tr'] = new_data[['high_diff', 'low_diff', 'close_diff']].max(axis=1).abs()

        # Calculate +DI and -DI
        new_data['+DI'] = 100 * (
            new_data['high'].diff(periods=1).where(new_data['high'].diff(periods=1) > new_data['low'].diff(periods=1), 0).rolling(window=period).mean() /
            new_data['tr'].rolling(window=period).mean()
        )
        new_data['-DI'] = 100 * (
            new_data['low'].diff(periods=1).where(new_data['low'].diff(periods=1) > new_data['high'].diff(periods=1), 0).rolling(window=period).mean() /
            new_data['tr'].rolling(window=period).mean()
        )

        # Calculate ADX
        new_data['adx'] = 100 * (new_data['+DI'] - new_data['-DI']).abs().rolling(window=period).mean()

        # Merge the new data with the original data
        data.update(new_data)

    # Add traces to the figure if provided
    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['adx'], mode='lines', name='ADX'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['+DI'], mode='lines', name='+DI'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['-DI'], mode='lines', name='-DI'), row=3, col=1)

# Parabolic SAR

def calculate_parabolic_sar_and_add_trace(data, af=0.02, max_af=0.2, fig=None):
    high = data['high']
    low = data['low']
    close = data['close']

    sar = [close[0]]
    trend = 1  # 1 for uptrend, -1 for downtrend
    ep = high[0] if trend == 1 else low[0]
    af_value = af

    for i in range(1, len(data)):
        prev_sar = sar[-1]

        if trend == 1:
            sar.append(prev_sar + af_value * (ep - prev_sar))
            if low[i] < sar[-1]:
                trend = -1
                sar[-1] = ep
                ep = low[i]
                af_value = af
            else:
                if high[i] > ep:
                    ep = high[i]
                    af_value = min(af_value + af, max_af)
        else:
            sar.append(prev_sar + af_value * (ep - prev_sar))
            if high[i] > sar[-1]:
                trend = 1
                sar[-1] = ep
                ep = high[i]
                af_value = af
            else:
                if low[i] < ep:
                    ep = low[i]
                    af_value = min(af_value + af, max_af)

    data['parabolic_sar'] = sar

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['parabolic_sar'], mode='markers', name='Parabolic SAR', marker=dict(color='red')), row=3, col=1)

def calculate_parabolic_sar_and_add_trace_incremental(data, af=0.02, max_af=0.2, fig=None):
    if 'parabolic_sar' in data.columns and not data['parabolic_sar'].isnull().all():
        last_calculated_index = data['parabolic_sar'].last_valid_index()
        start_index = last_calculated_index + 1
        sar = data['parabolic_sar'].tolist()[:start_index]
        trend = 1 if data['close'].iloc[last_calculated_index] > data['parabolic_sar'].iloc[last_calculated_index] else -1
        ep = data['high'].iloc[last_calculated_index] if trend == 1 else data['low'].iloc[last_calculated_index]
        af_value = af
    else:
        start_index = 1
        sar = [data['close'][0]]
        trend = 1  # 1 for uptrend, -1 for downtrend
        ep = data['high'][0] if trend == 1 else data['low'][0]
        af_value = af

    for i in range(start_index, len(data)):
        prev_sar = sar[-1]

        if trend == 1:
            sar.append(prev_sar + af_value * (ep - prev_sar))
            if data['low'][i] < sar[-1]:
                trend = -1
                sar[-1] = ep
                ep = data['low'][i]
                af_value = af
            else:
                if data['high'][i] > ep:
                    ep = data['high'][i]
                    af_value = min(af_value + af, max_af)
        else:
            sar.append(prev_sar + af_value * (ep - prev_sar))
            if data['high'][i] > sar[-1]:
                trend = 1
                sar[-1] = ep
                ep = data['high'][i]
                af_value = af
            else:
                if data['low'][i] < ep:
                    ep = data['low'][i]
                    af_value = min(af_value + af, max_af)

    data['parabolic_sar'] = sar

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['parabolic_sar'], mode='markers', name='Parabolic SAR', marker=dict(color='red')), row=3, col=1)


 #ON BALANCE VOLUME(OBV)
def calculate_obv(data, fig=None):
    """Calculate the On-Balance Volume (OBV) and optionally plot it."""
    obv = [0]  # Start with OBV of 0
    for i in range(1, len(data)):
        if data['close'].iloc[i] > data['close'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['close'].iloc[i] < data['close'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv
    
    if fig:
        # Add OBV line to the third subplot
        fig.add_trace(go.Scatter(x=data['Date'], y=data['OBV'], mode='lines', name='OBV'), row=3, col=1)

# Candlestick Patterns

def find_and_plot_candlestick_patterns(data, fig=None):
    required_columns = ['Open', 'close', 'high', 'low']
    
    # Check if all required columns are in the DataFrame
    for col in required_columns:
        if col not in data.columns:
            raise KeyError(f"'{col}' column is missing from the DataFrame")
    
    data['candlestick_pattern'] = None
    
    for i in range(1, len(data)):
        open_price = data['Open'].iloc[i]
        close_price = data['close'].iloc[i]
        high_price = data['high'].iloc[i]
        low_price = data['low'].iloc[i]
        
        if abs(open_price - close_price) < ((high_price - low_price) * 0.1):
            data.at[i, 'candlestick_pattern'] = 'doji'
        elif open_price > close_price and (open_price - close_price) > ((high_price - low_price) * 0.6) and (low_price == min(open_price, close_price)):
            data.at[i, 'candlestick_pattern'] = 'hammer'
        elif close_price > open_price and data['close'].iloc[i-1] < data['Open'].iloc[i-1] and (close_price > data['Open'].iloc[i-1]) and (open_price < data['close'].iloc[i-1]):
            data.at[i, 'candlestick_pattern'] = 'bullish_engulfing'
        elif open_price > close_price and data['close'].iloc[i-1] > data['Open'].iloc[i-1] and (open_price > data['close'].iloc[i-1]) and (close_price < data['Open'].iloc[i-1]):
            data.at[i, 'candlestick_pattern'] = 'bearish_engulfing'
        elif close_price < open_price and (open_price - close_price) > ((high_price - low_price) * 0.6) and (high_price == max(open_price, close_price)):
            data.at[i, 'candlestick_pattern'] = 'shooting_star'
    
    if fig:
        patterns = ['doji', 'hammer', 'bullish_engulfing', 'bearish_engulfing', 'shooting_star']
        colors = {
            'doji': 'yellow',
            'hammer': 'green',
            'bullish_engulfing': 'blue',
            'bearish_engulfing': 'red',
            'shooting_star': 'purple'
        }

        for pattern in patterns:
            pattern_data = data[data['candlestick_pattern'] == pattern]
            fig.add_trace(
                go.Scatter(
                    x=pattern_data['Date'],
                    y=pattern_data['close'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=colors[pattern],
                        symbol='circle'
                    ),
                    name=pattern.capitalize()
                ),
                row=3, col=1
            )

def find_and_plot_candlestick_patterns_incremental(data, fig=None):
    required_columns = ['Open', 'close', 'high', 'low', 'Date']
    
    # Check if all required columns are in the DataFrame
    for col in required_columns:
        if col not in data.columns:
            raise KeyError(f"'{col}' column is missing from the DataFrame")
    
    if 'candlestick_pattern' not in data.columns:
        data['candlestick_pattern'] = None
    
    latest_date = data['Date'].max()
    
    start_index = 0
    if data['candlestick_pattern'].notna().any():
        latest_indicator_date = data.loc[data['candlestick_pattern'].notna(), 'Date'].max()
        start_index = data[data['Date'] == latest_indicator_date].index[0] + 1
    
    for i in range(start_index, len(data)):
        open_price = data['Open'].iloc[i]
        close_price = data['close'].iloc[i]
        high_price = data['high'].iloc[i]
        low_price = data['low'].iloc[i]
        
        if abs(open_price - close_price) < ((high_price - low_price) * 0.1):
            data.at[i, 'candlestick_pattern'] = 'doji'
        elif open_price > close_price and (open_price - close_price) > ((high_price - low_price) * 0.6) and (low_price == min(open_price, close_price)):
            data.at[i, 'candlestick_pattern'] = 'hammer'
        elif close_price > open_price and data['close'].iloc[i-1] < data['Open'].iloc[i-1] and (close_price > data['Open'].iloc[i-1]) and (open_price < data['close'].iloc[i-1]):
            data.at[i, 'candlestick_pattern'] = 'bullish_engulfing'
        elif open_price > close_price and data['close'].iloc[i-1] > data['Open'].iloc[i-1] and (open_price > data['close'].iloc[i-1]) and (close_price < data['Open'].iloc[i-1]):
            data.at[i, 'candlestick_pattern'] = 'bearish_engulfing'
        elif close_price < open_price and (open_price - close_price) > ((high_price - low_price) * 0.6) and (high_price == max(open_price, close_price)):
            data.at[i, 'candlestick_pattern'] = 'shooting_star'
    
    if fig:
        patterns = ['doji', 'hammer', 'bullish_engulfing', 'bearish_engulfing', 'shooting_star']
        colors = {
            'doji': 'yellow',
            'hammer': 'green',
            'bullish_engulfing': 'blue',
            'bearish_engulfing': 'red',
            'shooting_star': 'purple'
        }

        for pattern in patterns:
            pattern_data = data[data['candlestick_pattern'] == pattern]
            fig.add_trace(
                go.Scatter(
                    x=pattern_data['Date'],
                    y=pattern_data['close'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=colors[pattern],
                        symbol='circle'
                    ),
                    name=pattern.capitalize()
                ),
                row=3, col=1
            )
    
# HEAD AND SHOULDER
def calculate_head_and_shoulders(data, fig=None):
    # Identifying peaks and troughs
    data['peak'] = (data['close'] > data['close'].shift(1)) & (data['close'] > data['close'].shift(-1))
    data['trough'] = (data['close'] < data['close'].shift(1)) & (data['close'] < data['close'].shift(-1))
    
    peaks = data[data['peak']]
    troughs = data[data['trough']]
    
    shoulders = []
    head = None
    neckline = None
    
    # Detect Head and Shoulders pattern
    for i in range(1, len(peaks) - 1):
        if peaks['close'].iloc[i-1] < peaks['close'].iloc[i] and peaks['close'].iloc[i+1] < peaks['close'].iloc[i]:
            head = peaks.iloc[i]
            shoulders.append((peaks.iloc[i-1], peaks.iloc[i+1]))
            break
    
    if head is not None:
        for i in range(len(troughs) - 1):
            if troughs['Date'].iloc[i] < head['Date'] and troughs['Date'].iloc[i+1] > head['Date']:
                neckline = (troughs.iloc[i], troughs.iloc[i+1])
                break
    
    # Plotting if fig is provided
    if fig and head is not None and neckline is not None:
        fig.add_trace(go.Scatter(x=[shoulders[0][0]['Date'], head['Date'], shoulders[0][1]['Date']],
                                 y=[shoulders[0][0]['close'], head['close'], shoulders[0][1]['close']],
                                 mode='lines+markers', name='Head and Shoulders'), row=3, col=1)
        
        fig.add_trace(go.Scatter(x=[neckline[0]['Date'], neckline[1]['Date']],
                                 y=[neckline[0]['close'], neckline[1]['close']],
                                 mode='lines', name='Neckline'), row=3, col=1)
        
def calculate_head_and_shoulders_incremental(data, fig=None):
    # Check if 'peak' and 'trough' columns already exist and get the latest date
    if 'peak' in data.columns and 'trough' in data.columns:
        last_calculated_date = data.dropna(subset=['peak', 'trough']).index[-1]
        data_to_calculate = data[data.index > last_calculated_date]
    else:
        data_to_calculate = data.copy()
        data['peak'] = np.nan
        data['trough'] = np.nan

    # Identifying peaks and troughs
    data_to_calculate['peak'] = (data_to_calculate['close'] > data_to_calculate['close'].shift(1)) & (data_to_calculate['close'] > data_to_calculate['close'].shift(-1))
    data_to_calculate['trough'] = (data_to_calculate['close'] < data_to_calculate['close'].shift(1)) & (data_to_calculate['close'] < data_to_calculate['close'].shift(-1))
    
    data.update(data_to_calculate[['peak', 'trough']])
    
    peaks = data[data['peak']]
    troughs = data[data['trough']]
    
    shoulders = []
    head = None
    neckline = None
    
    # Detect Head and Shoulders pattern
    for i in range(1, len(peaks) - 1):
        if peaks['close'].iloc[i-1] < peaks['close'].iloc[i] and peaks['close'].iloc[i+1] < peaks['close'].iloc[i]:
            head = peaks.iloc[i]
            shoulders.append((peaks.iloc[i-1], peaks.iloc[i+1]))
            break
    
    if head is not None:
        for i in range(len(troughs) - 1):
            if troughs['Date'].iloc[i] < head['Date'] and troughs['Date'].iloc[i+1] > head['Date']:
                neckline = (troughs.iloc[i], troughs.iloc[i+1])
                break
    
    # Plotting if fig is provided
    if fig and head is not None and neckline is not None:
        fig.add_trace(go.Scatter(x=[shoulders[0][0]['Date'], head['Date'], shoulders[0][1]['Date']],
                                 y=[shoulders[0][0]['close'], head['close'], shoulders[0][1]['close']],
                                 mode='lines+markers', name='Head and Shoulders'), row=3, col=1)
        
        fig.add_trace(go.Scatter(x=[neckline[0]['Date'], neckline[1]['Date']],
                                 y=[neckline[0]['close'], neckline[1]['close']],
                                 mode='lines', name='Neckline'), row=3, col=1)


# Double Top/Bottom
def identify_double_top_bottom(data, fig=None):
    data['double_top'] = float('nan')
    data['double_bottom'] = float('nan')
    double_top_indices = []
    double_bottom_indices = []

    for i in range(1, len(data)-1):
        # Detecting Double Top
        if data['high'].iloc[i-1] < data['high'].iloc[i] and data['high'].iloc[i+1] < data['high'].iloc[i]:
            if i-2 >= 0 and data['high'].iloc[i-2] < data['high'].iloc[i-1] and data['high'].iloc[i+2] < data['high'].iloc[i+1]:
                double_top_indices.append(i)
                data['double_top'].iloc[i] = data['high'].iloc[i]

        # Detecting Double Bottom
        if data['low'].iloc[i-1] > data['low'].iloc[i] and data['low'].iloc[i+1] > data['low'].iloc[i]:
            if i-2 >= 0 and data['low'].iloc[i-2] > data['low'].iloc[i-1] and data['low'].iloc[i+2] > data['low'].iloc[i+1]:
                double_bottom_indices.append(i)
                data['double_bottom'].iloc[i] = data['low'].iloc[i]

    if fig:
        if double_top_indices:
            fig.add_trace(go.Scatter(
                x=[data['Date'].iloc[idx] for idx in double_top_indices],
                y=[data['high'].iloc[idx] for idx in double_top_indices],
                mode='markers',
                marker=dict(symbol='triangle-up', color='blue', size=10),
                name='Double Top'
            ))
        if double_bottom_indices:
            fig.add_trace(go.Scatter(
                x=[data['Date'].iloc[idx] for idx in double_bottom_indices],
                y=[data['low'].iloc[idx] for idx in double_bottom_indices],
                mode='markers',
                marker=dict(symbol='triangle-down', color='black', size=10),
                name='Double Bottom'
            ))

        # Add dotted lines connecting each double top to its corresponding double bottom
        for top_idx, bottom_idx in zip(double_top_indices, double_bottom_indices):
            fig.add_trace(go.Scatter(
                x=[data['Date'].iloc[top_idx], data['Date'].iloc[bottom_idx]],
                y=[data['high'].iloc[top_idx], data['low'].iloc[bottom_idx]],
                mode='lines',
                line=dict(color='green', width=2, dash='dot'),
                name='Double Top-Bottom Line',
                showlegend=False
            ))


def identify_double_top_bottom_incremental(data, fig=None):
    # Check if the indicator columns already exist
    if 'double_top' not in data.columns:
        data['double_top'] = float('nan')
    if 'double_bottom' not in data.columns:
        data['double_bottom'] = float('nan')

    double_top_indices = []
    double_bottom_indices = []

    # Find the last index where double_top or double_bottom is not NaN
    last_index = max(data['double_top'].last_valid_index() or 0, data['double_bottom'].last_valid_index() or 0)

    for i in range(last_index + 1, len(data)-1):
        # Detecting Double Top
        if data['high'].iloc[i-1] < data['high'].iloc[i] and data['high'].iloc[i+1] < data['high'].iloc[i]:
            if i-2 >= 0 and data['high'].iloc[i-2] < data['high'].iloc[i-1] and data['high'].iloc[i+2] < data['high'].iloc[i+1]:
                double_top_indices.append(i)
                data.at[i, 'double_top'] = data['high'].iloc[i]

        # Detecting Double Bottom
        if data['low'].iloc[i-1] > data['low'].iloc[i] and data['low'].iloc[i+1] > data['low'].iloc[i]:
            if i-2 >= 0 and data['low'].iloc[i-2] > data['low'].iloc[i-1] and data['low'].iloc[i+2] > data['low'].iloc[i+1]:
                double_bottom_indices.append(i)
                data.at[i, 'double_bottom'] = data['low'].iloc[i]

    if fig:
        if double_top_indices:
            fig.add_trace(go.Scatter(
                x=[data['Date'].iloc[idx] for idx in double_top_indices],
                y=[data['high'].iloc[idx] for idx in double_top_indices],
                mode='markers',
                marker=dict(symbol='triangle-up', color='blue', size=10),
                name='Double Top'
            ))
        if double_bottom_indices:
            fig.add_trace(go.Scatter(
                x=[data['Date'].iloc[idx] for idx in double_bottom_indices],
                y=[data['low'].iloc[idx] for idx in double_bottom_indices],
                mode='markers',
                marker=dict(symbol='triangle-down', color='black', size=10),
                name='Double Bottom'
            ))

        # Add dotted lines connecting each double top to its corresponding double bottom
        for top_idx, bottom_idx in zip(double_top_indices, double_bottom_indices):
            fig.add_trace(go.Scatter(
                x=[data['Date'].iloc[top_idx], data['Date'].iloc[bottom_idx]],
                y=[data['high'].iloc[top_idx], data['low'].iloc[bottom_idx]],
                mode='lines',
                line=dict(color='green', width=2, dash='dot'),
                name='Double Top-Bottom Line',
                showlegend=False
            ))

#VPT STRATEGY
def calculate_vpt(data, fig=None):
    """Calculate the Volume Price Trend (VPT) and optionally plot it."""
    vpt = [0]  # Start with VPT of 0
    for i in range(1, len(data)):
        vpt.append(vpt[-1] + (data['Volume'].iloc[i] * (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]))
    data['VPT'] = vpt
    
    if fig:
        # Add VPT line to the third subplot
        fig.add_trace(go.Scatter(x=data['Date'], y=data['VPT'], mode='lines', name='VPT'), row=3, col=1)


#CHAIKIN MONEY FLOW(CMF)
def calculate_cmf(data, fig=None):
    """Calculate the Chaikin Money Flow (CMF) and optionally plot it."""
    adl = []
    for i in range(len(data)):
        adl_value = ((data['close'].iloc[i] - data['low'].iloc[i]) - (data['high'].iloc[i] - data['close'].iloc[i])) / (data['high'].iloc[i] - data['low'].iloc[i]) * data['Volume'].iloc[i]
        adl.append(adl_value)
    
    data['ADL'] = adl
    data['MF_Multiplier'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
    data['MF_Volume'] = data['MF_Multiplier'] * data['Volume']
    
    cmf_values = []
    sum_mf_volume = 0
    sum_volume = 0
    
    for i in range(len(data)):
        sum_mf_volume += data['MF_Volume'].iloc[i]
        sum_volume += data['Volume'].iloc[i]
        
        if i >= 21:
            sum_volume_period = sum(data['Volume'].iloc[max(0, i-20):i+1])
            if sum_volume_period != 0:
                cmf = sum(data['MF_Volume'].iloc[max(0, i-20):i+1]) / (sum_volume_period + 1e-8)  # Adding a small epsilon to avoid division by zero
            else:
                cmf = 0  # Handle division by zero gracefully
            cmf_values.append(cmf)
        else:
            cmf_values.append(0)
    
    data['CMF'] = cmf_values
    
    if fig:
        # Add CMF line to the third subplot
        fig.add_trace(go.Scatter(x=data['Date'], y=data['CMF'], mode='lines', name='CMF'), row=3, col=1)


#Heikin Ashi Strategy
def calculate_heikin_ashi(data):
    """Calculate the Heikin-Ashi candlesticks and update the DataFrame in place."""
    heikin_ashi_close = (data['Open'] + data['high'] + data['low'] + data['close']) / 4
    heikin_ashi_open = [data['Open'].iloc[0]]  # Initialize the first HA open value as the traditional open value
    heikin_ashi_high = []
    heikin_ashi_low = []

    for i in range(len(data)):
        if i > 0:
            heikin_ashi_open.append((heikin_ashi_open[i-1] + heikin_ashi_close.iloc[i-1]) / 2)
        heikin_ashi_high.append(max(data['high'].iloc[i], heikin_ashi_open[i], heikin_ashi_close.iloc[i]))
        heikin_ashi_low.append(min(data['low'].iloc[i], heikin_ashi_open[i], heikin_ashi_close.iloc[i]))

    data['Heikin_Ashi_Open'] = heikin_ashi_open
    data['Heikin_Ashi_High'] = heikin_ashi_high
    data['Heikin_Ashi_Low'] = heikin_ashi_low
    data['Heikin_Ashi_Close'] = heikin_ashi_close



#ELLIOT WAVE STRATEGY
def identify_elliott_wave_patterns(df, fig=None):
    """
    Identify Elliott Wave patterns and optionally plot them.
    """
    df['Wave'] = None  # Initialize with None

    # Simplified pattern identification (for demonstration)
    for i in range(1, len(df)-1):
        if df['close'].iloc[i] > df['close'].iloc[i-1] and df['close'].iloc[i] > df['close'].iloc[i+1]:
            df['Wave'].iloc[i] = 'Impulse'
        elif df['close'].iloc[i] < df['close'].iloc[i-1] and df['close'].iloc[i] < df['close'].iloc[i+1]:
            df['Wave'].iloc[i] = 'Corrective'
    
    if fig:
        # Add wave patterns to the third subplot
        impulse_wave = df[df['Wave'] == 'Impulse']
        corrective_wave = df[df['Wave'] == 'Corrective']
        
        fig.add_trace(go.Scatter(x=impulse_wave['Date'], y=impulse_wave['close'], mode='markers+lines', name='Impulse Wave', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=corrective_wave['Date'], y=corrective_wave['close'], mode='markers+lines', name='Corrective Wave', line=dict(color='red')), row=3, col=1)

#DONCHIAN CHANNEL STARTEGY
def calculate_donchian_channels(data, n, fig=None):
    """Calculate the Donchian Channels and optionally plot them."""
    data['Upper_Channel'] = data['high'].rolling(window=n).max()
    data['Lower_Channel'] = data['low'].rolling(window=n).min()
    
    if fig:
        # Add Donchian Channels to the chart
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Upper_Channel'], mode='lines', name='Upper Channel'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Lower_Channel'], mode='lines', name='Lower Channel'), row=3, col=1)

#Flags and Pennants
def calculate_flag_and_add_trace(data, fig=None):
    data['flag_top'] = np.nan
    data['flag_bottom'] = np.nan

    min_periods = 5  # Number of periods to consider for flag/pennant identification
    
    for i in range(min_periods, len(data) - min_periods):
        # Identify potential flagpole
        if data['close'][i] > data['close'][i-1] * 1.05:  # 5% price increase as a placeholder
            flagpole_start = i-1
            flagpole_end = i
            flag_top = data['close'][i]
            flag_bottom = data['close'][flagpole_start]
            
            # Check for consolidation (flag/pennant formation)
            for j in range(i + 1, len(data)):
                if data['close'][j] < flag_top and data['close'][j] > flag_bottom:
                    continue
                else:
                    if data['close'][j] > flag_top:  # Breakout to the upside
                        data.loc[flagpole_start:j, 'flag_top'] = flag_top
                        data.loc[flagpole_start:j, 'flag_bottom'] = flag_bottom
                        break
                    else:
                        break
    
    data['flag_top'] = data['flag_top'].interpolate()
    data['flag_bottom'] = data['flag_bottom'].interpolate()
    
    if fig:
        # Add flag formation to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data['flag_top'], mode='lines', name='Flag Top'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['flag_bottom'], mode='lines', name='Flag Bottom'), row=3, col=1)

def calculate_flag_and_add_trace_incremental(data, fig=None, start_date=None):
    if 'flag_top' not in data.columns or 'flag_bottom' not in data.columns:
        data['flag_top'] = np.nan
        data['flag_bottom'] = np.nan

    min_periods = 5  # Number of periods to consider for flag/pennant identification
    
    start_idx = 0
    if start_date:
        start_idx = data.index[data['Date'] > start_date][0]
    
    for i in range(max(min_periods, start_idx), len(data) - min_periods):
        # Identify potential flagpole
        if data['close'][i] > data['close'][i-1] * 1.05:  # 5% price increase as a placeholder
            flagpole_start = i-1
            flagpole_end = i
            flag_top = data['close'][i]
            flag_bottom = data['close'][flagpole_start]
            
            # Check for consolidation (flag/pennant formation)
            for j in range(i + 1, len(data)):
                if data['close'][j] < flag_top and data['close'][j] > flag_bottom:
                    continue
                else:
                    if data['close'][j] > flag_top:  # Breakout to the upside
                        data.loc[flagpole_start:j, 'flag_top'] = flag_top
                        data.loc[flagpole_start:j, 'flag_bottom'] = flag_bottom
                        break
                    else:
                        break
    
    data['flag_top'] = data['flag_top'].interpolate()
    data['flag_bottom'] = data['flag_bottom'].interpolate()
    
    if fig:
        # Add flag formation to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data['flag_top'], mode='lines', name='Flag Top'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['flag_bottom'], mode='lines', name='Flag Bottom'), row=3, col=1)


# Triangles
def calculate_triangle_and_add_trace(data, min_periods ,fig=None):
    data[f'upper_trendline_{min_periods}'] = np.nan
    data[f'lower_trendline_{min_periods}'] = np.nan

    # min_periods : Number of periods to consider for local minima and maxima
    
    for i in range(min_periods, len(data) - min_periods):
        local_min = data['close'][i-min_periods:i+min_periods].min()
        local_max = data['close'][i-min_periods:i+min_periods].max()
        
        if data['close'][i] == local_min:
            data[f'lower_trendline_{min_periods}'][i] = local_min
        if data['close'][i] == local_max:
            data[f'upper_trendline_{min_periods}'][i] = local_max

    data[f'lower_trendline_{min_periods}'] = data[f'lower_trendline_{min_periods}'].interpolate()
    data[f'upper_trendline_{min_periods}'] = data[f'upper_trendline_{min_periods}'].interpolate()
    
    if fig:
        # Add upper trendline to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'upper_trendline_{min_periods}'], mode='lines', name='Upper Trendline'), row=1, col=1)
        # Add lower trendline to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'lower_trendline_{min_periods}'], mode='lines', name='Lower Trendline'), row=1, col=1)

def calculate_triangle_and_add_trace_incremental(data, min_periods, fig=None):
    upper_col = f'upper_trendline_{min_periods}'
    lower_col = f'lower_trendline_{min_periods}'
    
    if upper_col not in data.columns or lower_col not in data.columns:
        data[upper_col] = np.nan
        data[lower_col] = np.nan
    
    # Find the latest date for which the indicators are calculated
    last_calculated_index = data.dropna(subset=[upper_col, lower_col]).index.max() if not data.dropna(subset=[upper_col, lower_col]).empty else min_periods

    # Start calculation from the next day after the last calculated index
    start_index = last_calculated_index + 1 if last_calculated_index is not None else min_periods

    for i in range(start_index, len(data) - min_periods):
        local_min = data['close'][i-min_periods:i+min_periods].min()
        local_max = data['close'][i-min_periods:i+min_periods].max()
        
        if data['close'][i] == local_min:
            data[lower_col][i] = local_min
        if data['close'][i] == local_max:
            data[upper_col][i] = local_max

    data[lower_col] = data[lower_col].interpolate()
    data[upper_col] = data[upper_col].interpolate()
    
    if fig:
        # Add upper trendline to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[upper_col], mode='lines', name='Upper Trendline'), row=1, col=1)
        # Add lower trendline to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[lower_col], mode='lines', name='Lower Trendline'), row=1, col=1)

#GANN ANGLES
def calculate_gann_angles(data, key_price_points, angles, fig=None):
    """Calculate and plot Gann Angles from key price points."""
    for key_price in key_price_points:
        for angle in angles:
            # Calculate the slope based on the angle (assuming 1 unit of time = 1 unit of price)
            slope = np.tan(np.radians(angle))
            gann_line = [key_price + slope * (i - key_price_points.index(key_price)) for i in range(len(data))]
            data[f'Gann_{angle}_{key_price}'] = gann_line
            
            if fig:
                # Add Gann Angle line to the third subplot
                fig.add_trace(go.Scatter(x=data['Date'], y=gann_line, mode='lines', name=f'Gann {angle}° from {key_price}'), row=3, col=1)


#MOMENTUM INDICATOR
def calculate_momentum(data, n, fig=None):
    """Calculate the Momentum indicator and optionally plot it."""
    momentum = data['close'] - data['close'].shift(n)
    data['Momentum'] = momentum
    
    if fig:
        # Add Momentum line to the third subplot
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Momentum'], mode='lines', name='Momentum'), row=3, col=1)



#MONEY FLOW INDEX
import pandas as pd
import plotly.graph_objs as go

def calculate_mfi(data, n, fig=None):
    """Calculate the Money Flow Index (MFI) and optionally plot it."""
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    money_flow = typical_price * data['Volume']

    positive_flow = []
    negative_flow = []

    for i in range(1, len(data)):
        if typical_price[i] > typical_price[i-1]:
            positive_flow.append(money_flow[i])
            negative_flow.append(0)
        elif typical_price[i] < typical_price[i-1]:
            positive_flow.append(0)
            negative_flow.append(money_flow[i])
        else:
            positive_flow.append(0)
            negative_flow.append(0)

    positive_mf = pd.Series(positive_flow).rolling(window=n).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=n).sum()

    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    data['MFI'] = mfi

    if fig:
        # Add MFI to the chart
        fig.add_trace(go.Scatter(x=data['Date'], y=data['MFI'], mode='lines', name='MFI'), row=3, col=1)


#TRIX INDICATOR
def calculate_trix(data, n, fig=None):
    """Calculate the TRIX indicator and optionally plot it."""
    # Calculate the single smoothed EMA
    ema1 = data['close'].ewm(span=n, adjust=False).mean()
    # Calculate the double smoothed EMA
    ema2 = ema1.ewm(span=n, adjust=False).mean()
    # Calculate the triple smoothed EMA
    ema3 = ema2.ewm(span=n, adjust=False).mean()
    # Calculate the 1-period rate-of-change (ROC) of the triple smoothed EMA
    trix = ema3.pct_change() * 100
    data['TRIX'] = trix

    # Calculate the signal line (9-period EMA of the TRIX)
    signal_line = trix.ewm(span=9, adjust=False).mean()
    data['TRIX_Signal'] = signal_line

    if fig:
        # Add TRIX and signal line to the chart
        fig.add_trace(go.Scatter(x=data['Date'], y=data['TRIX'], mode='lines', name='TRIX'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['TRIX_Signal'], mode='lines', name='TRIX Signal'), row=3, col=1)


#Price Rate of Change (PROC) Strategy
def calculate_proc(data, n, fig=None):
    """Calculate the Price Rate of Change (PROC) and optionally plot it."""
    proc = ((data['close'] - data['close'].shift(n)) / data['close'].shift(n)) * 100
    data['PROC'] = proc

    if fig:
        # Add PROC to the chart
        fig.add_trace(go.Scatter(x=data['Date'], y=data['PROC'], mode='lines', name='PROC'), row=3, col=1)

#VORTEX INDICATOR STRATREGY
def calculate_vortex(data, n, fig=None):
    """Calculate the Vortex Indicator (VI) and optionally plot it."""
    tr = np.maximum(data['high'] - data['low'], np.maximum(abs(data['high'] - data['close'].shift(1)), abs(data['low'] - data['close'].shift(1))))
    atr = tr.rolling(window=n).sum()

    vm_plus = abs(data['high'] - data['low'].shift(1))
    vm_minus = abs(data['low'] - data['high'].shift(1))
    
    vi_plus = vm_plus.rolling(window=n).sum() / atr
    vi_minus = vm_minus.rolling(window=n).sum() / atr

    data['VI+'] = vi_plus
    data['VI-'] = vi_minus

    if fig:
        # Add VI+ and VI- to the chart
        fig.add_trace(go.Scatter(x=data['Date'], y=data['VI+'], mode='lines', name='VI+'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['VI-'], mode='lines', name='VI-'), row=3, col=1)

# Rate of Change
def calculate_roc_and_add_trace(data, window, fig=None):
    # Calculate ROC
    roc_col = f'roc_{window}'
    data[roc_col] = (data['close'].diff(window) / data['close'].shift(window)) * 100
    
    if fig:
        # Add ROC line to the fourth subplot (adjust the subplot as needed)
        fig.add_trace(go.Scatter(x=data['Date'], y=data[roc_col], mode='lines', name=f'ROC_{window}'), row=3, col=1)

def calculate_roc_and_add_trace_incremental(data, window, fig=None):
    # Calculate ROC
    roc_col = f'roc_{window}'
    
    # Check if the ROC column already exists
    if roc_col in data.columns:
        # Get the latest date for which ROC is defined
        last_defined_date = data[data[roc_col].notna()]['Date'].max()
        start_idx = data[data['Date'] > last_defined_date].index[0] if not pd.isnull(last_defined_date) else window
    else:
        start_idx = window
        data[roc_col] = float('nan')
    
    # Calculate ROC for the new data points
    data.loc[start_idx:, roc_col] = (data['close'].diff(window) / data['close'].shift(window)) * 100
    
    if fig:
        # Add ROC line to the fourth subplot (adjust the subplot as needed)
        fig.add_trace(go.Scatter(x=data['Date'], y=data[roc_col], mode='lines', name=f'ROC_{window}'), row=3, col=1)

# Commodity Channel Index
def calculate_cci_and_add_trace(data, window=20, fig=None):
    # Calculate Typical Price
    data['Typical_Price'] = (data['high'] + data['low'] + data['close']) / 3
    
    # Calculate the SMA of Typical Price
    data[f'SMA_Typical_Price_{window}'] = data['Typical_Price'].rolling(window=window).mean()
    
    # Calculate the Mean Deviation
    def mean_deviation(x):
        return np.mean(np.abs(x - np.mean(x)))
    
    data['Mean_Deviation'] = data['Typical_Price'].rolling(window=window).apply(mean_deviation)
    
    # Calculate CCI
    cci_col = f'cci_{window}'
    data[cci_col] = (data['Typical_Price'] - data[f'SMA_Typical_Price_{window}']) / (0.015 * data['Mean_Deviation'])
    
    if fig:
        # Add CCI line to the fourth subplot (adjust the subplot as needed)
        fig.add_trace(go.Scatter(x=data['Date'], y=data[cci_col], mode='lines', name=f'CCI_{window}'), row=3, col=1)

def calculate_cci_and_add_trace_incremental(data, window=20, fig=None):
    cci_col = f'cci_{window}'

    # Check if CCI already exists up to the latest date
    if cci_col in data.columns:
        last_calculated_date = data.dropna(subset=[cci_col]).index[-1]
        new_data = data.loc[last_calculated_date:]
    else:
        new_data = data

    if len(new_data) <= window:
        # Not enough data to calculate CCI
        return

    # Calculate Typical Price
    new_data['Typical_Price'] = (new_data['high'] + new_data['low'] + new_data['close']) / 3
    
    # Calculate the SMA of Typical Price
    new_data[f'SMA_Typical_Price_{window}'] = new_data['Typical_Price'].rolling(window=window).mean()
    
    # Calculate the Mean Deviation
    def mean_deviation(x):
        return np.mean(np.abs(x - np.mean(x)))
    
    new_data['Mean_Deviation'] = new_data['Typical_Price'].rolling(window=window).apply(mean_deviation)
    
    # Calculate CCI
    new_data[cci_col] = (new_data['Typical_Price'] - new_data[f'SMA_Typical_Price_{window}']) / (0.015 * new_data['Mean_Deviation'])
    
    # Update the original data with the new CCI values
    data.update(new_data)

    if fig:
        # Add CCI line to the fourth subplot (adjust the subplot as needed)
        fig.add_trace(go.Scatter(x=data['Date'], y=data[cci_col], mode='lines', name=f'CCI_{window}'), row=3, col=1)

# Willian %R
def calculate_williams_r_and_add_trace(data, window, fig=None):
    # Calculate Williams %R
    data[f'Highest_High_{window}'] = data['high'].rolling(window=window).max()
    data[f'Lowest_Low_{window}'] = data['low'].rolling(window=window).min()
    williams_r_col = f'williams_%R_{window}'
    data[williams_r_col] = (data[f'Highest_High_{window}'] - data['close']) / (data[f'Highest_High_{window}'] - data[f'Lowest_Low_{window}']) * -100

    if fig:
        # Add Williams %R line to the fourth subplot (adjust the subplot as needed)
        fig.add_trace(go.Scatter(x=data['Date'], y=data[williams_r_col], mode='lines', name=f'Williams %R_{window}'), row=3, col=1)

def calculate_williams_r_and_add_trace_incremental(data, window, fig=None):
    # Check if Williams %R is already calculated
    williams_r_col = f'williams_%R_{window}'
    if williams_r_col in data.columns:
        last_calculated_date = data.dropna(subset=[williams_r_col])['Date'].max()
        start_idx = data[data['Date'] == last_calculated_date].index[0] + 1
    else:
        start_idx = window
        data[f'Highest_High_{window}'] = float('nan')
        data[f'Lowest_Low_{window}'] = float('nan')
        data[williams_r_col] = float('nan')

    # Calculate Williams %R for new data
    for i in range(start_idx, len(data)):
        data.at[i, f'Highest_High_{window}'] = data['high'].iloc[i-window:i].max()
        data.at[i, f'Lowest_Low_{window}'] = data['low'].iloc[i-window:i].min()
        data.at[i, williams_r_col] = (data[f'Highest_High_{window}'].iloc[i] - data['close'].iloc[i]) / (data[f'Highest_High_{window}'].iloc[i] - data[f'Lowest_Low_{window}'].iloc[i]) * -100

    if fig:
        # Add Williams %R line to the fourth subplot (adjust the subplot as needed)
        fig.add_trace(go.Scatter(x=data['Date'], y=data[williams_r_col], mode='lines', name=f'Williams %R_{window}'), row=3, col=1)
# Pivot Points

def calculate_pivot_points_and_add_trace(data, fig=None):
    # Calculate Pivot Points, Support and Resistance Levels
    data['pivot_point'] = (data['high'] + data['low'] + data['close']) / 3
    data['support_1'] = (2 * data['pivot_point']) - data['high']
    data['resistance_1'] = (2 * data['pivot_point']) - data['low']
    data['support_2'] = data['pivot_point'] - (data['high'] - data['low'])
    data['resistance_2'] = data['pivot_point'] + (data['high'] - data['low'])
    data['support_3'] = data['low'] - 2 * (data['high'] - data['pivot_point'])
    data['resistance_3'] = data['high'] + 2 * (data['pivot_point'] - data['low'])
    
    if fig:
        # Add pivot points and support/resistance levels to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data['pivot_point'], mode='lines', name='Pivot Point'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['support_1'], mode='lines', name='Support 1'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['resistance_1'], mode='lines', name='Resistance 1'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['support_2'], mode='lines', name='Support 2'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['resistance_2'], mode='lines', name='Resistance 2'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['support_3'], mode='lines', name='Support 3'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['resistance_3'], mode='lines', name='Resistance 3'), row=3, col=1)

def calculate_pivot_points_and_add_trace_incremental(data, fig=None):
    # Check if pivot points are already calculated
    if 'pivot_point' in data.columns:
        # Find the last date where pivot points were calculated
        last_calculated_date = data.dropna(subset=['pivot_point'])['Date'].max()
        # Filter the data to calculate new values only from the next day
        new_data = data[data['Date'] > last_calculated_date]
    else:
        new_data = data

    # Calculate Pivot Points, Support and Resistance Levels for new data
    new_data['pivot_point'] = (new_data['high'] + new_data['low'] + new_data['close']) / 3
    new_data['support_1'] = (2 * new_data['pivot_point']) - new_data['high']
    new_data['resistance_1'] = (2 * new_data['pivot_point']) - new_data['low']
    new_data['support_2'] = new_data['pivot_point'] - (new_data['high'] - new_data['low'])
    new_data['resistance_2'] = new_data['pivot_point'] + (new_data['high'] - new_data['low'])
    new_data['support_3'] = new_data['low'] - 2 * (new_data['high'] - new_data['pivot_point'])
    new_data['resistance_3'] = new_data['high'] + 2 * (new_data['pivot_point'] - new_data['low'])

    # Update the original data with new calculations
    for col in ['pivot_point', 'support_1', 'resistance_1', 'support_2', 'resistance_2', 'support_3', 'resistance_3']:
        data.loc[data['Date'].isin(new_data['Date']), col] = new_data[col]

    if fig:
        # Add pivot points and support/resistance levels to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data['pivot_point'], mode='lines', name='Pivot Point'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['support_1'], mode='lines', name='Support 1'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['resistance_1'], mode='lines', name='Resistance 1'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['support_2'], mode='lines', name='Support 2'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['resistance_2'], mode='lines', name='Resistance 2'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['support_3'], mode='lines', name='Support 3'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['resistance_3'], mode='lines', name='Resistance 3'), row=3, col=1)

# ATR
def calculate_atr_and_add_trace(data, window, fig=None):
    # Calculate True Range (TR)
    data['high_low'] = data['high'] - data['low']
    data['high_close'] = abs(data['high'] - data['close'].shift(1))
    data['low_close'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)

    # Calculate ATR
    data[f'atr_{window}'] = data['true_range'].rolling(window=window).mean()

    # Debug prints to ensure calculations are correct
    print(data[['Date', 'true_range', f'atr_{window}']].head())

    if fig:
        # Add ATR line to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'atr_{window}'], mode='lines', name='ATR'), row=3, col=1)

def calculate_atr_and_add_trace_incremental(data, window, fig=None):
    # Calculate True Range (TR)
    data['high_low'] = data['high'] - data['low']
    data['high_close'] = abs(data['high'] - data['close'].shift(1))
    data['low_close'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)

    atr_col = f'atr_{window}'

    if atr_col in data.columns:
        last_calculated_date = data.dropna(subset=[atr_col]).iloc[-1]['Date']
        start_index = data[data['Date'] == last_calculated_date].index[0] + 1
    else:
        last_calculated_date = None
        start_index = 0

    # Calculate ATR
    data.loc[start_index:, atr_col] = data['true_range'].rolling(window=window).mean()

    # Debug prints to ensure calculations are correct
    print(data[['Date', 'true_range', atr_col]].head())

    if fig:
        # Add ATR line to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[atr_col], mode='lines', name='ATR'), row=3, col=1)



#Keltner Channels
def calculate_keltner_channels_and_add_trace(data, ema_window, atr_window, atr_multiplier, fig=None):
    # Calculate the 20-day EMA for the middle line
    data[f'ema_{ema_window}'] = data['close'].ewm(span=ema_window, adjust=False).mean()

    # Calculate True Range (TR)
    data['high_low'] = data['high'] - data['low']
    data['high_close'] = abs(data['high'] - data['close'].shift(1))
    data['low_close'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)

    # Calculate ATR
    data[f'atr_{atr_window}'] = data['true_range'].rolling(window=atr_window).mean()

    # Calculate Upper and Lower Channel Lines
    data[f'upper_channel'] = data[f'ema_{ema_window}'] + (atr_multiplier * data[f'atr_{atr_window}'])
    data[f'lower_channel'] = data[f'ema_{ema_window}'] - (atr_multiplier * data[f'atr_{atr_window}'])

    # Debug prints to ensure calculations are correct
    print(data[['Date', f'ema_{ema_window}', f'upper_channel', f'lower_channel']].head())

    if fig:
        # Add EMA (middle line) to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'ema_{ema_window}'], mode='lines', name='EMA'), row=1, col=1)
        # Add Upper Channel line to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'upper_channel'], mode='lines', name='Upper Channel'), row=1, col=1)
        # Add Lower Channel line to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'lower_channel'], mode='lines', name='Lower Channel'), row=1, col=1)

def calculate_keltner_channels_and_add_trace_incremental(data, ema_window, atr_window, atr_multiplier, fig=None):
    # Check if the required columns exist, if not, create them
    for col in [f'ema_{ema_window}', f'atr_{atr_window}', 'upper_channel', 'lower_channel']:
        if col not in data.columns:
            data[col] = float('nan')
    
    # Find the last index where all required values are non-NaN
    last_calculated_index = data.dropna(subset=[f'ema_{ema_window}', f'atr_{atr_window}', 'upper_channel', 'lower_channel']).index[-1] if not data.dropna(subset=[f'ema_{ema_window}', f'atr_{atr_window}', 'upper_channel', 'lower_channel']).empty else -1

    # Calculate the 20-day EMA for the middle line
    if last_calculated_index == -1:
        data[f'ema_{ema_window}'] = data['close'].ewm(span=ema_window, adjust=False).mean()
    else:
        data.loc[last_calculated_index+1:, f'ema_{ema_window}'] = data['close'].ewm(span=ema_window, adjust=False).mean().iloc[last_calculated_index+1:]

    # Calculate True Range (TR)
    data['high_low'] = data['high'] - data['low']
    data['high_close'] = abs(data['high'] - data['close'].shift(1))
    data['low_close'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)

    # Calculate ATR
    if last_calculated_index == -1:
        data[f'atr_{atr_window}'] = data['true_range'].rolling(window=atr_window).mean()
    else:
        data.loc[last_calculated_index+1:, f'atr_{atr_window}'] = data['true_range'].rolling(window=atr_window).mean().iloc[last_calculated_index+1:]

    # Calculate Upper and Lower Channel Lines
    if last_calculated_index == -1:
        data[f'upper_channel'] = data[f'ema_{ema_window}'] + (atr_multiplier * data[f'atr_{atr_window}'])
        data[f'lower_channel'] = data[f'ema_{ema_window}'] - (atr_multiplier * data[f'atr_{atr_window}'])
    else:
        data.loc[last_calculated_index+1:, f'upper_channel'] = data[f'ema_{ema_window}'] + (atr_multiplier * data[f'atr_{atr_window}'])
        data.loc[last_calculated_index+1:, f'lower_channel'] = data[f'ema_{ema_window}'] - (atr_multiplier * data[f'atr_{atr_window}'])

    # Debug prints to ensure calculations are correct
    print(data[['Date', f'ema_{ema_window}', f'upper_channel', f'lower_channel']].head())

    if fig:
        # Add EMA (middle line) to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'ema_{ema_window}'], mode='lines', name='EMA'), row=1, col=1)
        # Add Upper Channel line to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'upper_channel'], mode='lines', name='Upper Channel'), row=1, col=1)
        # Add Lower Channel line to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'lower_channel'], mode='lines', name='Lower Channel'), row=1, col=1)


# Price Channels

def calculate_price_channels_and_add_trace(data, window, fig=None):
    """
    Calculate the Price Channels indicator and add it to the plot.
    Upper Channel = Highest high over the last 'window' periods
    Lower Channel = Lowest low over the last 'window' periods
    """
    # Calculate the upper and lower price channels
    data[f'upper_channel'] = data['high'].rolling(window=window).max()
    data[f'lower_channel'] = data['low'].rolling(window=window).min()

    # Debug prints to ensure calculations are correct
    print(data[['Date', f'upper_channel', f'lower_channel']].head())

    if fig:
        # Add Upper Channel line to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'upper_channel'], mode='lines', name='Upper Channel'), row=3, col=1)
        # Add Lower Channel line to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'lower_channel'], mode='lines', name='Lower Channel'), row=3, col=1)

def calculate_price_channels_and_add_trace_incremental(data, window, start_date=None, fig=None):
    """
    Calculate the Price Channels indicator and add it to the plot.
    Upper Channel = Highest high over the last 'window' periods
    Lower Channel = Lowest low over the last 'window' periods
    """
    # Ensure start_date is a datetime object
    if start_date:
        start_date = pd.to_datetime(start_date)
        start_index = data[data['Date'] > start_date].index[0]
        data = data.loc[start_index-window+1:]
        
    data['upper_channel'] = data['high'].rolling(window=window).max()
    data['lower_channel'] = data['low'].rolling(window=window).min()

    # Debug prints to ensure calculations are correct
    print(data[['Date', 'upper_channel', 'lower_channel']].head())

    if fig:
        # Add Upper Channel line to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data['upper_channel'], mode='lines', name='Upper Channel'), row=3, col=1)
        # Add Lower Channel line to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data['lower_channel'], mode='lines', name='Lower Channel'), row=3, col=1)

#RVI Strategy


def calculate_RVI(data, window=10, fig=None):
    if 'RVI' not in data.columns:
        data['RVI'] = float('nan')

    last_index = data['RVI'].last_valid_index()
    start_index = window if last_index is None else last_index + 1

    close_open_diff = data['close'] - data['Open']
    high_low_diff = data['high'] - data['low']
    
    close_open_diff_roll = close_open_diff.rolling(window=window).mean()
    high_low_diff_roll = high_low_diff.rolling(window=window).mean()

    data.loc[start_index:, 'RVI'] = close_open_diff_roll[start_index:] / high_low_diff_roll[start_index:]

    # Fill NaN values to ensure the length matches
    data['RVI'].fillna(method='ffill', inplace=True)
    data['RVI'].fillna(0, inplace=True)

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['RVI'], mode='lines', name='RVI'), row=3, col=1)
        fig.add_shape(
            type="line", line=dict(color="black", width=1, dash="dash"),
            x0=data['Date'].iloc[0], y0=0, x1=data['Date'].iloc[-1], y1=0,
            row=3, col=1
        )

def implement_RVI(data, toPlot=False):
    calculate_RVI(data)
    
    # Example of generating buy/sell signals
    data['buy_signal'] = (data['RVI'] > 0).astype(int)
    data['sell_signal'] = (data['RVI'] < 0).astype(int)
    
    if toPlot:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
        calculate_RVI(data, fig)
        fig.show()

    return data



#Volume Oscillator


def calculate_volume_oscillator(data, short_window=14, long_window=28, fig=None):
    if 'Volume_Oscillator' not in data.columns:
        data['Volume_Oscillator'] = float('nan')

    last_index = data['Volume_Oscillator'].last_valid_index()
    start_index = long_window if last_index is None else last_index + 1

    volume_oscillator = data['Volume_Oscillator'].tolist()[:start_index]

    for i in range(start_index, len(data)):
        short_ma = data['Volume'].iloc[i-short_window+1:i+1].mean()
        long_ma = data['Volume'].iloc[i-long_window+1:i+1].mean()
        vol_osc = (short_ma - long_ma) / long_ma
        volume_oscillator.append(vol_osc)

    data['Volume_Oscillator'] = pd.Series(volume_oscillator, index=data.index)

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Volume_Oscillator'], mode='lines', name='Volume Oscillator'), row=3, col=1)


#CMO Strategy


def calculate_CMO(data, window=14, fig=None):
    if 'CMO' not in data.columns:
        data['CMO'] = float('nan')

    last_index = data['CMO'].last_valid_index()
    start_index = window if last_index is None else last_index + 1

    cmo = data['CMO'].tolist()[:start_index]

    for i in range(start_index, len(data)):
        close_diff = data['close'].iloc[i-window+1:i+1].diff()
        upward_changes = close_diff.clip(lower=0)
        downward_changes = -close_diff.clip(upper=0)
        sum_upward = upward_changes.sum()
        sum_downward = downward_changes.sum()
        cmo_value = (sum_upward - sum_downward) / (sum_upward + sum_downward) * 100
        cmo.append(cmo_value)

    data['CMO'] = pd.Series(cmo, index=data.index)

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['CMO'], mode='lines', name='CMO'), row=3, col=1)
        fig.add_shape(
            type="line", line=dict(color="black", width=1, dash="dash"),
            x0=data['Date'].iloc[0], y0=0, x1=data['Date'].iloc[-1], y1=0,
            row=3, col=1
        )



#Aroon Strategy


def calculate_aroon(data, window=25, fig=None):
    if 'Aroon Up' not in data.columns:
        data['Aroon Up'] = float('nan')
        data['Aroon Down'] = float('nan')

    last_index = data['Aroon Up'].last_valid_index()
    start_index = window if last_index is None else last_index + 1

    aroon_up = data['Aroon Up'].tolist()[:start_index]
    aroon_down = data['Aroon Down'].tolist()[:start_index]

    for i in range(start_index, len(data)):
        aroon_up_value = (window - data['high'].iloc[i-window+1:i+1].argmax()) / window * 100
        aroon_down_value = (window - data['low'].iloc[i-window+1:i+1].argmin()) / window * 100
        aroon_up.append(aroon_up_value)
        aroon_down.append(aroon_down_value)

    data['Aroon Up'] = pd.Series(aroon_up, index=data.index)
    data['Aroon Down'] = pd.Series(aroon_down, index=data.index)

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Aroon Up'], mode='lines', name='Aroon Up'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Aroon Down'], mode='lines', name='Aroon Down'), row=3, col=1)


#Ultimate Oscillator

def calculate_ultimate_oscillator(data, short_period=7, medium_period=14, long_period=28, fig=None):
    if 'Ultimate Oscillator' not in data.columns:
        data['Ultimate Oscillator'] = float('nan')

    last_index = data['Ultimate Oscillator'].last_valid_index()
    start_index = long_period if last_index is None else last_index + 1

    min_low_or_prev_close = data[['low', 'close']].min(axis=1).shift(1)
    true_range = data['high'].combine(min_low_or_prev_close, max) - data['low'].combine(min_low_or_prev_close, min)
    buying_pressure = data['close'] - min_low_or_prev_close

    ultimate_oscillator = data['Ultimate Oscillator'].tolist()[:start_index]

    for i in range(start_index, len(data)):
        avg_bp1 = buying_pressure.iloc[i-short_period+1:i+1].sum()
        avg_tr1 = true_range.iloc[i-short_period+1:i+1].sum()

        avg_bp2 = buying_pressure.iloc[i-medium_period+1:i+1].sum()
        avg_tr2 = true_range.iloc[i-medium_period+1:i+1].sum()

        avg_bp3 = buying_pressure.iloc[i-long_period+1:i+1].sum()
        avg_tr3 = true_range.iloc[i-long_period+1:i+1].sum()

        uo = 100 * ((4 * (avg_bp1 / avg_tr1)) + (2 * (avg_bp2 / avg_tr2)) + (avg_bp3 / avg_tr3)) / (4 + 2 + 1)
        ultimate_oscillator.append(uo)

    data['Ultimate Oscillator'] = pd.Series(ultimate_oscillator, index=data.index)

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Ultimate Oscillator'], mode='lines', name='Ultimate Oscillator'), row=3, col=1)
        fig.add_shape(
            type="line", line=dict(color="red", width=1, dash="dash"),
            x0=data['Date'].iloc[0], y0=70, x1=data['Date'].iloc[-1], y1=70,
            row=3, col=1
        )
        fig.add_shape(
            type="line", line=dict(color="blue", width=1, dash="dash"),
            x0=data['Date'].iloc[0], y0=30, x1=data['Date'].iloc[-1], y1=30,
            row=3, col=1
        )


#Chandelier Exit Strategy
def calculate_chandelier_exit(data, window=22, multiplier=3, fig=None):
    if 'Chandelier Exit Long' not in data.columns:
        data['Chandelier Exit Long'] = float('nan')
        data['Chandelier Exit Short'] = float('nan')

    last_index = data['Chandelier Exit Long'].last_valid_index()
    start_index = window if last_index is None else last_index + 1

    chandelier_exit_long = data['Chandelier Exit Long'].tolist()[:start_index]
    chandelier_exit_short = data['Chandelier Exit Short'].tolist()[:start_index]

    for i in range(start_index, len(data)):
        high_max = data['high'].iloc[i-window+1:i+1].max()
        low_min = data['low'].iloc[i-window+1:i+1].min()
        atr = data['high'].iloc[i-window+1:i+1].combine(data['low'].iloc[i-window+1:i+1], lambda x, y: abs(x - y)).mean()

        cel = high_max - (atr * multiplier)
        ces = low_min + (atr * multiplier)

        chandelier_exit_long.append(cel)
        chandelier_exit_short.append(ces)

    data['Chandelier Exit Long'] = pd.Series(chandelier_exit_long, index=data.index)
    data['Chandelier Exit Short'] = pd.Series(chandelier_exit_short, index=data.index)

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Chandelier Exit Long'], mode='lines', name='Chandelier Exit Long'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Chandelier Exit Short'], mode='lines', name='Chandelier Exit Short'), row=3, col=1)




#DMI Strategy


def calculate_dmi(data, window=14, fig=None):
    if '+DI' not in data.columns:
        data['+DI'] = float('nan')
        data['-DI'] = float('nan')
        data['ADX'] = float('nan')

    last_index = data['+DI'].last_valid_index()
    start_index = window if last_index is None else last_index + 1

    di_plus_list = data['+DI'].tolist()[:start_index]
    di_minus_list = data['-DI'].tolist()[:start_index]
    adx_list = data['ADX'].tolist()[:start_index]

    for i in range(start_index, len(data)):
        high = data['high'].iloc[i-window+1:i+1]
        low = data['low'].iloc[i-window+1:i+1]
        close = data['close'].iloc[i-window+1:i+1]

        tr = np.maximum(np.maximum(high - low, abs(high - close.shift(1))), abs(low - close.shift(1)))
        dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), np.maximum(high - high.shift(1), 0), 0)
        dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), np.maximum(low.shift(1) - low, 0), 0)

        atr = tr.rolling(window=window).mean()
        di_plus = (pd.Series(dm_plus).rolling(window=window).mean() / atr) * 100
        di_minus = (pd.Series(dm_minus).rolling(window=window).mean() / atr) * 100

        dx = 100 * np.abs((di_plus - di_minus) / (di_plus + di_minus))
        adx = dx.rolling(window=window).mean().iloc[-1]

        di_plus_list.append(di_plus.iloc[-1])
        di_minus_list.append(di_minus.iloc[-1])
        adx_list.append(adx)

    data['+DI'] = pd.Series(di_plus_list, index=data.index)
    data['-DI'] = pd.Series(di_minus_list, index=data.index)
    data['ADX'] = pd.Series(adx_list, index=data.index)

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['+DI'], mode='lines', name='+DI'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['-DI'], mode='lines', name='-DI'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['ADX'], mode='lines', name='ADX'), row=3, col=1)

        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="DMI", row=10, col=1)


        fig.add_annotation(
        dict(
        x=0.5,
        y=0.285,  # Adjust this value to position the title within the subplot
        xref='x3 domain',
        yref='paper',  # Use paper reference for y
        text="DMI",
        showarrow=False,
        font=dict(size=16),
        xanchor='center',
        yanchor='bottom'
    )
)






# ADL Strategy


def calculate_ADL(data, fig=None):
    if 'ADL' not in data.columns:
        data['ADL'] = float('nan')

    last_index = data['ADL'].last_valid_index()
    start_index = 0 if last_index is None else last_index + 1

    high = pd.to_numeric(data['high'])
    low = pd.to_numeric(data['low'])
    close = pd.to_numeric(data['close'])
    volume = pd.to_numeric(data['Volume'])

    mf_mult = ((close - low) - (high - close)) / (high - low)
    mf_vol = mf_mult * volume

    adl_list = data['ADL'].tolist()[:start_index]
    
    if start_index == 0:
        adl_list = mf_vol.cumsum().tolist()
    else:
        for i in range(start_index, len(data)):
            adl_list.append(adl_list[-1] + mf_vol.iloc[i])

    data['ADL'] = pd.Series(adl_list, index=data.index)

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['ADL'], mode='lines', name='ADL'), row=3, col=1)

#klinger volume oscillator


def calculate_kvo(data, fast_period=34, slow_period=55, signal_period=13, fig=None):
    if 'KVO' not in data.columns:
        data['KVO'] = float('nan')
        data['KVO Signal'] = float('nan')

    last_index = data['KVO'].last_valid_index()
    start_index = max(fast_period, slow_period) if last_index is None else last_index + 1

    close = pd.to_numeric(data['close'])
    volume = pd.to_numeric(data['Volume'])

    mfv = close - (close.shift(1) + close.shift(-1)) / 2
    mfv *= volume

    fast_emav = mfv.ewm(span=fast_period, min_periods=fast_period).mean()
    slow_emav = mfv.ewm(span=slow_period, min_periods=slow_period).mean()

    kvo_list = data['KVO'].tolist()[:start_index]
    signal_list = data['KVO Signal'].tolist()[:start_index]

    if start_index == max(fast_period, slow_period):
        kvo_list = (fast_emav - slow_emav).tolist()
        signal_list = pd.Series(kvo_list).ewm(span=signal_period, min_periods=signal_period).mean().tolist()
    else:
        for i in range(start_index, len(data)):
            kvo = fast_emav.iloc[i] - slow_emav.iloc[i]
            kvo_list.append(kvo)

            signal = pd.Series(kvo_list[-signal_period:]).ewm(span=signal_period, min_periods=signal_period).mean().iloc[-1]
            signal_list.append(signal)

    data['KVO'] = pd.Series(kvo_list, index=data.index)
    data['KVO Signal'] = pd.Series(signal_list, index=data.index)

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['KVO'], mode='lines', name='KVO'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['KVO Signal'], mode='lines', name='KVO Signal'), row=3, col=1)

# Elder Ray

def calculate_elder_ray(data, window=13, fig=None):
    # Ensure data columns are correctly named and types are consistent
    data['high'] = pd.to_numeric(data['high'])
    data['low'] = pd.to_numeric(data['low'])
    data['close'] = pd.to_numeric(data['close'])

    if 'Bull Power' not in data.columns:
        data['Bull Power'] = float('nan')
        data['Bear Power'] = float('nan')

    last_index = data['Bull Power'].last_valid_index()
    start_index = window if last_index is None else last_index + 1
    
    high = pd.Series(data['high'])
    low = pd.Series(data['low'])
    close = pd.Series(data['close'])

    ema = close.ewm(span=window, min_periods=window).mean()

    # Initialize bull_power and bear_power with existing values if present
    bull_power = data['Bull Power'].tolist()[:start_index]
    bear_power = data['Bear Power'].tolist()[:start_index]

    for i in range(start_index, len(data)):
        bull_power.append(high[i] - ema[i])
        bear_power.append(low[i] - ema[i])

    data['Bull Power'] = pd.Series(bull_power, index=data.index)
    data['Bear Power'] = pd.Series(bear_power, index=data.index)

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Bull Power'], mode='lines', name='Bull Power'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Bear Power'], mode='lines', name='Bear Power'), row=3, col=1)




#Swing Index


def calculate_swing_index(data, limit_move=0.05, fig=None):
    if 'SwingIndex' not in data.columns:
        data['SwingIndex'] = float('nan')
    
    last_index = data['SwingIndex'].last_valid_index()
    start_index = 1 if last_index is None else last_index + 1

    for i in range(start_index, len(data)):
        K = max(data['high'][i] - data['close'][i-1], data['low'][i] - data['close'][i-1])
        R = data['high'][i] - data['low'][i]
        C = data['close'][i] - data['close'][i-1]
        O = data['Open'][i] - data['close'][i-1]

        if K != 0:
            SI = (C + 0.5 * R + 0.25 * (data['close'][i-1] - data['Open'][i])) / K
        else:
            SI = 0
        
        data.loc[i, 'SwingIndex'] = SI

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['SwingIndex'], mode='lines', name='Swing Index'), row=3, col=1)


#Schaff Trend Cycle Strategy

def calculate_stc(data, short_window, long_window, signal_window, cycle_window, fig=None):
    """Calculate the Schaff Trend Cycle (STC) and optionally plot it."""
    # Use the existing MACD calculation function
    calculate_macd_and_add_trace(data, short_window, long_window, signal_window, fig)
    
    macd_col = f'macd_{short_window}_{long_window}'
    signal_col = f'signal_line_{short_window}_{long_window}'
    
    stc = []
    for i in range(len(data)):
        if i < cycle_window:
            stc.append(float('nan'))
        else:
            stc.append(data[macd_col].iloc[i] - data[signal_col].iloc[i])

    data['STC'] = pd.Series(stc).ewm(span=cycle_window, adjust=False).mean()

    if fig:
        # Add STC to the chart
        fig.add_trace(go.Scatter(x=data['Date'], y=data['STC'], mode='lines', name='STC'), row=3, col=1)


#DIVERGENCE ANALYSIS
def calculate_indicators_and_add_trace(data, short_window=12, long_window=26, signal_window=9, rsi_window=14, fig=None):
    # Calculate MACD
    data[f'ema_{short_window}'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data[f'ema_{long_window}'] = data['close'].ewm(span=long_window, adjust=False).mean()
    macd_col = f'macd_{short_window}_{long_window}'
    signal_col = f'signal_line_{short_window}_{long_window}'
    histogram_col = f'macd_histogram_{short_window}_{long_window}'
    data[macd_col] = data[f'ema_{short_window}'] - data[f'ema_{long_window}']
    data[signal_col] = data[macd_col].ewm(span=signal_window, adjust=False).mean()
    data[histogram_col] = data[macd_col] - data[signal_col]
    
    if fig:
        # Add MACD, Signal Line, and Histogram to the third subplot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[macd_col], mode='lines', name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data[signal_col], mode='lines', name='Signal Line'), row=3, col=1)
        fig.add_trace(go.Bar(x=data['Date'], y=data[histogram_col], name='MACD Histogram'), row=3, col=1)

    
    # Calculate RSI
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_window, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi_col = f'rsi_{rsi_window}'
    data[rsi_col] = 100 - (100 / (1 + rs))
    
    # Calculate OBV
    obv = (np.sign(data['close'].diff()) * data['Volume']).fillna(0).cumsum()
    obv_col = 'obv'
    data[obv_col] = obv

    if fig:
        # Add MACD traces
        fig.add_trace(go.Scatter(x=data['Date'], y=data[macd_col], mode='lines', name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data[signal_col], mode='lines', name='Signal Line'), row=3, col=1)
        fig.add_trace(go.Bar(x=data['Date'], y=data[histogram_col], name='MACD Histogram'), row=3, col=1)
        
        # Add RSI trace
        fig.add_trace(go.Scatter(x=data['Date'], y=data[rsi_col], mode='lines', name='RSI'), row=3, col=1)
        
        # Add OBV trace
        fig.add_trace(go.Scatter(x=data['Date'], y=data[obv_col], mode='lines', name='OBV'), row=3, col=1)


#Senkou Span
def calculate_ichimoku(data, fig=None):
    if 'tenkan_sen' not in data.columns:
        data['tenkan_sen'] = float('nan')
    if 'kijun_sen' not in data.columns:
        data['kijun_sen'] = float('nan')
    if 'senkou_span_a' not in data.columns:
        data['senkou_span_a'] = float('nan')
    if 'senkou_span_b' not in data.columns:
        data['senkou_span_b'] = float('nan')

    last_index = data['tenkan_sen'].last_valid_index()
    start_index = 52 if last_index is None else last_index + 1

    high_9 = data['high'].rolling(window=9).max()
    low_9 = data['low'].rolling(window=9).min()
    high_26 = data['high'].rolling(window=26).max()
    low_26 = data['low'].rolling(window=26).min()
    high_52 = data['high'].rolling(window=52).max()
    low_52 = data['low'].rolling(window=52).min()

    data['tenkan_sen'] = (high_9 + low_9) / 2
    data['kijun_sen'] = (high_26 + low_26) / 2
    data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)
    data['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['senkou_span_a'], mode='lines', name='Senkou Span A'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['senkou_span_b'], mode='lines', name='Senkou Span B'), row=3, col=1)



#ZgZag Indicator

def calculate_zigzag(data, threshold=5, fig=None):
    if 'zigzag' not in data.columns:
        data['zigzag'] = float('nan')

    last_index = data['zigzag'].last_valid_index()
    start_index = 1 if last_index is None else last_index + 1

    zigzag = data['zigzag'].tolist()[:start_index]
    last_extreme = data['close'].iloc[start_index-1] if start_index > 1 else data['close'].iloc[0]
    direction = None if start_index == 1 else ('up' if zigzag[-1] > last_extreme else 'down')

    for i in range(start_index, len(data)):
        price_change = (data['close'].iloc[i] - last_extreme) / last_extreme * 100

        if direction is None:
            if abs(price_change) > threshold:
                direction = 'up' if price_change > 0 else 'down'
                last_extreme = data['close'].iloc[i]
                zigzag.append(data['close'].iloc[i])
            else:
                zigzag.append(float('nan'))

        elif direction == 'up':
            if price_change < -threshold:
                direction = 'down'
                last_extreme = data['close'].iloc[i]
                zigzag.append(data['close'].iloc[i])
            else:
                if data['close'].iloc[i] > last_extreme:
                    last_extreme = data['close'].iloc[i]
                zigzag.append(last_extreme)

        elif direction == 'down':
            if price_change > threshold:
                direction = 'up'
                last_extreme = data['close'].iloc[i]
                zigzag.append(data['close'].iloc[i])
            else:
                if data['close'].iloc[i] < last_extreme:
                    last_extreme = data['close'].iloc[i]
                zigzag.append(last_extreme)

    data['zigzag'] = pd.Series(zigzag, index=data.index)

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['zigzag'], mode='lines+markers', name='Zig Zag'), row=3, col=1)



#ATR Strategy
def calculate_atr(data, window=14, fig=None):
    if 'tr' not in data.columns:
        data['tr'] = float('nan')
        data['atr'] = float('nan')

    last_index = data['atr'].last_valid_index()
    start_index = window if last_index is None else last_index + 1

    tr = data['tr'].tolist()[:start_index]
    atr = data['atr'].tolist()[:start_index]

    for i in range(start_index, len(data)):
        tr_value = max(data['high'].iloc[i] - data['low'].iloc[i],
                       abs(data['high'].iloc[i] - data['close'].iloc[i-1]),
                       abs(data['low'].iloc[i] - data['close'].iloc[i-1]))
        tr.append(tr_value)
        atr_value = pd.Series(tr[-window:]).mean()
        atr.append(atr_value)

    data['tr'] = pd.Series(tr, index=data.index)
    data['atr'] = pd.Series(atr, index=data.index)

    data['upper_band'] = data['close'] + data['atr']
    data['lower_band'] = data['close'] - data['atr']

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['upper_band'], mode='lines', name='Upper Band'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['lower_band'], mode='lines', name='Lower Band'), row=3, col=1)


#Envelope Band

def calculate_envelope_channel(data, window=20, offset=0.02, fig=None):
    if 'ma' not in data.columns:
        data['ma'] = float('nan')
        data['upper_band'] = float('nan')
        data['lower_band'] = float('nan')

    last_index = data['ma'].last_valid_index()
    start_index = window if last_index is None else last_index + 1

    ma = data['ma'].tolist()[:start_index]
    upper_band = data['upper_band'].tolist()[:start_index]
    lower_band = data['lower_band'].tolist()[:start_index]

    for i in range(start_index, len(data)):
        ma_value = data['close'].iloc[i-window+1:i+1].mean()
        ma.append(ma_value)
        upper_band.append(ma_value * (1 + offset))
        lower_band.append(ma_value * (1 - offset))

    data['ma'] = pd.Series(ma, index=data.index)
    data['upper_band'] = pd.Series(upper_band, index=data.index)
    data['lower_band'] = pd.Series(lower_band, index=data.index)

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['ma'], mode='lines', name='Moving Average'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['upper_band'], mode='lines', name='Upper Band'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['lower_band'], mode='lines', name='Lower Band'), row=3, col=1)

