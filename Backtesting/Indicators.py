import plotly.graph_objects as go
import random
import pandas as pd

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
    
    return levels

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
                row=1, col=1
            )
    return data,fig

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
import pandas as pd
import plotly.graph_objects as go

def calculate_heikin_ashi(data, fig=None):
    """Calculate the Heikin-Ashi candlesticks and optionally plot them."""
    ha_data = data.copy()

    ha_data['HA_Close'] = (data['Open'] + data['high'] + data['low'] + data['close']) / 4
    ha_data['HA_Open'] = (ha_data['Open'].shift(1) + ha_data['close'].shift(1)) / 2
    ha_data['HA_Open'].iloc[0] = (data['Open'].iloc[0] + data['close'].iloc[0]) / 2
    ha_data['HA_High'] = ha_data[['high', 'HA_Open', 'HA_Close']].max(axis=1)
    ha_data['HA_Low'] = ha_data[['low', 'HA_Open', 'HA_Close']].min(axis=1)

    if fig:
        # Add Heikin-Ashi candlesticks to the third subplot
        fig.add_trace(go.Candlestick(x=ha_data['Date'],
                                     open=ha_data['HA_Open'],
                                     high=ha_data['HA_High'],
                                     low=ha_data['HA_Low'],
                                     close=ha_data['HA_Close'],
                                     name='Heikin-Ashi'), row=3, col=1)

    return ha_data

