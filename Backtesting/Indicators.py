import plotly.graph_objects as go
import random
import pandas as pd
import numpy as np


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
    
    return data, neckline

# Double Top/Bottom

def detect_double_top_bottom(data, fig=None):
    data['double_top'] = [float('nan')] * len(data)
    data['double_bottom'] = [float('nan')] * len(data)

    # Assuming a simple approach to detect double tops/bottoms
    peaks = (data['high'] > data['high'].shift(1)) & (data['high'] > data['high'].shift(-1))
    troughs = (data['low'] < data['low'].shift(1)) & (data['low'] < data['low'].shift(-1))

    for i in range(1, len(data) - 1):
        if peaks[i] and data['high'][i] >= data['high'][i-1] and data['high'][i] >= data['high'][i+1]:
            data['double_top'][i] = data['high'][i]
        elif troughs[i] and data['low'][i] <= data['low'][i-1] and data['low'][i] <= data['low'][i+1]:
            data['double_bottom'][i] = data['low'][i]

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['double_top'], mode='markers', name='Double Top', marker=dict(color='blue', symbol='triangle-up')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['double_bottom'], mode='markers', name='Double Bottom', marker=dict(color='black', symbol='triangle-down')), row=1, col=1)

    return data
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

#ELLIOT WAVE STRATEGY
def identify_elliott_wave_patterns(data, fig=None):
    """
    Identify Elliott Wave patterns and optionally plot them.
    """
    wave_data = data.copy()

    # Placeholder for actual wave pattern identification logic.
    wave_data['Wave'] = None  # Initialize with None

    # Simplified pattern identification (for demonstration)
    for i in range(1, len(wave_data)-1):
        if wave_data['close'].iloc[i] > wave_data['close'].iloc[i-1] and wave_data['close'].iloc[i] > wave_data['close'].iloc[i+1]:
            wave_data['Wave'].iloc[i] = 'Impulse'
        elif wave_data['close'].iloc[i] < wave_data['close'].iloc[i-1] and wave_data['close'].iloc[i] < wave_data['close'].iloc[i+1]:
            wave_data['Wave'].iloc[i] = 'Corrective'
    
    if fig:
        # Add wave patterns to the third subplot
        impulse_wave = wave_data[wave_data['Wave'] == 'Impulse']
        corrective_wave = wave_data[wave_data['Wave'] == 'Corrective']
        
        fig.add_trace(go.Scatter(x=impulse_wave['Date'], y=impulse_wave['close'], mode='markers+lines', name='Impulse Wave', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=corrective_wave['Date'], y=corrective_wave['close'], mode='markers+lines', name='Corrective Wave', line=dict(color='red')), row=3, col=1)
    
    return wave_data

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
    
    return data

# Triangles

def calculate_triangle_and_add_trace(data, fig=None):
    data['upper_trendline'] = np.nan
    data['lower_trendline'] = np.nan

    min_periods = 5  # Number of periods to consider for local minima and maxima
    
    for i in range(min_periods, len(data) - min_periods):
        local_min = data['close'][i-min_periods:i+min_periods].min()
        local_max = data['close'][i-min_periods:i+min_periods].max()
        
        if data['close'][i] == local_min:
            data['lower_trendline'][i] = local_min
        if data['close'][i] == local_max:
            data['upper_trendline'][i] = local_max

    data['lower_trendline'] = data['lower_trendline'].interpolate()
    data['upper_trendline'] = data['upper_trendline'].interpolate()
    
    if fig:
        # Add upper trendline to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data['upper_trendline'], mode='lines', name='Upper Trendline'), row=3, col=1)
        # Add lower trendline to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data['lower_trendline'], mode='lines', name='Lower Trendline'), row=3, col=1)
    
    return data


#GANN ANGLES
import plotly.graph_objs as go
import numpy as np

def calculate_gann_angles(data, key_price_points, angles, fig=None):
    """Calculate and plot Gann Angles from key price points."""
    for key_price in key_price_points:
        for angle in angles:
            # Calculate the slope based on the angle (assuming 1 unit of time = 1 unit of price)
            slope = np.tan(np.radians(angle))
            gann_line = [key_price + slope * (i - key_price_points.index(key_price)) for i in range(len(data))]
            data[f'Gann_{angle}_{key_price}'] = gann_line
            
            if fig:
                # Add Gann Angle line to the chart
                fig.add_trace(go.Scatter(x=data['Date'], y=gann_line, mode='lines', name=f'Gann {angle}° from {key_price}'))



#MOMENTUM INDICATOR
def calculate_momentum(data, n, fig=None):
    """Calculate the Momentum indicator and optionally plot it."""
    momentum = data['close'] - data['close'].shift(n)
    data['Momentum'] = momentum
    
    if fig:
        # Add Momentum line to the chart
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

    return data

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
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'ema_{ema_window}'], mode='lines', name='EMA'), row=3, col=1)
        # Add Upper Channel line to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'upper_channel'], mode='lines', name='Upper Channel'), row=3, col=1)
        # Add Lower Channel line to the plot
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'lower_channel'], mode='lines', name='Lower Channel'), row=3, col=1)

    return data

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

    return data

#RVI Strategy
import pandas as pd
import plotly.graph_objs as go

def calculate_RVI(data, window=10, fig=None):
    close_open_diff = data['close'] - data['Open']
    high_low_diff = data['high'] - data['low']
    data['RVI'] = close_open_diff.rolling(window=window).mean() / high_low_diff.rolling(window=window).mean()

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['RVI'], mode='lines', name='RVI'), row=3, col=1)
        fig.add_shape(
            type="line", line=dict(color="black", width=1, dash="dash"),
            x0=data['Date'].iloc[0], y0=0, x1=data['Date'].iloc[-1], y1=0,
            row=3, col=1
        )


#Volume Oscillator


def calculate_volume_oscillator(data, short_window=14, long_window=28, fig=None):
    short_ma = data['Volume'].rolling(window=short_window).mean()
    long_ma = data['Volume'].rolling(window=long_window).mean()
    data['Volume_Oscillator'] = (short_ma - long_ma) / long_ma

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Volume_Oscillator'], mode='lines', name='Volume Oscillator'), row=3, col=1)



#CMO Strategy


def calculate_CMO(data, window=14, fig=None):
    close_diff = data['close'].diff()
    upward_changes = close_diff.clip(lower=0)
    downward_changes = -close_diff.clip(upper=0)
    
    sum_upward = upward_changes.rolling(window=window).sum()
    sum_downward = downward_changes.rolling(window=window).sum()

    data['CMO'] = (sum_upward - sum_downward) / (sum_upward + sum_downward) * 100

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['CMO'], mode='lines', name='CMO'), row=3, col=1)
        fig.add_shape(
            type="line", line=dict(color="black", width=1, dash="dash"),
            x0=data['Date'].iloc[0], y0=0, x1=data['Date'].iloc[-1], y1=0,
            row=3, col=1
        )


#Aroon Strategy


def calculate_aroon(data, window=25, fig=None):
    data['Aroon Up'] = data['high'].rolling(window).apply(lambda x: (window - x.argmax()) / window * 100, raw=True)
    data['Aroon Down'] = data['low'].rolling(window).apply(lambda x: (window - x.argmin()) / window * 100, raw=True)

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Aroon Up'], mode='lines', name='Aroon Up'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Aroon Down'], mode='lines', name='Aroon Down'), row=3, col=1)



#Ultimate Oscillator


def calculate_ultimate_oscillator(data, short_period=7, medium_period=14, long_period=28, fig=None):
    min_low_or_prev_close = data[['low', 'close']].min(axis=1).shift(1)
    true_range = data['high'].combine(min_low_or_prev_close, max) - data['low'].combine(min_low_or_prev_close, min)
    buying_pressure = data['close'] - min_low_or_prev_close

    avg_bp1 = buying_pressure.rolling(window=short_period).sum()
    avg_tr1 = true_range.rolling(window=short_period).sum()

    avg_bp2 = buying_pressure.rolling(window=medium_period).sum()
    avg_tr2 = true_range.rolling(window=medium_period).sum()

    avg_bp3 = buying_pressure.rolling(window=long_period).sum()
    avg_tr3 = true_range.rolling(window=long_period).sum()

    ultimate_oscillator = 100 * ((4 * (avg_bp1 / avg_tr1)) + (2 * (avg_bp2 / avg_tr2)) + (avg_bp3 / avg_tr3)) / (4 + 2 + 1)
    data['Ultimate Oscillator'] = ultimate_oscillator

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
    high_max = data['high'].rolling(window=window).max()
    low_min = data['low'].rolling(window=window).min()
    atr = data['high'].combine(data['low'], lambda x, y: abs(x - y)).rolling(window=window).mean()

    chandelier_exit_long = high_max - (atr * multiplier)
    chandelier_exit_short = low_min + (atr * multiplier)

    data['Chandelier Exit Long'] = chandelier_exit_long
    data['Chandelier Exit Short'] = chandelier_exit_short

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Chandelier Exit Long'], mode='lines', name='Chandelier Exit Long'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Chandelier Exit Short'], mode='lines', name='Chandelier Exit Short'), row=3, col=1)


#DMI Strategy

def calculate_dmi(data, window=14, fig=None):
    # Ensure data columns are correctly named and types are consistent
    data['high'] = pd.to_numeric(data['high'])
    data['low'] = pd.to_numeric(data['low'])
    data['close'] = pd.to_numeric(data['close'])

    high = pd.Series(data['high'])
    low = pd.Series(data['low'])
    close = pd.Series(data['close'])

    # Calculate True Range (TR)
    tr = np.maximum(np.maximum(high - low, abs(high - close.shift(1))), abs(low - close.shift(1)))

    # Calculate Directional Movement (DM)
    dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), np.maximum(high - high.shift(1), 0), 0)
    dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), np.maximum(low.shift(1) - low, 0), 0)

    # Convert to pandas Series
    dm_plus = pd.Series(dm_plus)
    dm_minus = pd.Series(dm_minus)
    tr = pd.Series(tr)

    # Smoothed versions of TR, DM+, and DM-
    atr = tr.rolling(window=window).mean()
    di_plus = (dm_plus.rolling(window=window).mean() / atr) * 100
    di_minus = (dm_minus.rolling(window=window).mean() / atr) * 100

    # Calculate Average Directional Index (ADX)
    dx = 100 * np.abs((di_plus - di_minus) / (di_plus + di_minus))
    adx = dx.rolling(window=window).mean()

    data['+DI'] = di_plus
    data['-DI'] = di_minus
    data['ADX'] = adx

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['+DI'], mode='lines', name='+DI'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['-DI'], mode='lines', name='-DI'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['ADX'], mode='lines', name='ADX'), row=3, col=1)

# ADL Strategy


def calculate_ADL(data, fig=None):
    # Ensure data columns are correctly named and types are consistent
    data['high'] = pd.to_numeric(data['high'])
    data['low'] = pd.to_numeric(data['low'])
    data['close'] = pd.to_numeric(data['close'])
    data['Volume'] = pd.to_numeric(data['Volume'])

    high = pd.Series(data['high'])
    low = pd.Series(data['low'])
    close = pd.Series(data['close'])

    # Calculate Money Flow Multiplier (MF Multiplier) and Money Flow Volume (MFV)
    mf_mult = ((close - low) - (high - close)) / (high - low)
    mf_vol = mf_mult * data['Volume']

    # Accumulate the Money Flow Volume to get ADL
    data['ADL'] = mf_vol.cumsum()

    if fig:
        # Plot ADL on the chart
        fig.add_trace(go.Scatter(x=data['Date'], y=data['ADL'], mode='lines', name='ADL'), row=3, col=1)

#klinger volume oscillator


def calculate_kvo(data, fast_period=34, slow_period=55, signal_period=13, fig=None):
    close = pd.to_numeric(data['close'])
    volume = pd.to_numeric(data['Volume'])

    # Calculate True Range (TR)
    tr = np.abs(close - close.shift(1))

    # Calculate Money Flow Volume (MFV)
    mfv = close - (close.shift(1) + close.shift(-1)) / 2
    mfv *= volume

    # Calculate Fast and Slow EMAs of MFV
    fast_emav = mfv.ewm(span=fast_period, min_periods=fast_period).mean()
    slow_emav = mfv.ewm(span=slow_period, min_periods=slow_period).mean()

    # Calculate Klinger Volume Oscillator (KVO)
    kvo = fast_emav - slow_emav

    # Calculate Signal Line
    signal_line = kvo.ewm(span=signal_period, min_periods=signal_period).mean()

    data['KVO'] = kvo
    data['KVO Signal'] = signal_line

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['KVO'], mode='lines', name='KVO'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['KVO Signal'], mode='lines', name='KVO Signal'), row=3, col=1)

# Elder Ray




def calculate_elder_ray(data, window=13, fig=None):
    # Ensure data columns are correctly named and types are consistent
    data['high'] = pd.to_numeric(data['high'])
    data['low'] = pd.to_numeric(data['low'])
    data['close'] = pd.to_numeric(data['close'])

    high = pd.Series(data['high'])
    low = pd.Series(data['low'])
    close = pd.Series(data['close'])

    # Calculate Exponential Moving Average (EMA)
    ema = close.ewm(span=window, min_periods=window).mean()

    # Calculate Bull Power and Bear Power
    bull_power = high - ema
    bear_power = low - ema

    # Add Bull Power and Bear Power to the DataFrame
    data['Bull Power'] = bull_power
    data['Bear Power'] = bear_power

    if fig:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Bull Power'], mode='lines', name='Bull Power'), row=3, col=1)
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Bear Power'], mode='lines', name='Bear Power'), row=3, col=1)



#Swing Index





