import plotly.graph_objects as go
import random
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
    

def calculate_bollinger_bands(df, window, num_std_dev, fig=None):
    df = ma(window, df)
    rolling_std(df, window)
    
    upper_band = f'upper_band_{window}_{num_std_dev}'
    lower_band = f'lower_band_{window}_{num_std_dev}'
    band_width = f'band_width_{window}_{num_std_dev}'

    df[upper_band] = df[f'MA_{window}'] + (df[f'rolling_std_{window}'] * num_std_dev)
    df[lower_band] = df[f'MA_{window}'] - (df[f'rolling_std_{window}'] * num_std_dev)
    df[band_width] = df[upper_band] - df[lower_band]

    if fig:
        fig.add_trace(go.Scatter(x=df['Date'], y=df[upper_band], mode='lines', name=upper_band, line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df[lower_band], mode='lines', name=lower_band, line=dict(color='blue')))
    



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
    rsi = RSIIndicator(close=data['close'], window=window)
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