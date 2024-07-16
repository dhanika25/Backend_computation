from Backtesting import data_retriever_util as dr
from Backtesting import Indicators as ndct
from Backtesting import utils as btutil
from Backtesting import Backtest as sb_bt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import plotly.io as pio

def smaCross(shortma, longma, df, toPlot=False):
    ticker = df['ticker'].iloc[0]
    fig = pio.from_json(dr.plotGraph(df, ticker)) if toPlot else None
    df = ndct.ma(shortma, df, fig)
    df = ndct.ma(longma, df, fig)

    df['Buy'] = (df['MA' + str(shortma)] > df['MA' + str(longma)]).astype(int)
    df['Sell'] = (df['MA' + str(longma)] >= df['MA' + str(shortma)]).astype(int)

    df = btutil.getTriggerColumn(df)
    pnl_res = sb_bt.simpleBacktest(df)

    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res

def smaCross2(shortma, longma, df, toPlot=False):
    ticker = df['ticker'].iloc[0]
    fig = pio.from_json(dr.plotGraph(df, ticker)) if toPlot else None
    df = ndct.ma(shortma, df, fig)
    df = ndct.ma(longma, df, fig)

    df['Buy'] = (df['MA' + str(shortma)] > df['MA' + str(longma)]).astype(int)
    df['Sell'] = (df['MA' + str(longma)] >= df['MA' + str(shortma)]).astype(int)

    df = btutil.getTriggerColumn(df)
    pnl_res = sb_bt.simpleBacktest(df)

    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res
#----------------------------Function for BOLLINGER BAND SQUUEZE STRATEGY-----------------------------------------------------------------
def bollinger_band_squeeze(df, squeeze_threshold, stop_loss_percentage, bollinger_window, num_std_dev, toPlot=False):
    """
    Strategy to buy when the Bollinger Band squeeze condition is met and the price breaks above the upper band,
    and sell when the squeeze condition ends and the price falls below the lower band or hits the stop-loss level.
    """
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None
    ndct.calculate_bollinger_bands(df, window=bollinger_window, num_std_dev=num_std_dev, fig=fig)

    upper_band = f'upper_band_{bollinger_window}_{num_std_dev}'
    lower_band = f'lower_band_{bollinger_window}_{num_std_dev}'
    band_width = f'band_width_{bollinger_window}_{num_std_dev}'

    buy_signals = [float('nan')] * len(df)
    sell_signals = [float('nan')] * len(df)
    triggers = ['H'] * len(df)
    isHoldingStock = False
    buy_price = 0

    for i in range(1, len(df)):
        squeeze = df[band_width].iloc[i] / df[f'MA_{bollinger_window}'].iloc[i] < squeeze_threshold
        if not isHoldingStock:
            # Entry Condition for Bollinger Band Squeeze
            """Buy when the conditions are met:
            - squeeze condition is True
            - close price is above the upper band"""
            
            if squeeze and df['close'].iloc[i] > df[upper_band].iloc[i]:
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue

        else:
            # Exit Conditions for Bollinger Band Squeeze
            """Sell when any of the following conditions are met:
            - not squeeze and close price is below the lower band
            - close price is below stop-loss level"""
            if (not squeeze and df['close'].iloc[i] < df[lower_band].iloc[i]) or (df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage)):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        buy_signals[i] = float('nan')
        sell_signals[i] = float('nan')
        triggers[i] = 'H'

    # Assign lists to DataFrame columns
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res


# #----------------------------------------------------------MACD STRATEGY-----------------------------------------------------------
def implement_macd(df, short_window, long_window, signal_window, toPlot=False, stop_loss_percentage=0.1):
    
    """Compares the MACD line (difference between two EMAs) to a signal line (EMA of the MACD line)
        EMA is the Exponential Moving Average and MACD is the Moving average convergence/divergence"""

    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    # For calling the calculate_macd_and_add_trace(), fig is not passed if the indicators are to be calculated and fig is passed if the indicators are to be traced.
    #print("Printing data121333:",df)
    ndct.calculate_macd_and_add_trace(df, short_window, long_window, signal_window)  # Calculate MACD within this function

    buy_signals = [float('nan')] * len(df)  # Initialize with NaNs of dfFrame length
    sell_signals = [float('nan')] * len(df)  # Initialize with NaNs of dfFrame length
    triggers = ['H'] * len(df)  # Initialize with 'H' of dfFrame length
    isHoldingStock = False  # None means no isHoldingStock, 1 means holding stock, 0 means not holding stock
    buy_price = 0  # Track the price at which the stock was bought

    # Properly format the column names
    macd_col = f'macd_{short_window}_{long_window}'
    signal_col = f'signal_line_{short_window}_{long_window}'
    histogram_col = f'macd_histogram_{short_window}_{long_window}'

    for i in range(1, len(df)):
        if not isHoldingStock:
            # Entry Condition
            """Buy when the MACD line crosses above the signal line, and
            the macd histogram is positive, and
            macd line is positive"""

            if (df[macd_col].iloc[i] > df[signal_col].iloc[i] and 
                df[histogram_col].iloc[i] > 0 and
                df[macd_col].iloc[i] > 0):
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue

        else:
            # Exit Condition based on MACD and Stop-loss
            """macd line is below signal line, or
            macd histogram is negative, or
            macd line is negative, or
            close price is less than stop-loss line"""

            if (df[macd_col].iloc[i] < df[signal_col].iloc[i] or
                df[histogram_col].iloc[i] < 0 or
                df[macd_col].iloc[i] < 0 or
                df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage)):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        buy_signals[i] = float('nan')
        sell_signals[i] = float('nan')
        triggers[i] = 'H'

    # Assign lists to dfFrame columns
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    ndct.calculate_macd_and_add_trace(df, short_window, long_window,signal_window,fig) #Traces the MACD graph

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res

# #----------------------------------------------------------RSI Strategy-----------------------------------------------------------
def implement_RSI(data, overbought_threshold=70, oversold_threshold=30,toPlot=False, stop_loss_percentage=0.05):

    ticker = data['ticker'].iloc[0]
    fig = dr.plotGraph(data, ticker) if toPlot else None

    ndct.calculate_RSI(data,fig=fig)
    buy_signals = [float('nan')]  # Initialize with nan
    sell_signals = [float('nan')]  # Initialize with nan
    triggers = ['H']  # Initialize with 'Hold'
    position = None  # None means no position, 1 means holding stock, 0 means not holding stock
    buy_price = 0  # Track the price at which the stock was bought

    for i in range(1, len(data)):
        # Entry Condition (Buy)
        flag=False
        if data['RSI'].iloc[i - 1] < oversold_threshold and data['RSI'].iloc[i] >= oversold_threshold:
            flag=True
            if position != 1:
                buy_signals.append(data['close'].iloc[i])
                sell_signals.append(float('nan'))
                triggers.append('B')
                position = 1
                buy_price = data['close'].iloc[i]
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')
        
        # Exit Condition (Sell)
        elif data['RSI'].iloc[i - 1] > overbought_threshold and data['RSI'].iloc[i] <= overbought_threshold:
            flag=True
            if position == 1:
                buy_signals.append(float('nan'))
                sell_signals.append(data['close'].iloc[i])
                triggers.append('S')
                position = 0
                print(data['Date'].iloc[i],"-exit condition executed")

            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        # Exit Condition based on Stop-Loss
        if position == 1 and data['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
            flag=True
            buy_signals.append(float('nan'))
            sell_signals.append(data['close'].iloc[i])
            triggers.append('S')
            position = 0
            print(data['Date'].iloc[i],"-StopLoss executed")
        if flag==False:
            buy_signals.append(float('nan'))
            sell_signals.append(float('nan'))
            triggers.append('H')

    data['buy_signal'] = buy_signals
    data['sell_signal'] = sell_signals
    data['Trigger'] = triggers
    pnl_res = sb_bt.simpleBacktest(data)
    if toPlot:
        fig = btutil.addBuySell2Graph(data, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res

# -------------------------------------------------Stochastic Oscillator------------------------------------------------------------

def implement_stochastic(df, k_window, d_window, toPlot=False, stop_loss_percentage=0.1):
    """Implements Stochastic Oscillator strategy."""
    
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None
    
    # Calculate Stochastic Oscillator
    ndct.calculate_and_add_trace_stochastic_oscillator(df, k_window, d_window)
    
    buy_signals = [float('nan')] * len(df)  # Initialize with NaNs of df length
    sell_signals = [float('nan')] * len(df)  # Initialize with NaNs of df length
    triggers = ['H'] * len(df)  # Initialize with 'H' of df length
    isHoldingStock = False  # False means not holding stock
    buy_price = 0  # Track the price at which the stock was bought
    
    for i in range(1, len(df)):
        if not isHoldingStock:
            # Entry Condition
            """Buy when %K crosses above 20 from below."""
            if df[f'%K_{k_window}_{d_window}'].iloc[i] > 20 and df[f'%K_{k_window}_{d_window}'].iloc[i-1] <= 20:
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue
        
        else:
            # Exit Condition
            """Sell when %K crosses below 80 from above, or
            close price is less than stop-loss line."""
            if df[f'%K_{k_window}_{d_window}'].iloc[i] < 80 and df[f'%K_{k_window}_{d_window}'].iloc[i-1] >= 80 or df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue
        
        buy_signals[i] = float('nan')
        sell_signals[i] = float('nan')
        triggers[i] = 'H'
    
    # Assign lists to dataframe columns
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers
    
    ndct.calculate_and_add_trace_stochastic_oscillator(df,k_window, d_window, fig)  # Add stochastic trace to the graph
    
    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res


#-------------------------------------------------------ICHIMOKU STRATEGY------------------------------------------------------------------------------------------------------
def ichimoku_cloud_strategy(df, tenkan_sen_period, kijun_sen_period, senkou_span_b_period, senkou_shift, stop_loss_percentage, toPlot=False):
    """
    Strategy to buy when the price is above the cloud and the Tenkan-sen crosses above the Kijun-sen,
    and sell when the price falls below the cloud and the Tenkan-sen crosses below the Kijun-sen or hits the stop-loss level.
    """
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None
    df = ndct.calculate_ichimoku(df, tenkan_sen_period, kijun_sen_period, senkou_span_b_period, senkou_shift, fig)

    buy_signals = [float('nan')] * len(df)
    sell_signals = [float('nan')] * len(df)
    triggers = ['H'] * len(df)
    isHoldingStock = False
    buy_price = 0

    for i in range(1, len(df)):
        # Entry Condition for Ichimoku Cloud
        if not isHoldingStock:
            if df['close'].iloc[i] > df['senkou_span_a'].iloc[i] and df['close'].iloc[i] > df['senkou_span_b'].iloc[i] and \
               df['tenkan_sen'].iloc[i] > df['kijun_sen'].iloc[i] and df['tenkan_sen'].iloc[i - 1] <= df['kijun_sen'].iloc[i - 1]:
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue

        else:
            # Exit Conditions for Ichimoku Cloud
            if (df['close'].iloc[i] < df['senkou_span_a'].iloc[i] or df['close'].iloc[i] < df['senkou_span_b'].iloc[i] and \
                df['tenkan_sen'].iloc[i] < df['kijun_sen'].iloc[i] and df['tenkan_sen'].iloc[i - 1] >= df['kijun_sen'].iloc[i - 1]) or \
                (df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage)):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        buy_signals[i] = float('nan')
        sell_signals[i] = float('nan')
        triggers[i] = 'H'

    # Assign lists to DataFrame columns
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res

#------------------------------------------------OBV STRATEGY-----------------------------------------------------------------------------------
def implement_obv(df, stop_loss_percentage, toPlot=False):
    """Implements the OBV strategy with stop-loss."""
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    # Ensure 'volume' column is present
    if 'Volume' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'volume' column.")
    
    # Calculate OBV
    ndct.calculate_obv(df)  # Calculate OBV within this function

    buy_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    sell_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    triggers = ['H'] * len(df)  # Initialize with 'H' of DataFrame length
    isHoldingStock = False  # Boolean to check if holding stock
    buy_price = 0  # Track the price at which the stock was bought
    for i in range(1, len(df)):
        if not isHoldingStock:
            # Entry Condition
            """Buy when OBV rises with the price"""
            if df['OBV'].iloc[i] > df['OBV'].iloc[i-1] and df['close'].iloc[i] > df['close'].iloc[i-1]:
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue

        else:
            # Exit Condition based on OBV and Stop-loss
            """Sell when OBV falls with the price or close price is less than stop-loss line"""
            if df['OBV'].iloc[i] < df['OBV'].iloc[i-1] or df['close'].iloc[i] < df['close'].iloc[i-1] or df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        buy_signals[i] = float('nan')
        sell_signals[i] = float('nan')
        triggers[i] = 'H'

    # Assign lists to DataFrame columns
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    # Add OBV to the plot if required
    ndct.calculate_obv(df, fig)

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res
# ------------------------------------------Fibonacci Retracement-------------------------------------------------------------------

def  implement_fibonacci(df, toPlot=False, stop_loss_percentage=0.1):
    """Uses Fibonacci retracement levels for buy/sell signals with a stop-loss condition"""

    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    levels = ndct.calculate_and_add_fibonacci_levels(df, fig)

    buy_signals = [float('nan')] * len(df)
    sell_signals = [float('nan')] * len(df)
    triggers = ['H'] * len(df)
    isHoldingStock = False
    buy_price = 0

    for i in range(1, len(df)):
        close_price = df['close'].iloc[i]

        if not isHoldingStock:
            # Entry Conditions: Buy at Fibonacci retracement levels
            if (close_price <= levels['61.8%'] or close_price <= levels['50%'] or close_price <= levels['38.2%']):
                buy_signals[i] = close_price
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = close_price
                continue
        else:
            # Exit Conditions: Sell when price reaches the next Fibonacci level or stop-loss
            if (close_price >= levels['23.6%'] or close_price <= buy_price * (1 - stop_loss_percentage)):
                buy_signals[i] = float('nan')
                sell_signals[i] = close_price
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        buy_signals[i] = float('nan')
        sell_signals[i] = float('nan')
        triggers[i] = 'H'

    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    pnl_res = sb_bt.simpleBacktest(df)

    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
        return pnl_res

# ----------------------------------------------------ADX-------------------------------------------------------------------------

def implement_adx(df, period=14, toPlot=False, stop_loss_percentage=0.1):
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None
    
    # Calculate ADX within this function
    ndct.calculate_adx_and_add_trace(df, period)  

    buy_signals = [float('nan')] * len(df)
    sell_signals = [float('nan')] * len(df)
    triggers = ['H'] * len(df)
    isHoldingStock = False
    buy_price = 0

    for i in range(1, len(df)):
        if not isHoldingStock:
            # Entry Condition
            if (df['adx'].iloc[i] > 25 and 
                df['+DI'].iloc[i] > df['-DI'].iloc[i]):
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue

        else:
            # Exit Condition based on ADX and Stop-loss
            if (df['adx'].iloc[i] < 20 or 
                df['-DI'].iloc[i] > df['+DI'].iloc[i] or 
                df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage)):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        buy_signals[i] = float('nan')
        sell_signals[i] = float('nan')
        triggers[i] = 'H'

    # Assign lists to DataFrame columns
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

  
    ndct.calculate_adx_and_add_trace(df, period, fig)  # Trace the ADX graph

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res

# ---------------------------------------Parabolic SAR----------------------------------------------------------------------------

def implement_parabolic_sar(df, af=0.02, max_af=0.2, toPlot=False, stop_loss_percentage=0.1):
    
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    ndct.calculate_parabolic_sar_and_add_trace(df, af, max_af)  # Calculate Parabolic SAR and add to plot if fig is provided

    buy_signals = [float('nan')] * len(df)
    sell_signals = [float('nan')] * len(df)
    triggers = ['H'] * len(df)
    isHoldingStock = False
    buy_price = 0

    for i in range(1, len(df)):
        if not isHoldingStock:
            # Entry Condition: Buy when the SAR is below the price
            if df['parabolic_sar'].iloc[i] < df['close'].iloc[i]:
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue

        else:
            # Exit Condition: Sell when the SAR is above the price or stop-loss
            if (df['parabolic_sar'].iloc[i] > df['close'].iloc[i] or
                df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage)):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        buy_signals[i] = float('nan')
        sell_signals[i] = float('nan')
        triggers[i] = 'H'

    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    ndct.calculate_parabolic_sar_and_add_trace(df, af, max_af, fig)

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res


#-----------------------------------------VPT STRATEGY-----------------------------------------------------------------------------------
def implement_vpt(df, stop_loss_percentage, toPlot=False):
    """Implements the VPT strategy with stop-loss."""
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    # Ensure 'volume' column is present
    if 'Volume' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'volume' column.")
    
    # Calculate VPT
    ndct.calculate_vpt(df, fig)  # Calculate VPT within this function

    buy_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    sell_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    triggers = ['H'] * len(df)  # Initialize with 'H' of DataFrame length
    isHoldingStock = False  # Boolean to check if holding stock
    buy_price = 0  # Track the price at which the stock was bought

    for i in range(1, len(df)):
        if not isHoldingStock:
            # Entry Condition
            """Buy when VPT is rising with increasing volume"""
            if df['VPT'].iloc[i] > df['VPT'].iloc[i-1] and df['Volume'].iloc[i] > df['Volume'].iloc[i-1]:
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue

        else:
            # Exit Condition based on VPT and Stop-loss
            """Sell when VPT is falling with decreasing volume or close price is less than stop-loss line"""
            if df['VPT'].iloc[i] < df['VPT'].iloc[i-1] or df['Volume'].iloc[i] < df['Volume'].iloc[i-1] or df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        buy_signals[i] = float('nan')
        sell_signals[i] = float('nan')
        triggers[i] = 'H'

    # Assign lists to DataFrame columns
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    ndct.calculate_vpt(df, fig)

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res


#---------------------------------------CHAIKIN MONEY FLOW--------------------------------------------------------------------------
import Backtesting.Indicators as ndct
import Backtesting.Backtest as sb_bt
import plotly.io as pio
from Backtesting import utils as btutil

def implement_cmf(df, stop_loss_percentage, toPlot=False):
    """Implement the CMF (Chaikin Money Flow) strategy on the given DataFrame."""
    
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    # Calculate CMF and add it to DataFrame
    ndct.calculate_cmf(df, fig)  # Calculate CMF within this function

    buy_signals = [float('nan')] * len(df)
    sell_signals = [float('nan')] * len(df)
    triggers = ['H'] * len(df)
    isHoldingStock = False
    buy_price = 0

    for i in range(1, len(df)):
        if not isHoldingStock:
            # Entry Condition
            """Buy when CMF is positive and rising"""
            if df['CMF'].iloc[i] > 0 and df['CMF'].iloc[i] > df['CMF'].iloc[i - 1]:
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue

        else:
            # Exit Condition based on CMF and Stop-loss
            """Sell when CMF is negative and falling, or stop loss condition is met"""
            if df['CMF'].iloc[i] < 0 or df['CMF'].iloc[i] < df['CMF'].iloc[i - 1] or df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        buy_signals[i] = float('nan')
        sell_signals[i] = float('nan')
        triggers[i] = 'H'

    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    # Calculate backtest results
    pnl_res = sb_bt.simpleBacktest(df)

    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)

    return pnl_res


#--------------------------------------------------------HEIKIN ASHI STRATEGY-------------------------------------------------------------------
def implement_heikin_ashi(df, stop_loss_percentage, toPlot=False):
    """Implements the Heikin-Ashi strategy with stop-loss."""
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    # Calculate Heikin-Ashi candlesticks
    ha_data = ndct.calculate_heikin_ashi(df, fig)

    buy_signals = [float('nan')] * len(df)
    sell_signals = [float('nan')] * len(df)
    triggers = ['H'] * len(df)
    isHoldingStock = False
    buy_price = 0

    for i in range(1, len(ha_data)):
        if not isHoldingStock:
            if ha_data['HA_Close'].iloc[i] > ha_data['HA_Open'].iloc[i]:
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue
        else:
            if ha_data['HA_Close'].iloc[i] < ha_data['HA_Open'].iloc[i] or df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        buy_signals[i] = float('nan')
        sell_signals[i] = float('nan')
        triggers[i] = 'H'

    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res

#-------------------------------------------------Candelstick Patterns-------------------------------------------------------------
def implement_candlestick_strategy(df, toPlot=False, stop_loss_percentage=0.1):
    """Analyzes formations such as doji, hammer, and engulfing"""
    
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    ndct.find_and_plot_candlestick_patterns(df)

    buy_signals = [float('nan')] * len(df)
    sell_signals = [float('nan')] * len(df)
    triggers = ['H'] * len(df)
    isHoldingStock = False
    buy_price = 0

    for i in range(1, len(df)):
        if not isHoldingStock:
            # Entry Conditions: Bullish Patterns
            if df['candlestick_pattern'].iloc[i] in ['hammer', 'bullish_engulfing']:
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue
        else:
            # Exit Conditions: Bearish Patterns and Stop-Loss
            if df['candlestick_pattern'].iloc[i] in ['shooting_star', 'bearish_engulfing'] or df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue
        
        buy_signals[i] = float('nan')
        sell_signals[i] = float('nan')
        triggers[i] = 'H'

    # Assign lists to DataFrame columns
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    ndct.find_and_plot_candlestick_patterns(df,fig)
    pnl_res = sb_bt.simpleBacktest(df)

    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    
    return pnl_res

# --------------------------------------------------Head and Shoulder---------------------------------------------------------------
def implement_head_and_shoulders(df, toPlot=False, stop_loss_percentage=0.1):
    
    """Identify Head and Shoulders patterns and make buy/sell decisions
       based on neckline breaks."""
    
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    # Calculate Head and Shoulders pattern
    df, neckline = ndct.calculate_head_and_shoulders(df, fig)

    buy_signals = [float('nan')] * len(df)
    sell_signals = [float('nan')] * len(df)
    triggers = ['H'] * len(df)
    isHoldingStock = False
    buy_price = 0

    # Make sure neckline_level is not None
    if neckline is not None:
        neckline_level = (neckline[0]['close'] + neckline[1]['close']) / 2
    else:
        neckline_level = None

    for i in range(1, len(df)):
        if isHoldingStock:
            # Exit condition
            if neckline_level is not None and (df['close'].iloc[i] < neckline_level or df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage)):
                sell_signals[i] = df['close'].iloc[i]
                isHoldingStock = False
                triggers[i] = 'S'
            else:
                sell_signals[i] = float('nan')
                triggers[i] = 'H'
        else:
            # Entry condition
            if neckline_level is not None and df['close'].iloc[i] > neckline_level:
                buy_signals[i] = df['close'].iloc[i]
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                triggers[i] = 'B'
            else:
                buy_signals[i] = float('nan')
                triggers[i] = 'H'

    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res = sb_bt.simpleBacktest(df)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    else:
        pnl_res = sb_bt.simpleBacktest(df)

    return pnl_res

# ---------------------------------------------------Double Top/Down----------------------------------------------------------------
def implement_double_top_bottom(df, toPlot=False, stop_loss_percentage=0.1):
    
    """Implements Double Top/Bottom strategy"""

    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    # Detect Double Top/Bottom within this function
    df = ndct.detect_double_top_bottom(df)

    buy_signals = [float('nan')] * len(df)  # Initialize with NaNs of dfFrame length
    sell_signals = [float('nan')] * len(df)  # Initialize with NaNs of dfFrame length
    triggers = ['H'] * len(df)  # Initialize with 'H' of dfFrame length
    isHoldingStock = False  # None means no isHoldingStock, 1 means holding stock, 0 means not holding stock
    buy_price = 0  # Track the price at which the stock was bought

    for i in range(1, len(df)):
        if not isHoldingStock:
            # Entry Condition
            """Buy when the price breaks above the peak of a double bottom pattern"""

            if pd.notna(df['double_bottom'].iloc[i]) and df['close'].iloc[i] > df['double_bottom'].iloc[i]:
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue

        else:
            # Exit Condition based on Double Top and Stop-loss
            """Sell when the price breaks below the trough of a double top pattern or
            close price is less than stop-loss line"""

            if pd.notna(df['double_top'].iloc[i]) and df['close'].iloc[i] < df['double_top'].iloc[i] or \
               df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        buy_signals[i] = float('nan')
        sell_signals[i] = float('nan')
        triggers[i] = 'H'

    # Assign lists to dfFrame columns
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    ndct.detect_double_top_bottom(df, fig) # Draws the Double Top/Bottom on the graph

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res
#-----------------------------------------------------ELLIOT WAVE STRATEGY-----------------------------------------------------------------------
def implement_elliott_wave(df, stop_loss_percentage, toPlot=False):
    """Implements the Elliott Wave strategy with stop-loss."""
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    # Identify Elliott Wave patterns
    wave_data = ndct.identify_elliott_wave_patterns(df, fig)

    buy_signals = [float('nan')] * len(df)
    sell_signals = [float('nan')] * len(df)
    triggers = ['H'] * len(df)
    isHoldingStock = False
    buy_price = 0

    for i in range(1, len(wave_data)):
        if not isHoldingStock:
            if wave_data['Wave'].iloc[i] == 'Impulse':
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue
        else:
            if wave_data['Wave'].iloc[i] == 'Corrective' or df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        buy_signals[i] = float('nan')
        sell_signals[i] = float('nan')
        triggers[i] = 'H'

    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res

#-----------------------------------DONCHIAN CHANNEL------------------------------------------------------------------------------------------
def implement_donchian_channels(df, n, stop_loss_percentage, toPlot=False):
    """Implements the Donchian Channels strategy with stop-loss."""
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    # Calculate Donchian Channels
    ndct.calculate_donchian_channels(df, n)  # Calculate Donchian Channels within this function

    buy_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    sell_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    triggers = ['H'] * len(df)  # Initialize with 'H' of DataFrame length
    isHoldingStock = False  # Boolean to check if holding stock
    buy_price = 0  # Track the price at which the stock was bought

    for i in range(1, len(df)):
        if not isHoldingStock:
            # Entry Condition
            """Buy on a breakout above the upper channel"""
            if df['close'].iloc[i] > df['Upper_Channel'].iloc[i-1]:
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue

        else:
            # Exit Condition based on Donchian Channels and Stop-loss
            """Sell on a breakout below the lower channel or close price is less than stop-loss line"""
            if df['close'].iloc[i] < df['Lower_Channel'].iloc[i-1] or df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        if not isHoldingStock:
            buy_signals[i] = float('nan')
            sell_signals[i] = float('nan')
            triggers[i] = 'H'

    # Assign lists to DataFrame columns
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    # Add Donchian Channels to the plot if required
    ndct.calculate_donchian_channels(df, n, fig)

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res



# ----------------------------------------------------Triangles-------------------------------------------------------------------
def implement_triangle_strategy(df, toPlot=False, stop_loss_percentage=0.1):
    
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    # Calculate triangle patterns
    df = ndct.calculate_triangle_and_add_trace(df, fig)  

    buy_signals = [float('nan')] * len(df)  # Initialize with NaNs of dfFrame length
    sell_signals = [float('nan')] * len(df)  # Initialize with NaNs of dfFrame length
    triggers = ['H'] * len(df)  # Initialize with 'H' of dfFrame length
    isHoldingStock = False  # None means no isHoldingStock, 1 means holding stock, 0 means not holding stock
    buy_price = 0  # Track the price at which the stock was bought

    for i in range(1, len(df)):
        if not isHoldingStock:
            # Entry Condition
            """Buy when the close price breaks above the upper trendline"""
            if df['close'].iloc[i] > df['upper_trendline'].iloc[i]:
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue

        else:
            # Exit Condition based on breakout below the lower trendline or stop-loss
            """Sell when the close price breaks below the lower trendline or hits stop-loss"""
            if df['close'].iloc[i] < df['lower_trendline'].iloc[i] or df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        buy_signals[i] = float('nan')
        sell_signals[i] = float('nan')
        triggers[i] = 'H'

    # Assign lists to dfFrame columns
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers


    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res

#--------------------------------------------------------GANN ANGLES--------------------------------------------------------------------------------
def implement_gann_angles(df, key_price_points, angles, stop_loss_percentage, toPlot=False):
    """Implements the Gann Angles strategy with stop-loss."""
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    # Calculate Gann Angles
    ndct.calculate_gann_angles(df, key_price_points, angles, fig)  # Calculate Gann Angles within this function

    buy_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    sell_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    triggers = ['H'] * len(df)  # Initialize with 'H' of DataFrame length
    isHoldingStock = False  # Boolean to check if holding stock
    buy_price = 0  # Track the price at which the stock was bought

    for i in range(1, len(df)):
        if not isHoldingStock:
            # Entry Condition
            """Buy when the price is above a key Gann angle"""
            for key_price in key_price_points:
                for angle in angles:
                    if df['close'].iloc[i] > df[f'Gann_{angle}_{key_price}'].iloc[i]:
                        buy_signals[i] = df['close'].iloc[i]
                        sell_signals[i] = float('nan')
                        triggers[i] = 'B'
                        isHoldingStock = True
                        buy_price = df['close'].iloc[i]
                        break
                if isHoldingStock:
                    break

        else:
            # Exit Condition based on Gann Angles and Stop-loss
            """Sell when the price is below a key Gann angle or close price is less than stop-loss line"""
            for key_price in key_price_points:
                for angle in angles:
                    if df['close'].iloc[i] < df[f'Gann_{angle}_{key_price}'].iloc[i] or df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                        buy_signals[i] = float('nan')
                        sell_signals[i] = df['close'].iloc[i]
                        triggers[i] = 'S'
                        isHoldingStock = False
                        break
                if not isHoldingStock:
                    break

        if not isHoldingStock:
            buy_signals[i] = float('nan')
            sell_signals[i] = float('nan')
            triggers[i] = 'H'

    # Assign lists to DataFrame columns
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    # Add Gann Angles to the plot if required
    ndct.calculate_gann_angles(df, key_price_points, angles, fig)

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res


#-----------------------------------------------MOMENTUM INDICATOR----------------------------------------------------------------------------------
def implement_momentum(df, n, stop_loss_percentage, toPlot=False):
    """Implements the Momentum strategy with stop-loss."""
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    # Calculate Momentum
    ndct.calculate_momentum(df, n)  # Calculate Momentum within this function

    buy_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    sell_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    triggers = ['H'] * len(df)  # Initialize with 'H' of DataFrame length
    isHoldingStock = False  # Boolean to check if holding stock
    buy_price = 0  # Track the price at which the stock was bought
    for i in range(1, len(df)):
        if not isHoldingStock:
            # Entry Condition
            """Buy when momentum is positive and increasing"""
            if df['Momentum'].iloc[i] > 0 and df['Momentum'].iloc[i] > df['Momentum'].iloc[i-1]:
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue

        else:
            # Exit Condition based on Momentum and Stop-loss
            """Sell when momentum is negative and decreasing or close price is less than stop-loss line"""
            if df['Momentum'].iloc[i] < 0 and df['Momentum'].iloc[i] < df['Momentum'].iloc[i-1] or df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        buy_signals[i] = float('nan')
        sell_signals[i] = float('nan')
        triggers[i] = 'H'

    # Assign lists to DataFrame columns
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    # Add Momentum to the plot if required
    ndct.calculate_momentum(df, n, fig)

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res

#-----------------------------------------------MONEY FLOW INDEX------------------------------------------------------------------------------\
def implement_mfi_strategy(df, n, stop_loss_percentage, toPlot=False):
    """Implements the MFI strategy with stop-loss."""
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    # Calculate MFI
    ndct.calculate_mfi(df, n, fig)  # Calculate MFI within this function

    buy_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    sell_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    triggers = ['H'] * len(df)  # Initialize with 'H' of DataFrame length
    isHoldingStock = False  # Boolean to check if holding stock
    buy_price = 0  # Track the price at which the stock was bought

    for i in range(1, len(df)):
        if not isHoldingStock:
            # Entry Condition
            """Buy when MFI crosses above 20 from below"""
            if df['MFI'].iloc[i] > 20 and df['MFI'].iloc[i-1] <= 20:
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue

        else:
            # Exit Condition based on MFI and Stop-loss
            """Sell when MFI crosses below 80 from above or close price is less than stop-loss line"""
            if df['MFI'].iloc[i] < 80 and df['MFI'].iloc[i-1] >= 80 or df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        if not isHoldingStock:
            buy_signals[i] = float('nan')
            sell_signals[i] = float('nan')
            triggers[i] = 'H'

    # Assign lists to DataFrame columns
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    # Add MFI to the plot if required
    ndct.calculate_mfi(df, n, fig)

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res


#--------------------------------------TRIX INDICATOR---------------------------------------------------------------------------
def implement_trix_strategy(df, n, stop_loss_percentage, toPlot=False):
    """Implements the TRIX strategy with stop-loss."""
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    # Calculate TRIX
    ndct.calculate_trix(df, n, fig)  # Calculate TRIX within this function

    buy_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    sell_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    triggers = ['H'] * len(df)  # Initialize with 'H' of DataFrame length
    isHoldingStock = False  # Boolean to check if holding stock
    buy_price = 0  # Track the price at which the stock was bought

    for i in range(1, len(df)):
        if not isHoldingStock:
            # Entry Condition
            """Buy when TRIX crosses above its signal line"""
            if df['TRIX'].iloc[i] > df['TRIX_Signal'].iloc[i] and df['TRIX'].iloc[i-1] <= df['TRIX_Signal'].iloc[i-1]:
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue

        else:
            # Exit Condition based on TRIX and Stop-loss
            """Sell when TRIX crosses below its signal line or close price is less than stop-loss line"""
            if df['TRIX'].iloc[i] < df['TRIX_Signal'].iloc[i] and df['TRIX'].iloc[i-1] >= df['TRIX_Signal'].iloc[i-1] or df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        if not isHoldingStock:
            buy_signals[i] = float('nan')
            sell_signals[i] = float('nan')
            triggers[i] = 'H'

    # Assign lists to DataFrame columns
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    # Add TRIX to the plot if required
    ndct.calculate_trix(df, n, fig)

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res

#---------------------------------------------------Price Rate of Change-------------------------------------------------------------------------------
def implement_proc_strategy(df, n, stop_loss_percentage, toPlot=False):
    """Implements the PROC strategy with stop-loss."""
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    # Calculate PROC
    ndct.calculate_proc(df, n, fig)  # Calculate PROC within this function

    buy_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    sell_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    triggers = ['H'] * len(df)  # Initialize with 'H' of DataFrame length
    isHoldingStock = False  # Boolean to check if holding stock
    buy_price = 0  # Track the price at which the stock was bought

    for i in range(n, len(df)):
        if not isHoldingStock:
            # Entry Condition
            """Buy when PROC is positive and rising"""
            if df['PROC'].iloc[i] > 0 and df['PROC'].iloc[i] > df['PROC'].iloc[i-1]:
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue

        else:
            # Exit Condition based on PROC and Stop-loss
            """Sell when PROC is negative and falling or close price is less than stop-loss line"""
            if df['PROC'].iloc[i] < 0 and df['PROC'].iloc[i] < df['PROC'].iloc[i-1] or df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        if not isHoldingStock:
            buy_signals[i] = float('nan')
            sell_signals[i] = float('nan')
            triggers[i] = 'H'

    # Assign lists to DataFrame columns
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    # Add PROC to the plot if required
    ndct.calculate_proc(df, n, fig)

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res


#VORTEX INDICATOR STRATEGY
def implement_vortex_strategy(df, n, stop_loss_percentage, toPlot=False):
    """Implements the Vortex Indicator strategy with stop-loss."""
    ticker = df['ticker'].iloc[0]
    fig = dr.plotGraph(df, ticker) if toPlot else None

    # Calculate Vortex Indicator
    ndct.calculate_vortex(df, n, fig)  # Calculate VI within this function

    buy_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    sell_signals = [float('nan')] * len(df)  # Initialize with NaNs of DataFrame length
    triggers = ['H'] * len(df)  # Initialize with 'H' of DataFrame length
    isHoldingStock = False  # Boolean to check if holding stock
    buy_price = 0  # Track the price at which the stock was bought

    for i in range(n, len(df)):
        if not isHoldingStock:
            # Entry Condition
            """Buy when VI+ crosses above VI-"""
            if df['VI+'].iloc[i] > df['VI-'].iloc[i] and df['VI+'].iloc[i-1] <= df['VI-'].iloc[i-1]:
                buy_signals[i] = df['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                isHoldingStock = True
                buy_price = df['close'].iloc[i]
                continue

        else:
            # Exit Condition based on Vortex Indicator and Stop-loss
            """Sell when VI- crosses above VI+ or close price is less than stop-loss line"""
            if df['VI-'].iloc[i] > df['VI+'].iloc[i] and df['VI-'].iloc[i-1] <= df['VI+'].iloc[i-1] or df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                buy_signals[i] = float('nan')
                sell_signals[i] = df['close'].iloc[i]
                triggers[i] = 'S'
                isHoldingStock = False
                continue

        if not isHoldingStock:
            buy_signals[i] = float('nan')
            sell_signals[i] = float('nan')
            triggers[i] = 'H'

    # Assign lists to DataFrame columns
    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    # Add Vortex Indicator to the plot if required
    ndct.calculate_vortex(df, n, fig)

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res
# #----------------------------------------------------------RVI Strategy-----------------------------------------------------------

def implement_RVI(data, toPlot=False):
    ticker = data['ticker'].iloc[0]
    fig = dr.plotGraph(data, ticker) if toPlot else None

    ndct.calculate_RVI(data, fig=fig)

    buy_signals = [float('nan')]
    sell_signals = [float('nan')]
    triggers = ['H']
    position = None

    for i in range(1, len(data)):
        flag = False
        if data['RVI'].iloc[i] > 0 and data['RVI'].iloc[i - 1] <= 0:
            flag = True
            if position != 1:
                buy_signals.append(data['close'].iloc[i])
                sell_signals.append(float('nan'))
                triggers.append('B')
                position = 1
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        elif data['RVI'].iloc[i] < 0 and data['RVI'].iloc[i - 1] >= 0:
            flag = True
            if position == 1:
                buy_signals.append(float('nan'))
                sell_signals.append(data['close'].iloc[i])
                triggers.append('S')
                position = 0
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        if flag == False:
            buy_signals.append(float('nan'))
            sell_signals.append(float('nan'))
            triggers.append('H')

    data['buy_signal'] = buy_signals
    data['sell_signal'] = sell_signals
    data['Trigger'] = triggers
    pnl_res = sb_bt.simpleBacktest(data)

    if toPlot:
        fig = btutil.addBuySell2Graph(data, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)

    return pnl_res

# #----------------------------------------------------------Volume Oscillator Strategy-----------------------------------------------------------



def implement_volume_oscillator(data, toPlot=False):
    ticker = data['ticker'].iloc[0]
    fig = dr.plotGraph(data, ticker) if toPlot else None

    ndct.calculate_volume_oscillator(data, fig=fig)

    buy_signals = [float('nan')]
    sell_signals = [float('nan')]
    triggers = ['H']
    position = None

    for i in range(1, len(data)):
        flag = False
        if data['Volume_Oscillator'].iloc[i] > 0 and data['Volume_Oscillator'].iloc[i - 1] <= 0:
            flag = True
            if position != 1:
                buy_signals.append(data['close'].iloc[i])
                sell_signals.append(float('nan'))
                triggers.append('B')
                position = 1
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        elif data['Volume_Oscillator'].iloc[i] < 0 and data['Volume_Oscillator'].iloc[i - 1] >= 0:
            flag = True
            if position == 1:
                buy_signals.append(float('nan'))
                sell_signals.append(data['close'].iloc[i])
                triggers.append('S')
                position = 0
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        if flag == False:
            buy_signals.append(float('nan'))
            sell_signals.append(float('nan'))
            triggers.append('H')

    data['buy_signal'] = buy_signals
    data['sell_signal'] = sell_signals
    data['Trigger'] = triggers
    pnl_res = sb_bt.simpleBacktest(data)

    if toPlot:
        fig = btutil.addBuySell2Graph(data, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)

    return pnl_res


# #----------------------------------------------------------CMO Strategy-----------------------------------------------------------


def implement_CMO(data, toPlot=False):
    ticker = data['ticker'].iloc[0]
    fig = dr.plotGraph(data, ticker) if toPlot else None

    ndct.calculate_CMO(data, fig=fig)

    buy_signals = [float('nan')]
    sell_signals = [float('nan')]
    triggers = ['H']
    position = None

    for i in range(1, len(data)):
        flag = False
        if data['CMO'].iloc[i] > 0 and data['CMO'].iloc[i - 1] <= 0:
            flag = True
            if position != 1:
                buy_signals.append(data['close'].iloc[i])
                sell_signals.append(float('nan'))
                triggers.append('B')
                position = 1
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        elif data['CMO'].iloc[i] < 0 and data['CMO'].iloc[i - 1] >= 0:
            flag = True
            if position == 1:
                buy_signals.append(float('nan'))
                sell_signals.append(data['close'].iloc[i])
                triggers.append('S')
                position = 0
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        if flag == False:
            buy_signals.append(float('nan'))
            sell_signals.append(float('nan'))
            triggers.append('H')

    data['buy_signal'] = buy_signals
    data['sell_signal'] = sell_signals
    data['Trigger'] = triggers
    pnl_res = sb_bt.simpleBacktest(data)

    if toPlot:
        fig = btutil.addBuySell2Graph(data, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)

    return pnl_res

# #----------------------------------------------------------Aroon Strategy-----------------------------------------------------------

from Backtesting import data_retriever_util as dr
from Backtesting import Indicators as ndct
from Backtesting import utils as btutil
from Backtesting import Backtest as sb_bt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

def implement_aroon(data, toPlot=False):
    ticker = data['ticker'].iloc[0]
    fig = dr.plotGraph(data, ticker) if toPlot else None

    ndct.calculate_aroon(data, fig=fig)

    buy_signals = [float('nan')]
    sell_signals = [float('nan')]
    triggers = ['H']
    position = None

    for i in range(1, len(data)):
        flag = False
        if data['Aroon Up'].iloc[i] > data['Aroon Down'].iloc[i] and data['Aroon Up'].iloc[i - 1] <= data['Aroon Down'].iloc[i - 1]:
            flag = True
            if position != 1:
                buy_signals.append(data['close'].iloc[i])
                sell_signals.append(float('nan'))
                triggers.append('B')
                position = 1
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        elif data['Aroon Down'].iloc[i] > data['Aroon Up'].iloc[i] and data['Aroon Down'].iloc[i - 1] <= data['Aroon Up'].iloc[i - 1]:
            flag = True
            if position == 1:
                buy_signals.append(float('nan'))
                sell_signals.append(data['close'].iloc[i])
                triggers.append('S')
                position = 0
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        if flag == False:
            buy_signals.append(float('nan'))
            sell_signals.append(float('nan'))
            triggers.append('H')

    data['buy_signal'] = buy_signals
    data['sell_signal'] = sell_signals
    data['Trigger'] = triggers
    pnl_res = sb_bt.simpleBacktest(data)

    if toPlot:
        fig = btutil.addBuySell2Graph(data, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)

    return pnl_res


# #----------------------------------------------------------Ultimate Oscillator Strategy-----------------------------------------------------------


def implement_ultimate_oscillator(data, toPlot=False):
    ticker = data['ticker'].iloc[0]
    fig = dr.plotGraph(data, ticker) if toPlot else None

    ndct.calculate_ultimate_oscillator(data, fig=fig)

    buy_signals = [float('nan')]
    sell_signals = [float('nan')]
    triggers = ['H']
    position = None

    for i in range(1, len(data)):
        flag = False
        if data['Ultimate Oscillator'].iloc[i] > 30 and data['Ultimate Oscillator'].iloc[i - 1] <= 30:
            flag = True
            if position != 1:
                buy_signals.append(data['close'].iloc[i])
                sell_signals.append(float('nan'))
                triggers.append('B')
                position = 1
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        elif data['Ultimate Oscillator'].iloc[i] < 70 and data['Ultimate Oscillator'].iloc[i - 1] >= 70:
            flag = True
            if position == 1:
                buy_signals.append(float('nan'))
                sell_signals.append(data['close'].iloc[i])
                triggers.append('S')
                position = 0
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        if flag == False:
            buy_signals.append(float('nan'))
            sell_signals.append(float('nan'))
            triggers.append('H')

    data['buy_signal'] = buy_signals
    data['sell_signal'] = sell_signals
    data['Trigger'] = triggers
    pnl_res = sb_bt.simpleBacktest(data)

    if toPlot:
        fig = btutil.addBuySell2Graph(data, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)

    return pnl_res


# #----------------------------------------------------------Chandelier Exit Strategy-----------------------------------------------------------




def implement_chandelier_exit(data, toPlot=False, stop_loss_percentage=0.05):
    ticker = data['ticker'].iloc[0]
    fig = dr.plotGraph(data, ticker) if toPlot else None

    ndct.calculate_chandelier_exit(data, fig=fig)

    buy_signals = [float('nan')]
    sell_signals = [float('nan')]
    triggers = ['H']
    position = None

    for i in range(1, len(data)):
        flag = False
        if data['close'].iloc[i] > data['Chandelier Exit Long'].iloc[i]:
            flag = True
            if position != 1:
                buy_signals.append(data['close'].iloc[i])
                sell_signals.append(float('nan'))
                triggers.append('B')
                position = 1
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        elif data['close'].iloc[i] < data['Chandelier Exit Short'].iloc[i]:
            flag = True
            if position == 1:
                buy_signals.append(float('nan'))
                sell_signals.append(data['close'].iloc[i])
                triggers.append('S')
                position = 0
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        if flag == False:
            buy_signals.append(float('nan'))
            sell_signals.append(float('nan'))
            triggers.append('H')

    data['buy_signal'] = buy_signals
    data['sell_signal'] = sell_signals
    data['Trigger'] = triggers
    pnl_res = sb_bt.simpleBacktest(data)

    if toPlot:
        fig = btutil.addBuySell2Graph(data, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)

    return pnl_res


# #----------------------------------------------------------DMI Strategy-----------------------------------------------------------



def implement_dmi(data, toPlot=False):
    ticker = data['ticker'].iloc[0]
    fig = dr.plotGraph(data, ticker) if toPlot else None

    ndct.calculate_dmi(data, fig=fig)

    buy_signals = [float('nan')]
    sell_signals = [float('nan')]
    triggers = ['H']
    position = None

    for i in range(1, len(data)):
        flag = False
        if data['+DI'].iloc[i] > data['-DI'].iloc[i] and data['+DI'].iloc[i - 1] <= data['-DI'].iloc[i - 1]:
            flag = True
            if position != 1:
                buy_signals.append(data['close'].iloc[i])
                sell_signals.append(float('nan'))
                triggers.append('B')
                position = 1
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        elif data['-DI'].iloc[i] > data['+DI'].iloc[i] and data['-DI'].iloc[i - 1] <= data['+DI'].iloc[i - 1]:
            flag = True
            if position == 1:
                buy_signals.append(float('nan'))
                sell_signals.append(data['close'].iloc[i])
                triggers.append('S')
                position = 0
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        if flag == False:
            buy_signals.append(float('nan'))
            sell_signals.append(float('nan'))
            triggers.append('H')

    data['buy_signal'] = buy_signals
    data['sell_signal'] = sell_signals
    data['Trigger'] = triggers
    pnl_res = sb_bt.simpleBacktest(data)

    if toPlot:
        fig = btutil.addBuySell2Graph(data, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)

    return pnl_res


# #----------------------------------------------------------ADL Strategy-----------------------------------------------------------



def implement_ADL(data, toPlot=False):
    ticker = data['ticker'].iloc[0]
    fig = dr.plotGraph(data, stockName=ticker) if toPlot else None

    ndct.calculate_ADL(data, fig=fig)

    # Implement your strategy logic using ADL signals
    buy_signals = [float('nan')] * len(data)
    sell_signals = [float('nan')] * len(data)
    triggers = ['H'] * len(data)
    position = None
    buy_price = 0

    for i in range(1, len(data)):
        flag = False
        # Example strategy: Buy when ADL is rising, sell when ADL is falling
        if data['ADL'].iloc[i] > data['ADL'].iloc[i - 1]:
            flag = True
            if position != 1:
                buy_signals[i] = data['close'].iloc[i]
                triggers[i] = 'B'
                position = 1
                buy_price = data['close'].iloc[i]
            else:
                buy_signals[i] = float('nan')
                triggers[i] = 'H'

        elif data['ADL'].iloc[i] < data['ADL'].iloc[i - 1]:
            flag = True
            if position == 1:
                sell_signals[i] = data['close'].iloc[i]
                triggers[i] = 'S'
                position = 0
            else:
                sell_signals[i] = float('nan')
                triggers[i] = 'H'

        if flag == False:
            buy_signals[i] = float('nan')
            sell_signals[i] = float('nan')
            triggers[i] = 'H'

    data['buy_signal'] = buy_signals
    data['sell_signal'] = sell_signals
    data['Trigger'] = triggers

    # Perform backtesting or other evaluation based on your strategy
    pnl_res = sb_bt.simpleBacktest(data)

    if toPlot:
        fig = btutil.addBuySell2Graph(data, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)

    return pnl_res


 ##----------------------------------------------------------Klinger volume Strategy-----------------------------------------------------------


def implement_kvo(data, toPlot=False):
    ticker = data['ticker'].iloc[0]
    fig = dr.plotGraph(data, ticker) if toPlot else None

    ndct.calculate_kvo(data, fig=fig)

    buy_signals = [float('nan')]
    sell_signals = [float('nan')]
    triggers = ['H']
    position = None

    for i in range(1, len(data)):
        flag = False
        if data['KVO'].iloc[i] > 0 > data['KVO'].iloc[i - 1]:
            flag = True
            if position != 1:
                buy_signals.append(data['close'].iloc[i])
                sell_signals.append(float('nan'))
                triggers.append('B')
                position = 1
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        elif data['KVO'].iloc[i] < 0 < data['KVO'].iloc[i - 1]:
            flag = True
            if position == 1:
                buy_signals.append(float('nan'))
                sell_signals.append(data['close'].iloc[i])
                triggers.append('S')
                position = 0
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        if flag == False:
            buy_signals.append(float('nan'))
            sell_signals.append(float('nan'))
            triggers.append('H')

    data['buy_signal'] = buy_signals
    data['sell_signal'] = sell_signals
    data['Trigger'] = triggers
    pnl_res = sb_bt.simpleBacktest(data)

    if toPlot:
        fig = btutil.addBuySell2Graph(data, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)

    return pnl_res


# #----------------------------------------------------------Elder Ray Strategy-----------------------------------------------------------


def implement_elder_ray(data, toPlot=False):
    ticker = data['ticker'].iloc[0]
    fig = dr.plotGraph(data, ticker) if toPlot else None

    ndct.calculate_elder_ray(data, fig=fig)

    buy_signals = [float('nan')]
    sell_signals = [float('nan')]
    triggers = ['H']
    position = None

    for i in range(1, len(data)):
        flag = False
        if data['Bull Power'].iloc[i] > 0 and data['Bull Power'].iloc[i] > data['Bull Power'].iloc[i - 1]:
            flag = True
            if position != 1:
                buy_signals.append(data['close'].iloc[i])
                sell_signals.append(float('nan'))
                triggers.append('B')
                position = 1
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        elif data['Bear Power'].iloc[i] < 0 and data['Bear Power'].iloc[i] < data['Bear Power'].iloc[i - 1]:
            flag = True
            if position == 1:
                buy_signals.append(float('nan'))
                sell_signals.append(data['close'].iloc[i])
                triggers.append('S')
                position = 0
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        if flag == False:
            buy_signals.append(float('nan'))
            sell_signals.append(float('nan'))
            triggers.append('H')

    data['buy_signal'] = buy_signals
    data['sell_signal'] = sell_signals
    data['Trigger'] = triggers
    pnl_res = sb_bt.simpleBacktest(data)

    if toPlot:
        fig = btutil.addBuySell2Graph(data, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)

    return pnl_res


# #----------------------------------------------------------Swing Index Strategy-----------------------------------------------------------







# #----------------------------------------------------------Senkou Span Strategy-----------------------------------------------------------


# #----------------------------------------------------------Zig Zag Strategy-----------------------------------------------------------


# #----------------------------------------------------------Average True Range Bands Strategy-----------------------------------------------------------


# #----------------------------------------------------------Envelope Strategy-----------------------------------------------------------
