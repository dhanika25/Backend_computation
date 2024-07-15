from Backtesting import data_retriever_util as dr
from Backtesting import Indicators as ndct
from Backtesting import utils as btutil
from Backtesting import Backtest as sb_bt

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

def implement_stochastic(df, k_window=14, d_window=3, toPlot=False, stop_loss_percentage=0.1):
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

# ------------------------------------------Fibonacci Retracement-------------------------------------------------------------------

def implement_fibonacci(df, toPlot=False, stop_loss_percentage=0.1):
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
