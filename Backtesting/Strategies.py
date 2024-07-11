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
def bollinger_band_squeeze(df, squeeze_threshold=0.1, stop_loss_percentage=0.02, toPlot=False):
    ticker = df['ticker'].iloc[0]
    fig = pio.from_json(dr.plotGraph(df, ticker)) if toPlot else None
    df = ndct.calculate_bollinger_bands(df, fig=fig)

    buy_signals = [float('nan')]
    sell_signals = [float('nan')]
    triggers = ['H']
    position = None
    buy_price = 0

    for i in range(1, len(df)):
        squeeze = df['band_width'].iloc[i] / df['MA' + str(20)].iloc[i] < squeeze_threshold
        condition_met = False

        if condition_met == False:
            if squeeze and df['close'].iloc[i] > df['upper_band'].iloc[i]:
                if position != 1:
                    buy_signals.append(df['close'].iloc[i])
                    sell_signals.append(float('nan'))
                    triggers.append('B')
                    position = 1
                    buy_price = df['close'].iloc[i]
                else:
                    buy_signals.append(float('nan'))
                    sell_signals.append(float('nan'))
                    triggers.append('H')
                condition_met = True

        if condition_met == False:
            if not squeeze and df['close'].iloc[i] < df['lower_band'].iloc[i]:
                if position == 1:
                    buy_signals.append(float('nan'))
                    sell_signals.append(df['close'].iloc[i])
                    triggers.append('S')
                    position = 0
                else:
                    buy_signals.append(float('nan'))
                    sell_signals.append(float('nan'))
                    triggers.append('H')
                condition_met = True

        if condition_met == False:
            if position == 1 and df['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                buy_signals.append(float('nan'))
                sell_signals.append(df['close'].iloc[i])
                triggers.append('S')
                position = 0
                condition_met = True

        if condition_met == False:
            buy_signals.append(float('nan'))
            sell_signals.append(float('nan'))
            triggers.append('H')

    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['Trigger'] = triggers

    pnl_res = sb_bt.simpleBacktest(df)
    if toPlot:
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res
# #----------------------------------------------------------MACD STRATEGY-----------------------------------------------------------
# Function to implement MACD strategy with Stop-Loss and print statements
def macd_strategy(data, stop_loss_percentage=0.1):
    data = ndct.calculate_macd(data)  # Calculate MACD within this function
    buy_signals = [float('nan')] * len(data)  # Initialize with NaNs of DataFrame length
    sell_signals = [float('nan')] * len(data)  # Initialize with NaNs of DataFrame length
    triggers = ['H'] * len(data)  # Initialize with 'H' of DataFrame length
    position = None  # None means no position, 1 means holding stock, 0 means not holding stock
    buy_price = 0  # Track the price at which the stock was bought

    for i in range(1, len(data)):
        flag = False
        # Entry Condition
        if (data['macd_12_26'].iloc[i] > data['signal_line_12_26'].iloc[i] and 
            data['macd_histogram_12_26'].iloc[i] > 0 and
            data['macd_12_26'].iloc[i] > 0):
            flag = True
            if position != 1:
                buy_signals[i] = data['close'].iloc[i]
                sell_signals[i] = float('nan')
                triggers[i] = 'B'
                position = 1
                buy_price = data['close'].iloc[i]

        # Exit Condition based on MACD
        if (data['macd_12_26'].iloc[i] < data['signal_line_12_26'].iloc[i] or
              data['macd_histogram_12_26'].iloc[i] < 0 or
              data['macd_12_26'].iloc[i] < 0):
            flag = True
            if position == 1:
                buy_signals[i] = float('nan')
                sell_signals[i] = data['close'].iloc[i]
                triggers[i] = 'S'
                position = 0

        # Exit Condition based on Stop-Loss
        if data['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
            flag = True
            if position == 1:
                buy_signals[i] = float('nan')
                sell_signals[i] = data['close'].iloc[i]
                triggers[i] = 'S'
                position = 0

        if not flag:
            buy_signals[i] = float('nan')
            sell_signals[i] = float('nan')
            triggers[i] = 'H'

    # Assign lists to DataFrame columns
    data['buy_signal'] = buy_signals
    data['sell_signal'] = sell_signals
    data['Trigger'] = triggers

    return data

def implement_macd(df):

    df=macd_strategy(df)

    # Perform a simple backtest
    result = sb_bt.simpleBacktest(df)
    print(result)

    # Plot the graph
    fig = dr.plotGraph(df, stockName='TATAMOTORS.NS')

    # fig= pio.from_json(fig)


    # Add MACD line to the third subplot
    fig.add_trace(go.Scatter(x=df['Date'], y=df['macd_12_26'], mode='lines', name='MACD'), row=3, col=1)
    # Add signal line to the third subplot
    fig.add_trace(go.Scatter(x=df['Date'], y=df['signal_line_12_26'], mode='lines', name='Signal Line'), row=3, col=1)
    # Add MACD histogram to the third subplot
    fig.add_trace(go.Bar(x=df['Date'], y=df['macd_histogram_12_26'], name='MACD Histogram'), row=3, col=1)

    # # Additional lines for MACD
    # fig.update_layout(
    #     height=800,
    #     xaxis2_rangeslider_visible=False,
    #     showlegend=True
    # )

    # Add buy/sell signals to the graph
    fig = btutil.addBuySell2Graph(df, fig)  

    # Convert the figure to JSON
    plotly_json = pio.to_json(fig, pretty=True)
    result = {"plotlyJson": plotly_json}

    # Display the figure
    fig.show()
