from Backtesting import data_retriever_util as dr
from Backtesting import Indicators as ndct
from Backtesting import utils as btutil
from Backtesting import Backtest as sb_bt

import plotly.io as pio

def smaCross(shortma,longma,df,toPlot=False):
    ticker = df['ticker'].iloc[0]
    fig = pio.from_json(dr.plotGraph(df,ticker)) if toPlot else None
    # fig.show()

    df = ndct.ma(shortma, df, fig)
    df = ndct.ma(longma, df, fig)

    df['Buy'] = (df['MA'+str(shortma)] > df['MA'+str(longma)]).astype(int)
    df['Sell'] = (df['MA'+str(longma)] >= df['MA'+str(shortma)]).astype(int)
    
    df = btutil.getTriggerColumn(df)

    
    # df = removeRedundantRows(df)
    # print(df)
    
    pnl_res = sb_bt.simpleBacktest(df)
    
    if toPlot: 
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res

def smaCross2(shortma,longma,df,toPlot=False):
    ticker = df['ticker'].iloc[0]
    fig = pio.from_json(dr.plotGraph(df,ticker)) if toPlot else None
    # fig.show()

    df = ndct.ma(shortma, df, fig)
    df = ndct.ma(longma, df, fig)

    df['Buy'] = (df['MA'+str(shortma)] > df['MA'+str(longma)]).astype(int)
    df['Sell'] = (df['MA'+str(longma)] >= df['MA'+str(shortma)]).astype(int)
    
    df = btutil.getTriggerColumn(df)

    
    # df = removeRedundantRows(df)
    # print(df)
    
    pnl_res = sb_bt.simpleBacktest(df)
    
    if toPlot: 
        fig = btutil.addBuySell2Graph(df, fig)
        pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
    return pnl_res

# #MACD STRATEGY--------------------------

# def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
#     data['ema_short'] = data['close'].ewm(span=short_window, adjust=False).mean()
#     data['ema_long'] = data['close'].ewm(span=long_window, adjust=False).mean()
#     data['macd'] = data['ema_short'] - data['ema_long']
#     data['signal_line'] = data['macd'].ewm(span=signal_window, adjust=False).mean()
#     data['macd_histogram'] = data['macd'] - data['signal_line']
#     return data

# def implement_macd_strategy(data, stop_loss_percentage=0.05):
#     buy_signals = [float('nan')]  # Initialize with nan
#     sell_signals = [float('nan')]  # Initialize with nan
#     triggers = ['H']  # Initialize with 'Hold'
#     position = None  # None means no position, 1 means holding stock, 0 means not holding stock
#     buy_price = 0  # Track the price at which the stock was bought

#     for i in range(1, len(data)):
#         # Entry Condition
#         if (data['macd'].iloc[i] > data['signal_line'].iloc[i] and 
#             data['macd_histogram'].iloc[i] > 0 and
#             data['macd'].iloc[i] > 0):
#             if position != 1:
#                 buy_signals.append(data['close'].iloc[i])
#                 sell_signals.append(float('nan'))
#                 triggers.append('B')
#                 position = 1
#                 buy_price = data['close'].iloc[i]
#             else:
#                 buy_signals.append(float('nan'))
#                 sell_signals.append(float('nan'))
#                 triggers.append('H')
        
#         # Exit Condition based on MACD
#         elif (data['macd'].iloc[i] < data['signal_line'].iloc[i] or
#               data['macd_histogram'].iloc[i] < 0 or
#               data['macd'].iloc[i] < 0):
#             if position == 1:
#                 buy_signals.append(float('nan'))
#                 sell_signals.append(data['close'].iloc[i])
#                 triggers.append('S')
#                 position = 0
#             else:
#                 buy_signals.append(float('nan'))
#                 sell_signals.append(float('nan'))
#                 triggers.append('H')

#         # Exit Condition based on Stop-Loss
#         elif position == 1 and data['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
#             buy_signals.append(float('nan'))
#             sell_signals.append(data['close'].iloc[i])
#             triggers.append('S')
#             position = 0
        
#         else:
#             buy_signals.append(float('nan'))
#             sell_signals.append(float('nan'))
#             triggers.append('H')

#     data['buy_signal'] = buy_signals
#     data['sell_signal'] = sell_signals
#     data['Trigger'] = triggers

    
#     return data

# def macd_strategy(df, toPlot=False):
#     df = calculate_macd(df)
#     df = implement_macd_strategy(df)
    
#     ticker = df['ticker'].iloc[0]
#     fig = pio.from_json(dr.plotGraph(df, ticker)) if toPlot else None

#     df = btutil.getTriggerColumn(df)
#     pnl_res = sb_bt.simpleBacktest(df)
    
#     if toPlot: 
#         fig = btutil.addBuySell2Graph(df, fig)
#         pnl_res["plotlyJson"] = pio.to_json(fig, pretty=True)
#     return pnl_res

# # Optionally, replace smaCross and smaCross2 with macd_strategy if MACD is preferred
# def smaCross(shortma, longma, df, toPlot=False):
#     return macd_strategy(df, toPlot)

# def smaCross2(shortma, longma, df, toPlot=False):
#     return macd_strategy(df, toPlot)    