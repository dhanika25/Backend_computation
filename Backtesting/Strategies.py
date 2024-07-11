from Backtesting import data_retriever_util as dr
from Backtesting import Indicators as ndct
from Backtesting import utils as btutil
from Backtesting import Backtest as sb_bt
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
