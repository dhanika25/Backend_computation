import data_retriever_util as dr
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

