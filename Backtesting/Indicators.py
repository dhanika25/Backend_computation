import plotly.graph_objects as go
import random

def ma(n, df,fig=None):
    df['MA'+str(n)] = df['close'].rolling(window=n).mean()
    if not fig:
        pass
    else:
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'yellow']
        color = random.choice(colors)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA'+str(n)], mode='lines', name='MA'+str(n),line=dict(color=color, width=2)))
    return df

# New function to apply MACD strategy and optionally plot the results
# def macd_with_plot(df, fig=None, stop_loss_percentage=0.05):
#     df = bt.calculate_macd(df)
#     df = bt.implement_macd_strategy(df, stop_loss_percentage)
#     if fig:
#         fig.add_trace(go.Scatter(x=df['Date'], y=df['macd'], mode='lines', name='MACD', line=dict(color='blue', width=2)))
#         fig.add_trace(go.Scatter(x=df['Date'], y=df['signal_line'], mode='lines', name='Signal Line', line=dict(color='red', width=2)))
#         fig.add_trace(go.Bar(x=df['Date'], y=df['macd_histogram'], name='MACD Histogram', marker_color='green'))

#         buy_signals = df[df['buy_signal'].notna()]
#         sell_signals = df[df['sell_signal'].notna()]

#         fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['buy_signal'], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', color='green', size=10)))
#         fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['sell_signal'], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', color='red', size=10)))
#     return df, fig