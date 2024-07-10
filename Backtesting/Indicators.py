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
