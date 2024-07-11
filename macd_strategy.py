import sqlite3
import pandas as pd
from Backtesting import Backtest as bt, data_retriever_util as dr
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Connect to the source SQLite database
source_db_path = "C:\\Users\\Dhanika Dewan\\Documents\\GitHub\\StockBuddyGenAI\\src\\Data\\NSE_Yahoo_9_FEB_24.sqlite"
source_conn = sqlite3.connect(source_db_path)

# Read the data into a pandas DataFrame
query = "SELECT * FROM NSE WHERE ticker = 'TATAMOTORS.NS'"
df = pd.read_sql(query, source_conn, parse_dates=['Date'])

# Close the connection to the source database
source_conn.close()

# Calculate the MACD and Signal Line
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['ema_short'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data['ema_long'] = data['close'].ewm(span=long_window, adjust=False).mean()
    data['macd'] = data['ema_short'] - data['ema_long']
    data['signal_line'] = data['macd'].ewm(span=signal_window, adjust=False).mean()
    data['macd_histogram'] = data['macd'] - data['signal_line']
    return data

df = calculate_macd(df)

# Implementing the MACD Trend Following Strategy with Stop-Loss Levels and Trigger Column
def implement_macd_strategy(data, stop_loss_percentage=0.05):
    buy_signals = [float('nan')]  # Initialize with nan
    sell_signals = [float('nan')]  # Initialize with nan
    triggers = ['H']  # Initialize with 'Hold'
    position = None  # None means no position, 1 means holding stock, 0 means not holding stock
    buy_price = 0  # Track the price at which the stock was bought

    for i in range(1, len(data)):
        # Entry Condition
        if (data['macd'].iloc[i] > data['signal_line'].iloc[i] and 
            data['macd_histogram'].iloc[i] > 0 and
            data['macd'].iloc[i] > 0):
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
        
        # Exit Condition based on MACD
        elif (data['macd'].iloc[i] < data['signal_line'].iloc[i] or
              data['macd_histogram'].iloc[i] < 0 or
              data['macd'].iloc[i] < 0):
            if position == 1:
                buy_signals.append(float('nan'))
                sell_signals.append(data['close'].iloc[i])
                triggers.append('S')
                position = 0
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        # Exit Condition based on Stop-Loss
        elif position == 1 and data['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
            buy_signals.append(float('nan'))
            sell_signals.append(data['close'].iloc[i])
            triggers.append('S')
            position = 0
        
        else:
            buy_signals.append(float('nan'))
            sell_signals.append(float('nan'))
            triggers.append('H')

    data['buy_signal'] = buy_signals
    data['sell_signal'] = sell_signals
    data['Trigger'] = triggers
    return data

df = implement_macd_strategy(df)

# Perform a simple backtest
result = bt.simpleBacktest(df)
print(result)

## Plot function
# Plot function
def plotGraph(df, stockName="No name"):
    stockName = stockName[:-3]  # Assuming the stockname will be Ticker data.(stockName.NS)

    # Initialize the figure with subplots
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        subplot_titles=(stockName + " Historical Data", 'Volume', 'MACD'),
        row_heights=[0.6, 0.2, 0.2], specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # Define colors for increasing and decreasing candles
    increasing_color = 'green'
    decreasing_color = 'red'

    # Add the candlestick chart to the first subplot
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['high'], low=df['low'], close=df['close'],
        increasing=dict(line=dict(color=increasing_color, width=1), fillcolor=increasing_color),
        decreasing=dict(line=dict(color=decreasing_color, width=1), fillcolor=decreasing_color),
        name="Candlestick"
    ), row=1, col=1)

    # Add volume trace to the second subplot
    colors = ['green' if close >= open_ else 'red' for open_, close in zip(df['Open'], df['close'])]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    # Add MACD line to the third subplot
    fig.add_trace(go.Scatter(x=df['Date'], y=df['macd'], mode='lines', name='MACD'), row=3, col=1)
    # Add signal line to the third subplot
    fig.add_trace(go.Scatter(x=df['Date'], y=df['signal_line'], mode='lines', name='Signal Line'), row=3, col=1)
    # Add MACD histogram to the third subplot
    fig.add_trace(go.Bar(x=df['Date'], y=df['macd_histogram'], name='MACD Histogram'), row=3, col=1)

    # Customize layout
    fig.update_layout(
        height=800, title=stockName,
        xaxis_title='Date', yaxis_title='Price',
        xaxis_rangeslider_visible=False, showlegend=False
    )

    # Show the plot
    fig.show()
    return fig

# Call the plot function
plotGraph(df, stockName='TATAMOTORS.NS')