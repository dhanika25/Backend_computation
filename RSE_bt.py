import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from Backtesting import Backtest as bt
from ta.momentum import RSIIndicator  # Import RSIIndicator from ta

# Connect to the source SQLite database
source_db_path = r'C:\Users\burma\OneDrive\Documents\GitHub\StockBuddyGenAI\src\Data\NSE_Yahoo_9_FEB_24.sqlite'
source_conn = sqlite3.connect(source_db_path)

# Read the data into a pandas DataFrame
query = "SELECT * FROM NSE WHERE ticker = 'TATAMOTORS.NS'"
df = pd.read_sql(query, source_conn, parse_dates=['Date'])

# Close the connection to the source database
source_conn.close()

# Calculate RSI
def calculate_rsi(data, window=14):
    rsi = RSIIndicator(close=data['close'], window=window)
    data['RSI'] = rsi.rsi()
    return data

# Implementing the RSI Strategy with Stop-Loss Levels and Trigger Column
def implement_rsi_strategy(data, overbought_threshold=70, oversold_threshold=30, stop_loss_percentage=0.05):
    buy_signals = [float('nan')]  # Initialize with nan
    sell_signals = [float('nan')]  # Initialize with nan
    triggers = ['H']  # Initialize with 'Hold'
    position = None  # None means no position, 1 means holding stock, 0 means not holding stock
    buy_price = 0  # Track the price at which the stock was bought

    for i in range(1, len(data)):
        # Entry Condition (Buy)
        if data['RSI'].iloc[i - 1] < oversold_threshold and data['RSI'].iloc[i] >= oversold_threshold:
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

# Apply RSI calculation and strategy implementation
df = calculate_rsi(df)
df = implement_rsi_strategy(df)

# Perform a simple backtest
result = bt.simpleBacktest(df)
print("Backtest Results:")
print(result)

# Plot function with stop-loss visualization
def plotGraph(df, stockName="No name", stop_loss_percentage=0.05):
    stockName = stockName[:-3]  # Assuming the stockname will be Ticker data.(stockName.NS)

    # Initialize the figure with subplots
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        subplot_titles=(stockName + " Historical Data", 'Volume', 'RSI'),
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

    # Add RSI indicator to the third subplot
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI'), row=3, col=1)

    # Add overbought (upper bound) and oversold (lower bound) lines to the RSI subplot
    fig.add_shape(
        type="line", line=dict(color="red", width=1, dash="dash"),
        x0=df['Date'].iloc[0], y0=70, x1=df['Date'].iloc[-1], y1=70,  # Upper bound (overbought)
        row=3, col=1
    )
    fig.add_shape(
        type="line", line=dict(color="blue", width=1, dash="dash"),
        x0=df['Date'].iloc[0], y0=30, x1=df['Date'].iloc[-1], y1=30,  # Lower bound (oversold)
        row=3, col=1
    )

    # Add small red dots where stop-loss is triggered
    stop_loss_dates = df.loc[df['Trigger'] == 'S', 'Date']
    stop_loss_prices = df.loc[df['Trigger'] == 'S', 'close']
    fig.add_trace(go.Scatter(x=stop_loss_dates, y=stop_loss_prices, mode='markers', marker=dict(color='red', size=4),
                             name='Stop Loss'), row=1, col=1)

    # Add legend for RSI subplot
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Customize layout
    fig.update_layout(
        height=800, title=stockName,
        xaxis_title='Date', yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    return fig

# Call the plot function
fig = plotGraph(df, stockName='TATAMOTORS.NS', stop_loss_percentage=0.05)

# Convert the figure to JSON
plotly_json = pio.to_json(fig, pretty=True)
result["plotlyJson"] = plotly_json

# Display the figure
fig.show()

# Print backtest results
print("Backtest Results:")
print(result)
