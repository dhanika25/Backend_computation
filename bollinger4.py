import sqlite3
import pandas as pd

from Backtesting import Backtest as bt, data_retriever_util as dr, utils as btutil
import plotly.io as pio
import plotly.graph_objects as go

# Connect to the source SQLite database
source_db_path = "C:\\Users\\Lenovo\\Downloads\\StockBuddyGenAI\\src\\Data\\NSE_Yahoo_9_FEB_24.sqlite"
source_conn = sqlite3.connect(source_db_path)

# Read the data into a pandas DataFrame
query = "SELECT * FROM NSE WHERE ticker = 'TATAMOTORS.NS'"
df = pd.read_sql(query, source_conn, parse_dates=['Date'])

# Close the connection to the source database
source_conn.close()

# Calculate the Bollinger Bands
def calculate_bollinger_bands(data, window= 20, num_std_dev=2):
    data['rolling_mean'] = data['close'].rolling(window=window).mean()
    data['rolling_std'] = data['close'].rolling(window=window).std()
    data['upper_band'] = data['rolling_mean'] + (data['rolling_std'] * num_std_dev)
    data['lower_band'] = data['rolling_mean'] - (data['rolling_std'] * num_std_dev)
    data['band_width'] = data['upper_band'] - data['lower_band']
    return data

df = calculate_bollinger_bands(df)

# Implementing the Bollinger Band Squeeze Strategy with Stop-Loss Levels and Trigger Column
def implement_bollinger_band_squeeze_strategy(data, squeeze_threshold=0.1, stop_loss_percentage=0.02):
    buy_signals = [float('nan')]  # Initialize with nan
    sell_signals = [float('nan')]  # Initialize with nan
    triggers = ['H']  # Initialize with 'Hold'
    position = None  # None means no position, 1 means holding stock, 0 means not holding stock
    buy_price = 0  # Track the price at which the stock was bought

    for i in range(1, len(data)):
        # Identify the squeeze condition
        squeeze = data['band_width'].iloc[i] / data['rolling_mean'].iloc[i] < squeeze_threshold
        condition_met = False

        # Entry Condition
        if condition_met == False:
            if squeeze and data['close'].iloc[i] > data['upper_band'].iloc[i]:
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
                condition_met = True

        # Exit Condition
        if condition_met == False:
            if not squeeze and data['close'].iloc[i] < data['lower_band'].iloc[i]:
                if position == 1:
                    buy_signals.append(float('nan'))
                    sell_signals.append(data['close'].iloc[i])
                    triggers.append('S')
                    position = 0
                    print(data["Date"].iloc[i], "normalexit")
                else:
                    buy_signals.append(float('nan'))
                    sell_signals.append(float('nan'))
                    triggers.append('H')
                condition_met = True

        # Exit Condition based on Stop-Loss
        if condition_met == False:
            if position == 1 and data['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
                buy_signals.append(float('nan'))
                sell_signals.append(data['close'].iloc[i])
                triggers.append('S')
                position = 0
                print(data["Date"].iloc[i], "stoploss")
                condition_met = True

        # Default case when no condition is met
        if condition_met == False:
            buy_signals.append(float('nan'))
            sell_signals.append(float('nan'))
            triggers.append('H')

    data['buy_signal'] = buy_signals
    data['sell_signal'] = sell_signals
    data['Trigger'] = triggers

    return data

df = implement_bollinger_band_squeeze_strategy(df)
result = bt.simpleBacktest(df)
print(result)

# Create a Plotly figure
fig = pio.from_json(dr.plotGraph(df))
fig = btutil.addBuySell2Graph(df, fig)

# Add traces for the Bollinger Bands
fig.add_trace(go.Scatter(x=df['Date'], y=df['close'], mode='lines', name='Close Price', line=dict(color='black')))
fig.add_trace(go.Scatter(x=df['Date'], y=df['upper_band'], mode='lines', name='Upper Bollinger Band', line=dict(color='red')))
fig.add_trace(go.Scatter(x=df['Date'], y=df['lower_band'], mode='lines', name='Lower Bollinger Band', line=dict(color='blue')))
fig.show()
