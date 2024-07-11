import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from Backtesting import Backtest as bt, data_retriever_util as dr
import plotly.io as pio

# Connect to the source SQLite database
source_db_path = "C:\\Users\\Lenovo\\Downloads\\StockBuddyGenAI\\src\\Data\\NSE_Yahoo_9_FEB_24.sqlite"
source_conn = sqlite3.connect(source_db_path)

# Read the data into a pandas DataFrame
query = "SELECT * FROM NSE WHERE ticker = 'TATAMOTORS.NS'"
df = pd.read_sql(query, source_conn, parse_dates=['Date'])

# Close the connection to the source database
source_conn.close()

# Calculate the Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std_dev=2):
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
        if data['band_width'].iloc[i] / data['rolling_mean'].iloc[i] < squeeze_threshold:
            squeeze = True
        else:
            squeeze = False

        # Entry Condition
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

        # Exit Condition
        elif not squeeze and data['close'].iloc[i] < data['lower_band'].iloc[i]:
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

df = implement_bollinger_band_squeeze_strategy(df)

# Print the unique values of the 'Trigger' column to debug
print(df['Trigger'].nunique())

# Print the first few rows of the DataFrame to check the strategy signals
print(df.head(20))

# Backtesting the strategy
result = bt.simpleBacktest(df)
print(result)

# Plotting the results
fig = pio.from_json(dr.plotGraph(df))
fig.show()

plt.figure(figsize=(12, 8))
plt.plot(df['close'], label='Close Price', alpha=0.5)
plt.plot(df['upper_band'], label='Upper Bollinger Band', color='red', alpha=0.5)
plt.plot(df['lower_band'], label='Lower Bollinger Band', color='blue', alpha=0.5)
plt.scatter(df.index, df['buy_signal'], label='Buy Signal', marker='^', color='green', alpha=1)
plt.scatter(df.index, df['sell_signal'], label='Sell Signal', marker='v', color='red', alpha=1)
plt.title('Bollinger Band Squeeze Strategy with Stop-Loss')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
