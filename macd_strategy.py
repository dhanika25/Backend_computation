import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from Backtesting import Backtest as bt, data_retriever_util as dr
import plotly.io as pio

# Connect to the source SQLite database
source_db_path = "C:\\Users\\Dhanika Dewan\\Documents\\GitHub\\StockBuddyGenAI\\src\\Data\\NSE_Yahoo_9_FEB_24.sqlite"
source_conn = sqlite3.connect(source_db_path)

# Read the data into a pandas DataFrame
query = "SELECT * FROM NSE WHERE ticker = 'TATAMOTORS.NS'"
df = pd.read_sql(query, source_conn,parse_dates=['Date'])

# Close the connection to the source database
source_conn.close()

# # Ensure the date column is in datetime format
# df['Date'] = pd.to_datetime(df['Date'])
# df.set_index('Date', inplace=True)

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

# Print the final DataFrame
# print(df)

result= bt.simpleBacktest(df)

print(result)

fig = pio.from_json(dr.plotGraph(df))
fig.show()

# Plotting the results
plt.figure(figsize=(12,8))
plt.plot(df['close'], label='Close Price', alpha=0.5)
plt.plot(df['macd'], label='MACD', color='red', alpha=0.5)
plt.plot(df['signal_line'], label='Signal Line', color='blue', alpha=0.5)
plt.scatter(df.index, df['buy_signal'], label='Buy Signal', marker='^', color='green', alpha=1)
plt.scatter(df.index, df['sell_signal'], label='Sell Signal', marker='v', color='red', alpha=1)
plt.title('MACD Trend Following Strategy with Stop-Loss')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()