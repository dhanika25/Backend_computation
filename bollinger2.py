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

# Calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['ema_short'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data['ema_long'] = data['close'].ewm(span=long_window, adjust=False).mean()
    data['macd'] = data['ema_short'] - data['ema_long']
    data['macd_signal'] = data['macd'].ewm(span=signal_window, adjust=False).mean()
    data['macd_histogram'] = data['macd'] - data['macd_signal']
    return data

df = calculate_macd(df)

# Implementing the Bollinger Band Squeeze Strategy with MACD Confirmation
def implement_bollinger_band_squeeze_strategy(data, squeeze_threshold=0.1, stop_loss_percentage=0.05, trailing_stop_percentage=0.05, max_days_in_trade=14):
    buy_signals = [float('nan')]
    sell_signals = [float('nan')]
    triggers = ['H']
    position = None
    buy_price = 0
    entry_date = None

    for i in range(1, len(data)):
        squeeze = data['band_width'].iloc[i] / data['rolling_mean'].iloc[i] < squeeze_threshold
        macd_crosses_above_signal = data['macd'].iloc[i] > data['macd_signal'].iloc[i] and data['macd'].iloc[i-1] <= data['macd_signal'].iloc[i-1]
        macd_histogram_positive = data['macd_histogram'].iloc[i] > 0
        macd_crosses_above_zero = data['macd'].iloc[i] > 0

        # Entry Condition
        if squeeze and macd_crosses_above_signal and macd_histogram_positive and macd_crosses_above_zero and data['close'].iloc[i] > data['upper_band'].iloc[i]:
            if position != 1:
                buy_signals.append(data['close'].iloc[i])
                sell_signals.append(float('nan'))
                triggers.append('B')
                position = 1
                buy_price = data['close'].iloc[i]
                entry_date = data['Date'].iloc[i]
                print(f"Buy at {data['Date'].iloc[i]} price {data['close'].iloc[i]}")
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))
                triggers.append('H')

        # Exit Condition based on Bollinger Bands
        elif position == 1 and data['close'].iloc[i] < data['lower_band'].iloc[i]:
            buy_signals.append(float('nan'))
            sell_signals.append(data['close'].iloc[i])
            triggers.append('S')
            position = 0
            entry_date = None
            print(f"Sell at {data['Date'].iloc[i]} price {data['close'].iloc[i]}")

        # Exit Condition based on MACD
        elif position == 1 and data['macd'].iloc[i] < data['macd_signal'].iloc[i]:
            buy_signals.append(float('nan'))
            sell_signals.append(data['close'].iloc[i])
            triggers.append('S')
            position = 0
            entry_date = None
            print(f"MACD Sell at {data['Date'].iloc[i]} price {data['close'].iloc[i]}")

        # Exit Condition based on Stop-Loss
        elif position == 1 and data['close'].iloc[i] < buy_price * (1 - stop_loss_percentage):
            buy_signals.append(float('nan'))
            sell_signals.append(data['close'].iloc[i])
            triggers.append('S')
            position = 0
            entry_date = None
            print(f"Stop-loss triggered at {data['Date'].iloc[i]} price {data['close'].iloc[i]}")

        # Exit Condition based on Trailing Stop
        elif position == 1 and data['close'].iloc[i] < data['close'].iloc[i-1] * (1 - trailing_stop_percentage):
            buy_signals.append(float('nan'))
            sell_signals.append(data['close'].iloc[i])
            triggers.append('S')
            position = 0
            entry_date = None
            print(f"Trailing Stop triggered at {data['Date'].iloc[i]} price {data['close'].iloc[i]}")

        # Exit Condition based on Time-Based Exit
        elif position == 1 and (data['Date'].iloc[i] - entry_date).days > max_days_in_trade:
            buy_signals.append(float('nan'))
            sell_signals.append(data['close'].iloc[i])
            triggers.append('S')
            position = 0
            entry_date = None
            print(f"Time-based exit at {data['Date'].iloc[i]} price {data['close'].iloc[i]}")

        else:
            buy_signals.append(float('nan'))
            sell_signals.append(float('nan'))
            triggers.append('H')

    data['buy_signal'] = buy_signals
    data['sell_signal'] = sell_signals
    data['Trigger'] = triggers  
    return data

df = implement_bollinger_band_squeeze_strategy(df)

# Print the final DataFrame
print(df)

# Check if 'Trigger' column has the correct signals
print(df['Trigger'].value_counts())

# Backtest the strategy
result = bt.simpleBacktest(df)
print(result)

# Plotting using plotly
fig = pio.from_json(dr.plotGraph(df))
fig.show()

# Plotting the results using matplotlib
plt.figure(figsize=(12, 8))
plt.plot(df['Date'], df['close'], label='Close Price', alpha=0.5)
plt.plot(df['Date'], df['upper_band'], label='Upper Bollinger Band', color='red', alpha=0.5)
plt.plot(df['Date'], df['lower_band'], label='Lower Bollinger Band', color='blue', alpha=0.5)
plt.scatter(df['Date'], df['buy_signal'], label='Buy Signal', marker='^', color='green', alpha=1)
plt.scatter(df['Date'], df['sell_signal'], label='Sell Signal', marker='v', color='red', alpha=1)
plt.title('Bollinger Band Squeeze Strategy with MACD Confirmation and Stop-Loss')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
