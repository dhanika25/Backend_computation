import sqlite3
import pandas as pd

# Paths to the original and new SQLite database files
original_db_path = 'C:\\Users\\Dhanika Dewan\\Documents\\VSCode\\Backend\\first50.sqlite'
new_db_path = 'output_with_indicators.sqlite'

# Connect to the original SQLite database
conn_original = sqlite3.connect(original_db_path)

# Load the full data from the NSE table
data_query = "SELECT * FROM NSE;"
data = pd.read_sql_query(data_query, conn_original)

# Ensure the data is sorted by Date
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')

# Define the periods for the moving averages
ma_periods = [5, 10, 20]

# Calculate moving averages and add them as new columns
for i, period in enumerate(ma_periods, start=1):
    data[f'ind{i}'] = data['Adj_Close'].rolling(window=period).mean()

# Initialize columns for entry and exit signals
data['entry_signal'] = False
data['exit_signal'] = False

# Define the entry condition
entry_condition = (data['ind1'] > data['ind2']) & (data['ind1'].shift(1) <= data['ind2'].shift(1))

# Define the exit condition
exit_condition = (data['ind1'] < data['ind2']) & (data['ind1'].shift(1) >= data['ind2'].shift(1))

# Apply the conditions to the DataFrame
data.loc[entry_condition, 'entry_signal'] = True
data.loc[exit_condition, 'exit_signal'] = True

# Connect to the new SQLite database
conn_new = sqlite3.connect(new_db_path)

# Store the resulting DataFrame in the new SQLite database
data.to_sql('NSE_with_indicators', conn_new, if_exists='replace', index=False)

# Optional: Verify the stored data
result_query = "SELECT * FROM NSE_with_indicators LIMIT 50;"
result = pd.read_sql_query(result_query, conn_new)
print(result.head(50))

# Close the connections
conn_original.close()
conn_new.close()