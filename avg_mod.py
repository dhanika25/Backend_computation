import sqlite3
import pandas as pd

# Paths to the original and new SQLite database files
original_db_path = 'C:\\Users\\Dhanika Dewan\\Documents\\GitHub\\StockBuddyGenAI\\src\\Data\\NSE_Yahoo_9_FEB_24.sqlite'
new_db_path = 'output.sqlite'

# Connect to the original SQLite database
conn_original = sqlite3.connect(original_db_path)

# Load the full data from the NSE table
data_query = "SELECT * FROM NSE;"
data = pd.read_sql_query(data_query, conn_original)

# Ensure the data is sorted by Date
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by=['ticker', 'Date'])

# Define the periods for the moving averages
ma_periods = [5, 10, 20]

# Initialize columns for entry and exit signals
data['C_5_MORE_THAN_10'] = False
data['C_10_MORE_THAN_5'] = False

# Function to calculate indicators and signals for the last row of each group
def calculate_indicators_signals(group):
    if len(group) >= max(ma_periods):
        for period in ma_periods:
            group[f'I_MA{period}'] = group['Adj_Close'].rolling(window=period).mean()
        entry_condition = (group['I_MA5'] > group['I_MA10']) & (group['I_MA5'].shift(1) <= group['I_MA10'].shift(1))
        exit_condition = (group['I_MA5'] < group['I_MA10']) & (group['I_MA5'].shift(1) >= group['I_MA10'].shift(1))
        group.loc[entry_condition, 'C_5_MORE_THAN_10'] = True
        group.loc[exit_condition, 'C_10_MORE_THAN_5'] = True
    return group

# Apply the function to each group
data = data.groupby('ticker', group_keys=False).apply(calculate_indicators_signals)

# Filter the DataFrame to only keep the last row of each group (unique stock)
last_rows = data.groupby('ticker', group_keys=False).tail(1)

# Connect to the new SQLite database
conn_new = sqlite3.connect(new_db_path)

# Store the resulting DataFrame in the new SQLite database
last_rows.to_sql('NSE_with_indicators', conn_new, if_exists='replace', index=False)

# Optional: Verify the stored data
result_query = "SELECT * FROM NSE_with_indicators LIMIT 50;"
result = pd.read_sql_query(result_query, conn_new)
print(result.head(50))

# Close the connections
conn_original.close()
conn_new.close()
