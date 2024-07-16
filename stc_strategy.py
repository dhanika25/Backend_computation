import sqlite3
import pandas as pd
from Backtesting import Backtest as bt, data_retriever_util as dr
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from Backtesting import utils as btutil
from Backtesting import Strategies as st

# Connect to the source SQLite database
source_db_path = r'C:\Users\Lenovo\Downloads\StockBuddyGenAI\src\Data\NSE_Yahoo_9_FEB_24.sqlite'
source_conn = sqlite3.connect(source_db_path)

# Read the data into a pandas DataFrame
query = "SELECT * FROM NSE WHERE ticker = 'TATAMOTORS.NS'"
df = pd.read_sql(query, source_conn, parse_dates=['Date'])

# Ensure 'Volume' column exists
if 'Volume' not in df.columns:
    print("The DataFrame does not contain a 'Volume' column.")
    exit()

# Close the connection to the source database
source_conn.close()

# Parameters for STC strategy
short_window = 12
long_window = 26
signal_window = 9
cycle_window = 10
stop_loss_percentage = 0.05
toPlot = True

# Apply STC strategy
result = st.implement_stc_strategy(df, short_window, long_window, signal_window, cycle_window, stop_loss_percentage, toPlot)

# Print the result summary
print({
    'Win Rate [%]': result['Win Rate [%]'],
    'Net Profit/Loss [$]': result['Net Profit/Loss [$]'],
    'Total Trades': result['Total Trades'],
    'Winning Trades': result['Winning Trades']
})

# Check if the result contains the plotlyJson key
if "plotlyJson" in result:
    # Convert JSON back to a Plotly figure
    fig = pio.from_json(result["plotlyJson"])
    # Display the figure
    fig.show()
else:
    print("Plotly JSON object not found in the result.")