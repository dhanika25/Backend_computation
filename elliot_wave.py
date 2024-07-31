import sqlite3
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.io as pio
import Backtesting.Strategies as st

# Connect to the source SQLite database
load_dotenv()
source_db_path = os.getenv('NSE_DB_PATH')

if source_db_path is None:
    raise ValueError("NSE_DB_PATH environment variable is not set")

source_conn = sqlite3.connect(source_db_path)

# Read the data into a pandas DataFrame
query = "SELECT * FROM NSE WHERE ticker = 'TATAMOTORS.NS'"
df = pd.read_sql(query, source_conn, parse_dates=['Date'])

# Close the connection to the source database
source_conn.close()

# Parameters for Elliott Wave strategy
stop_loss_percentage = 0.02
toPlot = True

# Apply Elliott Wave strategy
result = st.implement_elliott_wave(df, stop_loss_percentage, toPlot)

# Print the result summary
print({
    'Win Rate [%]': result['Win Rate [%]'],
    'Net Profit/Loss [$]': result['Net Profit/Loss [$]'],
    'Total Trades': result['Total Trades'],
    'Winning Trades': result['Winning Trades']
})

# Check if the result contains the plotlyJson key
if "plotlyJson" in result:
    fig = pio.from_json(result["plotlyJson"])
    fig.show()
else:
    print("Plotly JSON object not found in the result.")
