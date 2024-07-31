import sqlite3
import os
from dotenv import load_dotenv
import pandas as pd
from Backtesting import Backtest as bt, data_retriever_util as dr
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from Backtesting import utils as btutil
from Backtesting import Strategies as st

# Connect to the source SQLite database
load_dotenv()
source_db_path = os.getenv('NSE_DB_PATH')

if source_db_path is None:
    raise ValueError("NSE_DB_PATH environment variable is not set")

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

# Parameters for Gann Angles strategy
key_price_points = [100, 200, 300]  # Example key price points; should be passed as arguments to avoid hardcoding
angles = [45, 90, 135]  # Example angles; should be passed as arguments to avoid hardcoding
stop_loss_percentage = 0.05
toPlot = True

# Apply Gann Angles strategy
result = st.implement_gann_angles(df, key_price_points, angles, stop_loss_percentage, toPlot)

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
