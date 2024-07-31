import sqlite3
import pandas as pd
from Backtesting import Backtest as bt, data_retriever_util as dr
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from Backtesting import utils as btutil
from Backtesting import Strategies as st
import os
from dotenv import load_dotenv

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

# Parameters for Ichimoku Cloud strategy
tenkan_sen_period = 9
kijun_sen_period = 26
senkou_span_b_period = 52
senkou_shift = 26
stop_loss_percentage = 0.05
toPlot = True

# Apply Ichimoku Cloud strategy
result = st.ichimoku_cloud_strategy(df, tenkan_sen_period, kijun_sen_period, senkou_span_b_period, senkou_shift, stop_loss_percentage, toPlot)

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
