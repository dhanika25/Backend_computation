# ADL_strategy.py

import sqlite3
import pandas as pd
from Backtesting.Strategies import implement_ADL
from Backtesting import data_retriever_util as dr
import plotly.io as pio
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

# Implement the ADL strategy
result = implement_ADL(df, toPlot=True)

# Print or use the results as needed
print({
    'Win Rate [%]': result['Win Rate [%]'],
    'Net Profit/Loss [$]': result['Net Profit/Loss [$]'],
    'Total Trades': result['Total Trades'],
    'Winning Trades': result['Winning Trades']
})

# Display the plot if available
if "plotlyJson" in result:
    fig = pio.from_json(result["plotlyJson"])
    fig.show()
else:
    print("Plotly JSON object not found in the result.")
