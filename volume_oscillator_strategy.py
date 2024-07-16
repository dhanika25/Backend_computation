import sqlite3
import pandas as pd
from Backtesting import utils as btutil
from Backtesting.Strategies import implement_volume_oscillator
from Backtesting import data_retriever_util as dr
import plotly.io as pio

# Connect to the source SQLite database
source_db_path = r'C:\Users\burma\OneDrive\Documents\GitHub\StockBuddyGenAI\src\Data\NSE_Yahoo_9_FEB_24.sqlite'
source_conn = sqlite3.connect(source_db_path)

# Read the data into a pandas DataFrame
query = "SELECT * FROM NSE WHERE ticker = 'TATAMOTORS.NS'"
df = pd.read_sql(query, source_conn, parse_dates=['Date'])

# Close the connection to the source database
source_conn.close()

result = implement_volume_oscillator(df, toPlot=True)
print({
    'Win Rate [%]': result['Win Rate [%]'],
    'Net Profit/Loss [$]': result['Net Profit/Loss [$]'],
    'Total Trades': result['Total Trades'],
    'Winning Trades': result['Winning Trades']
})

if "plotlyJson" in result:
    fig = pio.from_json(result["plotlyJson"])
    fig.show()
else:
    print("Plotly JSON object not found in the result.")
