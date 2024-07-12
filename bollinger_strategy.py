import sqlite3
import pandas as pd
from Backtesting import Backtest as bt, data_retriever_util as dr
import plotly.graph_objects as go
import plotly.io as pio
from Backtesting import utils as btutil
from Backtesting import Strategies as st

# Connect to the source SQLite database
# source_db_path = "C:\\Users\\Lenovo\\Downloads\\StockBuddyGenAI\\src\\Data\\NSE_Yahoo_9_FEB_24.sqlite"
#source_db_path = r"D:\Dkafka\StockBuddyGenAI\src\Data\NSE_Yahoo_9_FEB_24.sqlite"
source_db_path = r'C:\Users\burma\OneDrive\Documents\GitHub\StockBuddyGenAI\src\Data\NSE_Yahoo_9_FEB_24.sqlite'
source_conn = sqlite3.connect(source_db_path)

# Read the data into a pandas DataFrame
query = "SELECT * FROM NSE WHERE ticker = 'TATAMOTORS.NS'"
df = pd.read_sql(query, source_conn, parse_dates=['Date'])

# Close the connection to the source database
source_conn.close()

# Run the strategy
result = st.bollinger_band_squeeze(df, toPlot=True)

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
