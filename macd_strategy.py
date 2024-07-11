import sqlite3
import pandas as pd
from Backtesting import Backtest as bt, data_retriever_util as dr
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from Backtesting import utils as btutil
from Backtesting import Strategies as st

# Connect to the source SQLite database
source_db_path = "C:\\Users\\Dhanika Dewan\\Documents\\GitHub\\StockBuddyGenAI\\src\\Data\\NSE_Yahoo_9_FEB_24.sqlite"
source_conn = sqlite3.connect(source_db_path)

# Read the data into a pandas DataFrame
query = "SELECT * FROM NSE WHERE ticker = 'TATAMOTORS.NS'"
df = pd.read_sql(query, source_conn, parse_dates=['Date'])

# Close the connection to the source database
source_conn.close()
    
macd_result= st.implement_macd(df)

