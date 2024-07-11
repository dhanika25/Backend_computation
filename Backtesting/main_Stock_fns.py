import pandas as pd
import numpy as np
import sqlite3
import backtesting as bt
from tqdm import tqdm
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random

import Indicators as ndct
import Strategies as strg

print("Loading NSE_Yahoo in memory...", end='')
# Connect to the SQLite database
conn = sqlite3.connect('C:\\Users\\Dhanika Dewan\\Documents\\GitHub\\StockBuddyGenAI\\src\\Data\\NSE_Yahoo_9_FEB_24.sqlite')

# Step 1: Read the entire table once
query = "SELECT * FROM NSE "
df = pd.read_sql_query(query, conn)
conn.close()

print(" Completed.")
print("Processing loaded DF...")
# Step 2: Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Step 3: Group by 'ticker' and create a dictionary of DataFrames
dfs_dict = {ticker: group for ticker, group in tqdm(df.groupby('ticker'), desc='Grouping by ticker')}

# Step 4: Convert the dictionary of DataFrames to a list of DataFrames and reset the index
# dfs_list = [group.reset_index(drop=True) for group in tqdm(dfs_dict.values())]
for ticker, group_df in tqdm(dfs_dict.items(), desc='Making separate df for each ticker'):
    dfs_dict[ticker] = group_df.reset_index(drop=True)

ticker_names = list(set(df['ticker']))
strategy_name = ["smaCross", "smaCross2"]

print("DF loading completed.")

def getBacktestingResult(tickerList, strategyList, toPlot):
    if not tickerList:
        tickerList = ticker_names  # Replace with your default ticker list

    if not strategyList:
        strategyList = strategy_name  # Assuming 'strategy_name' is a global variable or list

    results = []
    for ticker in tickerList:
        for strategy in strategyList:
            df_ticker = dfs_dict[ticker]

            match strategy:
                case "smaCross":
                    data = strategy["arguments"] # data means arguments needed in the function
                    data["df"] = df_ticker
                    data["toPlot"] = toPlot

                    row = strg.smaCross(strategy["short_window"], strategy["long_window"], strategy["df"], strategy["toPlot"])
                    row["ticker"] = ticker
                    row["strategy"] = strategy["Name"]
                    results.append(row)
                case _:
                    print(f"Invalid strategy name: {strategy}")
                    
    return results