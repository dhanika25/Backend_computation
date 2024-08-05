import sqlite3
import pandas as pd
from Backtesting import utils as btutil
from Backtesting.Strategies import implement_RSI
from Backtesting import data_retriever_util as dr
import plotly.io as pio
#import RSI
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


result= implement_RSI(df,toPlot=True)
print({
        'Win Rate [%]': result['Win Rate [%]'],
        'Net Profit/Loss [$]': result['Net Profit/Loss [$]'],
        'Total Trades': result['Total Trades'],
        'Winning Trades': result['Winning Trades']
    })
#print(result)


#fig =dr.plotGraph(df, stockName='TATAMOTORS.NS')

# Convert the figure to JSON
#fig = pio.from_json(fig)
#result["plotlyJson"] = plotly_json
#fig=btutil.addBuySell2Graph(df,fig)

# Display the figure
#fig.show()
# Check if the result contains the plotlyJson key
if "plotlyJson" in result:
    # Convert JSON back to a Plotly figure
    fig = pio.from_json(result["plotlyJson"])
    # Display the figure
    fig.show()
else:
    print("Plotly JSON object not found in the result.")