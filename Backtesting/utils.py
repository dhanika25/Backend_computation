import pandas as pd
import plotly.graph_objects as go

def addBuySell2Graph(df, fig=None):
    """
    Function to add buy and sell signals to a plotly figure.

    Parameters:
    - fig: The plotly figure object.
    - df: DataFrame containing the stock data with 'Trigger' column.

    Returns:
    - fig: The updated plotly figure object with buy and sell signals added.
    """
    # Add buy signals
    buy_indices = df.index[df['Trigger'] == 'B']
    buy_dates = df.loc[buy_indices, 'Date']
    buy_prices = df.loc[buy_indices, 'low']  # You can adjust the y-coordinate as needed
    fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', marker_symbol='triangle-up', marker=dict(color='green', size=10), name='Buy Signal'))

    # Add sell signals
    sell_indices = df.index[df['Trigger'] == 'S']
    sell_dates = df.loc[sell_indices, 'Date']
    sell_prices = df.loc[sell_indices, 'high']  # You can adjust the y-coordinate as needed
    fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', marker_symbol='triangle-down', marker=dict(color='red', size=10), name='Sell Signal'))

    # Add lines connecting buy to sell signals
    last_buy_index = None

    for index, row in df.iterrows():
        if row['Trigger'] == 'B':
            last_buy_index = index
        elif row['Trigger'] == 'S' and last_buy_index is not None:
            buy_date = df.at[last_buy_index, 'Date']
            buy_price = df.at[last_buy_index, 'low']
            sell_date = row['Date']
            sell_price = row['high']
            fig.add_trace(go.Scatter(
                x=[buy_date, sell_date],
                y=[buy_price, sell_price],
                mode='lines',
                line=dict(color='blue', dash='dot'),
                showlegend=False
            ))
            last_buy_index = None  # Reset last buy index after connecting to a sell

    return fig

def getTriggerColumn(df_ticker):
   
    df_ticker['Trigger'] = ''
    trigger_state = ''
    for i in range(1, len(df_ticker)):
        if trigger_state == '' or trigger_state == 'S' and df_ticker.at[i, 'Buy'] == 1:
            df_ticker.at[i, 'Trigger'] = 'B'
            trigger_state = 'B'
        elif trigger_state == 'B' and df_ticker.at[i, 'Sell'] == 1:
            df_ticker.at[i, 'Trigger'] = 'S'
            trigger_state = 'S'
        else:
            df_ticker.at[i, 'Trigger'] = 'H'

    
    # trigger_counts = df_ticker['Trigger'].value_counts()
    # # print(trigger_counts) 

    # trigger_counts2 = df_ticker['Buy'].value_counts()
    # # print(trigger_counts2)

    # trigger_counts3 = df_ticker['Sell'].value_counts()
    # # print(trigger_counts3)
       
    return df_ticker # Return the dataframe

def removeRedundantRows(df):
    """
    Remove rows from the DataFrame where the 'Trigger' column has the value 'H',
    and reset the dates starting from a specific date.

    Parameters:
    - df: The input DataFrame.

    Returns:
    - df_filtered: The DataFrame with redundant rows removed and dates reset.
    """
    # print(len(df))
    
    # Filter out rows where 'Trigger' column has the value 'H'
    df = df[df['Trigger'] != 'H'].reset_index(drop=True)
    df_filtered = df[df['Trigger'] != ''].reset_index(drop=True)
    
    # print(len(df_filtered))
    
    # Generate a new date range starting from a specific date
    start_date = pd.to_datetime('2001-01-01')  # Replace with your desired start date
    new_dates = pd.date_range(start_date, periods=len(df_filtered), freq='D')
    
    # Reset the 'Date' column with the new date range
    df_filtered['Date'] = new_dates
    
    return df_filtered