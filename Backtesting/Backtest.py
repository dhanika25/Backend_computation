import backtesting as bt

def simpleBacktest(df_ticker):
    pnl_results = []

    # Check for empty DataFrames and data sufficiency (same as before)
    if df_ticker.empty:
        print(f"Skipping empty DataFrame for ticker: {df_ticker['ticker'].iloc[0] if not df_ticker.empty else 'Unknown'}")
        return
    
    if 'Date' not in df_ticker.columns or df_ticker['Date'].nunique() < 2:
        print(f"Insufficient data or missing 'Date' column for ticker: {df_ticker['ticker'].iloc[0] if not df_ticker.empty else 'Unknown'}")
        return

    # Rename columns and set index
    column_mapping = {
        'Open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'Volume': 'Volume'
    }
    data = df_ticker.rename(columns=column_mapping).set_index('Date')[['Open', 'High', 'Low', 'Close', 'Trigger']]

    # Define a simple strategy class that only acts on 'Trigger'
    class backtest(bt.Strategy):   
        def init(self):
            pass  # Empty init method is required

        def next(self):
            if self.data['Trigger'][-1] == 'B':
                self.buy()
            elif self.data['Trigger'][-1] == 'S':
                self.sell()
        
    # Run the backtest
    bt_result = bt.Backtest(data, backtest, trade_on_close=False, exclusive_orders=True)
    result = bt_result.run()

    net_profit = result.loc["Equity Final [$]"] - 10000

    # print(result._trades)
    # Get winning trades
    winning_trades = result.loc["# Trades"] * result.loc["Win Rate [%]"] / 100

    # Gather the specific results you want
    return {
            'Win Rate [%]': result.loc["Win Rate [%]"],
            'Net Profit/Loss [$]': net_profit,
            'Total Trades': result.loc["# Trades"],
            'Winning Trades': winning_trades,
        }