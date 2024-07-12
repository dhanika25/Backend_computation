import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.io as pio

def plotGraph(df, stockName="No name"):
	stockName=stockName[:-3] # Assuming the stockname will be Ticker data.(stockName.NS)

    # Initialize the figure with subplots	
	fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02, subplot_titles=(stockName+" Historical Data", 'Volume'),
						row_heights=[0.5, 0.2, 0.3], specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]])

	# Define colors for increasing and decreasing candles
	increasing_color = 'green'
	decreasing_color = 'red'

	# Add the candlestick chart to the first subplot with matching outline and fill colors
	fig.add_trace(go.Candlestick(x=df['Date'],
								open=df['Open'],
								high=df['high'],
								low=df['low'],
								close=df['close'],
								increasing=dict(line=dict(color=increasing_color, width=1), fillcolor=increasing_color),
								decreasing=dict(line=dict(color=decreasing_color, width=1), fillcolor=decreasing_color),
								name="Candlestick"), row=1, col=1)


	# Add volume trace to the second subplot
	colors = ['green' if close >= open_ else 'red' for open_, close in zip(df['Open'], df['close'])]
	fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

	# Customize layout without setting xaxis_type to 'category'
	fig.update_layout(height=900, title=stockName,
					xaxis_title='Date',
					yaxis_title='Price',
					xaxis_rangeslider_visible=False,
					showlegend=True)
	
	# Show the plot
	# fig.show()
	# return pio.to_json(fig, pretty=True)
	return fig
