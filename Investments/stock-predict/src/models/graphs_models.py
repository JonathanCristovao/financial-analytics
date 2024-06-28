import pandas as pd 
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class StockChart:
    def __init__(self, data):
        self.data = data
        self.fig = make_subplots(rows=2, cols=1, vertical_spacing=0.01, shared_xaxes=True)

    def add_price_chart(self):
        self.fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Open'], name='Open Price', marker_color='#1F77B4'), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=self.data.index, y=self.data['High'], name='High Price', marker_color='#9467BD'), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Low'], name='Low Price', marker_color='#D62728'), row=1, col=1)
        self.fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Close'], name='Close Price', marker_color='#76B900'), row=1, col=1)

    
    def add_oversold_overbought_lines(self):
        self.fig.add_hline(y=30, line_dash='dash', line_color='limegreen', line_width=1, row=1, col=1)
        self.fig.add_hline(y=70, line_dash='dash', line_color='red', line_width=1, row=1, col=1)
        self.fig.update_yaxes(title_text='RSI Score', row=1, col=1)

    def add_volume_chart(self):
        colors = ['#9C1F0B' if row['Open'] - row['Close'] >= 0 else '#2B8308' for index, row in self.data.iterrows()]
        self.fig.add_trace(go.Bar(x=self.data.index, y=self.data['Volume'], showlegend=False, marker_color=colors), row=2, col=1)

    def render_chart(self):
        self.fig.update_layout(title='Historical Price and Volume', height=500, margin=dict(l=0, r=10, b=10, t=25))
        st.plotly_chart(self.fig, use_container_width=True)        

class Plots:
    def __init__(self, data):
        self.data = data

    def plot_predictions(self, predictions, future_predictions):

        predicted_dates = self.data.index[-len(predictions):]  
        future_dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=len(future_predictions), freq='B')
        predictions = [float(val) for val in predictions if pd.notna(val)]
        future_predictions = [float(val) for val in future_predictions if pd.notna(val)]

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Close'], mode='lines', name='Actual Stock Prices', marker_color='blue'))
        fig.add_trace(go.Scatter(x=predicted_dates, y=predictions, mode='lines', name='LSTM Predicted Prices', marker_color='red', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Future Predictions', marker_color='green', line=dict(dash='dot')))

        fig.update_layout(title='Comparison of Actual, Predicted, and Future Stock Prices', xaxis_title='Date', yaxis_title='Price', legend_title='Legend', height=500)
        st.plotly_chart(fig, use_container_width=True)



    