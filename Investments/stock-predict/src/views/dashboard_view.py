import pandas as pd 
import streamlit as st
from src.models.data_models import StockData
from src.models.graphs_models import StockChart

class DashboardView:
    def __init__(self):
        self.data_model = StockData()
        self.tickers = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN']
        self.period_map = {'1m': '1mo', '6m': '6mo', 'YTD': 'ytd', '1y': '1y', 'all': 'max'}

    def render_sidebar(self):
        st.sidebar.header("Choose your filter:")
        self.ticker = st.sidebar.selectbox('Choose Ticker', options=self.tickers, help='Select a ticker')
        self.selected_range = st.sidebar.selectbox('Select Period', options=list(self.period_map.keys()))

    def load_data(self):
        self.yf_data = self.data_model.load_data(self.ticker)
        self.df_history = self.yf_data.history(period=self.period_map[self.selected_range])
        self.current_price = self.yf_data.info.get('currentPrice', 'N/A')
        self.previous_close = self.yf_data.info.get('previousClose', 'N/A')
          
    def display_header(self):
        company_name = self.yf_data.info['shortName']
        symbol = self.yf_data.info['symbol']
        st.subheader(f'{company_name} ({symbol})')
        st.divider()
        if self.current_price != 'N/A' and self.previous_close != 'N/A':
            price_change = self.current_price - self.previous_close
            price_change_ratio = (abs(price_change) / self.previous_close * 100)
            price_change_direction = "+" if price_change > 0 else "-"
            st.metric(label='Current Price', value=f"{self.current_price:.2f}",
                      delta=f"{price_change:.2f} ({price_change_direction}{price_change_ratio:.2f}%)")

    def plot_data(self):
        chart = StockChart(self.df_history)
        chart.add_price_chart()
        chart.add_oversold_overbought_lines()
        chart.add_volume_chart()
        chart.render_chart()

    def run(self):
        self.render_sidebar()
        self.load_data()
        self.display_header()
        self.plot_data()
