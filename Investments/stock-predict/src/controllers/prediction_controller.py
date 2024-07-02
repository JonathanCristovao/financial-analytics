import streamlit as st
import pandas as pd
from src.models.graphs_models import Plots
from src.models.prediction_models import PredictionModel
from src.models.data_models import StockData

class PredictionController:
    def __init__(self):
        self.tickers = ['NVDA', 'AAPL', 'GOOGL', 'MSFT', 'AMZN']
        self.model = PredictionModel()
        self.data_model = StockData()
        self.data = None


    def load_data(self):
        self.ticker = st.sidebar.selectbox('Choose Stock Ticker', self.tickers)
        start_date = st.sidebar.date_input("Start date", value=pd.to_datetime('2020-01-01'))
        end_date = st.sidebar.date_input("End date", value=pd.to_datetime('today'))
        if st.sidebar.button('Load Data'):
           st.session_state['stock_data'] = self.data_model.load_data_period(self.ticker, start_date, end_date)
           st.write("Data loaded successfully.")
           return st.session_state['stock_data']
        return None

    def train_model(self):
        if st.button('Train Model'):
            if 'stock_data' in st.session_state:
                st.write("Training model...")
                self.data = st.session_state['stock_data']
                self.model.train(self.data)
                predictions = self.model.predict(self.data)
                future_predictions = self.model.forecast_future(self.data,days=5)
                self.display_predictions(self.data, predictions, future_predictions)
            else:
                st.write("Please load data before training the model.")


    def display_predictions(self, stock_data, predictions, future_predictions):
        plot_instance = Plots(stock_data)
        plot_instance.plot_predictions(predictions, future_predictions)

    def run(self):
        st.write("--------------------------------------------")
        st.write(f'<div style="font-size:50px">🤖 Real-Time Stock Prediction', unsafe_allow_html=True)
        self.load_data()
        self.train_model()
