import yfinance as yf

class StockData:
    def load_data_period(self, ticker, start, end):
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end)
        return data
    
    def load_data(self,ticker):
        return yf.Ticker(ticker)
    