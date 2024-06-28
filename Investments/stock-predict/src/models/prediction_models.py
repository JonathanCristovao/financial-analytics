import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


class PredictionModel:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
    def prepare_data(self,data, look_back=1):
        scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        def create_dataset(dataset):
            X, Y = [], []
            for i in range(len(dataset) - look_back):
                a = dataset[i:(i + look_back), 0]
                X.append(a)
                Y.append(dataset[i + look_back, 0])
            return np.array(X), np.array(Y)
        X, Y = create_dataset(scaled_data)
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        return X, Y    


    def train(self, data,epochs=5, batch_size=1):
        X, Y = self.prepare_data(data)
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(1, 1)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, data):
        X, _ = self.prepare_data(data)
        predictions = self.model.predict(X)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
    
    def forecast_future(self,data, days=5):
        last_value = data['Close'].values[-1:].reshape(-1, 1)
        last_scaled = self.scaler.transform(last_value)
        future_predictions = []
        for _ in range(days):
            prediction = self.model.predict(last_scaled.reshape(1, 1, 1))[0]
            future_predictions.append(prediction)
            last_scaled = prediction  
        future_predictions = self.scaler.inverse_transform(future_predictions)
        return future_predictions
