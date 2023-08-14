from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

class TrainModel:
    def __init__(self, data):
        self.data = data

    def train(self, params):
        self.model = SARIMAX(self.data,
                        order=(params['p'], params['d'], params['q']),
                        seasonal_order=(params['P'], params['D'], params['Q'], params['s']),
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit()
        return self.model
    
    def predict(self, test_data):
        self.test_data = test_data
        if not hasattr(self, 'model'):
            raise ValueError("Model has not been trained yet. Please call train() before predict().")
        self.pred = self.model.forecast(steps=test_data.shape[0])
        return self.pred
    
    def evaluate(self):
        return np.mean(np.abs((self.test_data - self.pred) / self.test_data)) * 100