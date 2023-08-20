from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from preprocessing.FeatureEngineering import Lag
import numpy as np

class SRX:
    def __init__(self, data, test_data):
        self.data = data
        self.test_data = test_data

    def fit(self, params):
        self.model = SARIMAX(self.data,
                        order=(params['p'], params['d'], params['q']),
                        seasonal_order=(params['P'], params['D'], params['Q'], params['m']),
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit()
        return self.model
    
    def predict(self):
        if not hasattr(self, 'model'):
            raise ValueError("Model has not been trained yet. Please call train() before predict().")
        self.pred = self.model.forecast(steps=self.test_data.shape[0])
        return self.pred.values

    
class RFR:
    def __init__(self, data):
        self.data = data

    def data_preprocess(self, seasonal_period):
        self.df = self.data
        for i in range(1, seasonal_period+1):
            self.df = Lag(self.df).lag_transform(i, self.df.columns[0])
        self.df = self.df.dropna()
        return self.df
    
    def train_test_split(self, test_size=0.25):
        x = np.array(self.df.iloc[:,1]).reshape(-1,1)
        for i in range(2, 8):
            xi = (np.array(self.df.iloc[:,i])).reshape(-1,1)
        x = np.concatenate((x, xi), axis=1)
        y = np.array(self.df.iloc[:,0])
        self.x_train = x[:int(len(x)*(1-test_size))]
        self.x_test = x[int(len(x)*(1-test_size)):]
        self.y_train = y[:int(len(y)*(1-test_size))]
        self.y_test = y[int(len(y)*(1-test_size)):]
        return self.x_train, self.x_test, self.y_train, self.y_test


    def fit(self, params):
        self.model = RandomForestRegressor(n_estimators=params['n_estimators'],
                                           max_depth=params['max_depth'],
                                           min_samples_split=params['min_samples_split'],
                                           min_samples_leaf=params['min_samples_leaf'],
                                           max_features=params['max_features'],
                                           bootstrap=params['bootstrap'],
                                           random_state=42).fit(self.x_train, self.y_train)
        return self.model
    
    def predict(self):
        self.pred = self.model.predict(self.x_test)
        return self.pred
