from hyperopt import hp,fmin,tpe,STATUS_OK,Trials
from hyperopt.pyll.base import scope
import statsmodels.api as sm
import numpy as np

class SarimaxHyperopt:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
    
    def fit(self):
        self.trained_model = sm.tsa.statespace.SARIMAX(
            self.train_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()
        return self.trained_model
    
    def predict(self):
        if not hasattr(self, 'trained_model'):
            raise ValueError("Model has not been trained yet. Please call fit() before predict().")
        
        pred = self.trained_model.forecast(steps=self.test_data.shape[0])
        return pred
    
    def objective(self, params):
        order = (params['p'], params['d'], params['q'])
        seasonal_order = (params['P'], params['D'], params['Q'], params['m'])
        model = sm.tsa.statespace.SARIMAX(
            self.train_data,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()
        y_pred = model.get_prediction(start=self.test_data.index[0], end=self.test_data.index[-1], dynamic=False).predicted_mean
        mape = self.evaluate(y_pred)
        return mape

    def hyperparameter_tune(self, num_evals=100):
        space = {
            'p': scope.int(hp.quniform('p', 0, 5, 1)),
            'd': scope.int(hp.quniform('d', 0, 2, 1)),
            'q': scope.int(hp.quniform('q', 0, 5, 1)),
            'P': scope.int(hp.quniform('P', 0, 5, 1)),
            'D': scope.int(hp.quniform('D', 0, 2, 1)),
            'Q': scope.int(hp.quniform('Q', 0, 5, 1)),
            'm': scope.int(hp.quniform('m', 6, 8, 1))
        }
        trials = Trials()
        best = fmin(fn=self.objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=num_evals,
                    trials=trials)
        self.order = (int(best['p']), int(best['d']), int(best['q']))
        self.seasonal_order = (int(best['P']), int(best['D']), int(best['Q']), int(best['m']))

    def evaluate(self, y_pred):
        return np.mean(np.abs((self.test_data - y_pred) / self.test_data)) * 100

    def best_params(self):
        return self.order, self.seasonal_order