from sklearn.ensemble import RandomForestRegressor
from preprocessing.FeatureEngineering import Lag
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials
from hyperopt.pyll.base import scope
import statsmodels.api as sm
import numpy as np
import optuna

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
    
class RFR_Optuna:
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

    def objective(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 2, 150)
        max_depth = trial.suggest_int('max_depth', 1, 50)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 2, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, bootstrap=bootstrap,
                               min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        output = reg.fit(self.x_train, self.y_train)
        predictions = output.predict(self.x_test)
    
        mape = np.mean(np.abs((self.y_test - predictions) / self.y_test)) * 100
        return mape
    
    def hyperparameter_tune(self, num_evals=100):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=num_evals)
        trial = study.best_trial
        self.params = trial.params
        self.mape = trial.value

    def fit(self):
        self.reg = RandomForestRegressor(n_estimators=self.params['n_estimators'], max_depth=self.params['max_depth'], max_features=self.params['max_features'], bootstrap=self.params['bootstrap'],
                               min_samples_leaf=self.params['min_samples_leaf'], min_samples_split=self.params['min_samples_split'])
        self.reg.fit(self.x_train, self.y_train)
        return self.reg
    
    def predict(self):
        self.pred = self.reg.predict(self.x_test)
        return self.pred
    
    def evaluate(self):
        return np.mean(np.abs((self.y_test - self.pred) / self.y_test)) * 100