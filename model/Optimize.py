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
        mape = np.mean(np.abs((self.test_data - y_pred) / self.test_data)) * 100
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
        self.best_params = {
                            'p': best['p'], 'd': best['d'], 'q': best['q'],
                            'P': best['P'], 'D': best['D'], 'Q': best['Q'],
                            'm': best['m']
                            }
        self.order = (int(best['p']), int(best['d']), int(best['q']))
        self.seasonal_order = (int(best['P']), int(best['D']), int(best['Q']), int(best['m']))

    
class RFR_Optuna:
    def __init__(self, data, seasonal_period, test_size=0.25):
        self.data = data

        # Data Preprocessing
        for i in range(1, seasonal_period+1):
            self.data = Lag(self.data).lag_transform(i, self.data.columns[0])
        self.data = self.data.dropna()

        # Train Test Split
        x = np.array(self.data.iloc[:,1]).reshape(-1,1)
        for i in range(2, 8):
            xi = (np.array(self.data.iloc[:,i])).reshape(-1,1)
        x = np.concatenate((x, xi), axis=1)
        y = np.array(self.data.iloc[:,0])
        self.x_train = x[:int(len(x)*(1-test_size))]
        self.x_test = x[int(len(x)*(1-test_size)):]
        self.y_train = y[:int(len(y)*(1-test_size))]
        self.y_test = y[int(len(y)*(1-test_size)):]

    def data_preprocess(self, seasonal_period):
        for i in range(1, seasonal_period+1):
            self.data = Lag(self.data).lag_transform(i, self.data.columns[0])
        self.data = self.data.dropna()
        return self.data
    
    def train_test_split(self, test_size=0.25):
        x = np.array(self.data.iloc[:,1]).reshape(-1,1)
        for i in range(2, 8):
            xi = (np.array(self.data.iloc[:,i])).reshape(-1,1)
        x = np.concatenate((x, xi), axis=1)
        y = np.array(self.data.iloc[:,0])
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
        self.best_params = trial.params
        self.mape = trial.value
