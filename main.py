import pandas as pd
import json

from preprocessing.ImputeMean import ImputeMean
from preprocessing.TrainTestSplit import TrainTestSplit
from preprocessing.DataAggregator import DataAggregator

from model.Train import SRX, RFR
from model.Optimize import SarimaxHyperopt, RFR_Optuna
from model.Evaluate import Evaluate

from joblib import Parallel, delayed


def save_to_json(filename, data, mode='w'):
    with open(f"./best_params/{filename}.json", mode) as f:
        json.dump(data, f, indent=4)

def read_from_json(filename,mode="r"):
    with open(f'./best_params/{filename}.json', mode) as f:
        # Load the JSON data into a Python dictionary
        params_data = json.load(f)
        return params_data
    

def hyperparameter_tuning(train_data, test_data, col, evals = 3):
    params_dict = [{}, {}]

    srx_hyperopt = SarimaxHyperopt(train_data[col], test_data[col])

    rfr_optuna = RFR_Optuna(train_data[[col]], 7 , 0.2)

    models = [srx_hyperopt, rfr_optuna]

    for i, model in enumerate(models):

        model.hyperparameter_tune(evals)

        params_dict[i][col] = model.best_params

    return params_dict


def model_training(eval_data, val_data, data, col, params, model_names, total_forecast):

    total_forecast[f'Original_{col}'] = val_data[col]

    srx_model = SRX(eval_data[col], val_data[[col]])

    rfr_model = RFR(data[[col]])

    rfr_model.data_preprocess(7)

    rfr_model.train_test_split(len(val_data)/len(data))

    models = [srx_model, rfr_model]

    for i, model in enumerate(models):

        model.fit(params[i][col])

        forecast = model.predict()

        total_forecast[f'{model_names[i]}_{col}'] = forecast
    
    return total_forecast


def main():
    # Read data
    calendar_df = pd.read_csv('E:/Documents/TanXor/Dataset/calendar.csv')
    validation_df = pd.read_csv('E:/Documents/TanXor/Dataset/sales_train_evaluation.csv')

    # Initializing Parameters
    date = calendar_df['date'].iloc[:1941]
    col1, col2 = 'store_id', 'dept_id'

    data = DataAggregator(validation_df)

    # Takes col1 and col2 and aggregates them into a new column
    data.aggregate(col1, col2)

    # Drops passed columns
    data.drop(col1)

    # Groups by the new column
    data.group_by()

    # Transforms the dataframe using '.T' function
    data.transform()

    # Sets the index to the date column
    data = data.set_datetime_index(date)

    # Replace zero sales with the mean of sales of that respective store and department
    ImputeMean(data, 0).imputer()

    #Split data into evaluation and validation sets
    eval_data, val_data = data.iloc[:1913, :], data.iloc[1913:, :]

    # Splits the data into train and test sets
    train_data, test_data = TrainTestSplit(eval_data, 0.2).data_split()

    # Sets the frequency of the data to daily
    eval_data.index.freq = val_data.index.freq = train_data.index.freq = test_data.index.freq = 'd'

    pp = int(input("Press 1 if you wish to use Joblib for parallel processing (Recommended)\n Press 2 if you do not want to perform parallel processing\n"))

    # Hyperparameter tuning
    if pp:
        # Parallel processing
        output = Parallel(n_jobs=-1)(
        delayed(hyperparameter_tuning)(train_data, test_data, col, 2)
        for col in train_data.columns)

    else:
        # Serial processing
        output = [{} for _ in range(len(train_data.columns))]

        for i, col in enumerate(train_data.columns):
            output[i] = hyperparameter_tuning(train_data, test_data, col, 2)

    # Saving the best parameters to json files 
    srx_best_ = output[0][0]
    rfr_best_ = output[0][1]

    for i in range(1, len(output)):
        srx_best_.update(output[i][0])
        rfr_best_.update(output[i][1])

    params_path = ['sarimax_best_params', 'rfr_best_params']
    params_data = [srx_best_, rfr_best_]

    for i, path in enumerate(params_path):
        save_to_json(path, params_data[i], 'w')

    # Reading the best parameters from json files
    srx_params = {}
    rfr_params = {}

    params = [srx_params, rfr_params]
    paths = ['sarimax_best_params', 'rfr_best_params']
    model_names = ['SARIMAX', 'Random_Forest']

    for i, path in enumerate(paths):
        params[i] =  read_from_json(path, mode="r")

    # Model training
    forecast = pd.DataFrame()

    if pp:
        # Parallel processing
        forecast = Parallel(n_jobs=-1)(
            delayed(model_training)(eval_data, val_data, data, col, params, model_names, forecast)
            for col in train_data.columns)

        forecast = pd.concat(forecast, axis=1)

    else:
        # Serial processing
        for col in train_data.columns:
            forecast = model_training(eval_data, val_data, data, col, params, model_names, forecast)

    # Seperating the forecasted data
    sarimax_forecast = forecast[[col for col in forecast.columns if "SARIMAX" in col]].copy()
    randomforest_forecast = forecast[[col for col in forecast.columns if "Random" in col]].copy()

    sarimax_forecast['Total'] = sarimax_forecast.sum(axis=1)
    randomforest_forecast['Total'] = randomforest_forecast.sum(axis=1)

    # Model evaluation
    future = val_data.sum(axis = 1).values
    model_eval = Evaluate(future)

    predictions = [sarimax_forecast, randomforest_forecast]
    preds = []

    for pred in predictions:
        preds.append(model_eval.mape(pred['Total']))

    best_model = model_names[preds.index(min(preds))]
    best_pred = predictions[preds.index(min(preds))]
    best_pred = best_pred.drop([best_pred.columns[0], best_pred.columns[-1]], axis=1)

    print('BEST MAPE:\n', best_model + ': ' + str(min(preds)))

    # Structure the forecasted data
    df = pd.DataFrame(val_data)

    forecast = pd.DataFrame()

    # Stack the columns column-wise
    transpose_data = df.transpose().stack()

    best_pred = best_pred.transpose().stack()

    forecast['store_id'] = transpose_data.index.get_level_values(0)

    forecast['month'] = transpose_data.index.get_level_values(1).strftime('%B')

    forecast['year'] = transpose_data.index.get_level_values(1).year

    forecast['original_sales'] = transpose_data.values

    forecast['forecast'] = best_pred.values

    forecast['timestap_columns'] = transpose_data.index.get_level_values(1)

    forecast.to_csv("E:/Documents/TanXor/Model/forecasted_data/forecast.csv", header=True)


if __name__ == "__main__":
    main()