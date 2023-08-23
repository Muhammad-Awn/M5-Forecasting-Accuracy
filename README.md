## <div align="center">M5 Forecasting Accuracy Competition (Walmart Sales Forecasting)</div>


**Introduction**

This repository contains a Python script for time series forecasting using parallel processing. The script uses various preprocessing techniques, hyperparameter tuning, and multiple forecasting models to predict future values of time series data. Parallel processing is implemented to speed up the hyperparameter tuning and model training processes.

The main focus of the script is to use Joblib for parallel processing, Optuna with Random Forest Regressor and HyperOpt with SARIMAX for hyperparameter tuning and model training.

---

**Usage**

1. **Installation**

    Before running the script, make sure you have the required packages installed. You can install the necessary packages using the following command:

    ```
    pip install joblib
    ```
    ```
    pip install optuna
    ```
    ```
    pip install hyperopt
    ```

2. **Dataset**

    The script uses two input CSV files: `calendar.csv` and `sales_train_evaluation.csv`. These files contain the calendar data and sales data for evaluation, respectively. Update the file paths accordingly:

    ```python
    calendar_df = pd.read_csv('path/to/calendar.csv')
    validation_df = pd.read_csv('path/to/sales_train_evaluation.csv')
    ```

3. **Preprocessing**

    The data preprocessing involves aggregation, transformation, and imputation of missing values. The steps are encapsulated in the `DataAggregator` and `ImputeMean` classes:

    ```python
    # Aggregating and transforming the data
    data = DataAggregator(validation_df)
    data.aggregate(col1, col2)
    data.drop(col1)
    data.group_by()
    data.transform()

    # Setting the datetime index and imputing missing values
    data = data.set_datetime_index(date)
    ImputeMean(data, 0).imputer()
    ```

4. **Hyperparameter Tuning**

    The script performs hyperparameter tuning for two forecasting models: SARIMA-X and Random Forest Regression. The tuning process is parallelized using the `Parallel` function from the `joblib` library:

    ```python
    # Parallel processing for hyperparameter tuning
    output = Parallel(n_jobs=-1)(
        delayed(hyperparameter_tuning)(train_data, test_data, col, 2)
        for col in train_data.columns)
    ```

5. **Model Training**

    The models are trained using the best parameters obtained from hyperparameter tuning. Similar to hyperparameter tuning, the model training process can also be parallelized using the `Parallel` function:

    ```python
    # Parallel processing for model training
    forecast = Parallel(n_jobs=-1)(
        delayed(model_training)(eval_data, val_data, data, col, params, model_names, forecast)
        for col in train_data.columns)
    ```

6. **Model Evaluation and Forecast Visualization**

    The script evaluates the forecasting performance using the Mean Absolute Percentage Error (MAPE). It then selects the model with the lowest MAPE for generating forecasts. The forecasts are structured and stored in a CSV file:

    ```python
    # Calculate MAPE and select the best model
    best_model = model_names[preds.index(min(preds))]
    best_pred = predictions[preds.index(min(preds))]

    # Structure the forecasted data and save to CSV
    df = pd.DataFrame(val_data)
    forecast = pd.DataFrame()

    # ... (Code for structuring and saving forecasted data)
    ```

7. **Running the Script**

    Run the script by executing the following command in your terminal:

    ```
    python script_name.py
    ```

---

**Conclusion**

This repository showcases a comprehensive approach to time series forecasting using parallel processing. It demonstrates the process of preprocessing, hyperparameter tuning, model training, and evaluation. By leveraging parallel processing, the script significantly reduces the time required for tuning and training, making it an efficient solution for large-scale time series forecasting tasks.

For more details, feel free to explore the code and customize it to your specific requirements.

---

**Author's Remarks**

This repository is my first official Github repository and also my first Time Series Analysis and Forecasting project.
This repository has much to improve and requires accurate and more precise optimization and preprocessing techniques.
Any feedback or contribution is highly appreciated.

Thank You
