import pandas as pd

class Lag:
    def __init__(self, df):
        self.df = df

    def lag_transform(self, lag, col):
        lag_data = self.df[[col]].shift(lag)
        lag_data.columns = [f"{col}_lag{lag}"]
        return pd.concat([self.df[[col]], lag_data], axis=1)