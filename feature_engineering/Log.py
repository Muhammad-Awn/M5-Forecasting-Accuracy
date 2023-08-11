import pandas as pd
import numpy as np

class Log:
    def __init__(self, data):
        self.data = data

    def log_transform(self, col):
        log_data = pd.DataFrame(np.log(self.data[col]), columns=[f"log_{col}"])
        return pd.concat([self.data, log_data], axis=1)