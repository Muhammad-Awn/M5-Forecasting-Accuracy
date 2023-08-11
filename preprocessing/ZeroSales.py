import pandas as pd

class ZeroSales:
    def __init__(self, data):
        self.data = data

    def zero_sales(self):
        df = pd.DataFrame()
        for i in self.data.columns[:70]:
            zero_values = pd.DataFrame(self.data[self.data[i] <= 0].index, columns=[i])
            df = pd.concat([df, zero_values], axis=1)
        return df