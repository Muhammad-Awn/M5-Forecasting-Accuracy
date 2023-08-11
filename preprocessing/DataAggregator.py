import pandas as pd
class DataAggregator:
    def __init__(self, data, col1, col2, index):
        self.data = data
        self.col1 = col1
        self.col2 = col2
        self.index = index

    def aggregate(self):
        self.data['new_col'] = self.data[self.col1] + '_' + self.data[self.col2]
        self.data = self.data.drop([self.col1, self.col2], axis=1)
        self.data = self.data.groupby('new_col').sum()
        return self.data
    
    def transform(self):
        new_data = self.data.T
        new_data.index=pd.to_datetime(self.index)
        new_data.columns.name = new_data.index.name = None
        return new_data
