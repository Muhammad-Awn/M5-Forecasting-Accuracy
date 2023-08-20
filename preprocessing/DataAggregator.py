import pandas as pd
class DataAggregator:
    def __init__(self, data):
        self.data = data

    def aggregate(self, col1, col2):
        self.col1, self.col2 = col1, col2
        self.col = self.col1 + '_' + self.col2
        self.data[self.col] = self.data[self.col1] + '_' + self.data[self.col2]
        return self.data
    
    def drop(self, cols):
        self.data = self.data.drop([cols], axis=1)
        return self.data
    
    def group_by(self):
        self.data = self.data.groupby(self.col).sum()
        return self.data
    
    def transform(self):
        self.data = self.data.T
        return self.data.T
    
    def set_datetime_index(self, index):
        self.data.index = pd.to_datetime(index)
        self.data.columns.name = self.data.index.name = None
        return self.data
