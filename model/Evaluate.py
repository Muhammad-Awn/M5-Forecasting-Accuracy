import numpy as np

class Evaluate:
    def __init__(self, data):
        self.data = data

    def mape(self, pred):
        pred = np.mean(np.abs((self.data - pred) / self.data)) * 100
        return pred