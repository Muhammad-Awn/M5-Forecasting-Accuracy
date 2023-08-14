from sklearn.model_selection import train_test_split

class TrainTestSplit:
    def __init__(self, X, test_size=0.25):
        self.X = X
        self.test_size = test_size

    def data_split(self):
        return train_test_split(self.X, test_size=self.test_size, random_state=0,
                                shuffle=False,stratify=None)
        