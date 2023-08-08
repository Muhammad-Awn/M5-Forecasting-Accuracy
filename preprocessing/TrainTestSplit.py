from sklearn.model_selection import train_test_split

class TrainTestSplit:
    def data_split(data, test_size):
        return train_test_split(data, test_size=test_size, random_state=42)