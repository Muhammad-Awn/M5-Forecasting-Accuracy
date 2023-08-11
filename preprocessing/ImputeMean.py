class ImputeMean:
    def __init__(self, data, value):
        self.data = data
        self.value = value

    def imputer(self):
        self.data.replace(self.value, self.data.mean(axis=0), inplace=True)

