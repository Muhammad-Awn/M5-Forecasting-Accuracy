class ImputeMean:
    def imputer(data, value):
        data.replace(value, data.mean(axis=0), inplace=True)

