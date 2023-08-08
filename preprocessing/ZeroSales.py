class ZeroSales:
    def zero_sales(data):
        for i in data.columns:
            print(i, ': ', data[data[i] <= 0].index, len(data[data[i] <= 0]))