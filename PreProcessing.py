

def pre_process_data(data):
    """

    """

   # 1. drop all rows that have any missing values
    original_length = len(data)

    # 1. Compute quantile bounds
    lower_limit = data['price'].quantile(0.01)
    upper_limit = data['price'].quantile(0.99)

    print(f"Price quantile bounds: 1% = {lower_limit:.2f}, 99% = {upper_limit:.2f}")

    # 2. Clip values outside bounds
    data[('price')] = data['price'].clip(lower=lower_limit, upper=upper_limit)

    # drop rain
    data = data.drop(columns=['rain_fc48h', 'rain_fc24h'])

    data = data.dropna()
    print(f"Total number of dropped Rows: {original_length - len(data)} ({(original_length - len(data)) / original_length * 100:.2f}%)")

    return data
