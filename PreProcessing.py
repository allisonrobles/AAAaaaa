

def pre_process_data(data):
    """

    """

   # 1. drop all rows that have any missing values
    original_length = len(data)
    data = data.dropna()
    print(f"Total number of dropped Rows: {original_length - len(data)} ({(original_length - len(data)) / original_length * 100:.2f}%)")
    return data
