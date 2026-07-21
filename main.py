from DataHandler import build_all, DatasetConfig
from PreProcessing import pre_process_data
from EDA import apply_eda

import pandas as pd

# Define the configuration
config = DatasetConfig(
    data_folder="data",
    region="NSW1",
    lead_days=(1, 2),
    decision_freq="30min",
    horizons=(30, 360, 1440, 2160),
    cache_dir="cache",
)

if True:
    # laod data
    data_df = pd.read_parquet('data.parquet')
else:
    # Build the dataset
    bundle = build_all(config)
    # Access the master DataFrame
    data_df = bundle["master"]


# Save data for faster usage
data_df.to_parquet('data.parquet', compression='snappy')

apply_eda(data_df)

# preprocess the data
data_df = pre_process_data(data_df)

print ("Preprocessed data shape:", data_df.shape)