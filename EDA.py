import matplotlib.pyplot as plt



def apply_eda(data_df):

    missing_values(data_df)


def missing_values(data_df):
    """
        Apply exploratory data analysis (EDA) on the dataset.
        This function will generate summary statistics and visualizations.
        """
    # count rows with NaN values
    missing_count = data_df.isna().any(axis=1).sum()

    # Visualize missing values over time
    plt.figure(figsize=(12, 4))
    plt.plot(data_df.index, data_df['temperature_2m_fc24h'].isna(), drawstyle='steps-post')
    plt.title('Missing Data Over Time (1 = Missing, 0 = Present)')
    plt.ylabel('Is Missing')
    plt.xlabel('Time')
    plt.text(0.5, 0.2, f"Total rows with NaN values: {missing_count}", ha='center', va='center', transform=plt.gca().transAxes)
    plt.show()