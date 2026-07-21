import pandas as pd


def fetch_sydney_sttm_gas_prices():
    """
    Downloads official Sydney STTM daily wholesale gas prices (AUD/GJ)
    directly from the Australian Energy Regulator / AEMO public data feed.
    """
    # Direct open-data CSV endpoint for STTM Hub Daily Prices
    url = "https://www.aer.gov.au/system/files/STTM%20-%20Daily%20weighted%20average%20prices%20by%20hub.csv"

    print("Fetching real Sydney STTM gas price data from AER/AEMO...")

    # Read directly from URL
    df = pd.read_csv(url)

    # Clean up column names and filter for Sydney hub
    df.columns = df.columns.str.strip().str.lower()

    # Filter for Sydney Hub (NSW)
    sydney_df = df[df['hub'].str.contains('Sydney', case=False, na=False)].copy()

    # Parse dates and sort
    sydney_df['date'] = pd.to_datetime(sydney_df['dates'], format='%d/%m/%Y', errors='coerce')
    sydney_df = sydney_df.dropna(subset=['date']).sort_values('date')

    # Select relevant columns (Ex-Ante price is the market schedule price)
    sydney_df = sydney_df[['date', 'ex-ante price ($/gj)', 'ex-post price ($/gj)']]
    sydney_df.columns = ['date', 'price_ex_ante_aud_gj', 'price_ex_post_aud_gj']

    return sydney_df.reset_index(drop=True)


# Fetch data
nsw_gas_df = fetch_sydney_sttm_gas_prices()
print(nsw_gas_df.tail(10))