import pandas as pd
from pathlib import Path

# =============================================================================
# 1. SINGLE COIN DATA LOADER MODULE
# =============================================================================

class SingleCoinDataLoader:
    """Handles loading, saving, and reloading a single cryptocurrency dataset."""

    def __init__(self, coin_name: str, data_dir: str = "../data/"):
        self.coin_name = coin_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data = None

    def load_coin_data(self, filename: str) -> pd.DataFrame:
        """Load and preprocess the coin CSV file."""
        filepath = self.data_dir / filename
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        daily_data = df.groupby('Date')['Close'].mean().reset_index()
        daily_data['Coin'] = self.coin_name
        self.data = daily_data
        print(f"Loaded {len(daily_data)} rows for {self.coin_name}")
        return self.data

    def save_coin_data(self, filename: str = None) -> None:
        """Save the coin data to a CSV file."""
        if self.data is not None:
            filename = filename or f"{self.coin_name}-PROCESSED.csv"
            path = self.data_dir / filename
            self.data.to_csv(path, index=False)
            print(f"Saved {self.coin_name} data to {path}")
        else:
            raise ValueError("No data to save.")

    def load_processed_data(self, path: str = None) -> pd.DataFrame:
        """Load previously saved processed coin data."""
        if not path:
            raise FileNotFoundError(f"{path} not found.")
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        self.data = df
        print(f"Reloaded {self.coin_name} data from {path}")
        return self.data

# if __name__ == "__main__":
#     loader = SingleCoinDataLoader(coin_name="BTC")
#     # # make sure to uncomment if running for first time
#     # loader.load_coin_data("BTC-USD.csv")
#     # loader.save_coin_data()
#     coin_data = loader.load_processed_data("../data/BTC-PROCESSED.csv")
