# Single-Coin LSTM Preprocessing Module

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class SingleCoinPreprocessor:
    """Handles preprocessing for a single cryptocurrency."""

    def __init__(self, prediction_days: int = 60):
        self.prediction_days = prediction_days
        self.scaler = None

    def prepare_data(self, df: pd.DataFrame) -> dict:
        """
        Prepares and scales training and test data.

        Args:
            df (pd.DataFrame): DataFrame with 'Date' and 'Close' columns.

        Returns:
            dict: Contains scaled train/test sets, original values, and dates.
        """
        df = df.sort_values('Date')
        prices = df['Close'].values.reshape(-1, 1)

        train_size = len(prices) - self.prediction_days
        if train_size <= 0:
            raise ValueError("Not enough data to split into train and test.")

        train_data = prices[:train_size]
        test_data = prices[train_size:]

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = self.scaler.fit_transform(train_data)
        scaled_test = self.scaler.transform(test_data)

        print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

        return {
            'train_data': train_data,
            'test_data': test_data,
            'scaled_train': scaled_train,
            'scaled_test': scaled_test,
            'dates': df['Date'].values
        }

    def save_preprocessor(self, filepath: str):
        """Save the scaler to disk."""
        joblib.dump({
            'scaler': self.scaler,
            'prediction_days': self.prediction_days
        }, filepath)
        print(f"Preprocessor saved to {filepath}")

    def load_preprocessor(self, filepath: str):
        """Load the scaler from disk."""
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.prediction_days = data['prediction_days']
        print(f"Preprocessor loaded from {filepath}")
