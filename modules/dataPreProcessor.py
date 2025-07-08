import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class SingleCoinPreprocessor:
    """Handles preprocessing for a single cryptocurrency with separate scalers for train and test."""

    def __init__(self, prediction_days: int = 60):
        self.prediction_days = prediction_days
        self.scaler_train = None
        self.scaler_test = None

    def prepare_data(self, df: pd.DataFrame) -> dict:
        """
        Prepares and scales training and test data using separate scalers.

        Args:
            df (pd.DataFrame): DataFrame with 'Date' and 'Close' columns.

        Returns:
            dict: Contains original and scaled train/test sets, scalers, and dates.
        """
        df = df.sort_values('Date')
        prices = df['Close'].values.reshape(-1, 1)

        train_size = len(prices) - self.prediction_days
        if train_size <= 0:
            raise ValueError("Not enough data to split into train and test.")

        train_data = prices[:train_size]
        test_data = prices[train_size:]

        # Fit separate scalers
        self.scaler_train = MinMaxScaler(feature_range=(0, 1))
        self.scaler_test = MinMaxScaler(feature_range=(0, 1))

        scaled_train = self.scaler_train.fit_transform(train_data)
        scaled_test = self.scaler_test.fit_transform(test_data)

        print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

        return {
            'train_data': train_data,
            'test_data': test_data,
            'scaled_train': scaled_train,
            'scaled_test': scaled_test,
            'dates': df['Date'].values
        }

    def save_preprocessor(self, filepath: str):
        """Save both scalers to disk."""
        joblib.dump({
            'scaler_train': self.scaler_train,
            'scaler_test': self.scaler_test,
            'prediction_days': self.prediction_days
        }, filepath)
        print(f"Preprocessor saved to {filepath}")

    def load_preprocessor(self, filepath: str):
        """Load both scalers from disk."""
        data = joblib.load(filepath)
        self.scaler_train = data['scaler_train']
        self.scaler_test = data['scaler_test']
        self.prediction_days = data['prediction_days']
        print(f"Preprocessor loaded from {filepath}")
