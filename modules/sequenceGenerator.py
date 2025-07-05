import numpy as np
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 4. SEQUENCE GENERATOR MODULE
# =============================================================================

class SingleCoinSequenceGenerator:
    """Generates LSTM sequences for a single cryptocurrency (without coin encoding)."""

    def __init__(self, lookback: int = 7):
        self.lookback = lookback

    def generate_sequences(self, scaled_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate (X, y) sequences for a single coin.

        Args:
            scaled_data (np.ndarray): Scaled price data, shape (n, 1)

        Returns:
            Tuple of np.ndarray: X shape = (samples, lookback, 1), y shape = (samples,)
        """
        X, y = [], []

        for i in range(len(scaled_data) - self.lookback):
            sequence = scaled_data[i:(i + self.lookback), 0]
            label = scaled_data[i + self.lookback, 0]
            X.append(sequence.reshape(-1, 1))
            y.append(label)

        return np.array(X), np.array(y)
