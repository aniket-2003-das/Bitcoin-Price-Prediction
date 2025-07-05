# Multi-Coin LSTM Forecasting Framework
# Modular implementation for scalable cryptocurrency prediction
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' = Filter out INFO and WARNING, show only ERROR
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


# =============================================================================
# 5. MODEL MODULE
# =============================================================================

class SingleCoinLSTM:
    """LSTM model for single-coin price prediction"""

    def __init__(self, input_shape: Tuple[int, int]):
        self.input_shape = input_shape
        self.model = None

    def build_model(self, lstm_units: List[int] = [128, 64], dropout_rate: float = 0.2):
        """Build and compile the LSTM model"""
        model = Sequential()

        # First LSTM layer
        model.add(LSTM(
            units=lstm_units[0],
            activation='relu',
            return_sequences=len(lstm_units) > 1,
            input_shape=self.input_shape
        ))
        model.add(Dropout(dropout_rate))

        # Additional LSTM layers
        for i, units in enumerate(lstm_units[1:], start=1):
            return_seq = i < len(lstm_units) - 1
            model.add(LSTM(units=units, return_sequences=return_seq))
            model.add(Dropout(dropout_rate))

        # Output layer
        model.add(Dense(units=1))

        # Compile
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        return model

    def get_model_summary(self):
        """Display the model architecture"""
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet. Call build_model() first.")