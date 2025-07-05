# Modular implementation for scalable cryptocurrency prediction

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 7. PREDICTOR MODULE
# =============================================================================
class SingleCoinPredictor:
    """Handles prediction for a single coin"""

    def __init__(self, model, preprocessor, sequence_generator):
        self.model = model
        self.preprocessor = preprocessor
        self.sequence_generator = sequence_generator

    def predict(self, coin_data: Dict[str, Any], num_future_days: int = 7) -> Dict:
        """Make test and future predictions for the coin"""

        scaler = self.preprocessor.scaler  # Single scaler
        # Prepare test sequences
        test_X, test_y = self.sequence_generator.generate_sequences(coin_data['scaled_test'])

        # Test predictions
        if len(test_X) > 0:
            test_predictions = self.model.predict(test_X)
            test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1))
            test_actual = scaler.inverse_transform(test_y.reshape(-1, 1))
        else:
            test_predictions = np.array([])
            test_actual = np.array([])

        # Future predictions
        future_predictions = []
        if num_future_days > 0:
            last_sequence = coin_data['scaled_test'][-self.sequence_generator.lookback:]

            for _ in range(num_future_days):
                input_sequence = last_sequence.reshape(1, self.sequence_generator.lookback, 1)
                next_pred = self.model.predict(input_sequence, verbose=0)
                next_pred_price = scaler.inverse_transform(next_pred.reshape(-1, 1))
                future_predictions.append(next_pred_price[0, 0])

                # Update sequence with predicted value
                last_sequence = np.roll(last_sequence, -1)
                last_sequence[-1] = next_pred[0, 0]

        return {
            'test_predictions': test_predictions.flatten(),
            'test_actual': test_actual.flatten(),
            'future_predictions': np.array(future_predictions),
            'scaler': scaler
        }