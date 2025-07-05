# Modular implementation for scalable cryptocurrency prediction
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, r2_score



# =============================================================================
# 8. EVALUATOR MODULE
# =============================================================================


class SingleCoinEvaluator:
    """Evaluates and visualizes model performance for a single coin"""

    def __init__(self):
        self.metrics = {}

    def calculate_metrics(self, prediction: Dict) -> Dict:
        """Calculate performance metrics"""
        test_actual = prediction['test_actual']
        test_pred = prediction['test_predictions']

        if len(test_actual) > 0 and len(test_pred) > 0:
            rmse = math.sqrt(mean_squared_error(test_actual, test_pred))
            r2 = r2_score(test_actual, test_pred)

            # Daily returns
            daily_returns_actual = np.diff(test_actual) / test_actual[:-1] * 100
            daily_returns_pred = np.diff(test_pred) / test_pred[:-1] * 100

            self.metrics = {
                'rmse': rmse,
                'r2': r2,
                'mean_actual_return': np.mean(daily_returns_actual),
                'mean_predicted_return': np.mean(daily_returns_pred),
                'std_actual_return': np.std(daily_returns_actual),
                'std_predicted_return': np.std(daily_returns_pred)
            }

        return self.metrics

    def print_metrics_summary(self):
        """Print metrics summary"""
        if not self.metrics:
            print("No metrics calculated yet")
            return

        print("\n" + "=" * 50)
        print("SINGLE COIN PERFORMANCE SUMMARY")
        print("=" * 50)

        print(f"  RMSE: {self.metrics['rmse']:.2f}")
        print(f"  R²: {self.metrics['r2']:.3f}")
        print(f"  Mean Actual Return: {self.metrics['mean_actual_return']:.2f}%")
        print(f"  Mean Predicted Return: {self.metrics['mean_predicted_return']:.2f}%")
        print(f"  Std Actual Return: {self.metrics['std_actual_return']:.2f}")
        print(f"  Std Predicted Return: {self.metrics['std_predicted_return']:.2f}")

    def plot_predictions(self, prediction: Dict, figsize: Tuple[int, int] = (12, 6)):
        """Plot predictions and forecast"""
        if len(prediction['test_actual']) == 0:
            print("No test predictions to plot.")
            return

        plt.figure(figsize=figsize)

        # Test predictions
        plt.plot(prediction['test_actual'], label='Actual', marker='o', markersize=2)
        plt.plot(prediction['test_predictions'], label='Predicted', marker='s', markersize=2)

        # Future predictions
        if len(prediction['future_predictions']) > 0:
            future_start = len(prediction['test_actual'])
            future_x = range(future_start, future_start + len(prediction['future_predictions']))
            plt.plot(future_x, prediction['future_predictions'],
                     label='Future Forecast', marker='^', linestyle='--', markersize=3)
        
        assets_path = Path("assets")
        assets_path.mkdir(exist_ok=True)
        plt.title("Single Coin Price Prediction")
        plt.xlabel("Time")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = assets_path / "price_prediction.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"prediction plot saved to {plot_path}")


            