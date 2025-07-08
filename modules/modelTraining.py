# Multi-Coin LSTM Forecasting Framework
# Modular implementation for scalable cryptocurrency prediction
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' = Filter out INFO and WARNING, show only ERROR
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

# =============================================================================
# 6. TRAINER MODULE
# =============================================================================

class SingleCoinTrainer:
    """Trainer for single-coin LSTM model"""

    def __init__(self, model, model_save_path: str = "models/",
                 scaler_train=None, scaler_test=None):
        self.model = model
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.history = None

        # Optional: keep references to scalers (can be saved with model or reused later)
        self.scaler_train = scaler_train
        self.scaler_test = scaler_test

    def train(self, 
              X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32, 
              patience: int = 10, verbose: int = 1):
        """Train the LSTM model with callbacks"""

        checkpoint_path = self.model_save_path / "best_model.keras"

        checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )

        callbacks = [checkpoint, early_stopping]

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=callbacks,
            verbose=verbose
        )

        return self.history

    def plot_training_history(self):
        """Plot and save training vs validation loss"""

        if self.history:
            assets_path = Path("assets")
            assets_path.mkdir(exist_ok=True)

            plt.figure(figsize=(10, 4))
            plt.plot(self.history.history['loss'], label='Train Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title("Training History")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plot_path = assets_path / "training_history.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"Training history plot saved to {plot_path}")
        else:
            print("No training history available.")

    def load_best_model(self):
        """Load the saved best model"""
        checkpoint_path = self.model_save_path / "best_model.keras"
        if checkpoint_path.exists():
            self.model = load_model(str(checkpoint_path))
            print("Best model loaded successfully.")
        else:
            print("No saved model found at:", checkpoint_path)
