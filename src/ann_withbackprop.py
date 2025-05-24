import os
import json
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import random


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("ann_with_backprop")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "ann_with_backprop.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def safe_metric(history, key):
    
    return float(history.history[key][-1]) if key in history.history else None

def create_and_train_model(X_train, y_train, model_path: str, metrics_path: str):
    try:
        logger.info("Creating ANN model...")
        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        logger.info("Starting training...")
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        logger.info("Training completed successfully.")

        
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        model.save(model_path)
        logger.info(f"Model saved at {model_path}")

       
        metrics_dir = os.path.dirname(metrics_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)

        
        final_metrics = {
            'final_loss': safe_metric(history, 'loss'),
            'final_val_loss': safe_metric(history, 'val_loss'),
            'final_mae': safe_metric(history, 'mean_absolute_error') or safe_metric(history, 'mae'),
            'final_val_mae': safe_metric(history, 'val_mean_absolute_error') or safe_metric(history, 'val_mae')
        }

        with open(metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)
        logger.info(f"Metrics saved at {metrics_path}")

    except Exception as e:
        logger.error(f"Error in create_and_train_model function: {e}")
        raise

def main():
    try:
        data_path = 'datasets/train.csv'
        model_path = 'models/ann_model.h5'
        metrics_path = 'metrics/ann_metrics.json'

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")

        logger.info("Loading data...")
        df = load_data(data_path)

        if 'Employee_Satisfaction_Score' not in df.columns:
            raise ValueError("Target column 'Employee_Satisfaction_Score' not found in data.")

        X = df.drop(columns=['Employee_Satisfaction_Score'])
        y = df['Employee_Satisfaction_Score']

       
        X = X.values.astype(np.float32)
        y = y.values.astype(np.float32)

        logger.info(f"Data shape: {df.shape}, Features: {X.shape}, Target: {y.shape}")

        create_and_train_model(X, y, model_path, metrics_path)

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
