import os
import logging
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import dvclive
from dvclive import Live

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "model_evaluation.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Corrected function name
def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    try:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.debug(f"Model evaluation metrics - MAE: {mae}, MSE: {mse}, R2: {r2}")
        return {
            'MAE': mae,
            'MSE': mse,
            'R2': r2
        }
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise

def save_metrics(metrics: dict, output_path: str):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug(f"Metrics saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error saving metrics to {output_path}: {e}")
        raise

import glob

def main():
    try:
        model_dir = 'models'
        data_path = 'datasets/test.csv'
        metrics_dir = 'metrics'
        os.makedirs(metrics_dir, exist_ok=True)

       
        df = load_data(data_path)
        X_test = df.drop(columns=['Employee_Satisfaction_Score'])
        y_test = df['Employee_Satisfaction_Score']

        
        model_files = glob.glob(os.path.join(model_dir, '*.pkl'))
        if not model_files:
            logger.warning(f"No model files found in {model_dir}")
            return

       
        all_metrics = {}

        for model_path in model_files:
            model_name = os.path.splitext(os.path.basename(model_path))[0]

            logger.info(f"Evaluating model: {model_name}")
            model = load_model(model_path)
            metrics = evaluate_model(model, X_test, y_test)
            all_metrics[model_name] = metrics

            
            metrics_output_path = os.path.join(metrics_dir, f"{model_name}_metrics.json")
            save_metrics(metrics, metrics_output_path)

        
        combined_metrics_path = os.path.join(metrics_dir, "all_models_metrics.json")
        save_metrics(all_metrics, combined_metrics_path)
        logger.info("All model metrics saved successfully.")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()