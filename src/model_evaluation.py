import os
import logging
import pandas as pd
import pickle
import json
import glob
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Logger setup
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

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

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

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        logger.debug(
            f"Model evaluation metrics - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}"
        )
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
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

def save_top_models(all_metrics, model_dir='models/temp', top_dir='models/traditional', top_k=3):
    # Sort by highest accuracy (descending)
    sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]['Accuracy'], reverse=True)
    top_models = sorted_models[:top_k]

    os.makedirs(top_dir, exist_ok=True)

    for model_name, metrics in top_models:
        src_path = os.path.join(model_dir, f"{model_name}.pkl")
        dest_path = os.path.join(top_dir, f"{model_name}.pkl")
        try:
            shutil.copy2(src_path, dest_path)
            logger.info(f"Saved top model {model_name} to {dest_path}")
        except Exception as e:
            logger.error(f"Failed to save top model {model_name}: {e}")

def main():
    try:
        model_dir = 'models/temp'  # Directory where your models are saved
        X_test_path = 'datasets/X_test.csv'  # Test features CSV (classification)
        y_test_path = 'datasets/y_test.csv'  # Test target CSV (classification)
        metrics_dir = 'metrics/eval'

        os.makedirs(metrics_dir, exist_ok=True)

        X_test = load_data(X_test_path)
        y_test_df = load_data(y_test_path)
        y_test = y_test_df.iloc[:, 0]  # Assuming target is first column

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

            print(
                f"Metrics for {model_name}: Accuracy={metrics['Accuracy']:.4f}, "
                f"Precision={metrics['Precision']:.4f}, Recall={metrics['Recall']:.4f}, F1={metrics['F1']:.4f}"
            )

            metrics_output_path = os.path.join(metrics_dir, f"{model_name}_metrics.json")
            save_metrics(metrics, metrics_output_path)

        combined_metrics_path = os.path.join(metrics_dir, "all_models_metrics.json")
        save_metrics(all_metrics, combined_metrics_path)
        logger.info("All model metrics saved successfully.")

        save_top_models(all_metrics, model_dir=model_dir, top_dir='models/traditional', top_k=3)

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
