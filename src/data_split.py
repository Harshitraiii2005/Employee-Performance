import os
import logging
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_split")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "data_split.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_params(path: str = "params.yaml"):
    if not os.path.exists(path):
        logger.error(f"Params file not found: {path}")
        raise FileNotFoundError(f"Params file not found: {path}")
    with open(path, 'r') as f:
        params = yaml.safe_load(f)
    if 'target_column' not in params:
        logger.error("'target_column' not found in params.yaml")
        raise KeyError("'target_column' not found in params.yaml")
    return params


def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in dataframe")
        raise KeyError(f"Target column '{target_col}' not found in dataframe")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.debug(f"Data split with test_size={test_size}")
    logger.debug(f"Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_split_data(X_train, X_test, y_train, y_test, output_dir='datasets'):
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    logger.info(f"Split data saved in directory: {output_dir}")


def main():
    params = load_params()
    data_path = params['paths']['processed_data']
    target_col = params['target_column']
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}")
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    logger.info(f"Data loaded from {data_path} with shape {df.shape}")

    X_train, X_test, y_train, y_test = split_data(df, target_col)
    logger.info("Data split into training and testing sets")

    save_split_data(X_train, X_test, y_train, y_test, output_dir=params['paths'].get('split_data_dir', 'datasets'))


if __name__ == "__main__":
    main()
