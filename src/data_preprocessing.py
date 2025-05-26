import os
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import yaml
import pickle
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import argparse
import time


def setup_logging(log_dir: str, log_file: str, log_level: int = logging.INFO) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("data_preprocessing")
    logger.setLevel(log_level)

    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        file_handler = RotatingFileHandler(os.path.join(log_dir, log_file), maxBytes=5 * 1024 * 1024, backupCount=5)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


def load_params(path: str = "params.yaml") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Params file not found: {path}")

    with open(path, 'r') as file:
        params = yaml.safe_load(file)

    required_keys = ['paths', 'numerical_features', 'categorical_features', 'preprocessing', 'target_column']
    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        raise KeyError(f"Missing required keys in params.yaml: {missing_keys}")

    return params


def split_features_target(df: pd.DataFrame, target_col: str, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in data columns: {df.columns.tolist()}")
        raise KeyError(f"Target column '{target_col}' not found in dataframe")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.debug(f"Split data into X (shape: {X.shape}) and y (shape: {y.shape}).")
    return X, y


def preprocess_features(
    X: pd.DataFrame, 
    numerical_features: list, 
    categorical_features: list, 
    ohe_params: dict,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, ColumnTransformer]:

    logger.debug(f"Preprocessing features: {len(numerical_features)} numerical, {len(categorical_features)} categorical.")

    X_num = X[numerical_features].apply(pd.to_numeric, errors='coerce')
    X_cat = X[categorical_features].astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(**ohe_params, sparse=False, handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )

    X_processed = preprocessor.fit_transform(pd.concat([X_num, X_cat], axis=1))

    ohe = preprocessor.named_transformers_['cat']
    ohe_columns = ohe.get_feature_names_out(categorical_features)
    all_columns = numerical_features + list(ohe_columns)

    X_processed_df = pd.DataFrame(X_processed, columns=all_columns, index=X.index)

    logger.info("Feature preprocessing completed.")
    logger.debug(f"Processed features shape: {X_processed_df.shape}")

    return X_processed_df, preprocessor


def save_preprocessor(preprocessor: ColumnTransformer, save_path: str, logger: logging.Logger):
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        logger.info(f"Preprocessor saved at {save_path}")
    except Exception as e:
        logger.error(f"Failed to save preprocessor: {e}")
        raise


def main(params_path: str):
    start_time = time.time()

    params = load_params(params_path)

    log_dir = params['paths']['log_dir']
    log_file = params['paths'].get('log_file', 'preprocessing.log')
    logger = setup_logging(log_dir, log_file, logging.DEBUG)

    try:
        raw_data_path = params['paths']['raw_data']
        processed_data_path = params['paths']['processed_data']
        preprocessor_save_path = params['paths'].get('preprocessor_save_path', 'preprocessor.pkl')

        target_col = params.get('target_column', 'Employee_Satisfaction_Score')
        numerical_features = params['numerical_features']
        categorical_features = params['categorical_features']
        ohe_params = params['preprocessing'].get('onehotencoder', {})

        df = pd.read_csv(raw_data_path)
        logger.info(f"Raw data loaded from {raw_data_path} with shape {df.shape}.")

        X, y = split_features_target(df, target_col, logger)
        X_processed, preprocessor = preprocess_features(X, numerical_features, categorical_features, ohe_params, logger)

        processed_df = X_processed.copy()
        processed_df[target_col] = y

        processed_df.to_csv(processed_data_path, index=False)
        logger.info(f"Preprocessed data saved at {processed_data_path}.")

        save_preprocessor(preprocessor, preprocessor_save_path, logger)

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

    elapsed_time = time.time() - start_time
    logger.info(f"Data preprocessing completed in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    parser.add_argument(
        '--params', '-p', type=str, default='params.yaml',
        help="Path to the params.yaml file"
    )
    args = parser.parse_args()
    main(args.params)
