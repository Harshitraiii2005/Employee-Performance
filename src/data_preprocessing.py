import os
import logging
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# -------------------------------
# Logging Setup
# -------------------------------

def setup_logging(log_dir, log_file):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("data_preprocessing")
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


# -------------------------------
# Load Params
# -------------------------------

def load_params(path="params.yaml"):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


# -------------------------------
# Split Features and Target
# -------------------------------

def split_features_target(df: pd.DataFrame, logger):
    try:
        X = df.drop(columns=['Employee_Satisfaction_Score'])
        y = df['Employee_Satisfaction_Score']
        logger.debug("Split data into features and target.")
        return X, y
    except Exception as e:
        logger.error(f"Error in split_features_target: {e}")
        raise


# -------------------------------
# Preprocess Features
# -------------------------------

def preprocess_features(X: pd.DataFrame, params, logger):
    try:
        numerical_features = params['numerical_features']
        categorical_features = params['categorical_features']
        ohe_params = params['preprocessing']['onehotencoder']

        logger.debug(f"Numerical features: {numerical_features}")
        logger.debug(f"Categorical features: {categorical_features}")
        logger.debug(f"OHE Params: {ohe_params}")

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(**ohe_params), categorical_features)
            ],
            remainder='drop'
        )

        X_processed = preprocessor.fit_transform(X)

        ohe = preprocessor.named_transformers_['cat']
        ohe_columns = ohe.get_feature_names_out(categorical_features)
        all_columns = numerical_features + list(ohe_columns)

        X_processed_df = pd.DataFrame(X_processed, columns=all_columns, index=X.index)

        logger.info("Features preprocessing completed.")
        logger.debug(f"Processed features shape: {X_processed_df.shape}")

        return X_processed_df

    except Exception as e:
        logger.error(f"Error in preprocess_features: {e}")
        raise


# -------------------------------
# Main Execution
# -------------------------------

def main():
    params = load_params()
    log_dir = params['paths']['log_dir']
    log_file = params['paths']['log_file']
    logger = setup_logging(log_dir, log_file)

    try:
        raw_data_path = params['paths']['raw_data']
        processed_data_path = params['paths']['processed_data']

        df = pd.read_csv(raw_data_path)
        logger.info(f"Raw data loaded from {raw_data_path} with shape {df.shape}.")

        X, y = split_features_target(df, logger)
        X_processed = preprocess_features(X, params, logger)

        X_processed['Employee_Satisfaction_Score'] = y
        X_processed.to_csv(processed_data_path, index=False)
        logger.info(f"Preprocessed data saved at {processed_data_path}.")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
