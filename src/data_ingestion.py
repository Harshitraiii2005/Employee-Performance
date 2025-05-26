import pandas as pd
import os
import logging

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.info(f"Data loaded successfully from {data_url}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {data_url}: {e}")
        raise


def preprocess_data(df: pd.DataFrame, null_threshold: float = 0.3) -> pd.DataFrame:
    logger.info("Starting preprocessing")

    
    null_percent = df.isnull().mean()
    cols_to_drop = null_percent[null_percent > null_threshold].index.tolist()
    if cols_to_drop:
        logger.info(f"Dropping columns with more than {null_threshold*100}% null values: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)
    else:
        logger.info("No columns dropped due to null values threshold")

    
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                fill_val = df[col].median()
                df[col].fillna(fill_val, inplace=True)
                logger.info(f"Filled nulls in numeric column '{col}' with median value {fill_val}")
            else:
                fill_val = df[col].mode()[0]
                df[col].fillna(fill_val, inplace=True)
                logger.info(f"Filled nulls in categorical column '{col}' with mode value '{fill_val}'")

    
    before_cols = df.shape[1]
    df = df.loc[:, ~df.T.duplicated()]
    after_cols = df.shape[1]
    dropped = before_cols - after_cols
    if dropped > 0:
        logger.info(f"Dropped {dropped} duplicate columns")
    else:
        logger.info("No duplicate columns found to drop")

    logger.info("Preprocessing completed")
    return df


def save_processed_data(df: pd.DataFrame, save_path: str):
    try:
        df.to_csv(save_path, index=False)
        logger.info(f"Processed data saved successfully at {save_path}")
    except Exception as e:
        logger.error(f"Error saving processed data at {save_path}: {e}")
        raise


def main():
    try:
        data_path = 'datasets/Extended_Employee_Performance_and_Productivity_Data.csv'
        save_path = 'datasets/Extended_Employee_Performance_and_Productivity_Data_Raw.csv'

        df = load_data(data_path)
        logger.info(f"Initial data shape: {df.shape}")

        df_processed = preprocess_data(df)
        logger.info(f"Processed data shape: {df_processed.shape}")
        logger.info(f"Processed data columns: {df_processed.columns.tolist()}")
        logger.info(f"Processed data head:\n{df_processed.head()}")

        save_processed_data(df_processed, save_path)

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
