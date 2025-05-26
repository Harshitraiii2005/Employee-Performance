import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Set up logger
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


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    try:
        X = df.drop(columns=['Employee_Satisfaction_Score'])
        y = df['Employee_Satisfaction_Score']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logger.debug("Dataset split into training and testing sets.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error in split_data function: {e}")
        raise


def save_split_data(X_train, X_test, y_train, y_test, output_dir='datasets'):
    try:
        os.makedirs(output_dir, exist_ok=True)

        train_features_path = os.path.join(output_dir, "X_train.csv")
        train_target_path = os.path.join(output_dir, "y_train.csv")
        test_features_path = os.path.join(output_dir, "X_test.csv")
        test_target_path = os.path.join(output_dir, "y_test.csv")

        X_train.to_csv(train_features_path, index=False)
        y_train.to_csv(train_target_path, index=False)
        X_test.to_csv(test_features_path, index=False)
        y_test.to_csv(test_target_path, index=False)

        logger.info(f"Training features saved to {train_features_path}")
        logger.info(f"Training target saved to {train_target_path}")
        logger.info(f"Testing features saved to {test_features_path}")
        logger.info(f"Testing target saved to {test_target_path}")

    except Exception as e:
        logger.error(f"Error saving split data: {e}")
        raise


def main():
    try:
        data_path = 'data/processed_data.csv'
        if not os.path.exists(data_path):
            logger.error(f"Data file not found at {data_path}")
            raise FileNotFoundError(f"Data file not found at {data_path}")

        df = pd.read_csv(data_path)
        logger.info("Data Loaded Successfully")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Data columns: {df.columns.tolist()}")
        logger.info(f"Data head:\n{df.head()}")

        X_train, X_test, y_train, y_test = split_data(df)
        logger.info("Data split into training and testing sets.")

        save_split_data(X_train, X_test, y_train, y_test)
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
