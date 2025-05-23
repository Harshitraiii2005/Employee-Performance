import pandas as pd
import os
import logging
import yaml

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

def main():
    try:
        data_path = 'datasets/Extended_Employee_Performance_and_Productivity_Data.csv'
        df = load_data(data_path)
        logger.info("Data Loaded Successfully")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Data columns: {df.columns.tolist()}")
        logger.info(f"Data head:\n{df.head()}")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
