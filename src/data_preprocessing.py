import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    log_file_path = os.path.join(log_dir, "data_preprocessing.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def split_data(df: pd.DataFrame):
    try:
        X = df.drop(columns=['Employee_Satisfaction_Score'])
        y = df['Employee_Satisfaction_Score']
        logger.debug("Dataset split into target and features.")
        return X, y
    except Exception as e:
        logger.error(f"Error in split_data function: {e}")
        raise


def preprocess_data(X: pd.DataFrame, y: pd.Series):
    try:

        numerical_features = ['Age', 'Years_At_Company', 'Performance_Score',
                              'Monthly_Salary', 'Work_Hours_Per_Week', 'Projects_Handled',
                              'Overtime_Hours', 'Sick_Days', 'Remote_Work_Frequency',
                              'Team_Size', 'Training_Hours', 'Promotions']

        categorical_features = ['Department', 'Job_Title', 'Education_Level']

        logger.debug(f"Numerical features: {numerical_features}")
        logger.debug(f"Categorical features: {categorical_features}")

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )

        X_processed = preprocessor.fit_transform(X)

        
        ohe = preprocessor.named_transformers_['cat']
        ohe_columns = ohe.get_feature_names_out(categorical_features)
        all_columns = numerical_features + ohe_columns.tolist()

        X_processed_df = pd.DataFrame(
            X_processed,
            columns=all_columns,
            index=X.index
        )

        logger.info("Data preprocessing completed successfully.")
        logger.debug(f"Processed data shape: {X_processed_df.shape}")

        return X_processed_df, y

    except Exception as e:
        logger.error(f"Error in preprocess_data function: {e}")
        raise


def main():
    try:
        df = pd.read_csv('datasets/Extended_Employee_Performance_and_Productivity_Data.csv')
        logger.info("Data loaded successfully.")

        X, y = split_data(df)
        X_processed, y = preprocess_data(X, y)

        data_path = "data"
        os.makedirs(data_path, exist_ok=True)

        X_processed['Employee_Satisfaction_Score'] = y
        X_processed.to_csv(os.path.join(data_path, "processed_data.csv"), index=False)
        logger.info("Preprocessed data saved successfully.")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
