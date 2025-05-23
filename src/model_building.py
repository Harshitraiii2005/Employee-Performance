import os
import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, "model_building.log")
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


def train_models(X_train: pd.DataFrame, y_train: pd.Series):
    try:
       
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'ElasticNet Regression': ElasticNet(),
            'Random Forest': RandomForestRegressor(n_jobs=-1, n_estimators=100),
            'Extra Trees': ExtraTreesRegressor(n_jobs=-1, n_estimators=100),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100),
            'XGBoost': XGBRegressor(n_jobs=-1, eval_metric='rmse', tree_method='hist'),
            'LightGBM': LGBMRegressor(n_jobs=-1),
            'CatBoost': CatBoostRegressor(verbose=0)
        }

        trained_models = {}
        for name, model in models.items():
            logger.debug(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            logger.debug(f"{name} model trained successfully.")

       
        stacking_model = StackingRegressor(
            estimators=[(name, model) for name, model in trained_models.items()],
            final_estimator=LinearRegression()
        )
        stacking_model.fit(X_train, y_train)
        trained_models['Stacking Regressor'] = stacking_model
        logger.debug("Stacking Regressor trained successfully.")

        
        voting_model = VotingRegressor(
            estimators=[(name, model) for name, model in trained_models.items()]
        )
        voting_model.fit(X_train, y_train)
        trained_models['Voting Regressor'] = voting_model
        logger.debug("Voting Regressor trained successfully.")

        return trained_models

    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise


def save_model(model, file_path: str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"Model saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {e}")
        raise


def main():
    try:
        data_path = 'datasets/train.csv'
        df = load_data(data_path)

        X = df.drop(columns=['Employee_Satisfaction_Score'])
        y = df['Employee_Satisfaction_Score']
        logger.info("Data split into features and target variable.")

        trained_models = train_models(X, y)
        logger.info("All models trained successfully.")

        for name, model in trained_models.items():
            model_path = os.path.join("models", f"{name.replace(' ', '_').lower()}.pkl")
            save_model(model, model_path)
            logger.info(f"{name} model saved at {model_path}.")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
