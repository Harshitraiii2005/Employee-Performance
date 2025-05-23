import os
import logging
import pickle
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Logger setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, "model_building.log"))
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Model mapping
MODEL_MAPPING = {
    'LinearRegression': LinearRegression,
    'Ridge': Ridge,
    'Lasso': Lasso,
    'ElasticNet': ElasticNet,
    'RandomForestRegressor': RandomForestRegressor,
    'ExtraTreesRegressor': ExtraTreesRegressor,
    'GradientBoostingRegressor': GradientBoostingRegressor,
    'XGBRegressor': XGBRegressor,
    'LGBMRegressor': LGBMRegressor,
    'CatBoostRegressor': CatBoostRegressor,
}

def load_params(path="params.yaml"):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
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

def train_models(X_train, y_train, models_config):
    trained_models = {}
    for model_name, params in models_config.items():
        model_class = MODEL_MAPPING.get(model_name)
        if not model_class:
            logger.warning(f"Model {model_name} is not supported.")
            continue

        logger.debug(f"Training {model_name} with params: {params}")
        model = model_class(**params)
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        logger.debug(f"{model_name} trained successfully.")

    # Stacking Regressor
    stacking_model = StackingRegressor(
        estimators=[(name, model) for name, model in trained_models.items()],
        final_estimator=LinearRegression()
    )
    stacking_model.fit(X_train, y_train)
    trained_models['StackingRegressor'] = stacking_model
    logger.debug("StackingRegressor trained successfully.")

    # Voting Regressor
    voting_model = VotingRegressor(
        estimators=[(name, model) for name, model in trained_models.items()]
    )
    voting_model.fit(X_train, y_train)
    trained_models['VotingRegressor'] = voting_model
    logger.debug("VotingRegressor trained successfully.")

    return trained_models

def main():
    try:
        params = load_params()
        data_path = 'datasets/train.csv'
        df = load_data(data_path)

        X = df.drop(columns=['Employee_Satisfaction_Score'])
        y = df['Employee_Satisfaction_Score']
        logger.info("Data split into features and target variable.")

        trained_models = train_models(X, y, params['models'])
        logger.info("All models trained successfully.")

        # Save all models to models/traditional/
        output_dir = os.path.join("models", "traditional")
        os.makedirs(output_dir, exist_ok=True)

        for name, model in trained_models.items():
            model_file = f"{name.replace(' ', '_').lower()}.pkl"
            model_path = os.path.join(output_dir, model_file)
            save_model(model, model_path)
            logger.info(f"{name} model saved at {model_path}.")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
