import os
from pathlib import Path
import logging
import pickle
import yaml
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

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

def load_params(path: Path) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def setup_logger(name: str, log_dir: Path, log_file: str, level_str: str) -> logging.Logger:
    level = getattr(logging, level_str.upper(), logging.DEBUG)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fh = logging.FileHandler(log_dir / log_file)
        fh.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger

def create_preprocessor(params: dict, logger: logging.Logger) -> ColumnTransformer:
    num_features = params.get("numerical_features", [])
    cat_features = params.get("categorical_features", [])

    ohe_params = params.get("preprocessing", {}).get("onehotencoder", {}).copy()
    # Adapt to scikit-learn version differences
    if 'sparse_output' in ohe_params:
        ohe_params['sparse'] = ohe_params.pop('sparse_output')

    logger.debug(f"Numerical features: {num_features}")
    logger.debug(f"Categorical features: {cat_features}")
    logger.debug(f"OneHotEncoder params: {ohe_params}")

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(**ohe_params)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='drop',
        n_jobs=-1
    )
    return preprocessor

def load_data(file_path: Path, logger: logging.Logger) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def save_model(model, file_path: Path, logger: logging.Logger):
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"Model saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {e}")
        raise

def train_models(X_train, y_train, models_config: dict, logger: logging.Logger):
    trained_models = {}
    for model_name, params in models_config.items():
        model_class = MODEL_MAPPING.get(model_name)
        if not model_class:
            logger.warning(f"Model {model_name} is not supported. Skipping.")
            continue

        logger.debug(f"Training {model_name} with params: {params}")
        try:
            model = model_class(**params)
            model.fit(X_train, y_train)
            trained_models[model_name] = model
            logger.debug(f"{model_name} trained successfully.")
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")

    if trained_models:
        # Stacking Regressor
        try:
            stacking_model = StackingRegressor(
                estimators=[(name, model) for name, model in trained_models.items()],
                final_estimator=LinearRegression(),
                n_jobs=-1
            )
            stacking_model.fit(X_train, y_train)
            trained_models['StackingRegressor'] = stacking_model
            logger.debug("StackingRegressor trained successfully.")
        except Exception as e:
            logger.error(f"Failed to train StackingRegressor: {e}")

        # Voting Regressor
        try:
            voting_model = VotingRegressor(
                estimators=[(name, model) for name, model in trained_models.items()]
            )
            voting_model.fit(X_train, y_train)
            trained_models['VotingRegressor'] = voting_model
            logger.debug("VotingRegressor trained successfully.")
        except Exception as e:
            logger.error(f"Failed to train VotingRegressor: {e}")

    return trained_models

def main():
    base_path = Path(__file__).parent
    params_path = base_path / "params.yaml"

    params = load_params(params_path)

    log_dir = Path(params.get('paths', {}).get('log_dir', 'logs'))
    log_file = params.get('paths', {}).get('log_file', 'data_preprocessing.log')
    log_level = params.get('logging', {}).get('level', 'DEBUG')

    logger = setup_logger("model_building", log_dir, log_file, log_level)

    try:
        raw_data_path = base_path / params['paths']['raw_data']
        processed_data_path = base_path / params['paths']['processed_data']

        df = load_data(raw_data_path, logger)

        target_col = 'Target'
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in data.")
            return

        preprocessor = create_preprocessor(params, logger)

        X = df[params['numerical_features'] + params['categorical_features']]
        y = df[target_col]

        logger.info("Starting preprocessing pipeline fit_transform...")
        X_processed = preprocessor.fit_transform(X)
        logger.info("Preprocessing complete.")

        # Convert to DataFrame for saving
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
        processed_df = pd.DataFrame(X_processed)
        processed_df[target_col] = y.values

        processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(processed_data_path, index=False)
        logger.info(f"Preprocessed data saved to {processed_data_path}")

        # Train models
        trained_models = train_models(X_processed, y, params.get('models', {}), logger)
        logger.info(f"Trained models: {list(trained_models.keys())}")

        # Save trained models
        models_output_dir = base_path / "models" / "temp"
        models_output_dir.mkdir(parents=True, exist_ok=True)
        for name, model in trained_models.items():
            model_file = f"{name.replace(' ', '_').lower()}.pkl"
            model_path = models_output_dir / model_file
            save_model(model, model_path, logger)
            logger.info(f"Saved {name} model to {model_path}")

    except Exception as e:
        logger.error(f"Exception in main: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
