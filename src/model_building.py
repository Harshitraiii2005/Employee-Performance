import os
import pickle
import yaml
import logging
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    StackingClassifier, VotingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json


MODEL_MAPPING = {
    'LogisticRegression': LogisticRegression,
    'RidgeClassifier': RidgeClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'ExtraTreesClassifier': ExtraTreesClassifier,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'XGBClassifier': XGBClassifier,
    'LGBMClassifier': LGBMClassifier,
    'CatBoostClassifier': CatBoostClassifier,
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
        fh = logging.FileHandler(log_dir / log_file)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger

def load_data(file_path: Path, logger: logging.Logger) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Loaded data from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise

def save_model(model, file_path: Path, logger: logging.Logger):
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"Saved model to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save model to {file_path}: {e}")
        raise



def save_metrics(metrics: dict, file_path: Path, logger: logging.Logger):
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug(f"Saved metrics to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics to {file_path}: {e}")
        raise

def train_and_evaluate_models(X_train, y_train, X_test, y_test, models_config: dict, logger: logging.Logger):
    trained_models = {}
    model_scores = {}
    model_metrics = {}

    for model_name, params in models_config.items():
        model_class = MODEL_MAPPING.get(model_name)
        if not model_class:
            logger.warning(f"{model_name} is not supported. Skipping.")
            continue

        try:
            logger.debug(f"Training {model_name} with params: {params}")
            model = model_class(**params)
            model.fit(X_train, y_train)
            trained_models[model_name] = model

            # Evaluate on test set
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred).tolist()  # convert numpy array to list for JSON serialization

            model_scores[model_name] = acc
            model_metrics[model_name] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm
            }
            logger.info(f"{model_name} trained with test accuracy: {acc:.4f}")
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")

    # Train stacking and voting classifiers if at least 2 models trained
    if len(trained_models) >= 2:
        try:
            stacking_model = StackingClassifier(
                estimators=[(name, model) for name, model in trained_models.items()],
                final_estimator=LogisticRegression(max_iter=1000),
                n_jobs=-1
            )
            stacking_model.fit(X_train, y_train)
            trained_models['StackingClassifier'] = stacking_model

            y_pred_stack = stacking_model.predict(X_test)
            acc_stack = accuracy_score(y_test, y_pred_stack)
            prec_stack = precision_score(y_test, y_pred_stack, average='weighted', zero_division=0)
            rec_stack = recall_score(y_test, y_pred_stack, average='weighted', zero_division=0)
            f1_stack = f1_score(y_test, y_pred_stack, average='weighted', zero_division=0)
            cm_stack = confusion_matrix(y_test, y_pred_stack).tolist()

            model_scores['StackingClassifier'] = acc_stack
            model_metrics['StackingClassifier'] = {
                "accuracy": acc_stack,
                "precision": prec_stack,
                "recall": rec_stack,
                "f1_score": f1_stack,
                "confusion_matrix": cm_stack
            }
            logger.info(f"StackingClassifier trained with test accuracy: {acc_stack:.4f}")
        except Exception as e:
            logger.error(f"Failed to train StackingClassifier: {e}")

        try:
            voting_model = VotingClassifier(
                estimators=[(name, model) for name, model in trained_models.items()],
                voting='soft'
            )
            voting_model.fit(X_train, y_train)
            trained_models['VotingClassifier'] = voting_model

            y_pred_vote = voting_model.predict(X_test)
            acc_vote = accuracy_score(y_test, y_pred_vote)
            prec_vote = precision_score(y_test, y_pred_vote, average='weighted', zero_division=0)
            rec_vote = recall_score(y_test, y_pred_vote, average='weighted', zero_division=0)
            f1_vote = f1_score(y_test, y_pred_vote, average='weighted', zero_division=0)
            cm_vote = confusion_matrix(y_test, y_pred_vote).tolist()

            model_scores['VotingClassifier'] = acc_vote
            model_metrics['VotingClassifier'] = {
                "accuracy": acc_vote,
                "precision": prec_vote,
                "recall": rec_vote,
                "f1_score": f1_vote,
                "confusion_matrix": cm_vote
            }
            logger.info(f"VotingClassifier trained with test accuracy: {acc_vote:.4f}")
        except Exception as e:
            logger.error(f"Failed to train VotingClassifier: {e}")

    # Sort models by test accuracy descending and pick top 3
    top3 = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    logger.info(f"Top 3 models by test accuracy: {top3}")
    print("Top 3 models (name, test accuracy):", top3)

    # Save metrics to file
    metrics_path = Path("models") / "metrics.json"
    save_metrics(model_metrics, metrics_path, logger)

    return trained_models, top3


def main():
    params_path = "params.yaml"
    params = load_params(Path(params_path))

    log_dir = Path(params.get('paths', {}).get('log_dir', 'logs'))
    log_file = params.get('paths', {}).get('log_file', 'model_building.log')
    log_level = params.get('logging', {}).get('level', 'DEBUG')
    logger = setup_logger("model_building", log_dir, log_file, log_level)

    try:
        x_train_path = Path("datasets") / "X_train.csv"
        y_train_path = Path("datasets") / "y_train.csv"
        x_test_path = Path("datasets") / "X_test.csv"
        y_test_path = Path("datasets") / "y_test.csv"

        X_train = load_data(x_train_path, logger)
        y_train_df = load_data(y_train_path, logger)
        X_test = load_data(x_test_path, logger)
        y_test_df = load_data(y_test_path, logger)

        if y_train_df.shape[1] == 1 and y_test_df.shape[1] == 1:
            y_train = y_train_df.iloc[:, 0]
            y_test = y_test_df.iloc[:, 0]
        else:
            logger.error("y_train and y_test should have only one column each.")
            return

        trained_models, top3 = train_and_evaluate_models(
            X_train, y_train, X_test, y_test, params.get("models", {}), logger
        )

        # Save top 3 models
        models_output_dir = Path("models") / "temp"
        for name, _ in top3:
            model = trained_models.get(name)
            if model:
                model_file = f"{name.lower()}.pkl"
                save_model(model, models_output_dir / model_file, logger)

    except Exception as e:
        logger.error(f"Exception in main: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
