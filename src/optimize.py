import os
import joblib
import json
import logging
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

logging.getLogger("mlflow").setLevel(logging.WARNING)

def load_processed_data(train_path: str, test_path: str, target_col: str):
    print(f"Loading data from {train_path} and {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    return X_train, X_test, y_train, y_test

def build_model(model_type: str, params: dict, random_state: int):
    if model_type == "RandomForest":
        return RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    if model_type == "XGBoost":
        return XGBRegressor(random_state=random_state, n_jobs=-1, **params)
    raise ValueError(f"Unknown model_type='{model_type}'.")

def evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

def suggest_params(trial: optuna.Trial, cfg: DictConfig):
    model_type = cfg.model.type
    if model_type == "RandomForest":
        space = cfg.hpo.random_forest
        return {
            "n_estimators": trial.suggest_int("n_estimators", space.n_estimators.low, space.n_estimators.high, step=space.n_estimators.step),
            "max_depth": trial.suggest_int("max_depth", space.max_depth.low, space.max_depth.high),
            "min_samples_split": trial.suggest_int("min_samples_split", space.min_samples_split.low, space.min_samples_split.high),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", space.min_samples_leaf.low, space.min_samples_leaf.high),
        }
    if model_type == "XGBoost":
        space = cfg.hpo.xgboost
        return {
            "n_estimators": trial.suggest_int("n_estimators", space.n_estimators.low, space.n_estimators.high, step=space.n_estimators.step),
            "max_depth": trial.suggest_int("max_depth", space.max_depth.low, space.max_depth.high),
            "learning_rate": trial.suggest_float("learning_rate", space.learning_rate.low, space.learning_rate.high, log=True),
        }
    raise ValueError(f"Unknown model_type='{model_type}'.")

def objective_factory(cfg: DictConfig, X_train, X_test, y_train, y_test):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, cfg)
        
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("model_type", cfg.model.type)
            
            mlflow.log_params(params)
            
            model = build_model(cfg.model.type, params, cfg.seed)
            rmse, r2 = evaluate(model, X_train, y_train, X_test, y_test)
            
            mlflow.log_metrics({"rmse": rmse, "r2": r2})
            
            return rmse if cfg.hpo.direction == "minimize" else r2
    return objective

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    X_train, X_test, y_train, y_test = load_processed_data(cfg.data.train_path, cfg.data.test_path, cfg.data.target_col)
    
    with mlflow.start_run(run_name=f"HPO_{cfg.model.type}_{cfg.hpo.sampler}") as parent_run:
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config_resolved.json")
        
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)
        study = optuna.create_study(direction=cfg.hpo.direction, sampler=sampler)
        
        objective = objective_factory(cfg, X_train, X_test, y_train, y_test)
        
        print(f"Starting optimization for {cfg.model.type} ({cfg.hpo.n_trials} trials)...")
        study.optimize(objective, n_trials=cfg.hpo.n_trials)
        
        best_trial = study.best_trial
        print(f"\nOptimization completed! Best trial: {best_trial.number}")
        print(f"Best {cfg.hpo.metric}: {best_trial.value:.2f}")
        
        mlflow.log_metric(f"best_{cfg.hpo.metric}", best_trial.value)
        mlflow.log_dict(best_trial.params, "best_params.json")
        
        with open("best_params.json", "w", encoding="utf-8") as f:
            json.dump(best_trial.params, f, indent=4)
        
        best_model = build_model(cfg.model.type, best_trial.params, cfg.seed)
        evaluate(best_model, X_train, y_train, X_test, y_test)
        
        os.makedirs("models", exist_ok=True)
        model_path = f"models/best_{cfg.model.type.lower()}.pkl"
        joblib.dump(best_model, model_path)
        
        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(best_model, artifact_path="model")
            print(f"Final model saved and logged to MLflow.")

if __name__ == "__main__":
    main()