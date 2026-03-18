import os
import argparse
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_processed_data(train_path: str, test_path: str, target_col: str = 'Rent'):
    """Loads prepared train and test datasets."""
    print(f"Loading data from {train_path} and {test_path}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    
    return X_train, X_test, y_train, y_test

def build_model(model_type: str, params: dict, random_state: int):
    """Initializes the model with suggested hyperparameters."""
    if model_type == "RandomForest":
        return RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)
    if model_type == "XGBoost":
        return XGBRegressor(random_state=random_state, n_jobs=-1, **params)
    
    raise ValueError(f"Unknown model_type='{model_type}'. Expected 'RandomForest' or 'XGBoost'.")

def evaluate(model, X_train, y_train, X_test, y_test):
    """Trains the model and returns the RMSE score."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

def suggest_params(trial: optuna.Trial, model_type: str):
    """Defines the hyperparameter search space for Optuna."""
    if model_type == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
    if model_type == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
    raise ValueError(f"Unknown model_type='{model_type}'.")

def objective_factory(args, X_train, X_test, y_train, y_test):
    """Creates the objective function for Optuna study."""
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, args.model_type)
        
        # Create nested run for each trial
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("model_type", args.model_type)
            mlflow.set_tag("optimization", "optuna")
            
            mlflow.log_params(params)
            
            model = build_model(args.model_type, params, args.random_state)
            rmse, r2 = evaluate(model, X_train, y_train, X_test, y_test)
            
            mlflow.log_metrics({"rmse": rmse, "r2": r2})
            
            # Return RMSE to minimize it
            return rmse
    return objective

def main(args):
    mlflow.set_experiment("House_Rent_Prediction_Lab1")
    
    X_train, X_test, y_train, y_test = load_processed_data(args.train_path, args.test_path)
    
    # Start parent MLflow run
    with mlflow.start_run(run_name=f"HPO_{args.model_type}") as parent_run:
        mlflow.set_tag("model_type", args.model_type)
        mlflow.log_param("n_trials", args.n_trials)
        
        # Initialize Optuna study
        sampler = optuna.samplers.TPESampler(seed=args.random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        
        objective = objective_factory(args, X_train, X_test, y_train, y_test)
        
        print(f"Starting optimization for {args.model_type} ({args.n_trials} trials)...")
        study.optimize(objective, n_trials=args.n_trials)
        
        # Log best results
        best_trial = study.best_trial
        print(f"\nOptimization completed! Best trial: {best_trial.number}")
        print(f"Best RMSE: {best_trial.value:.2f}")
        print("Best Params:", best_trial.params)
        
        mlflow.log_metric("best_rmse", best_trial.value)
        mlflow.log_dict(best_trial.params, "best_params.json")
        
        # Train final best model
        print("\nTraining final model with best parameters...")
        best_model = build_model(args.model_type, best_trial.params, args.random_state)
        best_rmse, best_r2 = evaluate(best_model, X_train, y_train, X_test, y_test)
        
        # Save and log the final model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/best_{args.model_type.lower()}.pkl"
        joblib.dump(best_model, model_path)
        
        mlflow.sklearn.log_model(best_model, artifact_path="model")
        print(f"Final model saved to {model_path} and logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization Pipeline")
    
    # Data paths
    parser.add_argument("--train_path", type=str, required=True, help="Path to prepared train.csv")
    parser.add_argument("--test_path", type=str, required=True, help="Path to prepared test.csv")
    
    # Optimization parameters
    parser.add_argument("--model_type", type=str, choices=["RandomForest", "XGBoost"], default="XGBoost", help="Algorithm to optimize")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials")
    
    # Execution parameters
    parser.add_argument("--random_state", type=int, default=42, help="Seed for reproducibility")
    
    args = parser.parse_args()
    main(args)