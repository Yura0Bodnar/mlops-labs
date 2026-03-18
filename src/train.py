import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main(args):
    print(f"Loading prepared data from: {args.train_path} and {args.test_path}")
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    
    target_col = 'Rent'

    # Separate features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    mlflow.set_experiment("House_Rent_Prediction_Lab1")

    print(f"Training {args.model_type} with n_estimators={args.n_estimators}, max_depth={args.max_depth}...")
    run_name = f"{args.model_type}_CLI_Run"
    
    with mlflow.start_run(run_name=run_name):
        
        # Set MLflow tags
        mlflow.set_tag("author", "yura")
        mlflow.set_tag("model_type", args.model_type)
        mlflow.set_tag("dataset_version", "prepared_split")
        
        # Initialize model
        if args.model_type == "RandomForest":
            model = RandomForestRegressor(
                n_estimators=args.n_estimators, 
                max_depth=args.max_depth, 
                random_state=args.random_state,
                n_jobs=args.n_jobs
            )
        elif args.model_type == "XGBoost":
            model = XGBRegressor(
                n_estimators=args.n_estimators, 
                max_depth=args.max_depth, 
                learning_rate=args.learning_rate,
                random_state=args.random_state,
                n_jobs=args.n_jobs
            )
        else:
            raise ValueError(f"Unsupported model_type: {args.model_type}")

        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")
        
        # Log parameters and metrics
        logged_params = {
            "model_type": args.model_type,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "random_state": args.random_state,
            "n_jobs": args.n_jobs
        }
        
        if args.model_type == "XGBoost":
            logged_params["learning_rate"] = args.learning_rate
            
        mlflow.log_params(logged_params)
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2_score": r2})
        mlflow.sklearn.log_model(model, f"{args.model_type.lower()}_model")
        
        # Generate and log feature importance plot
        feature_importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)[:10]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importances.values, y=feature_importances.index, palette="viridis")
        plt.title(f"Top 10 Feature Importances ({args.model_type})")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        
        plot_filename = f"feature_importance_{args.model_type}.png"
        plt.savefig(plot_filename)
        plt.close() 
        
        mlflow.log_artifact(plot_filename)
        
        metrics_dict = {
            "rmse": float(rmse), 
            "mae": float(mae), 
            "r2": float(r2)
        }
        with open("metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
            
        joblib.dump(model, "model.pkl")
        
        print(f"Artifacts (model.pkl, metrics.json, {plot_filename}) saveds for CI/CD!")
        # ---------------------------------------
        
    print(f"Training and logging for {args.model_type} completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Training Pipeline")
    
    # Data paths
    parser.add_argument("--train_path", type=str, required=True, help="Path to prepared train.csv")
    parser.add_argument("--test_path", type=str, required=True, help="Path to prepared test.csv")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, choices=["RandomForest", "XGBoost"], default="RandomForest", help="Algorithm selection")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of boosting rounds or trees")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum tree depth")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate (XGBoost only)")
    
    # Execution parameters
    parser.add_argument("--random_state", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--n_jobs", type=int, default=2, help="Number of parallel threads")
    
    args = parser.parse_args()
    main(args)