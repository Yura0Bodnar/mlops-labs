import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor  # <-- Added XGBoost
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # 1. Load data
    data_path = "/home/yura/PolProjects/mlops_labs/mlops_lab_1/data/raw/House_Rent_10M_balanced_40cities.csv"
    print("Loading data...")
    df = pd.read_csv(data_path, nrows=50000)
    target_col = 'Rent'

    # 2. Data Preprocessing
    print("Preprocessing data...")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # DROP high-cardinality columns to avoid memory overload
    cols_to_drop = ['Property ID', 'Area Locality', 'Posted On']
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors='ignore')

    # Handle missing values
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # Encode categorical variables (One-Hot Encoding)
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. MLflow Experiment Setup
    mlflow.set_experiment("House_Rent_Hyperparameter_Tuning")

    # === HYPERPARAMETER TUNING & MODEL COMPARISON ===
    depths = [2, 5, 10, 15, 20, 25]
    
    # Define the models and their base parameters to compare
    models_to_test = {
        "RandomForest": {
            "model_class": RandomForestRegressor,
            "base_params": {"n_estimators": 50, "random_state": 42, "n_jobs": 2}
        },
        "XGBoost": {
            "model_class": XGBRegressor,
            "base_params": {"n_estimators": 50, "learning_rate": 0.1, "random_state": 42, "n_jobs": 2}
        }
    }

    print(f"Starting tuning for models: {list(models_to_test.keys())}")
    print(f"Testing max_depth values: {depths}\n")

    # Iterate over each model type
    for model_name, model_info in models_to_test.items():
        print(f"{'='*40}")
        print(f"Evaluating {model_name}")
        print(f"{'='*40}")
        
        # Iterate over each depth for the current model
        for depth in depths:
            run_name = f"{model_name}_max_depth_{depth}"
            
            with mlflow.start_run(run_name=run_name):
                print(f"---> Training {model_name} with max_depth={depth}...")
                
                # Combine base parameters with current max_depth
                ModelClass = model_info["model_class"]
                params = model_info["base_params"].copy()
                params["max_depth"] = depth
                
                # Initialize and train the model
                model = ModelClass(**params)
                model.fit(X_train, y_train)
                
                # Predictions for TRAINING set
                pred_train = model.predict(X_train)
                rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
                r2_train = r2_score(y_train, pred_train)
                
                # Predictions for TESTING set
                pred_test = model.predict(X_test)
                rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
                r2_test = r2_score(y_test, pred_test)
                
                print(f"Train RMSE: {rmse_train:.2f} | Test RMSE: {rmse_test:.2f}")
                
                # MLflow Logging
                mlflow.set_tag("experiment_type", "tuning")
                mlflow.set_tag("model_type", model_name)
                
                mlflow.log_param("model_type", model_name)
                mlflow.log_params(params)
                
                mlflow.log_metrics({
                    "train_rmse": rmse_train,
                    "test_rmse": rmse_test,
                    "train_r2": r2_train,
                    "test_r2": r2_test
                })

    print("\nTuning and comparison completed! Please check the MLflow UI.")

if __name__ == "__main__":
    main()

