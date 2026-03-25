import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main(args):
    print(f"Loading raw data from: {args.input_path}")
    df = pd.read_csv(args.input_path, nrows=args.nrows)

    target_col = "Rent"

    print("Executing data preprocessing pipeline...")
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Drop high-cardinality columns to optimize memory usage
    cols_to_drop = ["Property ID", "Area Locality", "Posted On"]
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns], errors="ignore")

    # Impute missing values
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    # Recombine features and target for saving
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")

    print(f"Saving prepared data to {args.output_dir}...")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Data preparation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preparation Pipeline")

    parser.add_argument("--input_path", type=str, required=True, help="Path to raw CSV data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save prepared data")
    parser.add_argument("--nrows", type=int, default=50000, help="Number of rows to load")
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")

    args = parser.parse_args()
    main(args)
