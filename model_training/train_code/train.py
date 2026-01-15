"""College Admissions Classification Model Training Script.

This script trains a Random Forest classifier to predict college admission decisions
based on applicant data. It loads training and testing datasets, preprocesses the data,
trains the model, evaluates its performance, and logs the model to MLflow.

Usage:
    python train.py --train_data <path> --test_data <path> --n_estimators <int>

Command Line Arguments:
    --train_data: Path to training data MLTable directory
    --test_data: Path to test data MLTable directory
    --n_estimators: Number of trees in the random forest
    --random_state: Random state for reproducibility
    --n_jobs: Number of CPU cores to use
    --min_samples_split: Minimum samples required to split a node
    --min_samples_leaf: Minimum samples required at a leaf node
    --max_features: Number of features for best split
    --artifact_path_name: Name of the registered model
"""
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import mltable
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a Random Forest classifier for college admissions."
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="model_training/train_data",
        help="Path to training data MLTable directory"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="model_training/test_data",
        help="Path to test data MLTable directory"
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=1000,
        help="Number of trees in the random forest"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of CPU cores to use for training (-1 uses all available cores)"
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=2,
        help="Minimum number of samples required to split an internal node"
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=1,
        help="Minimum number of samples required to be at a leaf node"
    )
    parser.add_argument(
        "--max_features",
        type=str,
        default="sqrt",
        help="Number of features to consider when looking for the best split (sqrt, log2, or None)"
    )
    parser.add_argument(
        "--artifact_path_name",
        type=str,
        help="Name of the registered model"
    )
    return parser.parse_args()


def load_data(train_path, test_path):
    """Load training and testing datasets from MLTable data assets.

    Args:
        train_path (str): Path to training data MLTable directory.
        test_path (str): Path to test data MLTable directory.

    Returns:
        tuple: (X_train, y_train, X_test, y_test) - Training and testing data splits.
    """
    # Convert to absolute paths if relative paths are provided
    train_path_obj = Path(train_path)
    if not train_path_obj.is_absolute():
        project_root = Path(__file__).parent.parent
        train_path_obj = project_root / train_path

    test_path_obj = Path(test_path)
    if not test_path_obj.is_absolute():
        project_root = Path(__file__).parent.parent
        test_path_obj = project_root / test_path
    
    # Load training data from MLTable
    print(f"Loading training data from MLTable: {train_path_obj}")
    train_tbl = mltable.load(str(train_path_obj))
    train_df = train_tbl.to_pandas_dataframe()

    # Load test data from MLTable
    print(f"Loading test data from MLTable: {test_path_obj}")
    test_tbl = mltable.load(str(test_path_obj))
    test_df = test_tbl.to_pandas_dataframe()

    # Extract target variable
    y_train = train_df["Accept"].values
    y_test = test_df["Accept"].values

    # Drop target variable from features
    X_train = train_df.drop("Accept", axis=1)
    X_test = test_df.drop("Accept", axis=1)

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """Train a Random Forest classifier.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (np.array): Training labels.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of the trees.
        random_state (int): Random state for reproducibility.

    Returns:
        RandomForestClassifier: Trained model.
    """
    print(f"Training Random Forest classifier with {n_estimators} trees...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on test data.

    Args:
        model: Trained model (can be a Pipeline or classifier).
        X_test (pd.DataFrame): Test features.
        y_test (np.array): Test labels.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    # Print metrics
    print("\nModel Performance Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    return metrics


def main():
    """Main function to orchestrate the training process.

    This function:
        1. Parses command line arguments
        2. Sets up MLflow tracking
        3. Loads and preprocesses data
        4. Trains a Random Forest classifier pipeline
        5. Evaluates model performance
        6. Logs model and metrics to MLflow
    """
    # Parse command line arguments
    args = parse_args()

    # Set up MLflow tracking
    mlflow.start_run()

    # Log parameters
    print("Logging parameters to MLflow...")
    params = {
        "train_data": args.train_data,
        "test_data": args.test_data,
        "n_estimators": args.n_estimators,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features
    }
    mlflow.log_params(params)

    # Load data
    X_train, y_train, X_test, y_test = load_data(
        args.train_data, args.test_data
    )
    
    # Define numerical and categorical features
    numerical_features = [
        "GPA", "SAT", "Age", "ExtracurricularScore",
        "EssayScore", "RecommendationScore", "InterviewScore", "FinancialAid"
    ]

    # Get categorical features (all columns except numerical ones)
    categorical_features = [
        col for col in X_train.columns if col not in numerical_features
    ]

    # Create preprocessing pipeline
    print("Creating preprocessing pipeline with ColumnTransformer...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", "passthrough", categorical_features)
        ],
        verbose_feature_names_out=False
    )
    
    # Create a pipeline that combines preprocessing and classification
    print("Creating full pipeline with preprocessor and classifier...")
    pipeline_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features
        ))
    ])

    # Fit the entire pipeline on training data
    print(f"Training pipeline with {args.n_estimators} trees...")
    pipeline_model.fit(X_train, y_train)

    # Evaluate model performance
    metrics = evaluate_model(pipeline_model, X_test, y_test)

    # Log metrics to MLflow
    print("Logging metrics to MLflow...")
    mlflow.log_metrics(metrics)
    
    # Log model to MLflow
    print("Logging model to MLflow...")

    # Generate model signature
    signature = infer_signature(X_test, pipeline_model.predict(X_test))

    # Generate sample input example
    input_example = X_test.iloc[0:1]

    # Log the pipeline model
    print(
        f"Logging pipeline model to MLflow with artifact path: "
        f"{args.artifact_path_name}"
    )
    mlflow.sklearn.log_model(
        sk_model=pipeline_model,
        signature=signature,
        input_example=input_example,
        artifact_path=args.artifact_path_name,
    )

    # End the MLflow run
    mlflow.end_run()

    print("✓ Training completed successfully!")


if __name__ == "__main__":
    main()
