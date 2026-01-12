
"""
College Admissions Classification Model Training Script.

This script trains a Random Forest classifier to predict college admission decisions
based on applicant data. It loads training and testing datasets, preprocesses the data,
trains the model, evaluates its performance, and saves the trained model.

"""

import os
import mlflow
import argparse
import numpy as np
import pandas as pd
import mltable
import mlflow.sklearn
from pathlib import Path
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def parse_args():
    """Parse command line arguments."""
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
    parser.add_argument( #This can be deleted or needs to be updated so that I can add it to command_job.py
        "--max_depth",
        type=int,
        default=None,
        help="Maximum depth of the trees"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )
    parser.add_argument(
        "--artifact_path_name",
        type=str,
        help="Name of the registered model"
    )
    return parser.parse_args()


def load_data(train_path, test_path):
    """
    Load training and testing datasets from MLTable data assets.
    
    Args:
        train_path (str): Path to training data MLTable directory
        test_path (str): Path to test data MLTable directory
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test) - training and testing data splits
    """
    # Convert to absolute paths if relative paths are provided
    # This ensures the script works regardless of execution directory
    train_path_obj = Path(train_path)
    if not train_path_obj.is_absolute():
        # Get the script's directory and construct absolute path from project root
        script_dir = Path(__file__).parent.parent
        train_path_obj = script_dir / train_path
    
    test_path_obj = Path(test_path)
    if not test_path_obj.is_absolute():
        script_dir = Path(__file__).parent.parent
        test_path_obj = script_dir / test_path
    
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
    """
    Train a Random Forest classifier.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (np.array): Training labels
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of the trees
        random_state (int): Random state for reproducibility
        
    Returns:
        RandomForestClassifier: Trained model
    """
    print(f"Training Random Forest classifier with {n_estimators} trees...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model (can be a Pipeline or classifier)
        X_test (pd.DataFrame): Test features
        y_test (np.array): Test labels
        
    Returns:
        dict: Dictionary of evaluation metrics
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
    """Main function to orchestrate the training process."""
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
        "max_depth": args.max_depth if args.max_depth is not None else "None",
        "random_state": args.random_state
    }
    mlflow.log_params(params)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data(
        args.train_data, args.test_data
    )
    
    # Define numerical and categorical features
    numerical_features = [
        'GPA', 'SAT', 'Age', 'ExtracurricularScore', 
        'EssayScore', 'RecommendationScore', 'InterviewScore', 'FinancialAid'
    ]
    
    # Get categorical features (all columns except numerical ones and target)
    categorical_features = [col for col in X_train.columns if col not in numerical_features]
    
    # Create preprocessing pipeline using ColumnTransformer
    # This scales numerical features and passes through categorical features unchanged
    print("Creating preprocessing pipeline with ColumnTransformer...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', 'passthrough', categorical_features)
        ],
        verbose_feature_names_out=False
    )
    
    # Create a pipeline that combines preprocessing and classification
    # This allows the model to handle raw input data directly
    print("Creating full pipeline with preprocessor and classifier...")
    pipeline_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1
        ))
    ])
    
    # Fit the entire pipeline on raw training data
    print(f"Training pipeline with {args.n_estimators} trees...")
    pipeline_model.fit(X_train, y_train)
    
    # Evaluate model using the pipeline on raw test data
    # This ensures the evaluation matches how the model will be used in production
    metrics = evaluate_model(pipeline_model, X_test, y_test)
    
    # Log metrics to MLflow
    print("Logging metrics to MLflow...")
    mlflow.log_metrics(metrics)
    
    # Log model to MLflow manually
    print("Logging model to MLflow...")
    
    # Generate model signature using RAW input data and pipeline model predictions
    # This is critical for the Responsible AI Dashboard to work correctly
    signature = infer_signature(X_test, pipeline_model.predict(X_test))
    
    # Generate a sample input example using RAW input data
    # This shows consumers of the model what format the input should be in
    input_example = X_test.iloc[0:1]
    
    # Log the PIPELINE model with signature and input example
    print(f"Logging pipeline model to MLflow with artifact path: {args.artifact_path_name}")
    mlflow.sklearn.log_model(
        sk_model=pipeline_model,  # Log the pipeline instead of just the classifier
        signature=signature,      # Signature based on raw input data
        input_example=input_example,  # Example using raw input data
        artifact_path=args.artifact_path_name,
    )
    
    # End the MLflow run
    mlflow.end_run()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
