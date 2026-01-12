"""
Split the admissions dataset into training and test sets.

This script ensures equal representation of sensitive features (Gender and Race)
in both training and test sets.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# File paths
INPUT_FILE = "full_admissions_dataset.csv"
TRAIN_OUTPUT = "G:\\My Drive\\Admissions Model\\model_training\\train_data\\train_admissions_dataset.csv"
TEST_OUTPUT = "G:\\My Drive\\Admissions Model\\model_training\\test_data\\test_admissions_dataset.csv"

# Test set size (percentage)
TEST_SIZE = 0.2

# Read the full dataset
print(f"Reading dataset from {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
print(f"Total samples: {len(df)}")

# Define gender and race column names based on one-hot encoding
GENDER_COLUMNS = ['Gender_Male', 'Gender_Female']
RACE_COLUMNS = ['Race_Asian', 'Race_Black', 'Race_Hispanic', 'Race_White', 'Race_Other']

def stratified_split_with_equal_representation(data, test_percentage):
    """
    Split the dataset into train and test sets while maintaining equal representation
    of sensitive features (Gender and Race) in both sets.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The input dataset with one-hot encoded features.
    test_percentage : float
        The proportion of the dataset to include in the test split.
        
    Returns
    -------
    tuple
        (train_data, test_data) - Split datasets with equal representation.
    """
    # Create empty train and test dataframes
    train_data = pd.DataFrame(columns=data.columns)
    test_data = pd.DataFrame(columns=data.columns)
    
    # Process gender groups (Male/Female)
    for gender_col in GENDER_COLUMNS:
        # Get subset of data for this gender
        gender_subset = data[data[gender_col] == 1]
        
        if len(gender_subset) > 0:
            # For each gender, ensure equal representation across races
            for race_col in RACE_COLUMNS:
                # Get subset for this gender and race combination
                race_subset = gender_subset[gender_subset[race_col] == 1]
                
                if len(race_subset) > 0:
                    # Split this subset
                    subset_train, subset_test = train_test_split(
                        race_subset, 
                        test_size=test_percentage,
                        random_state=42,
                        stratify=race_subset['Admitted']  # Stratify by admission decision
                    )
                    
                    # Add to respective datasets
                    train_data = pd.concat([train_data, subset_train])
                    test_data = pd.concat([test_data, subset_test])
    
    return train_data, test_data

def main():
    """Execute the main dataset splitting process."""
    # Split the data
    print("Splitting data into train and test sets...")
    train_df, test_df = stratified_split_with_equal_representation(df, TEST_SIZE)
    
    # Shuffle the datasets
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Print statistics
    print(f"\nTrain set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    return train_df, test_df

def print_statistics(train_df, test_df):
    """Print statistics about the train and test datasets.
    
    Parameters
    ----------
    train_df : pandas.DataFrame
        The training dataset.
    test_df : pandas.DataFrame
        The test dataset.
    """
    # Verify balance of sensitive attributes in train set
    print("\nTrain set - Gender distribution:")
    for col in GENDER_COLUMNS:
        print(f"{col}: {train_df[col].sum()}")
    
    print("\nTrain set - Race distribution:")
    for col in RACE_COLUMNS:
        print(f"{col}: {train_df[col].sum()}")
    
    print("\nTrain set - Admission rates by gender:")
    for col in GENDER_COLUMNS:
        print(f"{col}: {train_df[train_df[col] == 1]['Admitted'].mean():.4f}")
    
    print("\nTrain set - Admission rates by race:")
    for col in RACE_COLUMNS:
        print(f"{col}: {train_df[train_df[col] == 1]['Admitted'].mean():.4f}")
    
    # Verify balance of sensitive attributes in test set
    print("\nTest set - Gender distribution:")
    for col in GENDER_COLUMNS:
        print(f"{col}: {test_df[col].sum()}")
    
    print("\nTest set - Race distribution:")
    for col in RACE_COLUMNS:
        print(f"{col}: {test_df[col].sum()}")
    
    print("\nTest set - Admission rates by gender:")
    for col in GENDER_COLUMNS:
        print(f"{col}: {test_df[test_df[col] == 1]['Admitted'].mean():.4f}")
    
    print("\nTest set - Admission rates by race:")
    for col in RACE_COLUMNS:
        print(f"{col}: {test_df[test_df[col] == 1]['Admitted'].mean():.4f}")

def save_datasets(train_df, test_df):
    """Save the train and test datasets to CSV files.
    
    Parameters
    ----------
    train_df : pandas.DataFrame
        The training dataset to save.
    test_df : pandas.DataFrame
        The test dataset to save.
    """
    print(f"\nSaving train set to {TRAIN_OUTPUT}...")
    train_df.to_csv(TRAIN_OUTPUT, index=False)
    
    print(f"Saving test set to {TEST_OUTPUT}...")
    test_df.to_csv(TEST_OUTPUT, index=False)
    
    print("Done!")


if __name__ == "__main__":
    train_df, test_df = main()
    print_statistics(train_df, test_df)
    save_datasets(train_df, test_df)
