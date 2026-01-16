import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Set random seed for reproducibility
np.random.seed(42)

# Define dataset parameters
n_samples =7500# Total number of samples
train_output_file = parent_dir / "model_training" / "train_data" / "train_admissions_dataset.csv"
test_output_file = parent_dir / "model_training" / "test_data" / "test_admissions_dataset.csv"
test_size = 0.2  # 20% for test set, 80% for train set

# Define feature distributions
# For GPA: Range 2.0-4.0, with most students between 2.5-3.8
# For SAT: Range 800-1600, with most students between 1000-1400
# For Age: Range 17-25, with most students between 17-19

# Define categories for sensitive attributes
genders = ["Male", "Female"]
race_ethnicity_categories = [
    "American_Indian_or_Alaska_Native",
    "Asian",
    "Black_or_African_American",
    "Native_Hawaiian_or_Other_Pacific_Islander",
    "White",
    "Hispanic_or_Latino"
]

# Create equal representation for sensitive features
n_per_gender = n_samples // len(genders)
n_per_race_ethnicity = n_samples // len(race_ethnicity_categories)

# Generate data with equal representation
gender_list = []
race_ethnicity_list = []

for gender in genders:
    gender_list.extend([gender] * n_per_gender)

for category in race_ethnicity_categories:
    race_ethnicity_list.extend([category] * n_per_race_ethnicity)

# Adjust lists to match n_samples exactly
if len(gender_list) < n_samples:
    gender_list.extend([genders[0]] * (n_samples - len(gender_list)))
elif len(gender_list) > n_samples:
    gender_list = gender_list[:n_samples]

if len(race_ethnicity_list) < n_samples:
    race_ethnicity_list.extend([race_ethnicity_categories[0]] * (n_samples - len(race_ethnicity_list)))
elif len(race_ethnicity_list) > n_samples:
    race_ethnicity_list = race_ethnicity_list[:n_samples]

# Shuffle the lists
np.random.shuffle(gender_list)
np.random.shuffle(race_ethnicity_list)

# Create Gender feature: 0=Female, 1=Male
gender = np.zeros(n_samples, dtype=int)

for i, gender_str in enumerate(gender_list):
    if gender_str == "Male":
        gender[i] = 1
    else:  # Female
        gender[i] = 0

# Create one-hot encoded features for Race and Ethnicity
race_american_indian_or_alaska_native = np.zeros(n_samples, dtype=int)
race_asian = np.zeros(n_samples, dtype=int)
race_black_or_african_american = np.zeros(n_samples, dtype=int)
race_native_hawaiian_or_other_pacific_islander = np.zeros(n_samples, dtype=int)
race_white = np.zeros(n_samples, dtype=int)
ethnicity_hispanic_or_latino = np.zeros(n_samples, dtype=int)

for i, category in enumerate(race_ethnicity_list):
    if category == "American_Indian_or_Alaska_Native":
        race_american_indian_or_alaska_native[i] = 1
    elif category == "Asian":
        race_asian[i] = 1
    elif category == "Black_or_African_American":
        race_black_or_african_american[i] = 1
    elif category == "Native_Hawaiian_or_Other_Pacific_Islander":
        race_native_hawaiian_or_other_pacific_islander[i] = 1
    elif category == "White":
        race_white[i] = 1
    elif category == "Hispanic_or_Latino":
        ethnicity_hispanic_or_latino[i] = 1

# Generate other features
# Generate GPA with only two decimal places
gpa = np.clip(np.random.normal(3.2, 0.5, n_samples), 2.0, 4.0)
gpa = np.round(gpa, 2)  # Round to 2 decimal places
sat = np.clip(np.random.normal(1200, 150, n_samples), 800, 1600).astype(int)
age = np.random.choice(range(17, 26), n_samples, p=[0.3, 0.4, 0.15, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01])

# Generate additional features colleges might consider
extracurricular_score = np.clip(np.random.normal(7, 2, n_samples), 1, 10).astype(int)
essay_score = np.clip(np.random.normal(7, 2, n_samples), 1, 10).astype(int)
recommendation_score = np.clip(np.random.normal(7, 2, n_samples), 1, 10).astype(int)
interview_score = np.clip(np.random.normal(7, 2, n_samples), 1, 10).astype(int)
# Generate categorical features with one-hot encoding (No=0, Yes=1)
legacy_status = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])  # 0=No, 1=Yes
first_generation = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 0=No, 1=Yes

# Create string versions for display/debugging if needed
legacy_status_str = np.array(['No', 'Yes'])[legacy_status]
first_generation_str = np.array(['No', 'Yes'])[first_generation]

financial_aid = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1])  # 0=None, 1=Low, 2=Medium, 3=High

# Create base dataframe (without admission decision yet)
df = pd.DataFrame({
    'GPA': gpa,
    'SAT': sat,
    'Age': age,
    'Gender': gender,  # 0=Female, 1=Male
    'EssayScore': essay_score,
    'InterviewScore': interview_score,
    'ExtracurricularScore': extracurricular_score,
    'RecommendationScore': recommendation_score,
    'LegacyStatus': legacy_status,  # One-hot encoded: 0=No, 1=Yes
    'FinancialAid': financial_aid,
    'FirstGeneration': first_generation,  # One-hot encoded: 0=No, 1=Yes
    'Race_American_Indian_or_Alaska_Native': race_american_indian_or_alaska_native,
    'Race_Asian': race_asian,
    'Race_Black_or_African_American': race_black_or_african_american,
    'Race_Native_Hawaiian_or_Other_Pacific_Islander': race_native_hawaiian_or_other_pacific_islander,
    'Race_White': race_white,
    'Ethnicity_Hispanic_or_Latino': ethnicity_hispanic_or_latino
})

# Create a combined stratification column for both gender and race/ethnicity
# This ensures equal representation of all demographic groups in train and test sets
race_ethnicity_label = []
for i in range(len(df)):
    if df.loc[i, 'Race_American_Indian_or_Alaska_Native'] == 1:
        race_ethnicity_label.append('American_Indian_or_Alaska_Native')
    elif df.loc[i, 'Race_Asian'] == 1:
        race_ethnicity_label.append('Asian')
    elif df.loc[i, 'Race_Black_or_African_American'] == 1:
        race_ethnicity_label.append('Black_or_African_American')
    elif df.loc[i, 'Race_Native_Hawaiian_or_Other_Pacific_Islander'] == 1:
        race_ethnicity_label.append('Native_Hawaiian_or_Other_Pacific_Islander')
    elif df.loc[i, 'Race_White'] == 1:
        race_ethnicity_label.append('White')
    elif df.loc[i, 'Ethnicity_Hispanic_or_Latino'] == 1:
        race_ethnicity_label.append('Hispanic_or_Latino')

df['Race_Ethnicity_Label'] = race_ethnicity_label

# Create combined stratification key: Gender_RaceEthnicity
# This ensures both gender AND race/ethnicity proportions are maintained
df['Stratify_Key'] = df['Gender'].astype(str) + '_' + df['Race_Ethnicity_Label']

# Split data into train and test sets BEFORE applying admission decisions
# This ensures both sets have representative samples
train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['Stratify_Key'])

# Reset indices for both dataframes
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Drop the temporary stratification columns (not needed in final datasets)
train_df = train_df.drop(columns=['Race_Ethnicity_Label', 'Stratify_Key'])
test_df = test_df.drop(columns=['Race_Ethnicity_Label', 'Stratify_Key'])

def apply_admission_logic(dataset):
    """
    Apply admission logic with Female and Asian biases to a dataset.
    This ensures the same unfairness is represented in both train and test sets.
    """
    # Calculate base admission probability based on academic and other factors
    # Scale features for probability calculation
    scaler = StandardScaler()
    academic_features = scaler.fit_transform(dataset[['GPA', 'SAT', 'ExtracurricularScore', 'EssayScore', 
                                               'RecommendationScore', 'InterviewScore']].values)
    
    # Base probability calculation (weighted sum of features)
    base_prob = (
        0.3 * academic_features[:, 0] +  # GPA
        0.25 * academic_features[:, 1] +  # SAT
        0.1 * academic_features[:, 2] +  # Extracurricular
        0.1 * academic_features[:, 3] +  # Essay
        0.1 * academic_features[:, 4] +  # Recommendation
        0.1 * academic_features[:, 5] +  # Interview
        0.05 * dataset['LegacyStatus']  # Legacy bonus (applied when LegacyStatus=1)
    )
    
    # Normalize to 0-1 range
    base_prob = (base_prob - base_prob.min()) / (base_prob.max() - base_prob.min())
    
    # Introduce unfairness for Gender (females slightly more likely to be accepted)
    gender_bias = np.zeros(len(dataset))
    gender_bias[dataset['Gender'] == 0] = 0.001  # 10% boost for females (Gender=0)
    
    # Introduce unfairness for Race (If you are Asian you are slightly more likely to be accepted)
    race_bias = np.zeros(len(dataset))
    race_bias[dataset['Race_Asian'] == 1] = 0.001  # 10% boost for Asians
    
    # Combine base probability with biases
    final_prob = np.clip(base_prob + gender_bias + race_bias, 0, 1)
    
    # Generate admission decision based on final probability
    admission_threshold = 0.5
    admission_decision = (final_prob >= admission_threshold).astype(int)
    
    return admission_decision

# Apply admission logic to both train and test sets
train_df['Accept'] = apply_admission_logic(train_df)
test_df['Accept'] = apply_admission_logic(test_df)

# Print statistics for both datasets
print("="*80)
print("TRAIN DATASET STATISTICS")
print("="*80)
print(f"\nTotal train samples: {len(train_df)}")
print(f"Train admission rate: {train_df['Accept'].mean():.4f}")

print("\nTrain Gender distribution:")
print(f"Male (1): {(train_df['Gender'] == 1).sum()}")
print(f"Female (0): {(train_df['Gender'] == 0).sum()}")

print("\nTrain Race/Ethnicity distribution:")
print(f"American Indian or Alaska Native: {train_df['Race_American_Indian_or_Alaska_Native'].sum()}")
print(f"Asian: {train_df['Race_Asian'].sum()}")
print(f"Black or African American: {train_df['Race_Black_or_African_American'].sum()}")
print(f"Native Hawaiian or Other Pacific Islander: {train_df['Race_Native_Hawaiian_or_Other_Pacific_Islander'].sum()}")
print(f"White: {train_df['Race_White'].sum()}")
print(f"Hispanic or Latino (Ethnicity): {train_df['Ethnicity_Hispanic_or_Latino'].sum()}")

print("\nTrain Admission rates by gender:")
print(f"Male (1): {train_df[train_df['Gender'] == 1]['Accept'].mean():.4f}")
print(f"Female (0): {train_df[train_df['Gender'] == 0]['Accept'].mean():.4f}")
print(f"Female advantage: {train_df[train_df['Gender'] == 0]['Accept'].mean() - train_df[train_df['Gender'] == 1]['Accept'].mean():.4f}")

print("\nTrain Admission rates by race/ethnicity:")
print(f"American Indian or Alaska Native: {train_df[train_df['Race_American_Indian_or_Alaska_Native'] == 1]['Accept'].mean():.4f}")
print(f"Asian: {train_df[train_df['Race_Asian'] == 1]['Accept'].mean():.4f}")
print(f"Black or African American: {train_df[train_df['Race_Black_or_African_American'] == 1]['Accept'].mean():.4f}")
print(f"Native Hawaiian or Other Pacific Islander: {train_df[train_df['Race_Native_Hawaiian_or_Other_Pacific_Islander'] == 1]['Accept'].mean():.4f}")
print(f"White: {train_df[train_df['Race_White'] == 1]['Accept'].mean():.4f}")
print(f"Hispanic or Latino (Ethnicity): {train_df[train_df['Ethnicity_Hispanic_or_Latino'] == 1]['Accept'].mean():.4f}")

print("\n" + "="*80)
print("TEST DATASET STATISTICS")
print("="*80)
print(f"\nTotal test samples: {len(test_df)}")
print(f"Test admission rate: {test_df['Accept'].mean():.4f}")

print("\nTest Gender distribution:")
print(f"Male (1): {(test_df['Gender'] == 1).sum()}")
print(f"Female (0): {(test_df['Gender'] == 0).sum()}")

print("\nTest Race/Ethnicity distribution:")
print(f"American Indian or Alaska Native: {test_df['Race_American_Indian_or_Alaska_Native'].sum()}")
print(f"Asian: {test_df['Race_Asian'].sum()}")
print(f"Black or African American: {test_df['Race_Black_or_African_American'].sum()}")
print(f"Native Hawaiian or Other Pacific Islander: {test_df['Race_Native_Hawaiian_or_Other_Pacific_Islander'].sum()}")
print(f"White: {test_df['Race_White'].sum()}")
print(f"Hispanic or Latino (Ethnicity): {test_df['Ethnicity_Hispanic_or_Latino'].sum()}")

print("\nTest Admission rates by gender:")
print(f"Male (1): {test_df[test_df['Gender'] == 1]['Accept'].mean():.4f}")
print(f"Female (0): {test_df[test_df['Gender'] == 0]['Accept'].mean():.4f}")
print(f"Female advantage: {test_df[test_df['Gender'] == 0]['Accept'].mean() - test_df[test_df['Gender'] == 1]['Accept'].mean():.4f}")

print("\nTest Admission rates by race/ethnicity:")
print(f"American Indian or Alaska Native: {test_df[test_df['Race_American_Indian_or_Alaska_Native'] == 1]['Accept'].mean():.4f}")
print(f"Asian: {test_df[test_df['Race_Asian'] == 1]['Accept'].mean():.4f}")
print(f"Black or African American: {test_df[test_df['Race_Black_or_African_American'] == 1]['Accept'].mean():.4f}")
print(f"Native Hawaiian or Other Pacific Islander: {test_df[test_df['Race_Native_Hawaiian_or_Other_Pacific_Islander'] == 1]['Accept'].mean():.4f}")
print(f"White: {test_df[test_df['Race_White'] == 1]['Accept'].mean():.4f}")
print(f"Hispanic or Latino (Ethnicity): {test_df[test_df['Ethnicity_Hispanic_or_Latino'] == 1]['Accept'].mean():.4f}")

# Save datasets to CSV
train_df.to_csv(train_output_file, index=False)
test_df.to_csv(test_output_file, index=False)

print("\n" + "="*80)
print(f"Train dataset saved to {train_output_file}")
print(f"Test dataset saved to {test_output_file}")
print(f"Total samples: {n_samples} (Train: {len(train_df)}, Test: {len(test_df)})")
print("="*80)
