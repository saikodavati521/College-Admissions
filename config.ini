"""Central Configuration File for College Admissions Model Project.

This file contains all shared constants, names, and configuration values
used across the project. Import from this file instead of duplicating values.

Usage:
    from config import MODEL_NAME, EXPERIMENT_NAME, TRAIN_DATA_NAME
"""

import json

# ============================================================================
# AZURE ML RESOURCE NAMES
# ============================================================================

# Model and Artifact Names
MODEL_NAME = "admissions_model"  # MLflow artifact path name
REGISTERED_MODEL_NAME = "admissions_model"  # Registered model name in Azure ML

# Environment Name
ENVIRONMENT_NAME = "admissions_environment"

# Data Asset Names
TRAIN_DATA_NAME = "admissions-train-data"
TEST_DATA_NAME = "admissions-test-data"

# Compute Target
COMPUTE_TARGET = "admissions-compute"

# Endpoint and Deployment Names
ONLINE_ENDPOINT_NAME = "admissions-endpoint"
DEPLOYMENT_NAME = "admissions-deployment"


# ============================================================================
# EXPERIMENT NAMES
# ============================================================================

# Training Experiment
TRAINING_EXPERIMENT_NAME = "model_training"

# Responsible AI Experiment
RAI_EXPERIMENT_NAME = "Responsible_AI_Insights_Admissions_Model"


# ============================================================================
# MODEL FEATURES AND SCHEMA
# ============================================================================

# Target Column
TARGET_COLUMN = "Accept"

# Required Input Columns (for inference)
REQUIRED_COLUMNS = [
    'GPA', 'SAT', 'Age', 'Gender',
    'EssayScore', 'InterviewScore', 'ExtracurricularScore', 
    'RecommendationScore',
    'LegacyStatus', 'FinancialAid', 'FirstGeneration',
    'Race_American_Indian_or_Alaska_Native', 'Race_Asian', 
    'Race_Black_or_African_American',
    'Race_Native_Hawaiian_or_Other_Pacific_Islander', 'Race_White', 
    'Ethnicity_Hispanic_or_Latino'
]

# Categorical Features (for RAI Dashboard)
CATEGORICAL_FEATURES = [
    "Gender",
    "Race_American_Indian_or_Alaska_Native",
    "Race_Asian",
    "Race_Black_or_African_American",
    "Race_Native_Hawaiian_or_Other_Pacific_Islander",
    "Race_White",
    "Ethnicity_Hispanic_or_Latino",
    "LegacyStatus",
    "FirstGeneration"
]

# Treatment Features (for RAI Causal Analysis)
TREATMENT_FEATURES = [
    "GPA", "SAT", "Gender",
    "Race_American_Indian_or_Alaska_Native", "Race_Asian", 
    "Race_Black_or_African_American",
    "Race_Native_Hawaiian_or_Other_Pacific_Islander", "Race_White", 
    "Ethnicity_Hispanic_or_Latino",
    "LegacyStatus", "FirstGeneration"
]

# Sensitive Features (for fairness validation)
SENSITIVE_FEATURES = [
    "Gender",
    "Race_American_Indian_or_Alaska_Native",
    "Race_Asian",
    "Race_Black_or_African_American",
    "Race_Native_Hawaiian_or_Other_Pacific_Islander",
    "Race_White",
    "Ethnicity_Hispanic_or_Latino"
]

# Race/Ethnicity Features (subset of sensitive features)
RACE_FEATURES = [
    "Race_American_Indian_or_Alaska_Native",
    "Race_Asian",
    "Race_Black_or_African_American",
    "Race_Native_Hawaiian_or_Other_Pacific_Islander",
    "Race_White",
    "Ethnicity_Hispanic_or_Latino"
]


# ============================================================================
# HYPERPARAMETERS (Default Values)
# ============================================================================

# RandomForestClassifier Hyperparameters
DEFAULT_N_ESTIMATORS = 1100
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_JOBS = -1
DEFAULT_MIN_SAMPLES_SPLIT = 20
DEFAULT_MIN_SAMPLES_LEAF = 10
DEFAULT_MAX_FEATURES = "sqrt"


# ============================================================================
# RESPONSIBLE AI CONFIGURATION
# ============================================================================

# SHAP Threshold for Sensitive Features
SHAP_THRESHOLD = 0.4  # Maximum absolute SHAP value for sensitive features

# Race Feature Distribution Threshold
RACE_STDEV_THRESHOLD = 2.0  # Number of standard deviations for outlier detection

# RAI Dashboard Label
RAI_LABEL = "latest"

# RAI Component Timeout (seconds)
RAI_TIMEOUT = 7200  # 2 hours

# RAI Output Download Path
RAI_DOWNLOAD_PATH = "./rai_outputs"
RAI_OUTPUT_NAME = "ux_json"


# ============================================================================
# DEPLOYMENT CONFIGURATION
# ============================================================================

# Instance Type for Online Deployment
DEPLOYMENT_INSTANCE_TYPE = "Standard_D2as_v4"

# Instance Count
DEPLOYMENT_INSTANCE_COUNT = 1

# Enable Application Insights
DEPLOYMENT_APP_INSIGHTS_ENABLED = True


# ============================================================================
# JSON SERIALIZED VERSIONS (for Azure ML components)
# ============================================================================

# These are pre-serialized for use in Azure ML components that require JSON strings
CATEGORICAL_FEATURES_JSON = json.dumps(CATEGORICAL_FEATURES)
TREATMENT_FEATURES_JSON = json.dumps(TREATMENT_FEATURES)
REQUIRED_COLUMNS_JSON = json.dumps(REQUIRED_COLUMNS)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_latest_version(ml_client, asset_type, asset_name):
    """
    Get the latest version of an Azure ML asset (model, data, environment).
    
    Args:
        ml_client: Azure ML client instance
        asset_type: Type of asset ('model', 'data', 'environment')
        asset_name: Name of the asset
        
    Returns:
        int: Latest version number
    """
    if asset_type == 'model':
        versions = [int(m.version) for m in ml_client.models.list(name=asset_name)]
    elif asset_type == 'data':
        versions = [int(d.version) for d in ml_client.data.list(name=asset_name)]
    elif asset_type == 'environment':
        versions = [int(e.version) for e in ml_client.environments.list(name=asset_name)]
    else:
        raise ValueError(f"Unknown asset type: {asset_type}")
    
    return max(versions) if versions else None


def get_azureml_uri(asset_type, asset_name, version):
    """
    Construct an Azure ML URI for an asset.
    
    Args:
        asset_type: Type of asset ('model', 'data', 'environment')
        asset_name: Name of the asset
        version: Version number or 'latest'
        
    Returns:
        str: Azure ML URI (e.g., 'azureml:model_name:1')
    """
    return f"azureml:{asset_name}:{version}"


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate that all configuration values are properly set."""
    errors = []
    
    # Check that all required names are non-empty
    if not MODEL_NAME:
        errors.append("MODEL_NAME is not set")
    if not ENVIRONMENT_NAME:
        errors.append("ENVIRONMENT_NAME is not set")
    if not TRAIN_DATA_NAME:
        errors.append("TRAIN_DATA_NAME is not set")
    if not TEST_DATA_NAME:
        errors.append("TEST_DATA_NAME is not set")
    
    # Check that feature lists are not empty
    if not REQUIRED_COLUMNS:
        errors.append("REQUIRED_COLUMNS is empty")
    if not CATEGORICAL_FEATURES:
        errors.append("CATEGORICAL_FEATURES is empty")
    
    # Check that hyperparameters are valid
    if DEFAULT_N_ESTIMATORS <= 0:
        errors.append("DEFAULT_N_ESTIMATORS must be positive")
    if DEFAULT_MIN_SAMPLES_SPLIT < 2:
        errors.append("DEFAULT_MIN_SAMPLES_SPLIT must be >= 2")
    if DEFAULT_MIN_SAMPLES_LEAF < 1:
        errors.append("DEFAULT_MIN_SAMPLES_LEAF must be >= 1")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


# Run validation when module is imported
if __name__ != "__main__":
    validate_config()


# ============================================================================
# DISPLAY CONFIGURATION (for debugging)
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("COLLEGE ADMISSIONS MODEL - CONFIGURATION")
    print("="*80)
    print("\n📦 AZURE ML RESOURCES:")
    print(f"  Model Name: {MODEL_NAME}")
    print(f"  Environment: {ENVIRONMENT_NAME}")
    print(f"  Train Data: {TRAIN_DATA_NAME}")
    print(f"  Test Data: {TEST_DATA_NAME}")
    print(f"  Compute: {COMPUTE_TARGET}")
    print(f"  Endpoint: {ONLINE_ENDPOINT_NAME}")
    print(f"  Deployment: {DEPLOYMENT_NAME}")
    
    print("\n🧪 EXPERIMENTS:")
    print(f"  Training: {TRAINING_EXPERIMENT_NAME}")
    print(f"  RAI: {RAI_EXPERIMENT_NAME}")
    
    print("\n📊 MODEL SCHEMA:")
    print(f"  Target Column: {TARGET_COLUMN}")
    print(f"  Required Columns: {len(REQUIRED_COLUMNS)} features")
    print(f"  Categorical Features: {len(CATEGORICAL_FEATURES)} features")
    print(f"  Sensitive Features: {len(SENSITIVE_FEATURES)} features")
    
    print("\n⚙️ HYPERPARAMETERS:")
    print(f"  n_estimators: {DEFAULT_N_ESTIMATORS}")
    print(f"  random_state: {DEFAULT_RANDOM_STATE}")
    print(f"  min_samples_split: {DEFAULT_MIN_SAMPLES_SPLIT}")
    print(f"  min_samples_leaf: {DEFAULT_MIN_SAMPLES_LEAF}")
    print(f"  max_features: {DEFAULT_MAX_FEATURES}")
    
    print("\n🛡️ RESPONSIBLE AI:")
    print(f"  SHAP Threshold: {SHAP_THRESHOLD}")
    print(f"  Race StdDev Threshold: {RACE_STDEV_THRESHOLD}")
    print(f"  RAI Timeout: {RAI_TIMEOUT}s")
    
    print("\n✅ Configuration validated successfully!")
    print("="*80)
